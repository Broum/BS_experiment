"""Backends with one transpile/run API: VLQ (qaas >= 1.0, LEXIS auth), IQM Sirius (Resonance) and FakeVLQ."""

from __future__ import annotations

import os
from pathlib import Path

from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMFakeBackend
from qiskit import QuantumCircuit


_SESSION = None  # one LEXIS login per kernel


def get_token() -> str:
    """LEXIS access token: VLQ_TOKEN env if set, else from a LexisSession (login once, then auto-refreshed).

    The login prints a URL: open it in your browser and log in within ~2 minutes.
    """
    global _SESSION
    if os.environ.get("VLQ_TOKEN"):
        return os.environ["VLQ_TOKEN"]
    if _SESSION is None:
        from py4lexis.session import LexisSession

        _SESSION = LexisSession(reraise_exceptions=True, log_file="lexis_logs.log")
    token = _SESSION.get_access_token()
    if not token:
        _SESSION = None
        raise RuntimeError("LEXIS login failed (no token); rerun and open the printed URL within ~2 minutes")
    return token


def vlq_backend(project: str | None = None, resource: str | None = None, token: str | None = None,
                use_metrics: bool = True):
    """Real VLQ backend. Defaults come from env: VLQ_PROJECT, VLQ_RESOURCE, VLQ_TOKEN.

    Without a token, logs in via :func:`get_token`.
    ``use_metrics`` loads calibration metrics so the transpiler picks good qubits.
    """
    from qheappe_client import QProvider

    provider = QProvider(lexis_project=project or os.environ["VLQ_PROJECT"], token=token or get_token())
    return provider.get_backend(resource or os.environ.get("VLQ_RESOURCE") or None, use_metrics=use_metrics)


SIRIUS_URL = "https://resonance.iqm.tech/"


def sirius_backend(token: str | None = None, use_metrics: bool = False):
    """IQM Sirius via IQM Resonance. Token: argument, else ``.IQM_token`` file, else ``IQM_TOKEN`` env."""
    from iqm.qiskit_iqm import IQMProvider

    if token is None:
        f = Path(__file__).resolve().parent.parent / ".IQM_token"
        token = f.read_text().strip() if f.exists() else os.environ.get("IQM_TOKEN")
    if not token:
        raise RuntimeError("no IQM Resonance token: put it in .IQM_token or IQM_TOKEN")
    for var in ("IQM_TOKEN", "IQM_TOKENS_FILE"):  # iqm-client refuses a token given twice
        os.environ.pop(var, None)
    return IQMProvider(SIRIUS_URL, quantum_computer="sirius", token=token).get_backend(use_metrics=use_metrics)


def get_backend(name: str):
    """'VLQ', 'Sirius' or 'FakeVLQ'."""
    return {"VLQ": vlq_backend, "Sirius": sirius_backend, "FakeVLQ": fake_vlq}[name]()


def fake_vlq() -> IQMFakeBackend:
    """Noisy VLQ simulator from the calibration snapshot in ``utils/fake_vlq_backend.py``."""
    from utils.fake_vlq_backend import fake_vlq_backend

    return fake_vlq_backend()


def transpile(circuits: list[QuantumCircuit], backend, layout: list[str] | None = None, seed: int | None = None,
              optimization_level: int = 3) -> tuple[list[QuantumCircuit], list[str]]:
    """Transpiles all circuits onto the same physical qubits.

    ``layout`` is a list of qubit names (e.g. ``["QB5", "QB7", "QB9", "QB12"]``); if ``None``, the layout chosen
    for the first circuit is reused for the rest. Returns the transpiled circuits and the layout used.
    """
    out = []
    for qc in circuits:
        initial = None if layout is None else [backend.qubit_name_to_index(q) for q in layout]
        t = transpile_to_IQM(qc, backend, initial_layout=initial, optimization_level=optimization_level,
                             seed_transpiler=seed)
        if layout is None:
            layout = [backend.index_to_qubit_name(i) for i in t.layout.initial_index_layout(filter_ancillas=True)]
        out.append(t)
    return out, layout


def submit(backend, transpiled: list[QuantumCircuit], shots: int, seed: int | None = None):
    """Starts one job for all circuits; returns the job (FakeVLQ: the counts directly)."""
    if isinstance(backend, IQMFakeBackend):
        return _run_fake(backend, transpiled, shots, seed)
    job = backend.run(transpiled, shots=shots)
    print(f"{backend.name} job id: {job.job_id()}")
    return job


def collect(job) -> list[dict[str, int]]:
    """Waits for a job from :func:`submit`; counts per circuit."""
    if isinstance(job, list):
        return job
    vlq_job = type(job).__module__.startswith("qheappe_client")
    result = job.result(qiskit_legacy=True) if vlq_job else job.result()
    return [result.get_counts(i) for i in range(len(result.results))]


def run(backend, transpiled: list[QuantumCircuit], shots: int, seed: int | None = None) -> list[dict[str, int]]:
    """Submits and waits: counts per circuit."""
    return collect(submit(backend, transpiled, shots, seed))


def _run_fake(backend: IQMFakeBackend, transpiled: list[QuantumCircuit], shots: int,
              seed: int | None) -> list[dict[str, int]]:
    """Like ``IQMFakeBackend.run`` but seeded and with MOVE noise applied (labelled MOVE unitary)."""
    from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit
    from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY
    from qiskit.circuit.library import UnitaryGate
    from qiskit_aer import AerSimulator

    move = UnitaryGate(MOVE_GATE_UNITARY, label="move")
    circuits = []
    for t in transpiled:
        validate_circuit(t, backend)
        c = t.copy_empty_like()
        for inst in t.data:
            c.append(move if inst.operation.name == "move" else inst.operation, inst.qubits, inst.clbits)
        circuits.append(c)
    result = AerSimulator(noise_model=backend.noise_model).run(circuits, shots=shots, seed_simulator=seed).result()
    return [result.get_counts(i) for i in range(len(circuits))]


def run_ideal(circuits: list[QuantumCircuit], shots: int, seed: int | None = None) -> list[dict[str, int]]:
    """Noise-free shots on Aer (untranspiled logical circuits)."""
    from qiskit import transpile as qk_transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    result = sim.run(qk_transpile(circuits, sim), shots=shots, seed_simulator=seed).result()
    return [result.get_counts(i) for i in range(len(circuits))]
