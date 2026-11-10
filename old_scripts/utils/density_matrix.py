"""Exact noisy outcome probabilities on an ``IQMFakeBackend``.

Mirrors the MOVE-lowering pipeline of ``IQMFakeBackend.run()`` but switches
Aer to ``method='density_matrix'``, yielding the exact (shot-noise-free)
distribution under the backend's calibrated noise model.

Limitations:
- No leakage out of the (qubit, resonator) computational subspace -- same
  limitation ``IQMFakeBackend.run()`` has.
- Memory scales as ``4**num_qubits`` (qubits + resonators as simulated). The
  default cap is 15 total qubits; override with ``max_total_qubits``.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator
from iqm.qiskit_iqm import IQMFakeBackend
from iqm.qiskit_iqm.iqm_backend import IQM_TO_QISKIT_GATE_NAME
from iqm.qiskit_iqm.iqm_transpilation import IQMReplaceGateWithUnitaryPass
from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY

GATE_TO_UNITARY = {"move": MOVE_GATE_UNITARY}


def exact_noisy_probabilities(
    circuit: QuantumCircuit,
    backend: IQMFakeBackend,
    trace_out_resonators: bool = True,
    max_total_qubits: int = 15,
) -> dict[str, float]:
    """Return exact noisy outcome probabilities for ``circuit`` on ``backend``.

    Args:
        circuit: A ``QuantumCircuit`` pre- or post-transpiled against ``backend``.
        backend: An ``IQMFakeBackend`` (e.g. ``FakeVLQ()``).
        trace_out_resonators: If ``True`` (default), partial-trace resonator
            subsystems out of the final density matrix so bitstring keys match
            what ``backend.run(...).get_counts()`` would produce. If ``False``,
            return the distribution over qubits + resonators.
        max_total_qubits: Fail-fast memory guard. Raises ``ValueError`` if
            ``backend.num_qubits`` exceeds this.

    Returns:
        Dict mapping bitstring (Qiskit little-endian) to probability. Sums to 1.
    """
    qc_t = transpile(circuit, backend)
    if qc_t.num_qubits > max_total_qubits:
        raise ValueError(
            f"Transpiled circuit spans {qc_t.num_qubits} qubits (incl. resonators); "
            f"density-matrix simulation is capped at {max_total_qubits} "
            f"(memory ~= 16 * 4**n bytes). "
            f"Override via `max_total_qubits=` or use a smaller backend."
        )

    qc_sub = qc_t
    for g in backend.noise_model.basis_gates:
        if g in IQM_TO_QISKIT_GATE_NAME.values():
            continue
        if g not in GATE_TO_UNITARY:
            raise NotImplementedError(
                f"No unitary substitution registered for IQM-specific gate '{g}'. "
                f"Update GATE_TO_UNITARY in simulation/density_matrix.py."
            )
        qc_sub = IQMReplaceGateWithUnitaryPass(g, GATE_TO_UNITARY[g])(qc_sub)

    qc_sub.remove_final_measurements()
    qc_sub.save_density_matrix()

    sim = AerSimulator(method="density_matrix", noise_model=backend.noise_model)
    rho = sim.run(qc_sub).result().data(0)["density_matrix"]

    if trace_out_resonators:
        resonator_indices = _resonator_indices(backend)
        if resonator_indices:
            rho = partial_trace(rho, resonator_indices)

    return {k: float(v) for k, v in rho.probabilities_dict().items()}


def _resonator_indices(backend: IQMFakeBackend) -> list[int]:
    """Return the integer indices of computational resonators on ``backend``.

    Uses the public ``backend.architecture`` (DynamicQuantumArchitecture) and
    ``backend.qubit_name_to_index`` rather than any mangled private attribute.
    """
    architecture = getattr(backend, "architecture", None)
    if architecture is None:
        return []
    return [backend.qubit_name_to_index(r) for r in architecture.computational_resonators]
