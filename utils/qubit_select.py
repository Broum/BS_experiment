"""Qubit selection from IQM calibration files (VLQ, Sirius): layouts ranked by estimated success probability (ESP)."""

from __future__ import annotations

import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from qiskit import QuantumCircuit

from utils.vlq import transpile

CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "calibration"
GATE_KEY = {"r": "prx", "prx": "prx", "cz": "cz", "move": "move", "measure": "readout"}


def load_calibration(source) -> tuple[dict[str, float], str, str]:
    """``({dut_field: value}, calibration set id, file name)`` from a JSON file, a folder (newest file) or a dict."""
    name = "live"
    if not isinstance(source, dict):
        path = Path(source)
        if path.is_dir():
            path = max(path.glob("*.json"))  # file names carry the timestamp
        source, name = json.loads(path.read_text()), path.name
    obs = {o["dut_field"]: float(o["value"]) for o in source["observations"]
           if not o.get("invalid") and o.get("value") is not None}
    return obs, str(source.get("describes_id")), name


def fetch_vlq_calibration(backend, save: bool = True) -> dict:
    """Current metrics of a real VLQ backend (qaas); saved to ``calibration/VLQ/``. Pass the result to load_calibration."""
    data = backend._qclient.get_machine_metrics()  # pylint: disable=protected-access
    if save:
        path = CALIBRATION_DIR / "VLQ" / f"{datetime.now(timezone.utc):%Y-%m-%dT%H_%M_%S}Z.json"
        path.write_text(json.dumps(data, indent=1))
        print("calibration saved:", path)
    return data


def check_calibration(backend, calibration_set_id: str) -> bool:
    """Warns if the file does not describe the calibration set the machine runs now."""
    current = str(backend.architecture.calibration_set_id)
    if current == calibration_set_id:
        print(f"  calibration file matches the machine ({current})")
        return True
    print(f"  WARNING: file describes calibration set {calibration_set_id}, the machine runs {current}: "
          "download a new calibration file")
    return False


def qubit_errors(obs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Per qubit: error (1 - fidelity) of prx, CZ and one MOVE with the resonator, readout; T1, T2 echo in us.

    The CZ metric of whichever implementation is calibrated (crf_crf, tgss, ...) is used.
    A missing metric counts as error 1, so that qubit is never picked.
    """
    qubits = sorted({m for k in obs for m in re.findall(r"QB\d+", k)}, key=lambda q: int(q[2:]))
    cz = {m[1]: v for k, v in obs.items() if (m := re.fullmatch(r"metrics\.irb\.cz\.\w+\.(QB\d+)__\w+\.fidelity", k))}
    move2 = {m[1]: v for k, v in obs.items()
             if (m := re.fullmatch(r"metrics\.irb\.move\.\w+\.(QB\d+)__\w+\.fidelity:n_interleaved_gates=2", k))}
    out = {}
    for q in qubits:
        p2 = (4 * move2.get(q, 0.0) - 1) / 3  # IRB of 2 moves -> one move (depolarizing polarizations)
        out[q] = {
            "prx": 1 - obs.get(f"metrics.rb.prx.drag_crf.{q}.fidelity:par=d1", 0.0),
            "cz": 1 - cz.get(q, 0.0),
            "move": 1 - (3 * max(p2, 0.0) ** 0.5 + 1) / 4,
            "readout": 1 - obs.get(f"metrics.ssro.measure.constant.{q}.fidelity", 0.0),
            "t1_us": obs.get(f"characterization.model.{q}.t1_time", 0.0) * 1e6,
            "t2_us": obs.get(f"characterization.model.{q}.t2_echo_time", 0.0) * 1e6,
        }
    return out


def score(e: dict[str, float]) -> float:
    """Per-qubit pre-ranking: readout + prx + one CZ through the resonator (cz + 2 moves)."""
    return e["readout"] + e["prx"] + e["cz"] + 2 * e["move"]


def print_errors(errors: dict[str, dict[str, float]], top: int | None = None) -> None:
    """Table of qubits, best (lowest score) first."""
    print(f"{'qubit':6}{'readout':>9}{'prx':>9}{'cz':>9}{'move':>9}{'T1 us':>8}{'T2 us':>8}{'score':>9}")
    for q in sorted(errors, key=lambda q: score(errors[q]))[:top]:
        e = errors[q]
        print(f"{q:6}{e['readout']:9.4f}{e['prx']:9.4f}{e['cz']:9.4f}{e['move']:9.4f}"
              f"{e['t1_us']:8.1f}{e['t2_us']:8.1f}{score(e):9.4f}")


def esp(transpiled: QuantumCircuit, backend, errors: dict[str, dict[str, float]]) -> float:
    """Product of (1 - error) over all gates and measurements (idling and crosstalk ignored)."""
    p = 1.0
    for inst in transpiled.data:
        key = GATE_KEY.get(inst.operation.name)
        if key is None:
            continue
        names = [backend.index_to_qubit_name(transpiled.find_bit(q).index) for q in inst.qubits]
        qb = next(n for n in names if n in errors)  # the qubit, not the resonator
        p *= 1 - errors[qb][key]
    return p


def best_layouts(circuit: QuantumCircuit, backend, errors: dict[str, dict[str, float]], n_candidates: int = 6,
                 seed: int | None = None, top: int = 5) -> list[tuple[float, list[str]]]:
    """Every placement of the circuit on the ``n_candidates`` best-scored qubits, ``[(esp, layout)]``, best first.

    ``circuit`` needs numeric parameters (one representative theta).
    """
    device = set(backend.architecture.qubits)
    pool = sorted((q for q in errors if q in device), key=lambda q: score(errors[q]))[:n_candidates]
    ranked = []
    for layout in itertools.permutations(pool, circuit.num_qubits):
        tr, _ = transpile([circuit], backend, layout=list(layout), seed=seed)
        ranked.append((esp(tr[0], backend, errors), list(layout)))
    return sorted(ranked, reverse=True)[:top]
