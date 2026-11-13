"""Ramsey tests of the MOVE sandwich on VLQ: residual Z phase on a qubit q around resonator traffic.

For phi in 0, pi/2, pi, 3pi/2: r(pi/2, 0) q · <variant> · r(pi/2, phi) q, measure q. A sandwich is
MOVE x,R · k x CZ s,R · MOVE x,R. Phase from P1(phi) = A + B cos(phi - delta), relative to ``ref`` (no MOVE),
same sign as the rz phases of ``fit_move_phases.py``.

``--test single`` (s = next qubit in the list): ``czK`` = q's own sandwich with K CZs (s in |0>, ideal 0);
``cz1x`` = s in |1> (ideal pi).
``--test multi`` (a = next qubit, s = the one after), one CZ per sandwich of a:
``own`` = q's sandwich alone; ``idleK`` = K sandwiches of a while q idles in superposition (crosstalk);
``carryK`` = K sandwiches of a before q's own sandwich (resonator frame carried over); ``*2x`` = a in |1>.
Circuit ``key_order``: q0 = |1>, q1 = |0>, keys must read ``01``.

One job per frame-tracking mode (``--frame-tracking``; the BS runs used ``full``). Circuits are native: no
transpilation, so the sandwiches are not optimised away; barriers fix the order.

  .venv/bin/python scripts/move_sandwich_test.py --test multi                          # FakeVLQ
  .venv/bin/python scripts/move_sandwich_test.py --test multi --backend VLQ --dry-run  # validates, no job
  .venv/bin/python scripts/move_sandwich_test.py --test multi --backend VLQ            # submits, polls
  .venv/bin/python scripts/move_sandwich_test.py --fetch Results/move_sandwich_<date>  # resume polling
  .venv/bin/python scripts/move_sandwich_test.py --analyse Results/move_sandwich_<date>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")  # VLQ_PROJECT, VLQ_RESOURCE, VLQ_TOKEN, as in the notebook

from iqm.qiskit_iqm.move_gate import MoveGate

from utils import vlq
from utils.results_io import make_run_dir, save_run

PHIS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
QUBITS = ["QB9", "QB10", "QB13", "QB14", "QB17"]  # 08 Nov BS layout (3 runs)


def _v(mid=(), pre=(), excite=(), ideal=0.0) -> dict:
    """Variant: sandwiches (role, K CZs) before the Ramsey and between its pulses; roles prepared in |1>."""
    return {"mid": list(mid), "pre": list(pre), "excite": list(excite), "ideal": ideal}


TESTS = {
    "single": {"ref": _v(), **{f"cz{k}": _v(mid=[("q", k)]) for k in (0, 1, 2, 4)},
               "cz1x": _v(mid=[("q", 1)], excite=["s"], ideal=np.pi)},
    "multi": {"ref": _v(), "own": _v(mid=[("q", 0)]),
              **{f"idle{k}": _v(mid=[("a", 1)] * k) for k in (1, 2, 4)},
              **{f"carry{k}": _v(pre=[("a", 1)] * k, mid=[("q", 0)]) for k in (1, 2, 4)},
              "idle2x": _v(mid=[("a", 1)] * 2, excite=["a"]),
              "carry2x": _v(pre=[("a", 1)] * 2, mid=[("q", 0)], excite=["a"])},
}
FAMILIES = {  # phase vs number of sandwiches / CZs -> slope
    "single": {"per CZ": [("cz0", 0), ("cz1", 1), ("cz2", 2), ("cz4", 4)]},
    "multi": {"idle/sw": [("ref", 0), ("idle1", 1), ("idle2", 2), ("idle4", 4)],
              "carry/sw": [("own", 0), ("carry1", 1), ("carry2", 2), ("carry4", 4)]},
}


def roles(qubits: list[str], test: str, spectator: str | None = None) -> list[dict[str, str]]:
    """Per Ramsey qubit q: passer a and CZ partner s, taken cyclically from the list."""
    n = len(qubits)
    if test == "single":
        return [{"q": q, "s": spectator or qubits[(i + 1) % n]} for i, q in enumerate(qubits)]
    if n < 3:
        raise ValueError("--test multi needs at least 3 qubits")
    return [{"q": q, "a": qubits[(i + 1) % n], "s": qubits[(i + 2) % n]} for i, q in enumerate(qubits)]


def sandwiches(c: QuantumCircuit, spec: list[tuple[str, int]], ix: dict[str, int], r: int) -> None:
    """Appends MOVE x,R · K x CZ s,R · MOVE x,R for each (role x, K)."""
    for role, k in spec:
        c.append(MoveGate(), [ix[role], r])
        for _ in range(k):
            c.cz(ix["s"], r)
        c.append(MoveGate(), [ix[role], r])


def build(backend, test: str, assignment: list[dict[str, str]]) -> tuple[list[QuantumCircuit], list[dict]]:
    """Native circuits and their labels."""
    idx = backend.qubit_name_to_index
    width = backend.target_with_resonators.num_qubits
    r = idx(backend.architecture.computational_resonators[0])  # VLQ: COMPR1, FakeVLQ: CR1
    circs, labels = [], []
    for names in assignment:
        ix = {role: idx(n) for role, n in names.items()}
        busy = sorted(set(ix.values()))

        for hold, v in TESTS[test].items():
            for phi in PHIS:
                c = QuantumCircuit(width, 1, name=f"{names['q']}_{hold}_{phi:.2f}")
                for role in v["excite"]:
                    c.r(np.pi, 0, ix[role])
                sandwiches(c, v["pre"], ix, r)
                c.barrier(busy)
                c.r(np.pi / 2, 0, ix["q"])
                c.barrier(busy)
                sandwiches(c, v["mid"], ix, r)
                c.barrier(busy)
                c.r(np.pi / 2, phi, ix["q"])
                c.measure(ix["q"], 0)
                circs.append(c)
                labels.append({"qubit": names["q"], "roles": names, "hold": hold, "phi": phi})
    q0, q1 = idx(assignment[0]["q"]), idx(assignment[1]["q"])
    c = QuantumCircuit(width, 2, name="key_order")
    c.r(np.pi, 0, q0)
    c.measure(q0, 0)
    c.measure(q1, 1)
    circs.append(c)
    labels.append({"check": "key_order", "expected": "01"})
    return circs, labels


def compilation_options(mode: str):
    from iqm.iqm_client import CircuitCompilationOptions, MoveGateFrameTrackingMode

    return CircuitCompilationOptions(move_gate_frame_tracking=MoveGateFrameTrackingMode(mode))


def phase(p1: list[float]) -> tuple[float, float]:
    """(delta, contrast B) from P1 at PHIS."""
    c, s = (p1[0] - p1[2]) / 2, (p1[1] - p1[3]) / 2
    return float(np.arctan2(s, c)), float(np.hypot(c, s))


def wrap(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def analyse(d: dict) -> None:
    test = d["config"].get("test", "single")
    variants, families = TESTS[test], FAMILIES[test]
    for mode, counts in d["counts"].items():
        print(f"\n== {d['config']['backend']}, test {test!r}, frame tracking {mode!r}, job {d['jobs'].get(mode)}")
        p1: dict[tuple[str, str], list[float]] = {}
        for lab, c in zip(d["labels"], counts):
            if "check" in lab:
                top = max(c, key=c.get)
                ok = "OK" if top == lab["expected"] else "REVERSED?"
                print(f"  key order: most frequent {top!r} ({c[top] / sum(c.values()):.2f}), expected"
                      f" {lab['expected']!r}: {ok}")
                continue
            p1.setdefault((lab["qubit"], lab["hold"]), []).append(c.get("1", 0) / sum(c.values()))
        holds = [h for h in variants if h != "ref"]
        print("  phase (rad) relative to ref, minus the ideal phase; contrast in brackets; slopes in rad per step")
        print(f"  {'qubit':6} {'ref':>12} " + " ".join(f"{h:>13}" for h in holds)
              + "".join(f" {f:>9}" for f in families))
        for qn in d["config"]["qubits"]:
            d0, b0 = phase(p1[(qn, "ref")])
            rel = {"ref": 0.0}
            row = []
            for h in holds:
                dh, bh = phase(p1[(qn, h)])
                rel[h] = wrap(dh - d0 - variants[h]["ideal"])
                row.append(f"{rel[h]:+6.2f} ({bh:.2f})")
            slopes = [np.polyfit([k for _, k in f], np.unwrap([rel[h] for h, _ in f]), 1)[0] for f in families.values()]
            print(f"  {qn:6} {d0:+6.2f} ({b0:.2f}) " + " ".join(f"{x:>13}" for x in row)
                  + "".join(f" {x:+9.3f}" for x in slopes))


def poll(backend, d: dict, folder: Path, every: float = 20.0) -> None:
    """Waits for every unfetched job, collects its counts (bit-order fixed) and saves after each."""
    from qheappe_client.qiskit.qbackend import QJob
    from qheappe_client.utils import HEAppEJobState

    qclient = backend._qclient  # pylint: disable=protected-access
    while pending := [m for m in d["jobs"] if m not in d["counts"]]:
        for mode in pending:
            job_id = int(d["jobs"][mode])
            info = qclient.get_job_info(job_id)
            state = str(HEAppEJobState.readable(info.get("State", -1))).upper()
            if state == "FINISHED":
                job = QJob(backend=backend, heappe_job_id=job_id, submitted_job=info, shots=d["config"]["shots"])
                d["counts"][mode] = vlq.collect(job)
                save_run(folder, d)
                print(f"job {job_id} ({mode}): fetched")
            elif state in ("FAILED", "CANCELED"):
                raise RuntimeError(f"job {job_id} ({mode}) {state}: {info}")
        if [m for m in d["jobs"] if m not in d["counts"]]:
            time.sleep(every)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", default="single", choices=list(TESTS))
    ap.add_argument("--backend", default="FakeVLQ", choices=["FakeVLQ", "VLQ"])
    ap.add_argument("--qubits", nargs="+", default=QUBITS)
    ap.add_argument("--spectator", help="--test single: one spectator for all qubits (default: next qubit)")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--frame-tracking", nargs="+", default=["full"], choices=["full", "no_detuning_correction", "none"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch", type=Path, help="run folder whose VLQ jobs to (re)collect")
    ap.add_argument("--analyse", type=Path, help="run folder to analyse")
    a = ap.parse_args()

    if a.analyse:
        analyse(json.loads((a.analyse / "results.json").read_text()))
        return
    if a.fetch:
        d = json.loads((a.fetch / "results.json").read_text())
        poll(vlq.get_backend("VLQ"), d, a.fetch)
        analyse(d)
        return

    assignment = roles(a.qubits, a.test, a.spectator)
    backend = vlq.get_backend(a.backend)
    circs, labels = build(backend, a.test, assignment)
    print(f"{len(circs)} circuits x {a.shots} shots, frame tracking {a.frame_tracking}")
    if a.dry_run:
        if a.backend == "VLQ":
            for mode in a.frame_tracking:
                backend.create_circuit_job_definition(run_input=circs, shots=a.shots,
                                                      circuit_compilation_options=compilation_options(mode))
        else:
            from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit

            for c in circs:
                validate_circuit(c, backend)
        print("validated, nothing submitted")
        return

    folder = make_run_dir(f"move_sandwich_{a.test}", root=ROOT / "Results")
    d = {"config": {"backend": a.backend, "test": a.test, "qubits": a.qubits, "roles": assignment, "shots": a.shots,
                    "frame_tracking": a.frame_tracking, "phis": PHIS, "holds": list(TESTS[a.test]),
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "calibration_set_id": str(getattr(backend, "_calibration_set_id", None))},
         "labels": labels, "jobs": {}, "counts": {}}
    if a.backend == "FakeVLQ":
        d["counts"]["fake"] = vlq.run(backend, circs, a.shots, seed=1)
        d["jobs"]["fake"] = None
    else:
        for mode in a.frame_tracking:
            job = vlq.submit(backend, circs, a.shots, circuit_compilation_options=compilation_options(mode))
            d["jobs"][mode] = str(job.job_id())
            save_run(folder, d)  # job ids on disk before waiting
        print(f"saved to {folder}; if interrupted: --fetch {folder.relative_to(ROOT)}")
        poll(backend, d, folder)
    save_run(folder, d)
    print(f"saved to {folder}")
    analyse(d)


if __name__ == "__main__":
    main()
