"""Does CZ(P, R) act as a swap when R holds another qubit's state? Population test of the BS endpoint motif.

All VLQ circuits that fail (BS theta = 0, pi/2 at level 3; theta = 0 at level 1) contain MOVE QB7,R · CZ QB17,R ·
MOVE QB7,R; no working one does. A fit replacing that CZ by a full |01> <-> |10> exchange restores theta = 0.

Motif circuits (native, no transpilation), for every ordered pair holder H != partner P of ``--qubits``:
X on the prepared qubits · MOVE H,R · CZ P,R · MOVE H,R · measure H (clbit 0), P (clbit 1).
``H1``: H = |1> -> ideal ``01`` (swap: ``10``); ``P1``: P = |1> -> ideal ``10`` (swap: ``01``); ``HP``: both -> ``11``.
Plus the BS theta = 0 and pi/2 level-3 circuits (input |1>|1>), as transpiled and with a barrier after every gate.

  .venv/bin/python scripts/cz_motif_test.py                           # FakeVLQ
  .venv/bin/python scripts/cz_motif_test.py --backend VLQ --dry-run   # validates, no job
  .venv/bin/python scripts/cz_motif_test.py --backend VLQ             # submits, polls
  .venv/bin/python scripts/cz_motif_test.py --fetch Results/cz_motif_<date>
  .venv/bin/python scripts/cz_motif_test.py --analyse Results/cz_motif_<date>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qiskit import QuantumCircuit, qpy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from iqm.qiskit_iqm.move_gate import MoveGate  # noqa: E402

from scripts.move_sandwich_test import poll  # noqa: E402
from utils import vlq  # noqa: E402
from utils.bs_circuits import bind, counts_to_fock, exact_fock_probs, make_bs, postselect  # noqa: E402
from utils.metrics import _sbc_dicts  # noqa: E402
from utils.results_io import make_run_dir, save_run  # noqa: E402

QUBITS = ["QB17", "QB9", "QB7", "QB14"]  # 13 Nov BS layout
PREPS = {"H1": (("H",), "01", "10"), "P1": (("P",), "10", "01"), "HP": (("H", "P"), "11", "11")}  # ideal, swapped


def motif(backend, h: str, p: str, prep: str) -> QuantumCircuit:
    idx = backend.qubit_name_to_index
    r = idx(backend.architecture.computational_resonators[0])
    ix = {"H": idx(h), "P": idx(p)}
    c = QuantumCircuit(backend.target_with_resonators.num_qubits, 2, name=f"{h}_{p}_{prep}")
    for role in PREPS[prep][0]:
        c.r(np.pi, 0, ix[role])
    c.barrier([ix["H"], ix["P"], r])
    c.append(MoveGate(), [ix["H"], r])
    c.cz(ix["P"], r)
    c.append(MoveGate(), [ix["H"], r])
    c.barrier([ix["H"], ix["P"], r])
    c.measure(ix["H"], 0)
    c.measure(ix["P"], 1)
    return c


def with_barriers(t: QuantumCircuit) -> QuantumCircuit:
    """Same circuit with a full barrier after every gate (forces program order on hardware)."""
    c = t.copy_empty_like()
    for inst in t.data:
        c.append(inst)
        if inst.operation.name not in ("barrier", "measure"):
            c.barrier()
    return c


def build(backend, qubits: list[str]):
    circs, labels = [], []
    for h, p in itertools.permutations(qubits, 2):
        for prep, (_, ideal, swapped) in PREPS.items():
            circs.append(motif(backend, h, p, prep))
            labels.append({"kind": "motif", "H": h, "P": p, "prep": prep, "ideal": ideal, "swapped": swapped})
    tr, _ = vlq.transpile(bind(make_bs(n1=1, n2=1), [0.0, np.pi / 2]), backend, layout=qubits, seed=42)
    for t, theta in zip(tr, (0.0, np.pi / 2)):
        for barriers in (False, True):
            circs.append(with_barriers(t) if barriers else t)
            labels.append({"kind": "bs", "theta": theta, "barriers": barriers})
    return circs, labels


def analyse(d: dict) -> None:
    for key, counts in d["counts"].items():
        print(f"\n{key}: motif MOVE H,R · CZ P,R · MOVE H,R; P(ideal) / P(swapped)")
        qs = d["config"]["qubits"]
        print("  H \\ P     " + "".join(f"{p:>22s}" for p in qs))
        rows = {}
        for lab, c in zip(d["labels"], counts):
            if lab["kind"] == "motif":
                s = sum(c.values())
                rows.setdefault(lab["H"], {}).setdefault(lab["P"], []).append(
                    f"{lab['prep']}:{c.get(lab['ideal'], 0) / s:.2f}/{c.get(lab['swapped'], 0) / s:.2f}"
                    if lab["prep"] != "HP" else f"HP:{c.get('11', 0) / s:.2f}")
        for h in qs:
            print(f"  {h:8s}" + "".join(f"{' '.join(rows[h][p]) if p in rows.get(h, {}) else '-':>34s}" for p in qs))
        print("  BS |1>|1>:")
        for lab, c in zip(d["labels"], counts):
            if lab["kind"] == "bs":
                p = counts_to_fock(c)
                ps, kept = postselect(p, 2)
                ex = {str(k): v for k, v in exact_fock_probs(lab["theta"], 1, 1).items()}
                post = _sbc_dicts({str(k): v for k, v in ps.items()}, ex)
                print(f"    theta {lab['theta']:.4f}  barriers {lab['barriers']!s:5s}  kept {kept:.3f}  SBC_post {post:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="FakeVLQ", choices=["FakeVLQ", "VLQ"])
    ap.add_argument("--qubits", nargs="+", default=QUBITS)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch", type=Path)
    ap.add_argument("--analyse", type=Path)
    a = ap.parse_args()

    if a.analyse:
        analyse(json.loads((a.analyse / "results.json").read_text()))
        return
    if a.fetch:
        d = json.loads((a.fetch / "results.json").read_text())
        poll(vlq.get_backend("VLQ"), d, a.fetch)
        analyse(d)
        return

    backend = vlq.get_backend(a.backend)
    circs, labels = build(backend, a.qubits)
    print(f"{len(circs)} circuits x {a.shots} shots")
    if a.dry_run:
        if a.backend == "VLQ":
            backend.create_circuit_job_definition(run_input=circs, shots=a.shots)
        print("validated, nothing submitted")
        return

    folder = make_run_dir("cz_motif", root=ROOT / "Results")
    with open(folder / "circuits.qpy", "wb") as f:
        qpy.dump(circs, f)
    d = {"config": {"backend": a.backend, "qubits": a.qubits, "shots": a.shots,
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "calibration_set_id": str(getattr(backend, "_calibration_set_id", None))},
         "labels": labels, "jobs": {}, "counts": {}}
    if a.backend == "FakeVLQ":
        d["counts"]["fake"] = vlq.run(backend, circs, a.shots, seed=1)
        d["jobs"]["fake"] = None
    else:
        job = vlq.submit(backend, circs, a.shots)
        d["jobs"]["full"] = str(job.job_id())
        save_run(folder, d)
        print(f"saved to {folder}; if interrupted: --fetch {folder.relative_to(ROOT)}")
        poll(backend, d, folder)
    save_run(folder, d)
    print(f"saved to {folder}")
    analyse(d)


if __name__ == "__main__":
    main()
