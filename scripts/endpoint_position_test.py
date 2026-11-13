"""Why do the BS circuits at theta = 0 and pi/2 fail on VLQ? Circuit content vs position in the batch.

In every VLQ BS run the endpoints were both the level-3 simplified circuits (theta = 0: 10 CZ, pi/2: 23 CZ,
others 28) and the first/last circuit of the job. One job separates the two: the level-3 endpoint circuits sit at
the batch edges and in the middle, the level-1 (unsimplified, 28 CZ) endpoint circuits in the middle.

- endpoints bad wherever they sit, level 1 fine -> the simplified circuits (content) fail on VLQ;
- only the batch edges bad (incl. the theta = pi/4 controls placed there) -> batch position (server/playlist).

Transpiled circuits (QPY) and the calibration set id are saved with the counts.

  .venv/bin/python scripts/endpoint_position_test.py                                  # FakeVLQ
  .venv/bin/python scripts/endpoint_position_test.py --backend VLQ --dry-run          # validates, no job
  .venv/bin/python scripts/endpoint_position_test.py --backend VLQ                    # submits, polls
  .venv/bin/python scripts/endpoint_position_test.py --fetch Results/endpoint_position_<date>
  .venv/bin/python scripts/endpoint_position_test.py --analyse Results/endpoint_position_<date>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qiskit import qpy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from scripts.move_sandwich_test import poll  # noqa: E402
from utils import vlq  # noqa: E402
from utils.bs_circuits import bind, counts_to_fock, exact_fock_probs, make_bs, postselect  # noqa: E402
from utils.metrics import _sbc_dicts  # noqa: E402
from utils.results_io import make_run_dir, save_run  # noqa: E402

LAYOUT = ["QB17", "QB9", "QB7", "QB14"]  # 13 Nov runs
P = np.pi
# (theta, optimization level); first and last entries are the batch edges
ORDER = [(0, 3), (P / 4, 3), (0, 3), (P / 2, 3), (P / 32, 3), (0, 1), (P / 2, 1), (15 * P / 32, 3), (0, 3),
         (P / 2, 3), (P / 4, 3), (P / 2, 3)]


def build(backend, n1: int, n2: int, layout: list[str], order=ORDER):
    """Transpiled circuits in ``order`` (seed 42, fixed layout) and their labels."""
    circs, labels = [], []
    for i, (theta, level) in enumerate(order):
        tr, _ = vlq.transpile(bind(make_bs(n1=n1, n2=n2), [theta]), backend, layout=layout, seed=42,
                              optimization_level=level)
        ops = tr[0].count_ops()
        circs.append(tr[0])
        labels.append({"pos": i, "theta": float(theta), "level": level, "cz": ops.get("cz", 0),
                       "move": ops.get("move", 0), "edge": i in (0, len(order) - 1)})
    return circs, labels


def analyse(d: dict) -> None:
    n1, n2 = d["config"]["input"]
    for key, counts in d["counts"].items():
        print(f"\n{key}  input |{n1}>|{n2}>  layout {d['config']['layout']}")
        print(" pos  theta   lvl  CZ  edge  kept   SBC_raw  SBC_post  top outcomes")
        for lab, c in zip(d["labels"], counts):
            p = counts_to_fock(c)
            ps, kept = postselect(p, n1 + n2)
            ex = {str(k): v for k, v in exact_fock_probs(lab["theta"], n1, n2).items()}
            raw = _sbc_dicts({str(k): v for k, v in p.items()}, ex)
            post = _sbc_dicts({str(k): v for k, v in ps.items()}, ex)
            s = sum(c.values())
            top = " ".join(f"{k}:{v / s:.2f}" for k, v in sorted(c.items(), key=lambda x: -x[1])[:3])
            print(f" {lab['pos']:3d}  {lab['theta']:.4f}  {lab['level']}  {lab['cz']:3d}  {'yes' if lab['edge'] else '   '}"
                  f"   {kept:.3f}  {raw:.3f}    {post:.3f}     {top}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="FakeVLQ", choices=["FakeVLQ", "VLQ"])
    ap.add_argument("--input", nargs=2, type=int, default=[1, 1], metavar=("N1", "N2"))
    ap.add_argument("--layout", nargs=4, default=LAYOUT)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fetch", type=Path, help="run folder whose VLQ job to (re)collect")
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

    backend = vlq.get_backend(a.backend)
    circs, labels = build(backend, *a.input, a.layout)
    print(f"{len(circs)} circuits x {a.shots} shots on {a.layout}")
    if a.dry_run:
        if a.backend == "VLQ":
            backend.create_circuit_job_definition(run_input=circs, shots=a.shots)
        print("validated, nothing submitted")
        return

    folder = make_run_dir(f"endpoint_position_in{a.input[0]}{a.input[1]}", root=ROOT / "Results")
    with open(folder / "transpiled.qpy", "wb") as f:
        qpy.dump(circs, f)
    d = {"config": {"backend": a.backend, "input": a.input, "layout": a.layout, "shots": a.shots,
                    "date": datetime.now().isoformat(timespec="seconds"),
                    "calibration_set_id": str(getattr(backend, "_calibration_set_id", None))},
         "labels": labels, "jobs": {}, "counts": {}}
    if a.backend == "FakeVLQ":
        d["counts"]["fake"] = vlq.run(backend, circs, a.shots, seed=1)
        d["jobs"]["fake"] = None
    else:
        job = vlq.submit(backend, circs, a.shots)
        d["jobs"]["full"] = str(job.job_id())
        save_run(folder, d)  # job id on disk before waiting
        print(f"saved to {folder}; if interrupted: --fetch {folder.relative_to(ROOT)}")
        poll(backend, d, folder)
    save_run(folder, d)
    print(f"saved to {folder}")
    analyse(d)


if __name__ == "__main__":
    main()
