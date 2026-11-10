"""Rebuilds results.json of run bs_full_in11_2026-11-09_163711 (not saved by the notebook).

VLQ (job 275): sector probabilities, kept, SBC from the table printed in bs_11_experiment.ipynb (3 decimals).
FakeVLQ: re-run with the same seed, layout and calibration snapshot; checked against the printed table.

Run from the project root: .venv/bin/python scripts/recover_run_2026-11-09_163711.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import vlq  # noqa: E402
from utils.bs_circuits import bind, counts_to_fock, exact_fock_probs, make_bs, postselect  # noqa: E402
from utils.fake_vlq_backend import CALIBRATION_FILE as FAKE_CALIBRATION  # noqa: E402
from utils.metrics import SBhattC, tvd  # noqa: E402
from utils.results_io import save_run  # noqa: E402

RUN_DIR = ROOT / "Results" / "bs_full_in11_2026-11-09_163711"
LAYOUT = ["QB11", "QB13", "QB17", "QB14"]
THETAS = [float(np.pi / 32 * n) for n in range(17)]
SHOTS, SEED, VLQ_JOB = 4000, 42, "275"
FOCK = [(a, b) for a in range(4) for b in range(4)]
SECTOR = [(0, 2), (1, 1), (2, 0)]
ROW = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def printed_tables(notebook: Path) -> dict[str, list[dict]]:
    """Backend -> rows of the results table printed by the notebook's analysis cell."""
    cells = json.loads(notebook.read_text())["cells"]
    text = next("".join(o.get("text", "")) for c in cells for o in c.get("outputs", [])
                if "SBC_raw SBC_post" in "".join(o.get("text", "")))
    tables, name = {}, None
    for line in text.splitlines():
        if m := ROW.match(line):
            t, p02, p11, p20, kept, sbc_raw, sbc_post = map(float, m.groups())
            tables[name].append({"theta": t, "raw": {"(0, 2)": p02, "(1, 1)": p11, "(2, 0)": p20},
                                 "kept": kept, "SBC_raw": sbc_raw, "SBC_post": sbc_post})
        elif line.strip() and not line.startswith(" theta"):
            name = line.strip()
            tables[name] = []
    return tables


def analyse(counts: list[dict[str, int]]) -> list[dict]:
    """Same metrics as the notebook's analysis cell."""
    rows = []
    for t, c in zip(THETAS, counts):
        p = counts_to_fock(c)
        ps, kept = postselect(p, 2)
        ex = exact_fock_probs(t, 1, 1)
        rows.append({
            "raw": {str(k): p.get(k, 0.0) for k in FOCK},
            "post": {str(k): ps.get(k, 0.0) for k in SECTOR},
            "kept": kept,
            "SBC_raw": SBhattC([p.get(k, 0.0) for k in FOCK], [ex.get(k, 0.0) for k in FOCK]),
            "SBC_post": SBhattC([ps.get(k, 0.0) for k in SECTOR], [ex.get(k, 0.0) for k in SECTOR]),
            "TVD_post": tvd([ps.get(k, 0.0) for k in SECTOR], [ex.get(k, 0.0) for k in SECTOR]),
        })
    return rows


def vlq_rows(table: list[dict]) -> list[dict]:
    """Printed VLQ rows completed with post-selected probabilities and TVD."""
    rows = []
    for t, r in zip(THETAS, table):
        assert np.isclose(r["theta"], t, atol=6e-4), (r["theta"], t)
        post = {k: v / r["kept"] for k, v in r["raw"].items()}
        ex = exact_fock_probs(t, 1, 1)
        rows.append({"raw": r["raw"], "post": post, "kept": r["kept"], "SBC_raw": r["SBC_raw"],
                     "SBC_post": r["SBC_post"],
                     "TVD_post": tvd([post[str(k)] for k in SECTOR], [ex.get(k, 0.0) for k in SECTOR])})
    return rows


def main() -> None:
    tables = printed_tables(ROOT / "bs_11_experiment.ipynb")

    backend = vlq.fake_vlq()
    transpiled, layout = vlq.transpile(bind(make_bs(circuit="full", n1=1, n2=1), THETAS), backend,
                                       layout=LAYOUT, seed=SEED)
    fake_counts = vlq.run(backend, transpiled, SHOTS, seed=SEED)
    fake = analyse(fake_counts)

    # The re-run must reproduce the printed FakeVLQ table.
    for r, p in zip(fake, tables["FakeVLQ"]):
        diff = max(abs(r["raw"][k] - v) for k, v in p["raw"].items())
        assert diff < 6e-4 and abs(r["SBC_raw"] - p["SBC_raw"]) < 6e-4, (p["theta"], diff)
    print("FakeVLQ re-run matches the printed table")

    results = {
        "config": {"circuit": "full", "input": [1, 1], "thetas": THETAS, "shots": SHOTS, "seed": SEED,
                   "backends": ["VLQ", "FakeVLQ"], "fake_calibration": FAKE_CALIBRATION,
                   "layouts": {"VLQ": layout, "FakeVLQ": layout}, "vlq_job_id": VLQ_JOB,
                   "recovered": "VLQ: printed notebook table (3 decimals, no counts); FakeVLQ: seeded re-run"},
        "counts": {"FakeVLQ": fake_counts},
        "metrics": {"VLQ": vlq_rows(tables["VLQ"]), "FakeVLQ": fake},
    }
    print("saved:", save_run(RUN_DIR, results))


if __name__ == "__main__":
    main()
