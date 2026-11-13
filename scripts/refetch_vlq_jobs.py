"""Re-fetches finished VLQ jobs and saves the server's raw counts, incl. the per-circuit ``measurement_keys``.

Shows whether the bit order of old runs was a plain reversal (keys in clbit order in every circuit) or changed
between circuits. Each job is matched to the ``Results/`` run with that ``config.vlq_job_id``, else to the one whose
stored VLQ counts equal the old decoding (qaas default ``key_order_left_to_right=False``). Writes ``vlq_refetch_job<ID>.json`` into that run folder
(or ``Results/vlq_refetch/``); ``results.json`` is not touched.

  .venv/bin/python scripts/refetch_vlq_jobs.py 275 427
  .venv/bin/python scripts/refetch_vlq_jobs.py --scan 250 300   # list accessible jobs (state, times, circuits)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from utils import vlq
from utils.bs_circuits import counts_to_fock, exact_fock_probs
from utils.metrics import _sbc_dicts


def stored_runs() -> dict[str, dict]:
    """Run name -> results.json."""
    return {f.parent.name: json.loads(f.read_text()) for f in sorted((ROOT / "Results").glob("bs*/results.json"))}


def clbit_order(keys: list[str]) -> list[int]:
    """Clbit index of each key, in the order the server lists them (``<creg>_<size>_<reg idx>_<clbit idx>``)."""
    return [int(k.rsplit("_", 1)[1]) for k in keys]


def nonzero(counts: list[dict[str, int]]) -> list[dict[str, int]]:
    return [{k: v for k, v in c.items() if v} for c in counts]


def sbc(counts: dict[str, int], theta: float, n1: int, n2: int) -> float:
    p = {str(k): v for k, v in counts_to_fock(counts).items()}
    return _sbc_dicts(p, {str(k): v for k, v in exact_fock_probs(theta, n1, n2).items()})


def scan(backend, first: int, last: int) -> None:
    qclient = backend._qclient  # pylint: disable=protected-access
    for job_id in range(first, last + 1):
        try:
            info = qclient.get_job_info(job_id)
        except Exception as error:  # noqa: BLE001
            print(f"{job_id}: not accessible ({type(error).__name__})")
            continue
        times = {k: v for k, v in info.items() if k.endswith("Time") and v}
        print(f"{job_id}: state {info.get('State')} name {info.get('Name')!r} tasks {len(info.get('Tasks', []))} {times}")


def refetch(backend, job_id: int, runs: dict[str, dict]) -> None:
    from qheappe_client.qiskit.qbackend import QJob

    info = backend._qclient.get_job_info(job_id)  # pylint: disable=protected-access
    job = QJob(backend=backend, heappe_job_id=job_id, submitted_job=info)
    res = job.result(qiskit_legacy=True)  # qaas default: key_order_left_to_right=False, as in the old runs
    old = [res.get_counts(i) for i in range(len(res.results))]
    raw = job._get_artifacts(with_measurements=False)[QJob._COUNTS_ARTIFACT]  # pylint: disable=protected-access
    try:
        new = vlq.collect(job)  # fixed bit order; raises if keys are not in clbit order
    except RuntimeError as error:
        print(f"  collect(): {error}")
        new = None

    orders = [tuple(clbit_order(r["measurement_keys"])) for r in raw]
    print(f"\njob {job_id}: {len(raw)} circuits")
    print("  measurement_keys orders (clbit index per position: circuits):", {o: orders.count(o) for o in set(orders)})
    if raw:
        print(f"  first circuit keys: {raw[0]['measurement_keys']}")

    match = next((n for n, d in runs.items() if str(d["config"].get("vlq_job_id")) == str(job_id)), None) or next(
        (n for n, d in runs.items() if isinstance(d.get("counts"), dict) and d["counts"].get("VLQ")
         and nonzero(d["counts"]["VLQ"]) == nonzero(old)), None)
    print(f"  stored run (by vlq_job_id, else identical old-decoded counts): {match}")
    record = {"job_id": job_id, "run": match, "raw_counts_artifact": raw, "counts_fixed": new,
              "counts_old_default": old, "job_info": info}
    folder = ROOT / "Results" / (match or "vlq_refetch")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"vlq_refetch_job{job_id}.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"  saved {folder.relative_to(ROOT)}/vlq_refetch_job{job_id}.json")
    if match and new is not None:
        c = runs[match]["config"]
        stored = runs[match].get("counts")
        old = (stored.get("VLQ") if isinstance(stored, dict) else None) or old
        n1, n2 = c["input"]
        print("  theta   SBC stored   SBC fixed   SBC reversed-stored")
        for t, o, n in zip(c["thetas"], old, new):
            rev = {k[::-1]: v for k, v in o.items()}
            print(f"  {t:.3f}   {sbc(o, t, n1, n2):.3f}        {sbc(n, t, n1, n2):.3f}       {sbc(rev, t, n1, n2):.3f}")
        print(f"  mean    {np.mean([sbc(o, t, n1, n2) for t, o in zip(c['thetas'], old)]):.3f}"
              f"        {np.mean([sbc(n, t, n1, n2) for t, n in zip(c['thetas'], new)]):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job_ids", nargs="*", type=int)
    ap.add_argument("--scan", nargs=2, type=int, metavar=("FIRST", "LAST"))
    a = ap.parse_args()
    backend = vlq.get_backend("VLQ")
    if a.scan:
        scan(backend, *a.scan)
    runs = stored_runs()
    for job_id in a.job_ids:
        try:
            refetch(backend, job_id, runs)
        except Exception as error:  # noqa: BLE001
            print(f"\njob {job_id}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
