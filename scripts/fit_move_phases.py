"""Fits a residual Z phase per physical qubit per MOVE sandwich to the stored VLQ BS counts.

Model: p = (1 - eps) * p_circuit(phases) + eps / 16, circuits re-transpiled as in the runs (level 3, seed 42,
FakeVLQ topology). Lower eps with phases = more of the deviation is coherent. Takes ~10 min.

The stored VLQ keys of these runs are reversed (collect() before the 2026-11-12 fix); REVERSED_KEYS undoes it.

Run from the project root: .venv/bin/python scripts/fit_move_phases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY
from qiskit.circuit.library import CZGate, RGate

from utils import vlq
from utils.bs_circuits import bind, make_bs

REVERSED_KEYS = {"bs_full_in11_2026-11-08_193135", "bs_full_in02_2026-11-08_200127", "bs_full_in20_2026-11-08_200400",
                 "bs_full_in11_2026-11-09_125141", "bs_full_in11_2026-11-09_125551"}
GROUPS = {  # runs sharing physical qubits on one day share the phases
    "08 Nov": ["bs_full_in11_2026-11-08_193135", "bs_full_in02_2026-11-08_200127", "bs_full_in20_2026-11-08_200400"],
    "09 Nov a": ["bs_full_in11_2026-11-09_125141"],
    "09 Nov b": ["bs_full_in11_2026-11-09_125551"],
}
CZ = CZGate().to_matrix()
MOVE = np.asarray(MOVE_GATE_UNITARY)
REVERSE = [int(f"{k:04b}"[::-1], 2) for k in range(16)]

Run = tuple[list[tuple[list, dict]], list[np.ndarray]]


def rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def load(run: str, backend) -> Run:
    """Transpiled circuits as (ops on qubit names, clbit -> qubit) and VLQ counts (index = int of Qiskit key)."""
    d = json.loads((ROOT / "Results" / run / "results.json").read_text())
    n1, n2 = d["config"]["input"]
    tr, _ = vlq.transpile(bind(make_bs(n1=n1, n2=n2), d["config"]["thetas"]), backend,
                          layout=d["config"]["layout"], seed=42)
    circs = []
    for t in tr:
        names = [backend.index_to_qubit_name(t.find_bit(q).index) for q in t.qubits]
        ops, meas = [], {}
        for inst in t.data:
            qs = [names[t.find_bit(q).index] for q in inst.qubits]
            if inst.operation.name == "measure":
                meas[t.find_bit(inst.clbits[0]).index] = qs[0]
            elif inst.operation.name in ("r", "cz", "move"):
                ops.append((inst.operation.name, qs, [float(p) for p in inst.operation.params]))
        circs.append((ops, meas))
    counts = []
    for cc in d["counts"]["VLQ"]:
        c = np.zeros(16)
        for k, v in cc.items():
            c[int(k, 2)] += v
        counts.append(c[REVERSE] if run in REVERSED_KEYS else c)
    return circs, counts


def simulate(ops: list, meas: dict, phases: dict[str, float]) -> np.ndarray:
    """Output probabilities; rz(phases[q]) when q's state comes back from the resonator."""
    qubits = sorted({q for _, qs, _ in ops for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(qubits)}
    psi = np.zeros([2] * len(qubits), complex)
    psi[(0,) * len(qubits)] = 1
    held: dict[str, str] = {}  # resonator -> qubit whose state it holds

    def apply(u: np.ndarray, qs: list[str]) -> None:
        nonlocal psi
        ax, k = [pos[q] for q in qs], len(qs)
        psi = np.moveaxis(np.tensordot(u.reshape([2] * 2 * k), psi, axes=(list(range(k, 2 * k)), ax)),
                          list(range(k)), ax)

    for name, qs, p in ops:
        if name == "r":
            apply(RGate(*p).to_matrix(), qs)
        elif name == "cz":
            apply(CZ, qs)
        else:
            q, r = qs
            apply(MOVE, qs)
            if held.get(r) == q:
                del held[r]
                if q in phases:
                    apply(rz(phases[q]), [q])
            else:
                held[r] = q
    probs = np.abs(psi) ** 2
    out = np.zeros(16)
    for idx in np.ndindex(*probs.shape):
        out[sum(idx[pos[meas[cb]]] << cb for cb in meas)] += probs[idx]
    return out


def score(runs: list[Run], phases: dict[str, float], eps: float | None = None) -> tuple[float, float, float]:
    """(log-loss per shot, eps, mean fidelity of the model to the data); eps on a grid unless given."""
    preds = [[simulate(ops, meas, phases) for ops, meas in circs] for circs, _ in runs]
    best = None
    for e in [eps] if eps is not None else np.linspace(0, 0.99, 34):
        ll, n, fid = 0.0, 0.0, []
        for pr, (_, counts) in zip(preds, runs):
            for p, c in zip(pr, counts):
                q = (1 - e) * p + e / 16
                ll -= (c * np.log(q + 1e-12)).sum()
                n += c.sum()
                fid.append(np.sum(np.sqrt(q * c / c.sum())) ** 2)
        if best is None or ll / n < best[0]:
            best = (ll / n, float(e), float(np.mean(fid)))
    return best


def fit(runs: list[Run], qubits: list[str], seed: int = 0) -> dict[str, float]:
    """Phases minimising the log-loss (eps fixed at 0.3 during the search), best of 12 random starts."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(12):
        r = minimize(lambda x: score(runs, dict(zip(qubits, x)), eps=0.3)[0], rng.uniform(-np.pi, np.pi, len(qubits)),
                     method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-2, "fatol": 1e-4})
        best = r if best is None or r.fun < best.fun else best
    return {q: float((v + np.pi) % (2 * np.pi) - np.pi) for q, v in zip(qubits, best.x)}


def show(label: str, s: tuple[float, float, float]) -> None:
    print(f"  {label:42} log-loss/shot {s[0]:.3f}  eps {s[1]:.2f}  fidelity {s[2]:.3f}")


def main() -> None:
    backend = vlq.fake_vlq()
    data = {r: load(r, backend) for runs in GROUPS.values() for r in runs}
    print(f"uniform noise: log-loss/shot {np.log(16):.3f}")
    for group, runs in GROUPS.items():
        rs = [data[r] for r in runs]
        qubits = sorted({q for circs, _ in rs for ops, _ in circs for _, qs, _ in ops for q in qs if q != "CR1"},
                        key=lambda q: int(q[2:]))
        phases = fit(rs, qubits)
        print(f"\n{group}: {runs}")
        show("no phases", score(rs, {}))
        show("fitted phases", score(rs, phases))
        print("  phases (rad):", ", ".join(f"{q} {v:+.2f}" for q, v in phases.items()))

    # out of sample: phases from the 08 Nov in11 run predict the in02 / in20 runs (QB17 not in the in11 layout)
    train = "bs_full_in11_2026-11-08_193135"
    phases = fit([data[train]], ["QB9", "QB10", "QB13", "QB14"])
    print(f"\nout of sample, phases from {train}:", ", ".join(f"{q} {v:+.2f}" for q, v in phases.items()))
    for run in ("bs_full_in02_2026-11-08_200127", "bs_full_in20_2026-11-08_200400"):
        show(f"{run[8:12]} no phases", score([data[run]], {}))
        show(f"{run[8:12]} predicted", score([data[run]], phases))


if __name__ == "__main__":
    main()
