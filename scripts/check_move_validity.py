"""Noise-free check of the transpiled BS circuits on the VLQ star (FakeVLQ topology, level 3, seed 42).

Before each MOVE it reports P(qubit = 1 and resonator = 1): there the MOVE gate is undefined on hardware, but
FakeVLQ's MOVE unitary leaves |11> unchanged and hides it. "HF leak" drops that amplitude, as hardware would.

Run from the project root: .venv/bin/python scripts/check_move_validity.py [optimization levels, default 3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY

from utils import vlq
from utils.bs_circuits import bind, counts_to_fock, exact_fock_probs, make_bs
from utils.metrics import _sbc_dicts

MOVE = UnitaryGate(MOVE_GATE_UNITARY)
THETAS = [np.pi / 32 * n for n in range(17)]
CASES = [((1, 1), ["QB11", "QB13", "QB17", "QB14"]), ((1, 1), ["QB10", "QB13", "QB14", "QB9"]),
         ((1, 1), ["QB2", "QB17", "QB18", "QB20"]), ((1, 1), ["QB15", "QB16", "QB21", "QB24"]),
         ((2, 0), ["QB14", "QB17", "QB13", "QB9"]), ((0, 2), ["QB10", "QB13", "QB14", "QB9"])]


def compact(t: QuantumCircuit) -> tuple[QuantumCircuit, dict[int, int]]:
    """Only the qubits that carry operations; returns the circuit and old -> new index."""
    used = sorted({t.find_bit(q).index for i in t.data for q in i.qubits if i.operation.name != "barrier"})
    idx = {o: n for n, o in enumerate(used)}
    c = QuantumCircuit(len(used), t.num_clbits)
    for i in t.data:
        if i.operation.name != "barrier":
            c.append(i.operation, [idx[t.find_bit(q).index] for q in i.qubits], [t.find_bit(b).index for b in i.clbits])
    return c, idx


def analyse(t: QuantumCircuit, res_index: int, leak: bool) -> tuple[dict[str, float], list[float], int]:
    """Output probabilities (Qiskit keys), P(qubit = 1 and resonator = 1) per MOVE, 1q gates on the resonator."""
    c, idx = compact(t)
    r = idx.get(res_index)
    sv = Statevector.from_label("0" * c.num_qubits)
    p11, on_res, meas = [], 0, {}
    for inst in c.data:
        qs = [c.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "measure":
            meas[c.find_bit(inst.clbits[0]).index] = qs[0]
            continue
        if r is not None and qs == [r]:
            on_res += 1
        if inst.operation.name == "move":
            q = qs[0] if qs[1] == r else qs[1]
            ii = np.arange(len(sv.data))
            both = ((ii >> q) & 1) & ((ii >> r) & 1) == 1
            p11.append(float((np.abs(sv.data[both]) ** 2).sum()))
            if leak:
                d = sv.data.copy()
                d[both] = 0
                sv = Statevector(d)
            sv = sv.evolve(MOVE, qs)
        else:
            sv = sv.evolve(inst.operation, qs)
    out: dict[str, float] = {}
    for i, p in enumerate(np.abs(sv.data) ** 2):
        if p > 1e-12:
            key = "".join(str((i >> meas[cb]) & 1) for cb in sorted(meas, reverse=True))
            out[key] = out.get(key, 0.0) + p
    return out, p11, on_res


def main() -> None:
    backend = vlq.fake_vlq()
    res_index = backend.qubit_name_to_index("CR1")
    for (n1, n2), layout in CASES:
        for level in [int(a) for a in sys.argv[1:]] or [3]:
            circs = bind(make_bs(n1=n1, n2=n2), THETAS)
            tr, _ = vlq.transpile(circs, backend, layout=layout, seed=42, optimization_level=level)
            print(f"== in{n1}{n2} {layout} level {level}")
            for t, c in zip(THETAS, tr):
                ideal, p11, on_res = analyse(c, res_index, leak=False)
                leaked, _, _ = analyse(c, res_index, leak=True)
                exact = {str(k): v for k, v in exact_fock_probs(t, n1, n2).items()}
                kept = sum(leaked.values())
                hf = _sbc_dicts({str(k): v for k, v in counts_to_fock(ideal).items()}, exact)
                hf_leak = _sbc_dicts({str(k): v for k, v in counts_to_fock(leaked).items()}, exact)  # renormalised
                ops = c.count_ops()
                print(f"  θ={t:.3f}  move {ops.get('move', 0):3d}  cz {ops.get('cz', 0):3d}  1q on resonator {on_res}"
                      f"  illegal MOVEs {sum(p > 1e-9 for p in p11):2d} (max P11 {max(p11, default=0):.2f})"
                      f"  HF {hf:.3f}  leaked {1 - kept:.3f}  HF leak {hf_leak:.3f}")


if __name__ == "__main__":
    main()
