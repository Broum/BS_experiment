"""Reusable experiment-circuit builders.

Lifted verbatim from `experiment_N2.1.ipynb` (cells `helpers`, `e5a236ea`,
`analytical-formula`) and `experiment_N4.1.ipynb` (cell `f6b501f1`).
The only new addition is `build_ghz()`.
"""

from __future__ import annotations

from math import pi as _pi

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ----------------------------------------------------------------------
# Pyramid beam-splitter experiment (N2.1)
# ----------------------------------------------------------------------

def qc_bs(a: int, b: int, n: int) -> QuantumCircuit:
    """Beam-splitter gadget: H-CX-Ry(-pi/4)^2-CX-H on qubits (a, b) in an n-qubit register.

    Implements a 50:50 beam splitter that preserves total particle number
    (excitation count). Maps computational basis states as:
      |00> -> |00>,  |01> -> (|01>+|10>)/sqrt(2),
      |10> -> (|01>-|10>)/sqrt(2),  |11> -> |11>.
    """
    qc = QuantumCircuit(n)
    qc.h(a)
    qc.cx(a, b)
    qc.ry(-_pi / 4, a)
    qc.ry(-_pi / 4, b)
    qc.cx(a, b)
    qc.h(a)
    return qc


def build_pyramid(L: int) -> QuantumCircuit:
    """Pyramid BS circuit with L layers on nq=2L qubits, excitation at center=L-1."""
    nq = 2 * L
    center = L - 1
    qc = QuantumCircuit(nq)
    qc.x(center)
    qc.barrier()
    for k in range(1, L + 1):
        start = center - (k - 1)
        for i in range(k):
            a = start + 2 * i
            qc = qc.compose(qc_bs(a, a + 1, nq))
        qc.barrier()
    qc.measure_all()
    return qc


def analytical_probabilities(L: int) -> tuple[dict[str, float], list[float]]:
    """Exact noise-free output of the pyramid via Qiskit's Statevector simulator.

    Returns `(probs_dict, hw_prob)` where `probs_dict` is bitstring -> probability
    and `hw_prob[k]` is the total probability mass at Hamming weight k.
    """
    nq = 2 * L
    qc = build_pyramid(L).remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc)
    probs = {k: float(v) for k, v in sv.probabilities_dict().items()}
    hw_prob = [0.0] * (nq + 1)
    for bitstring, p in probs.items():
        k = bitstring.count("1")
        hw_prob[k] += p
    return probs, hw_prob


def pyramid_statevector(n_layers: int) -> np.ndarray:
    """Full 2**N statevector after a size-n BS pyramid on a single excitation,
    computed analytically (no Qiskit, no statevector simulator).

    The pyramid preserves total excitation number; on a single-excitation input
    the dynamics live in an N-dimensional subspace, so we propagate N complex
    amplitudes through O(n^2) 2x2 BS multiplications and embed back into 2**N.
    """
    N = 2 * n_layers
    p0 = n_layers - 1  # initial excitation qubit

    c = np.zeros(N, dtype=complex)
    c[p0] = 1.0

    BS = np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2)
    for L in range(1, n_layers + 1):
        for off in range(-(L - 1), L, 2):
            i = p0 + off
            c[i], c[i + 1] = BS @ c[i:i + 2]

    # Embed into the full Hilbert space using Qiskit's little-endian convention,
    # i.e. qubit k corresponds to bit k of the integer index. This is what
    # `Statevector.probabilities_dict()` uses, so the resulting bitstrings line
    # up directly with `analytical_probabilities(L)[0]`.
    sv = np.zeros(2 ** N, dtype=complex)
    for k in range(N):
        sv[1 << k] = c[k]   # qubit k = bit k (little-endian)
    return sv


def pyramid_probabilities_analytical(n_layers: int) -> dict[str, float]:
    """Bitstring -> probability dict derived from `pyramid_statevector`.
    Same key format as `analytical_probabilities(L)[0]` (Qiskit little-endian).
    """
    sv = pyramid_statevector(n_layers)
    nq = 2 * n_layers
    probs = {}
    for idx, amp in enumerate(sv):
        p = float(abs(amp) ** 2)
        if p > 0:
            probs[format(idx, f"0{nq}b")] = p
    return probs


# ----------------------------------------------------------------------
# Generic helpers used in both N2.1 and N4.1
# ----------------------------------------------------------------------

def prepare_basis_state(qc: QuantumCircuit, bitstring: str) -> None:
    """Prepare |bitstring> on `qc` using X gates only (in place).

    Qiskit uses little-endian ordering: bitstring 'abcd' means q0=d, q1=c, q2=b, q3=a.
    """
    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            qc.x(i)


# ----------------------------------------------------------------------
# GHZ circuit (new in this module — referenced by experiment_GHZ.ipynb)
# ----------------------------------------------------------------------

def build_ghz(n_qubits: int) -> QuantumCircuit:
    """Standard GHZ-N preparation: H on qubit 0, then a linear chain of CXs.

    Produces the state (|0...0> + |1...1>) / sqrt(2) on `n_qubits` qubits.
    Includes a measurement on all qubits so it can be run on `backend.run(...)`.
    """
    if n_qubits < 2:
        raise ValueError(f"GHZ needs >= 2 qubits, got {n_qubits}")
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for q in range(n_qubits - 1):
        qc.cx(q, q + 1)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc
