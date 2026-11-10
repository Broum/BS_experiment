"""Truncated quantum beam splitter (Fock states |0>..|3>) as 4-qubit circuits, after the GBS4EO BS paper.

Encoding: mode 1 = (q0, q1), mode 2 = (q2, q3); n = 2 * q_msb + q_lsb.
"""

from __future__ import annotations

import numpy as np
from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from scipy.linalg import expm

FOCK_CUTOFF = 4
THETA = Parameter("θ")  # symbolic theta for the general circuit
MODE_QUBITS = ((0, 1), (2, 3))  # (msb, lsb) per mode


def phase_shift(qc: QuantumCircuit, phi: float, msb: int, lsb: int) -> None:
    """U(phi) = exp(-i phi N) on one mode (up to global phase)."""
    qc.rz(-2 * phi, msb)
    qc.rz(-phi, lsb)


def m_gate(qc: QuantumCircuit, a: int = 2, b: int = 3) -> None:
    """M: S on a, S H on b, CNOT(b -> a)."""
    qc.s(a)
    qc.s(b)
    qc.h(b)
    qc.cx(b, a)


def m_dagger(qc: QuantumCircuit, a: int = 2, b: int = 3) -> None:
    """M^dagger."""
    qc.cx(b, a)
    qc.h(b)
    qc.sdg(b)
    qc.sdg(a)


def v_odd(qc: QuantumCircuit, theta: float) -> None:
    """V_odd: sectors U1, U3, U5 (acts for q1 = 1)."""
    qc.rx(-pi / 4, 2)
    qc.rx(5 * pi / 12, 3)
    qc.cy(0, 2)
    qc.cy(0, 3)
    qc.rx(-pi / 4, 2)
    qc.rx(pi / 12, 3)
    qc.cry(-2 * theta, 1, 2)
    qc.cry(-4 * theta, 1, 3)
    qc.rx(pi / 4, 2)
    qc.rx(-pi / 12, 3)
    qc.cy(0, 2)
    qc.cy(0, 3)
    qc.rx(pi / 4, 2)
    qc.rx(-5 * pi / 12, 3)


def xi(theta: float) -> tuple[float, float]:
    """xi_+/- = -(sqrt(3) +/- 1) / 2 * theta."""
    return -(np.sqrt(3) + 1) / 2 * theta, -(np.sqrt(3) - 1) / 2 * theta


def v_even(qc: QuantumCircuit, theta: float, q1_is_zero: bool = False) -> None:
    """V_even: sectors U2, U4 (acts for q1 = 0).

    ``q1_is_zero`` drops the q1-controlled CNOTs (valid when q1 is known to be 0, e.g. input |1>|1>).
    """
    xp, xm = xi(theta)
    qc.cy(0, 2)
    qc.rx(-pi / 4, 2)
    qc.rx(pi / 4, 3)
    for t in (2, 3):
        if q1_is_zero:
            qc.ry(2 * xp, t)
            qc.cx(0, t)
            qc.ry(2 * xm, t)
            qc.cx(0, t)
        else:
            qc.ry(xp, t)
            qc.cx(1, t)
            qc.ry(xp, t)
            qc.cx(0, t)
            qc.ry(xm, t)
            qc.cx(1, t)
            qc.ry(xm, t)
            qc.cx(0, t)
    qc.rx(-pi / 4, 3)
    qc.rx(pi / 4, 2)
    qc.cy(0, 2)


def bs_circuit(theta: float, phi: float = 0.0) -> QuantumCircuit:
    """Full U_BS(theta, phi) on 4 qubits (no state preparation, no measurement)."""
    qc = QuantumCircuit(4, name="BS")
    if phi:
        phase_shift(qc, phi, *MODE_QUBITS[0])
    qc.cx(2, 0)
    qc.cx(3, 1)
    m_gate(qc)
    v_odd(qc, theta)
    m_dagger(qc)
    qc.cx(3, 0)
    m_gate(qc)
    v_even(qc, theta)
    m_dagger(qc)
    qc.cx(3, 0)
    qc.cx(3, 1)
    qc.cx(2, 0)
    if phi:
        phase_shift(qc, -phi, *MODE_QUBITS[0])
    return qc


def prepare_fock(qc: QuantumCircuit, n1: int, n2: int) -> None:
    """X gates preparing |n1>|n2>."""
    for n, (msb, lsb) in zip((n1, n2), MODE_QUBITS):
        if n & 2:
            qc.x(msb)
        if n & 1:
            qc.x(lsb)


def bs_experiment(theta: float, n1: int = 1, n2: int = 1, phi: float = 0.0) -> QuantumCircuit:
    """|n1>|n2> -> U_BS(theta, phi) -> measure all."""
    qc = QuantumCircuit(4)
    prepare_fock(qc, n1, n2)
    qc.barrier()
    qc.compose(bs_circuit(theta, phi), inplace=True)
    qc.measure_all()
    return qc


def bs_11_circuit(theta: float) -> QuantumCircuit:
    """Simplified circuit for input |1>|1>: only the even sector contributes.

    After the first CNOTs the state is |q0 q1 q2 q3> = |0001>, so V_odd (and M, M^dagger around it) is the
    identity; the remaining CNOT(q3 -> q0) gives |1001>, prepared directly here.
    """
    qc = QuantumCircuit(4, name="BS11")
    qc.x(0)
    qc.x(3)
    qc.barrier()
    m_gate(qc)
    v_even(qc, theta, q1_is_zero=True)
    m_dagger(qc)
    qc.cx(3, 0)
    qc.cx(3, 1)
    qc.cx(2, 0)
    qc.measure_all()
    return qc


def make_bs(theta=THETA, circuit: str = "full", n1: int = 1, n2: int = 1) -> QuantumCircuit:
    """Measured BS circuit: 'full' (any input |n1>|n2>) or 'simplified' (|1>|1> only).

    ``theta`` is a number or, by default, the symbolic ``THETA``; give it values with :func:`bind`.
    """
    if circuit == "simplified":
        if (n1, n2) != (1, 1):
            raise ValueError("the simplified circuit needs input |1>|1>; use circuit='full'")
        return bs_11_circuit(theta)
    if circuit == "full":
        return bs_experiment(theta, n1, n2)
    raise ValueError(f"circuit must be 'full' or 'simplified', got {circuit!r}")


def bind(qc: QuantumCircuit, thetas) -> list[QuantumCircuit]:
    """One circuit per theta value from a circuit with the symbolic ``THETA``."""
    return [qc.assign_parameters({THETA: float(t)}) for t in thetas]


def bitstring_to_fock(bitstring: str) -> tuple[int, int]:
    """Qiskit key 'q3q2q1q0' -> (n1, n2)."""
    b = bitstring.replace(" ", "")[::-1]
    return 2 * int(b[0]) + int(b[1]), 2 * int(b[2]) + int(b[3])


def counts_to_fock(counts: dict[str, int]) -> dict[tuple[int, int], float]:
    """Counts -> probabilities over (n1, n2)."""
    shots = sum(counts.values())
    probs: dict[tuple[int, int], float] = {}
    for k, v in counts.items():
        key = bitstring_to_fock(k)
        probs[key] = probs.get(key, 0.0) + v / shots
    return probs


def theory_11(theta: float) -> dict[tuple[int, int], float]:
    """Paper eqs. (13)-(15) for input |1>|1>."""
    s = 0.5 * np.sin(2 * theta) ** 2
    return {(2, 0): s, (1, 1): np.cos(2 * theta) ** 2, (0, 2): s}


def exact_bs_unitary(theta: float, phi: float = 0.0) -> np.ndarray:
    """Truncated exp[theta (e^{i phi} b1^dag b2 - e^{-i phi} b1 b2^dag)], basis index 4 * n1 + n2."""
    b = np.diag(np.sqrt(np.arange(1, FOCK_CUTOFF)), 1)
    g = np.exp(1j * phi) * np.kron(b.T, b) - np.exp(-1j * phi) * np.kron(b, b.T)
    return expm(theta * g)


def exact_fock_probs(theta: float, n1: int = 1, n2: int = 1, phi: float = 0.0) -> dict[tuple[int, int], float]:
    """Exact output distribution over (n1, n2) for input |n1>|n2>."""
    amp = exact_bs_unitary(theta, phi)[:, 4 * n1 + n2]
    return {(i // 4, i % 4): float(abs(a) ** 2) for i, a in enumerate(amp) if abs(a) ** 2 > 1e-12}


def postselect(probs: dict[tuple[int, int], float], total: int = 2) -> tuple[dict[tuple[int, int], float], float]:
    """Keeps outcomes with n1 + n2 == total, renormalised; returns them and the kept fraction."""
    kept = {k: v for k, v in probs.items() if sum(k) == total}
    frac = sum(kept.values())
    return ({k: v / frac for k, v in kept.items()} if frac else kept), frac


def draw_ordered(qc: QuantumCircuit, **kwargs):
    """Draws gates in the order they were added (one column each), like the PDF figures.

    Qiskit's default drawing shifts gates on disjoint qubits left; the circuit itself is unchanged.
    """
    d = qc.copy_empty_like()
    for inst in qc.data:
        d.append(inst)
        d.barrier()
    return d.draw("mpl", fold=-1, plot_barriers=False, **kwargs)
