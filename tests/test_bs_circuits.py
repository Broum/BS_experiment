"""Checks the PDF circuits against the exact truncated beam-splitter unitary."""

import unittest

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from utils.bs_circuits import (
    bind,
    bs_11_circuit,
    bs_circuit,
    bs_experiment,
    exact_bs_unitary,
    exact_fock_probs,
    make_bs,
    theory_11,
)

THETAS = [0.0, 0.1, np.pi / 8, np.pi / 4, 0.7, np.pi / 2]


def fock_permutation() -> np.ndarray:
    """Maps Fock index 4 * n1 + n2 to the qiskit little-endian qubit index."""
    p = np.zeros((16, 16))
    for n1 in range(4):
        for n2 in range(4):
            q0, q1, q2, q3 = n1 >> 1, n1 & 1, n2 >> 1, n2 & 1
            p[q0 + 2 * q1 + 4 * q2 + 8 * q3, 4 * n1 + n2] = 1
    return p


def circuit_fock_probs(qc) -> dict:
    """Ideal probabilities over (n1, n2) of a measured circuit."""
    probs = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict()
    out: dict = {}
    for k, v in probs.items():
        b = k[::-1]
        key = (2 * int(b[0]) + int(b[1]), 2 * int(b[2]) + int(b[3]))
        out[key] = out.get(key, 0.0) + v
    return out


def assert_probs_equal(case, p, q):
    for k in set(p) | set(q):
        case.assertAlmostEqual(p.get(k, 0.0), q.get(k, 0.0), places=10, msg=str(k))


class BSCircuitTest(unittest.TestCase):
    def test_unitary_matches_exact(self):
        p = fock_permutation()
        for phi in (0.0, 0.4):
            for theta in THETAS:
                u = p.T @ Operator(bs_circuit(theta, phi)).data @ p
                e = exact_bs_unitary(theta, phi)
                ph = np.vdot(e.ravel(), u.ravel()) / 16  # global phase
                np.testing.assert_allclose(u, ph * e, atol=1e-10, err_msg=f"theta={theta}, phi={phi}")

    def test_all_inputs(self):
        for n1 in range(4):
            for n2 in range(4):
                assert_probs_equal(self, circuit_fock_probs(bs_experiment(0.3, n1, n2)), exact_fock_probs(0.3, n1, n2))

    def test_11_theory(self):
        for theta in THETAS:
            assert_probs_equal(self, circuit_fock_probs(bs_experiment(theta)), theory_11(theta))
            assert_probs_equal(self, circuit_fock_probs(bs_11_circuit(theta)), theory_11(theta))

    def test_symbolic_theta(self):
        for circuit, inputs in [("simplified", [(1, 1)]), ("full", [(1, 1), (2, 1), (3, 0)])]:
            for n1, n2 in inputs:
                general = make_bs(circuit=circuit, n1=n1, n2=n2)
                self.assertEqual(len(general.parameters), 1)
                for theta, qc in zip(THETAS, bind(general, THETAS)):
                    assert_probs_equal(self, circuit_fock_probs(qc), exact_fock_probs(theta, n1, n2))


if __name__ == "__main__":
    unittest.main()
