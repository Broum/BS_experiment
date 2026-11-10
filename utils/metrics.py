"""Shared distribution-comparison metrics used across the experiment notebooks.

Lifted verbatim from `experiment_N2.1.ipynb` (cells `e5a236ea`, `018073e8`,
the helper block in `f6b501f1`) and `experiment_N4.1.ipynb` (cell `40ec807b`)
so the same definitions can be imported anywhere instead of duplicated.
"""

from __future__ import annotations

from math import sqrt
from typing import Mapping, Sequence


def hamming_weight_hist(counts: Mapping[str, int], nq: int) -> list[int]:
    """Shot counts bucketed by Hamming distance to the all-zeros string 0^nq.

    Returns a list `hist` of length nq+1 where hist[k] is the total number of
    shots whose measured bitstring has exactly k ones.
    """
    hist = [0] * (nq + 1)
    for bitstring, c in counts.items():
        k = bitstring.replace(" ", "").count("1")
        hist[k] += c
    return hist


def tvd(p: Sequence[float], q: Sequence[float]) -> float:
    """Total Variation Distance between two aligned discrete distributions.

    TVD = sum(|p_i - q_i|) / 2, bounded in [0, 1].
    """
    return sum(abs(pi - qi) for pi, qi in zip(p, q)) / 2


def max_abs_diff(p: Sequence[float], q: Sequence[float]) -> float:
    """Largest absolute difference across bins: max_k |p_k - q_k|."""
    return max(abs(pi - qi) for pi, qi in zip(p, q))


def SBhattC(p: Sequence[float], q: Sequence[float]) -> float:
    """Squared Bhattacharyya coefficient (classical fidelity) between two distributions.

    F(p, q) = (sum_i sqrt(p_i * q_i))^2, bounded in [0, 1].
    F = 1 means identical distributions; F = 0 means disjoint support.
    """
    return sum(sqrt(pi * qi) for pi, qi in zip(p, q)) ** 2


def _sbc_dicts(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """SBC between two probability dicts over (possibly different) bitstring alphabets.
    Missing keys are treated as zero-probability outcomes.
    """
    keys = set(p) | set(q)
    return SBhattC(
        [p.get(k, 0.0) for k in keys],
        [q.get(k, 0.0) for k in keys],
    )


def _sbc_counts(counts_a: Mapping[str, int], counts_b: Mapping[str, int], shots: int) -> float:
    """SBC between two shot-count dicts normalised by `shots`."""
    all_keys = sorted(set(counts_a) | set(counts_b))
    p = [counts_a.get(k, 0) / shots for k in all_keys]
    q = [counts_b.get(k, 0) / shots for k in all_keys]
    return SBhattC(p, q)
