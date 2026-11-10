"""Bootstrap error bars and null-floor estimation for distribution comparisons.

Implements the recipes documented in
`VLQ_vs_FakeVLQ/iqm_metrics_and_error_bars.md` (sections 3-6). The default
statistic across these helpers is `qiskit.quantum_info.hellinger_fidelity`
which equals the squared Bhattacharyya coefficient (= classical fidelity).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


CountsDict = Mapping[str, int]
Statistic = Callable[[CountsDict, CountsDict], float]


def bootstrap_compare(counts_a: CountsDict,
                      counts_b: CountsDict,
                      statistic: Statistic,
                      n_boot: int = 2000,
                      rng: np.random.Generator | None = None) -> dict:
    """Two-sample non-parametric bootstrap.

    Resamples both datasets independently each iteration (independent shot
    noise on the two devices). Returns the point estimate, the bootstrap
    standard error, the 95% percentile CI, the plug-in bias, and the
    bias-corrected estimate.

    Captures within-job shot noise only -- not job-to-job drift.
    """
    rng = rng or np.random.default_rng()
    keys = sorted(set(counts_a) | set(counts_b))
    a = np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    b = np.array([counts_b.get(k, 0) for k in keys], dtype=float)
    Na, Nb = int(a.sum()), int(b.sum())
    p_a, p_b = a / Na, b / Nb

    theta_hat = float(statistic(counts_a, counts_b))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.multinomial(Na, p_a)
        rb = rng.multinomial(Nb, p_b)
        boot[i] = statistic(dict(zip(keys, ra.tolist())),
                            dict(zip(keys, rb.tolist())))
    bias = float(boot.mean() - theta_hat)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "estimate":       theta_hat,
        "std_error":      float(boot.std(ddof=1)),
        "ci95":           (lo, hi),
        "bias":           bias,
        "bias_corrected": theta_hat - bias,
        "n_boot":         n_boot,
        "Na":             Na,
        "Nb":             Nb,
    }


def null_floor(counts_ref: CountsDict,
               statistic: Statistic,
               N: int | None = None,
               n_rep: int = 2000,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Empirical distribution of `statistic` for two independent N-shot draws
    from the SAME distribution -- the value when the noise model is perfect.

    Compare a measured SBhattC against this band: values inside the band are
    consistent with a correct noise model (mismatch buried in shot noise),
    values clearly below indicate a real model/hardware discrepancy.
    """
    rng = rng or np.random.default_rng()
    keys = sorted(counts_ref)
    p = np.array([counts_ref[k] for k in keys], dtype=float)
    if N is None:
        N = int(p.sum())
    p = p / p.sum()
    out = np.empty(n_rep)
    for i in range(n_rep):
        a = rng.multinomial(N, p)
        b = rng.multinomial(N, p)
        out[i] = statistic(dict(zip(keys, a.tolist())),
                           dict(zip(keys, b.tolist())))
    return out


def shot_budget_sweep(probs_or_counts: Mapping[str, float],
                      candidates: Sequence[int] = (1024, 2048, 4096, 8192, 16384),
                      statistic: Statistic | None = None,
                      n_rep: int = 1000,
                      rng: np.random.Generator | None = None) -> dict:
    """For each candidate N, estimate the null-floor band (mean, std, 95% band).

    Use this BEFORE spending hardware time: pick the smallest N whose null band
    is narrow enough that the mismatch you care about would fall outside it.

    `probs_or_counts` is any {bitstring: weight} dict; weights are normalised.
    Defaults to Hellinger fidelity (which equals SBhattC = BC^2) when `statistic` is None.
    """
    if statistic is None:
        from qiskit.quantum_info import hellinger_fidelity
        statistic = hellinger_fidelity
    rng = rng or np.random.default_rng()
    keys = sorted(probs_or_counts)
    p = np.array([probs_or_counts[k] for k in keys], dtype=float)
    p = p / p.sum()
    report = {}
    for N in candidates:
        out = np.empty(n_rep)
        for i in range(n_rep):
            a = rng.multinomial(int(N), p)
            b = rng.multinomial(int(N), p)
            out[i] = statistic(dict(zip(keys, a.tolist())),
                               dict(zip(keys, b.tolist())))
        lo, hi = (float(x) for x in np.percentile(out, [2.5, 97.5]))
        report[int(N)] = {
            "mean": float(out.mean()),
            "std":  float(out.std(ddof=1)),
            "lo":   lo,
            "hi":   hi,
            "width": hi - lo,
        }
    return report


def effective_support(probs: Mapping[str, float], threshold: float = 1e-3) -> int:
    """Number of bitstrings with probability >= threshold (default 0.1%).

    Quick proxy for "how many appreciable bins" -- aim for >= ~100 counts per
    appreciable bin (so N >> 100 * effective_support) when targeting an absolute
    SBhattC measurement.
    """
    total = sum(probs.values())
    return sum(1 for v in probs.values() if v / total >= threshold)
