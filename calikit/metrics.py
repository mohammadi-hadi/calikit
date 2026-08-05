"""Scoring rules and calibration statistics."""

from __future__ import annotations

import math

from calikit.binning import Bin

LOG_EPS = 1e-12


def brier(probs: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted probability and outcome."""
    _check(probs, labels)
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def log_loss(probs: list[float], labels: list[int]) -> float:
    """Mean negative log likelihood, with probabilities clipped away from 0/1."""
    _check(probs, labels)
    total = 0.0
    for p, y in zip(probs, labels):
        p = min(1.0 - LOG_EPS, max(LOG_EPS, p))
        total += -math.log(p) if y else -math.log(1.0 - p)
    return total / len(probs)


def auc(probs: list[float], labels: list[int]) -> float:
    """Area under the ROC curve via the rank-sum formula, ties averaged."""
    _check(probs, labels)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both positive and negative labels")
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # ranks are 1-based
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def spiegelhalter(probs: list[float], labels: list[int]) -> tuple[float, float]:
    """Spiegelhalter's Z test for calibration: (z, two-sided p-value).

    Under the null that every stated probability is correct, the Brier score
    has known mean and variance; Z standardizes the observed score against
    them. Large |Z| means the miscalibration is too big to be sampling noise.
    """
    _check(probs, labels)
    num = sum((y - p) * (1.0 - 2.0 * p) for p, y in zip(probs, labels))
    var = sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p) for p in probs)
    if var <= 0.0:
        return 0.0, 1.0
    z = num / math.sqrt(var)
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))
    return z, p_value


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def decomposition(labels: list[int], bins: list[Bin]) -> tuple[float, float, float]:
    """Murphy decomposition from binned forecasts: (reliability, resolution, uncertainty).

    Brier = reliability - resolution + uncertainty, exactly when forecasts
    within a bin are identical and approximately otherwise. Reliability is the
    part you can remove by recalibrating; resolution is the discrimination the
    forecasts actually have; uncertainty is the base rate's irreducible floor.
    """
    n = sum(b.n for b in bins)
    if n != len(labels):
        raise ValueError(f"bins cover {n} items but there are {len(labels)} labels")
    base = sum(labels) / len(labels)
    rel = sum(b.n * (b.conf - b.acc) ** 2 for b in bins) / n
    res = sum(b.n * (b.acc - base) ** 2 for b in bins) / n
    unc = base * (1.0 - base)
    return rel, res, unc


def _check(probs: list[float], labels: list[int]) -> None:
    if not probs:
        raise ValueError("no predictions")
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} predictions but {len(labels)} labels")
