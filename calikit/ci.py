"""Bootstrap confidence intervals for calibration metrics."""

from __future__ import annotations

import random
from collections.abc import Callable


def bootstrap_ci(
    probs: list[float],
    labels: list[int],
    stat: Callable[[list[float], list[int]], float],
    level: float = 0.95,
    reps: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for any statistic of (probs, labels).

    Resamples items with replacement; a statistic that fails on a resample
    (e.g. AUC with a single-class draw) is skipped, and the interval reflects
    the resamples that succeeded.
    """
    n = len(probs)
    if n == 0:
        raise ValueError("no predictions")
    if not 0.5 < level < 1.0:
        raise ValueError(f"level must be in (0.5, 1), got {level}")
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            stats.append(stat([probs[i] for i in idx], [labels[i] for i in idx]))
        except ValueError:
            continue
    if len(stats) < reps // 2:
        raise ValueError("too many bootstrap resamples failed to compute the statistic")
    stats.sort()
    alpha = 1.0 - level
    lo = stats[int(alpha / 2.0 * (len(stats) - 1))]
    hi = stats[int((1.0 - alpha / 2.0) * (len(stats) - 1))]
    return lo, hi
