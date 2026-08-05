import random

import pytest

from calikit.ci import bootstrap_ci
from calikit.metrics import brier


def sample(n: int, seed: int) -> tuple[list[float], list[int]]:
    rng = random.Random(seed)
    probs = [rng.random() for _ in range(n)]
    labels = [int(rng.random() < p) for p in probs]
    return probs, labels


def test_bootstrap_is_deterministic():
    probs, labels = sample(200, 0)
    first = bootstrap_ci(probs, labels, brier, seed=7)
    second = bootstrap_ci(probs, labels, brier, seed=7)
    assert first == second
    assert bootstrap_ci(probs, labels, brier, seed=8) != first


def test_bootstrap_brackets_the_point_estimate():
    probs, labels = sample(300, 1)
    lo, hi = bootstrap_ci(probs, labels, brier, reps=500, seed=0)
    assert lo < brier(probs, labels) < hi


def test_bootstrap_narrows_with_more_data():
    small = sample(50, 2)
    large = sample(2000, 2)
    lo_s, hi_s = bootstrap_ci(*small, brier, reps=500, seed=0)
    lo_l, hi_l = bootstrap_ci(*large, brier, reps=500, seed=0)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_validation():
    with pytest.raises(ValueError, match="no predictions"):
        bootstrap_ci([], [], brier)
    probs, labels = sample(50, 3)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci(probs, labels, brier, level=0.4)


def test_bootstrap_reports_persistent_failure():
    def always_fails(probs, labels):
        raise ValueError("nope")

    probs, labels = sample(50, 4)
    with pytest.raises(ValueError, match="resamples failed"):
        bootstrap_ci(probs, labels, always_fails, reps=50)
