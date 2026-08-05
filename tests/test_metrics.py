import math
import random

import pytest

from calikit.binning import bin_predictions
from calikit.metrics import auc, brier, decomposition, log_loss, spiegelhalter


def test_brier_hand_computed():
    assert brier([1.0, 0.0, 0.5], [1, 0, 1]) == pytest.approx(0.25 / 3)


def test_log_loss_hand_computed():
    assert log_loss([0.5, 0.5], [0, 1]) == pytest.approx(math.log(2.0))


def test_log_loss_clips_impossible_predictions():
    assert math.isfinite(log_loss([0.0], [1]))


def test_auc_hand_computed():
    # Positive items outrank negatives in 3 of 4 cross pairs.
    assert auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_auc_ties_average():
    assert auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


def test_auc_perfect_and_inverted():
    assert auc([0.1, 0.9], [0, 1]) == 1.0
    assert auc([0.9, 0.1], [0, 1]) == 0.0


def test_auc_needs_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        auc([0.2, 0.8], [1, 1])


def test_spiegelhalter_hand_computed():
    z, p = spiegelhalter([0.8, 0.4], [1, 0])
    # num = (1-.8)(1-1.6) + (0-.4)(1-.8) = -0.2
    # var = (1-1.6)^2*.8*.2 + (1-.8)^2*.4*.6 = 0.0672
    assert z == pytest.approx(-0.2 / math.sqrt(0.0672))
    assert 0.0 < p < 1.0


def test_spiegelhalter_calibrated_model_is_insignificant():
    rng = random.Random(1)
    probs = [rng.random() for _ in range(4000)]
    labels = [int(rng.random() < p) for p in probs]
    z, _p = spiegelhalter(probs, labels)
    assert abs(z) < 3.0


def test_spiegelhalter_overconfident_model_is_significant():
    rng = random.Random(2)
    probs, labels = [], []
    for _ in range(2000):
        t = rng.random()
        labels.append(int(rng.random() < t))
        # Push the stated probability toward the extremes.
        probs.append(min(0.999, max(0.001, 0.5 + 1.6 * (t - 0.5))))
    z, p = spiegelhalter(probs, labels)
    assert abs(z) > 4.0
    assert p < 0.001


def test_spiegelhalter_degenerate_half_probabilities():
    z, p = spiegelhalter([0.5, 0.5], [0, 1])
    assert (z, p) == (0.0, 1.0)


def test_decomposition_identity_on_discrete_forecasts():
    # Forecasts take one distinct value per bin, so the Murphy decomposition
    # is exact: Brier = reliability - resolution + uncertainty.
    rng = random.Random(3)
    values = [0.15, 0.55, 0.85]
    probs = [rng.choice(values) for _ in range(300)]
    labels = [int(rng.random() < p * 0.8) for p in probs]
    bins = bin_predictions(probs, labels, k=10, scheme="width")
    rel, res, unc = decomposition(labels, bins)
    assert rel - res + unc == pytest.approx(brier(probs, labels))


def test_decomposition_size_mismatch():
    bins = bin_predictions([0.2, 0.8], [0, 1], k=2, scheme="width")
    with pytest.raises(ValueError, match="labels"):
        decomposition([0, 1, 1], bins)


def test_empty_inputs_error():
    with pytest.raises(ValueError, match="no predictions"):
        brier([], [])
    with pytest.raises(ValueError, match="labels"):
        log_loss([0.5], [])
