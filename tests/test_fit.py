import math
import random

import pytest

from calikit.fit import Mapping, fit_isotonic, fit_platt, fit_temperature
from calikit.metrics import log_loss


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def overconfident_sample(n: int, scale: float, seed: int) -> tuple[list[float], list[int]]:
    rng = random.Random(seed)
    probs, labels = [], []
    for _ in range(n):
        z = rng.gauss(0.0, 1.5)
        labels.append(int(rng.random() < sigmoid(z)))
        probs.append(sigmoid(scale * z))
    return probs, labels


def test_temperature_recovers_known_scale():
    # Probabilities use logits scaled by 1.6, so the optimal temperature
    # divides them back: T should land near 1.6.
    probs, labels = overconfident_sample(4000, scale=1.6, seed=0)
    mapping = fit_temperature(probs, labels)
    assert mapping.params["T"] == pytest.approx(1.6, abs=0.15)


def test_temperature_improves_log_loss():
    probs, labels = overconfident_sample(1000, scale=2.0, seed=1)
    mapping = fit_temperature(probs, labels)
    assert log_loss(mapping.apply(probs), labels) < log_loss(probs, labels)


def test_temperature_near_one_for_calibrated_input():
    probs, labels = overconfident_sample(4000, scale=1.0, seed=2)
    mapping = fit_temperature(probs, labels)
    assert mapping.params["T"] == pytest.approx(1.0, abs=0.1)


def test_platt_fixes_a_biased_model():
    rng = random.Random(3)
    probs, labels = [], []
    for _ in range(2000):
        z = rng.gauss(0.0, 1.5)
        labels.append(int(rng.random() < sigmoid(z)))
        probs.append(sigmoid(z + 1.0))  # shifted: systematically too high
    mapping = fit_platt(probs, labels)
    assert mapping.params["b"] == pytest.approx(-1.0, abs=0.25)
    assert mapping.params["a"] == pytest.approx(1.0, abs=0.15)
    assert log_loss(mapping.apply(probs), labels) < log_loss(probs, labels)


def test_isotonic_pava_hand_computed():
    mapping = fit_isotonic([0.1, 0.2, 0.3, 0.4] * 3, [0, 1, 0, 1] * 3)
    values = [b[2] for b in mapping.blocks]
    assert values == [0.0, 0.5, 1.0]
    assert mapping.apply_one(0.25) == pytest.approx(0.5)
    assert mapping.apply_one(0.05) == 0.0
    assert mapping.apply_one(0.95) == 1.0


def test_isotonic_blocks_are_monotone():
    rng = random.Random(4)
    probs = [rng.random() for _ in range(500)]
    labels = [int(rng.random() < p) for p in probs]
    mapping = fit_isotonic(probs, labels)
    values = [b[2] for b in mapping.blocks]
    assert values == sorted(values)


def test_isotonic_gap_interpolation():
    mapping = Mapping(
        method="isotonic", blocks=[(0.1, 0.1, 0.0), (0.2, 0.3, 0.5), (0.4, 0.4, 1.0)]
    )
    assert mapping.apply_one(0.15) == pytest.approx(0.25)
    assert mapping.apply_one(0.35) == pytest.approx(0.75)


@pytest.mark.parametrize("fitter", [fit_temperature, fit_platt, fit_isotonic])
def test_mapping_roundtrips_through_json_dict(fitter):
    probs, labels = overconfident_sample(200, scale=1.5, seed=5)
    mapping = fitter(probs, labels)
    clone = Mapping.from_dict(mapping.as_dict())
    test_points = [0.05, 0.3, 0.5, 0.7, 0.95]
    assert clone.apply(test_points) == mapping.apply(test_points)


def test_fit_validation():
    with pytest.raises(ValueError, match="at least 10"):
        fit_temperature([0.5] * 5, [0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="both positive and negative"):
        fit_platt([0.5] * 12, [1] * 12)
    with pytest.raises(ValueError, match="predictions but"):
        fit_isotonic([0.5] * 12, [0] * 11)


def test_mapping_from_dict_rejects_junk():
    with pytest.raises(ValueError, match="unknown mapping method"):
        Mapping.from_dict({"method": "magic", "params": {}})
    with pytest.raises(ValueError, match="malformed"):
        Mapping.from_dict({"params": {"T": 2.0}})
