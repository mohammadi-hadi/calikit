import pytest

from calikit.binning import bin_predictions, ece, mce


def test_width_bins_hand_computed():
    probs = [0.9, 0.8, 0.3, 0.2]
    labels = [1, 0, 0, 0]
    bins = bin_predictions(probs, labels, k=2, scheme="width")
    assert len(bins) == 2
    low, high = bins
    assert low.n == 2 and low.conf == pytest.approx(0.25) and low.acc == 0.0
    assert high.n == 2 and high.conf == pytest.approx(0.85) and high.acc == 0.5
    assert ece(bins) == pytest.approx(0.3)
    assert mce(bins) == pytest.approx(0.35)


def test_width_bin_edges_include_one():
    bins = bin_predictions([1.0, 0.0], [1, 0], k=10, scheme="width")
    assert {(b.lo, b.hi) for b in bins} == {(0.0, 0.1), (0.9, 1.0)}


def test_width_drops_empty_bins():
    bins = bin_predictions([0.05, 0.06, 0.95], [0, 0, 1], k=10, scheme="width")
    assert len(bins) == 2


def test_mass_bins_split_remainder():
    probs = [i / 10 for i in range(10)]
    labels = [0] * 10
    bins = bin_predictions(probs, labels, k=3, scheme="mass")
    assert [b.n for b in bins] == [4, 3, 3]


def test_mass_bins_are_sorted_ranges():
    probs = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2]
    labels = [1, 0, 1, 0, 1, 0]
    bins = bin_predictions(probs, labels, k=2, scheme="mass")
    assert bins[0].hi <= bins[1].lo
    assert bins[0].n + bins[1].n == 6


def test_bin_summaries_are_means():
    bins = bin_predictions([0.2, 0.4], [1, 0], k=2, scheme="mass")
    assert bins[0].conf == pytest.approx(0.2) and bins[0].acc == 1.0
    assert bins[1].conf == pytest.approx(0.4) and bins[1].acc == 0.0


def test_validation_errors():
    with pytest.raises(ValueError, match="no predictions"):
        bin_predictions([], [], k=2)
    with pytest.raises(ValueError, match="labels"):
        bin_predictions([0.5], [], k=2)
    with pytest.raises(ValueError, match="at least 2"):
        bin_predictions([0.5, 0.6], [0, 1], k=1)
    with pytest.raises(ValueError, match="scheme"):
        bin_predictions([0.5, 0.6], [0, 1], k=2, scheme="quantile")
    with pytest.raises(ValueError, match="more bins"):
        bin_predictions([0.5, 0.6], [0, 1], k=3, scheme="mass")
