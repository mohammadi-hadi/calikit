from calikit.binning import bin_predictions
from calikit.svg import reliability_svg, write_svg


def make_bins():
    probs = [i / 20 for i in range(1, 20)]
    labels = [int(p > 0.6) for p in probs]
    return bin_predictions(probs, labels, k=5, scheme="mass")


def test_svg_contains_one_point_per_bin():
    bins = make_bins()
    text = reliability_svg(bins)
    assert text.startswith("<svg")
    assert text.count("<circle") == len(bins)
    assert "ECE" in text


def test_write_svg(tmp_path):
    path = tmp_path / "diagram.svg"
    write_svg(str(path), make_bins())
    content = path.read_text(encoding="utf-8")
    assert content.startswith("<svg") and content.rstrip().endswith("</svg>")
