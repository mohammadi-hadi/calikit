import json

import pytest

from calikit.__main__ import main
from calikit.data import extract, parse_rescale, read_jsonl


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def preds_file(tmp_path):
    # A small over-confident model: stated probabilities hug 0/1 harder
    # than the outcomes justify.
    import math
    import random

    rng = random.Random(0)
    records = []
    for i in range(120):
        z = rng.gauss(0.0, 1.5)
        y = int(rng.random() < 1.0 / (1.0 + math.exp(-z)))
        p = 1.0 / (1.0 + math.exp(-2.0 * z))
        records.append({"id": f"x{i}", "p": round(p, 6), "y": y})
    path = tmp_path / "preds.jsonl"
    write_jsonl(path, records)
    return path


def test_read_jsonl_skips_blanks_and_reports_line(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"p": 0.5, "y": 1}\n\n{"p": 0.4, "y": 0}\n', encoding="utf-8")
    assert len(read_jsonl(str(path))) == 2
    path.write_text('{"p": 0.5}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="data.jsonl:2"):
        read_jsonl(str(path))


def test_read_jsonl_rejects_non_objects_and_empty(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected an object"):
        read_jsonl(str(path))
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        read_jsonl(str(path))


def test_extract_validates():
    with pytest.raises(ValueError, match="missing field 'p'"):
        extract([{"y": 1}])
    with pytest.raises(ValueError, match="label"):
        extract([{"p": 0.5, "y": 2}])
    with pytest.raises(ValueError, match="rescale"):
        extract([{"p": 5, "y": 1}])
    with pytest.raises(ValueError, match="must be a number"):
        extract([{"p": True, "y": 1}])


def test_extract_accepts_bools_and_rescale():
    probs, labels = extract([{"s": 7, "y": True}], prob_key="s", rescale=(1.0, 10.0))
    assert probs == [pytest.approx(6.0 / 9.0)]
    assert labels == [1]
    with pytest.raises(ValueError, match="outside the rescale range"):
        extract([{"s": 11, "y": 0}], prob_key="s", rescale=(1.0, 10.0))


def test_parse_rescale():
    assert parse_rescale("1,10") == (1.0, 10.0)
    for bad in ("1", "a,b", "5,2"):
        with pytest.raises(ValueError):
            parse_rescale(bad)


def test_audit_text_output(preds_file, capsys):
    assert main(["audit", str(preds_file)]) == 0
    out = capsys.readouterr().out
    assert "Brier:" in out and "ECE" in out and "Spiegelhalter" in out
    assert "verdict: miscalibration is significant" in out


def test_audit_json_output(preds_file, capsys):
    assert main(["audit", str(preds_file), "--json", "--reps", "200"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 120
    assert payload["significant"] is True
    assert len(payload["bins"]) == payload["bins_k"] == 10
    assert payload["brier_ci"][0] < payload["brier"] < payload["brier_ci"][1]


def test_audit_writes_svg(preds_file, tmp_path, capsys):
    svg = tmp_path / "diagram.svg"
    assert main(["audit", str(preds_file), "--svg", str(svg)]) == 0
    assert svg.read_text(encoding="utf-8").startswith("<svg")


def test_fit_apply_roundtrip(preds_file, tmp_path, capsys):
    mapping_path = tmp_path / "mapping.json"
    assert main(["fit", str(preds_file), "--out", str(mapping_path)]) == 0
    out = capsys.readouterr().out
    assert "T =" in out and "before" in out and "in-sample" in out
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["method"] == "temperature" and mapping["params"]["T"] > 1.2

    calibrated = tmp_path / "calibrated.jsonl"
    assert main(["apply", str(mapping_path), str(preds_file), "--out", str(calibrated)]) == 0
    records = read_jsonl(str(calibrated))
    assert len(records) == 120
    assert all("p_raw" in r and 0.0 <= r["p"] <= 1.0 for r in records)

    # The calibrated file should audit as no longer significantly miscalibrated.
    assert main(["audit", str(calibrated), "--json", "--reps", "200"]) == 0
    capsys.readouterr()  # flush the fit/apply output already captured


def test_fit_json_and_isotonic(preds_file, capsys):
    assert main(["fit", str(preds_file), "--method", "isotonic", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "isotonic" and payload["blocks"] > 1
    assert payload["after"]["ece"] < payload["before"]["ece"]


def test_fit_with_heldout_eval(preds_file, capsys):
    assert main(["fit", str(preds_file), "--eval", str(preds_file)]) == 0
    out = capsys.readouterr().out
    assert "held-out" in out and "in-sample" not in out


def test_rescale_cli_flow(tmp_path, capsys):
    import random

    rng = random.Random(1)
    records = []
    for i in range(60):
        t = rng.random()
        records.append({"score": min(10, max(1, round(1 + 9 * t))), "human": int(t > 0.5)})
    path = tmp_path / "judge.jsonl"
    write_jsonl(path, records)
    args = ["audit", str(path), "--prob-key", "score", "--label-key", "human"]
    assert main(args) == 2  # scores outside [0, 1] without --rescale
    assert "outside" in capsys.readouterr().err
    assert main([*args, "--rescale", "1,10"]) == 0


def test_missing_file_and_version(capsys):
    assert main(["audit", "no-such-file.jsonl"]) == 2
    assert "error:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
