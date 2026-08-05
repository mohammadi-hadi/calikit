"""Load predictions and labels from JSONL result files."""

from __future__ import annotations

import json
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines. Errors carry the line number."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{lineno}: expected an object, got {type(rec).__name__}")
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def _as_label(value: Any, where: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    raise ValueError(f"{where}: label must be 0/1 or true/false, got {value!r}")


def _as_prob(value: Any, where: str, rescale: tuple[float, float] | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: prediction must be a number, got {value!r}")
    p = float(value)
    if rescale is not None:
        lo, hi = rescale
        if not lo <= p <= hi:
            raise ValueError(f"{where}: value {p} outside the rescale range [{lo}, {hi}]")
        p = (p - lo) / (hi - lo)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"{where}: probability {p} outside [0, 1]"
            " (scores on another scale? use --rescale LO,HI)"
        )
    return p


def extract(
    records: list[dict[str, Any]],
    prob_key: str = "p",
    label_key: str = "y",
    rescale: tuple[float, float] | None = None,
) -> tuple[list[float], list[int]]:
    """Pull (probabilities, labels) out of records, validating as we go."""
    if rescale is not None and rescale[1] <= rescale[0]:
        raise ValueError(f"rescale range must have lo < hi, got {rescale}")
    probs, labels = [], []
    for i, rec in enumerate(records, start=1):
        if prob_key not in rec:
            raise ValueError(f"record {i}: missing field {prob_key!r}")
        if label_key not in rec:
            raise ValueError(f"record {i}: missing field {label_key!r}")
        probs.append(_as_prob(rec[prob_key], f"record {i} ({prob_key!r})", rescale))
        labels.append(_as_label(rec[label_key], f"record {i} ({label_key!r})"))
    return probs, labels


def parse_rescale(text: str) -> tuple[float, float]:
    """Parse a --rescale LO,HI argument."""
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"--rescale expects LO,HI (e.g. 1,10), got {text!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"--rescale expects two numbers, got {text!r}") from exc
    if hi <= lo:
        raise ValueError(f"--rescale range must have lo < hi, got {text!r}")
    return lo, hi
