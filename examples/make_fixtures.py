"""Regenerate the example data. Seeded, so the files are reproducible.

python examples/make_fixtures.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(name: str, records: list[dict]) -> None:
    path = HERE / name
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec) + "\n" for rec in records)
    print(f"wrote {path.name} ({len(records)} records)")


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def main() -> None:
    rng = random.Random(42)

    # A classifier scored on 500 items. The outcome follows the true signal,
    # but the model's probabilities use logits scaled by 1.8 with extra noise:
    # a confidently wrong model, the most common failure in the wild.
    preds = []
    for i in range(500):
        z_true = rng.gauss(0.0, 1.4)
        y = int(rng.random() < sigmoid(z_true))
        z_model = 1.8 * (0.85 * z_true + rng.gauss(0.0, 0.5))
        preds.append({"id": f"item-{i:03d}", "p": round(sigmoid(z_model), 6), "y": y})
    write("preds.jsonl", preds)

    # An LLM judge scoring 300 answers 1-10, with human pass/fail labels.
    # The judge pushes toward the extremes (over-confident) and is noisy.
    judged = []
    for i in range(300):
        t = rng.random()
        human = int(rng.random() < t)
        est = 0.5 + 1.25 * (t - 0.5) + rng.gauss(0.0, 0.15)
        est = min(1.0, max(0.0, est))
        score = min(10, max(1, round(1 + 9 * est)))
        judged.append({"id": f"ans-{i:03d}", "score": score, "human": human})
    write("judge_scores.jsonl", judged)


if __name__ == "__main__":
    main()
