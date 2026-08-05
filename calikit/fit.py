"""Fit recalibration mappings: temperature scaling, Platt scaling, isotonic regression."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

from calikit.metrics import log_loss

CLIP = 1e-6


def _logit(p: float) -> float:
    p = min(1.0 - CLIP, max(CLIP, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Mapping:
    """A fitted recalibration map that can be applied, saved, and reloaded."""

    method: str
    params: dict[str, float] = field(default_factory=dict)
    blocks: list[tuple[float, float, float]] | None = None  # isotonic: (lo, hi, value)

    def apply_one(self, p: float) -> float:
        if self.method == "temperature":
            return _sigmoid(_logit(p) / self.params["T"])
        if self.method == "platt":
            return _sigmoid(self.params["a"] * _logit(p) + self.params["b"])
        if self.method == "isotonic":
            return self._apply_isotonic(p)
        raise ValueError(f"unknown mapping method {self.method!r}")

    def apply(self, probs: list[float]) -> list[float]:
        return [self.apply_one(p) for p in probs]

    def _apply_isotonic(self, p: float) -> float:
        blocks = self.blocks or []
        if not blocks:
            raise ValueError("isotonic mapping has no blocks")
        if p <= blocks[0][1]:
            return blocks[0][2]
        if p >= blocks[-1][0]:
            return blocks[-1][2]
        # Find the last block starting at or below p.
        i = bisect.bisect_right([b[0] for b in blocks], p) - 1
        _lo, hi, value = blocks[i]
        if p <= hi:
            return value
        # p falls in the gap between block i and block i+1: interpolate.
        nlo, _, nvalue = blocks[i + 1]
        frac = (p - hi) / (nlo - hi)
        return value + frac * (nvalue - value)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "params": self.params}
        if self.blocks is not None:
            out["blocks"] = [list(b) for b in self.blocks]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mapping:
        try:
            method = data["method"]
            params = {k: float(v) for k, v in data.get("params", {}).items()}
            raw_blocks = data.get("blocks")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed mapping file: {exc}") from exc
        if method not in ("temperature", "platt", "isotonic"):
            raise ValueError(f"unknown mapping method {method!r}")
        blocks = None
        if raw_blocks is not None:
            blocks = [(float(b[0]), float(b[1]), float(b[2])) for b in raw_blocks]
        return cls(method=method, params=params, blocks=blocks)


def fit_temperature(probs: list[float], labels: list[int]) -> Mapping:
    """One-parameter fit: divide logits by T, chosen to minimize log loss.

    T > 1 softens over-confident predictions; T < 1 sharpens under-confident
    ones. The search is golden-section on log T, since the loss is smooth and
    unimodal in it.
    """
    _check_fit(probs, labels)
    logits = [_logit(p) for p in probs]

    def loss(log_t: float) -> float:
        t = math.exp(log_t)
        return log_loss([_sigmoid(z / t) for z in logits], labels)

    lo, hi = math.log(0.05), math.log(20.0)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, fd = loss(c), loss(d)
    for _ in range(200):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv_phi * (hi - lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv_phi * (hi - lo)
            fd = loss(d)
    return Mapping(method="temperature", params={"T": math.exp((lo + hi) / 2.0)})


def fit_platt(probs: list[float], labels: list[int]) -> Mapping:
    """Two-parameter logistic fit sigma(a*logit(p) + b), by damped Newton.

    Uses Platt's smoothed targets so a perfectly separable input does not
    drive the parameters to infinity.
    """
    _check_fit(probs, labels)
    logits = [_logit(p) for p in probs]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    targets = [t_pos if y else t_neg for y in labels]

    def nll(a: float, b: float) -> float:
        total = 0.0
        for z, t in zip(logits, targets):
            q = _sigmoid(a * z + b)
            q = min(1.0 - CLIP, max(CLIP, q))
            total += -(t * math.log(q) + (1.0 - t) * math.log(1.0 - q))
        return total

    a, b = 1.0, 0.0
    current = nll(a, b)
    for _ in range(100):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for z, t in zip(logits, targets):
            q = _sigmoid(a * z + b)
            w = q * (1.0 - q)
            g_a += (q - t) * z
            g_b += q - t
            h_aa += w * z * z
            h_ab += w * z
            h_bb += w
        h_aa += 1e-9
        h_bb += 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 0.0:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det
        scale = 1.0
        for _ in range(30):
            candidate = nll(a - scale * step_a, b - scale * step_b)
            if candidate <= current:
                break
            scale /= 2.0
        else:
            break
        a -= scale * step_a
        b -= scale * step_b
        if abs(current - candidate) < 1e-12:
            current = candidate
            break
        current = candidate
    return Mapping(method="platt", params={"a": a, "b": b})


def fit_isotonic(probs: list[float], labels: list[int]) -> Mapping:
    """Monotone step-function fit via pool-adjacent-violators (PAVA)."""
    _check_fit(probs, labels)
    order = sorted(range(len(probs)), key=lambda i: (probs[i], labels[i]))
    # Each block: [sum_y, n, lo_p, hi_p]; merge while means are non-increasing.
    blocks: list[list[float]] = []
    for i in order:
        blocks.append([float(labels[i]), 1.0, probs[i], probs[i]])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            sy, n, _lo, hi = blocks.pop()
            blocks[-1][0] += sy
            blocks[-1][1] += n
            blocks[-1][3] = hi
    out = [(b[2], b[3], b[0] / b[1]) for b in blocks]
    return Mapping(method="isotonic", blocks=out)


def _check_fit(probs: list[float], labels: list[int]) -> None:
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} predictions but {len(labels)} labels")
    if len(probs) < 10:
        raise ValueError(f"need at least 10 items to fit a mapping, got {len(probs)}")
    if len(set(labels)) < 2:
        raise ValueError("need both positive and negative labels to fit a mapping")
