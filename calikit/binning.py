"""Group predictions into confidence bins and summarize each bin."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bin:
    """One confidence bin: its range, size, mean confidence, and hit rate."""

    lo: float
    hi: float
    n: int
    conf: float  # mean predicted probability in the bin
    acc: float  # observed positive rate in the bin


def bin_predictions(
    probs: list[float],
    labels: list[int],
    k: int = 10,
    scheme: str = "mass",
) -> list[Bin]:
    """Split predictions into k bins.

    "mass" bins hold (nearly) equal numbers of items — the default, because
    equal-width bins can leave most bins empty when predictions cluster.
    "width" bins cut [0, 1] into k equal intervals.
    Empty bins are dropped, so fewer than k bins may come back.
    """
    n = len(probs)
    if n == 0:
        raise ValueError("no predictions to bin")
    if len(labels) != n:
        raise ValueError(f"{n} predictions but {len(labels)} labels")
    if k < 2:
        raise ValueError(f"need at least 2 bins, got {k}")
    if scheme not in ("mass", "width"):
        raise ValueError(f"scheme must be 'mass' or 'width', got {scheme!r}")

    groups: list[list[int]]
    if scheme == "width":
        groups = [[] for _ in range(k)]
        for i, p in enumerate(probs):
            j = min(k - 1, int(p * k))
            groups[j].append(i)
    else:
        if k > n:
            raise ValueError(f"more bins ({k}) than items ({n})")
        order = sorted(range(n), key=lambda i: probs[i])
        base, rem = divmod(n, k)
        groups, start = [], 0
        for j in range(k):
            size = base + (1 if j < rem else 0)
            groups.append(order[start : start + size])
            start += size

    bins = []
    for j, idx in enumerate(groups):
        if not idx:
            continue
        ps = [probs[i] for i in idx]
        ys = [labels[i] for i in idx]
        if scheme == "width":
            lo, hi = j / k, (j + 1) / k
        else:
            lo, hi = min(ps), max(ps)
        bins.append(Bin(lo=lo, hi=hi, n=len(idx), conf=sum(ps) / len(ps), acc=sum(ys) / len(ys)))
    return bins


def ece(bins: list[Bin]) -> float:
    """Expected calibration error: the size-weighted mean |accuracy - confidence|."""
    n = sum(b.n for b in bins)
    return sum(b.n * abs(b.acc - b.conf) for b in bins) / n


def mce(bins: list[Bin]) -> float:
    """Maximum calibration error: the worst bin's |accuracy - confidence|."""
    return max(abs(b.acc - b.conf) for b in bins)
