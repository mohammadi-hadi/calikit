"""Render a reliability diagram as a standalone SVG file (no dependencies)."""

from __future__ import annotations

from calikit.binning import Bin, ece

# One accent color on a paper-friendly light surface.
ACCENT = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

W, H = 640, 520
ML, MR, MT, MB = 64, 24, 56, 130  # margins; bottom holds the count histogram
PW, PH = W - ML - MR, H - MT - MB  # plot area
HIST_H = 64


def _x(v: float) -> float:
    return ML + v * PW


def _y(v: float) -> float:
    return MT + (1.0 - v) * PH


def reliability_svg(bins: list[Bin], title: str = "Reliability diagram") -> str:
    n = sum(b.n for b in bins)
    value = ece(bins)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">'
        ),
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="{ML}" y="26" font-size="16" font-weight="600" fill="{INK}">{title}</text>',
        (
            f'<text x="{ML}" y="44" font-size="12" fill="{MUTED}">'
            f"n = {n} &#183; ECE = {value:.4f} &#183; {len(bins)} bins</text>"
        ),
    ]
    # Grid and axes.
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{_x(0):.1f}" y1="{_y(t):.1f}" x2="{_x(1):.1f}" y2="{_y(t):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{_x(0) - 8:.1f}" y="{_y(t) + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{t:g}</text>'
        )
        parts.append(
            f'<text x="{_x(t):.1f}" y="{_y(0) + 16:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="middle">{t:g}</text>'
        )
    # Perfect-calibration diagonal.
    parts.append(
        f'<line x1="{_x(0):.1f}" y1="{_y(0):.1f}" x2="{_x(1):.1f}" y2="{_y(1):.1f}" '
        f'stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    # Per-bin gap sticks and points.
    for b in bins:
        parts.append(
            f'<line x1="{_x(b.conf):.1f}" y1="{_y(b.conf):.1f}" '
            f'x2="{_x(b.conf):.1f}" y2="{_y(b.acc):.1f}" '
            f'stroke="{ACCENT}" stroke-width="1.5" opacity="0.45"/>'
        )
    for b in bins:
        parts.append(
            f'<circle cx="{_x(b.conf):.1f}" cy="{_y(b.acc):.1f}" r="5" fill="{ACCENT}" '
            f'stroke="{SURFACE}" stroke-width="2"><title>conf {b.conf:.3f}, acc {b.acc:.3f}, '
            f"n {b.n}</title></circle>"
        )
    # Axis titles.
    parts.append(
        f'<text x="{ML + PW / 2:.1f}" y="{_y(0) + 36:.1f}" font-size="12" fill="{INK}" '
        f'text-anchor="middle">mean predicted probability</text>'
    )
    parts.append(
        f'<text x="18" y="{MT + PH / 2:.1f}" font-size="12" fill="{INK}" text-anchor="middle" '
        f'transform="rotate(-90 18 {MT + PH / 2:.1f})">observed frequency</text>'
    )
    # Count histogram under the plot.
    hist_top = H - HIST_H - 26
    max_n = max(b.n for b in bins)
    for b in bins:
        bar_h = (b.n / max_n) * (HIST_H - 14)
        bw = max(3.0, (b.hi - b.lo) * PW - 2.0)
        cx = _x((b.lo + b.hi) / 2.0)
        parts.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{hist_top + (HIST_H - 14) - bar_h:.1f}" '
            f'width="{bw:.1f}" height="{bar_h:.1f}" rx="2" fill="{MUTED}" opacity="0.5">'
            f"<title>[{b.lo:.3f}, {b.hi:.3f}]: {b.n} items</title></rect>"
        )
    parts.append(
        f'<text x="{ML}" y="{H - 10}" font-size="11" fill="{MUTED}">items per bin</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(path: str, bins: list[Bin], title: str = "Reliability diagram") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(reliability_svg(bins, title=title))
