"""Dependency-free SVG of a pilot loss log. No evaluation or convergence claim."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    if not rows or any(not math.isfinite(r["loss"]) for r in rows):
        raise SystemExit("Need finite recorded training losses")
    width, height = 1000, 510
    left, right, top, bottom = 80, 960, 105, 405
    maximum = max(r["loss"] for r in rows) * 1.08
    last = max(r["step"] for r in rows)

    def point(step, loss):
        return left + (right - left) * step / last, bottom - (bottom - top) * loss / maximum

    colors = {1: "#007f86", 2: "#6c62a8", 3: "#cb732c", 4: "#bd4460"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f7f3"/>',
        '<g font-family="Arial, sans-serif" fill="#253335">',
        f'<text x="80" y="40" font-size="24" font-weight="bold">A pilot: {len(rows)} optimizer steps</text>',
        '<text x="80" y="68" font-size="14">Training loss only · mixed examples and loop counts · not benchmark quality</text>',
    ]
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        svg.extend(
            [
                f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#d9ddda"/>',
                f'<text x="66" y="{y + 4}" text-anchor="end" font-size="12">{maximum * i / 5:.1f}</text>',
            ]
        )
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(r["step"], r["loss"]) for r in rows))
    svg.append(f'<polyline points="{points}" fill="none" stroke="#9aaba9" stroke-width="1"/>')
    for row in rows:
        x, y = point(row["step"], row["loss"])
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{colors[row["K"]]}"/>')
    rolling = []
    for i in range(9, len(rows)):
        rolling.append(point(rows[i]["step"], sum(r["loss"] for r in rows[i - 9 : i + 1]) / 10))
    if rolling:
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in rolling)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#233c50" stroke-width="2.5"/>')
    for step in range(0, last + 1, max(1, last // 5)):
        x, _ = point(step, 0)
        svg.append(f'<text x="{x}" y="428" text-anchor="middle" font-size="12">{step}</text>')
    svg.extend(
        [
            '<text x="500" y="450" text-anchor="middle" font-size="14">Optimizer step</text>',
            '<text x="24" y="260" text-anchor="middle" transform="rotate(-90 24 260)" font-size="14">Weighted soft-label cross entropy</text>',
            '<text x="80" y="485" font-size="12">Dot color: sampled passes</text>',
        ]
    )
    for k, color in colors.items():
        x = 260 + 60 * k
        svg.extend(
            [
                f'<circle cx="{x}" cy="481" r="4" fill="{color}"/>',
                f'<text x="{x + 10}" y="485" font-size="12">{k}</text>',
            ]
        )
    svg.extend(
        [
            '<text x="590" y="485" font-size="12">Dark line: 10-step rolling mean (descriptive only)</text>',
            "</g></svg>",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
