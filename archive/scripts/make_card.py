"""Kaggle writeup card (560 x 280) for the ARC Prize 2026 Paper Track.

Vector SVG (lossless at any scale) plus raster fallbacks. Text-light: a real grid
our TTT run solved, rendered as 3D cells, with the headline result (16.28%, full 120).

-> paper/card.svg  (use this; crisp at any size, declares 560x280)
   paper/card_560x280.png   (exact-size raster)
   paper/card_1680x840.png  (3x raster)
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = r"D:\ARC_AGI-2,3"

# A task our TTT run solved (its produced output grid). Background is azure (8), not
# black, so every cell is part of the answer and is drawn.
GRID = [
    [1, 8, 8, 8, 1],
    [8, 6, 6, 6, 8],
    [3, 8, 6, 8, 3],
    [3, 8, 9, 8, 3],
    [3, 8, 9, 8, 3],
    [3, 8, 6, 8, 3],
    [8, 6, 6, 6, 8],
    [1, 8, 8, 8, 1],
]

ARC = ["#101014", "#1E6FD0", "#E23B36", "#37B24D", "#F0C419",
       "#9AA0A6", "#D6489B", "#E8820C", "#5FC6E8", "#8A1B2E"]
INK, BLUE, GREY = "#1b1b1b", "#1E5FA8", "#6b6b6b"

# --- layout / view knobs ---
AX_RECT = [0.0, 0.10, 0.63, 0.86]
ELEV, AZIM = 30, -60
TEXT_X = 0.635


def blocks3d(ax, grid):
    h, w = len(grid), len(grid[0])
    xs, ys, zs, dx, dy, dz, cols = [], [], [], [], [], [], []
    for r in range(h):
        for c in range(w):
            k = grid[r][c]
            xs.append(c + 0.06); ys.append((h - 1 - r) + 0.06); zs.append(0)
            dx.append(0.88); dy.append(0.88); dz.append(1.0)
            cols.append(ARC[k])
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=cols, edgecolor="white",
             linewidth=0.5, shade=True)
    ax.set_proj_type("ortho")
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(0, 1.6)
    ax.set_box_aspect((w, h, 2.2))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()


def build():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "svg.fonttype": "path",   # embed text as outlines -> renders identically anywhere
    })
    fig = plt.figure(figsize=(560 / 72, 280 / 72))   # 1 point == 1 px in the viewBox
    fig.patch.set_facecolor("white")
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    ax = fig.add_axes(AX_RECT, projection="3d")
    blocks3d(ax, GRID)

    bg.text(TEXT_X, 0.60, "16.28%", fontsize=46, color=BLUE, weight="bold", va="center")
    bg.text(TEXT_X, 0.405, "ARC-AGI-2, full 120-task eval", fontsize=12.5, color=INK, va="center")
    bg.text(TEXT_X, 0.315, "up from 12.79% without TTT", fontsize=12.5, color=GREY, va="center")
    bg.text(TEXT_X, 0.10, "ARC Prize 2026  ·  Paper Track", fontsize=11.5, color=GREY, va="center")
    return fig


def main():
    fig = build()
    svg = os.path.join(ROOT, "paper", "card.svg")
    fig.savefig(svg)
    # declare exact 560x280 pixels on the root <svg> (matplotlib emits points)
    s = open(svg, encoding="utf-8").read()
    s = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r'\1 width="560"', s, count=1)
    s = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r'\1 height="280"', s, count=1)
    open(svg, "w", encoding="utf-8").write(s)

    fig.savefig(os.path.join(ROOT, "paper", "card_560x280.png"), dpi=72)    # 560x280
    fig.savefig(os.path.join(ROOT, "paper", "card_1680x840.png"), dpi=216)  # 3x
    fig.savefig(os.path.join(ROOT, "paper", "card_preview.png"), dpi=144)   # for review
    print("saved card.svg, card_560x280.png, card_1680x840.png")


if __name__ == "__main__":
    main()
