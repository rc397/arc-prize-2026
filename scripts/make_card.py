"""Kaggle writeup card/thumbnail (560 x 280, 2:1) for the ARC Prize 2026 Paper Track.

Left: a real ARC-AGI-2 task as 3D cells (the hero). Right: title + the headline
result. All numbers are measured. -> paper/card_560x280.png (+ 2x for retina).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = r"D:\ARC_AGI-2,3"
TASK = os.path.join(ROOT, "paper", "assets", "pool", "28a6681f.json")

ARC = ["#101014", "#1E6FD0", "#E23B36", "#37B24D", "#F0C419",
       "#9AA0A6", "#D6489B", "#E8820C", "#5FC6E8", "#8A1B2E"]
INK, BLUE, GREY = "#1b1b1b", "#1E5FA8", "#6b6b6b"


def load_grid():
    return json.load(open(TASK))["test"][0]["input"]


def blocks3d(ax, grid):
    h, w = len(grid), len(grid[0])
    xs, ys, zs, dx, dy, dz, cols = [], [], [], [], [], [], []
    for r in range(h):
        for c in range(w):
            k = grid[r][c]
            if k == 0:
                continue
            xs.append(c + 0.06); ys.append((h - 1 - r) + 0.06); zs.append(0)
            dx.append(0.88); dy.append(0.88); dz.append(1.0)
            cols.append(ARC[k])
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=cols, edgecolor="white",
             linewidth=0.3, shade=True)
    for i in range(w + 1):
        ax.plot([i, i], [0, h], [0, 0], color="#dcdcdc", lw=0.4, zorder=0)
    for j in range(h + 1):
        ax.plot([0, w], [j, j], [0, 0], color="#dcdcdc", lw=0.4, zorder=0)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(0, 2.2)
    ax.set_box_aspect((w, h, 3.4))
    ax.view_init(elev=32, azim=-58)
    ax.set_axis_off()


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })
    fig = plt.figure(figsize=(5.6, 2.8))
    fig.patch.set_facecolor("white")
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    grid = load_grid()
    ax = fig.add_axes([-0.05, -0.08, 0.64, 1.16], projection="3d")
    blocks3d(ax, grid)

    bg.add_line(Line2D([0.545, 0.545], [0.16, 0.84], color="#e3e3e3", lw=1.0))

    x = 0.60
    bg.text(x, 0.885, "ARC Prize 2026  ·  Paper Track", fontsize=8.5,
            color=GREY, va="top")
    bg.text(x, 0.80, "Test-time training\non one consumer GPU", fontsize=14.5,
            color=INK, weight="bold", va="top", linespacing=1.18)
    bg.add_line(Line2D([x, x + 0.13], [0.545, 0.545], color=INK, lw=1.3))

    bg.text(x, 0.40, "16.28%", fontsize=30, color=BLUE, weight="bold", va="center")
    bg.text(x, 0.245, "on ARC-AGI-2, full 120-task eval", fontsize=8.7,
            color=INK, va="center")
    bg.text(x, 0.165, "up from 12.79% without TTT", fontsize=8.7,
            color=GREY, va="center")
    bg.text(x, 0.075, "1× RTX 5070 Ti, 16 GB  ·  13× faster decode", fontsize=8.7,
            color=GREY, va="center")

    out = os.path.join(ROOT, "paper", "card_560x280.png")
    fig.savefig(out, dpi=100)                      # exactly 560 x 280
    fig.savefig(os.path.join(ROOT, "paper", "card_1120x560.png"), dpi=200)  # 2x
    print("saved", out, "and 2x")


if __name__ == "__main__":
    main()
