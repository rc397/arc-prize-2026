"""Paper-style title page for the ARC Prize 2026 write-up.

Mirrors the IMMC summary-sheet layout: serif type, a running header/footer, a real
3D plot of an ARC-AGI-2 task as the centerpiece, and the measured result plots
(adapter-scale curve + the TTT/augmentation matrix) below. All numbers are the
measured values from the runs; the hero grid is a real public ARC-AGI-2 eval task.

-> paper/cover_v2.png and paper/cover_v2.pdf
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

ROOT = r"D:\ARC_AGI-2,3"
TASK = os.path.join(ROOT, "paper", "assets", "pool", "28a6681f.json")
OUT = os.path.join(ROOT, "paper", "cover_v2")

# Canonical ARC colour palette (index 0..9). The domain's real palette.
ARC = ["#101014", "#1E6FD0", "#E23B36", "#37B24D", "#F0C419",
       "#9AA0A6", "#D6489B", "#E8820C", "#5FC6E8", "#8A1B2E"]
ARC_CMAP = ListedColormap(ARC)

BLUE, GREY = "#1E5FA8", "#9AA0A6"

# ---- measured results ---------------------------------------------------------
SCALE_PTS = [(0.50, 12.0), (0.70, 12.8), (0.85, 15.2), (1.00, 16.8), (1.25, 17.6)]
NO_TTT_84 = 12.8                       # dashed reference on the 84 trainable tasks
GROUPS = ["8 augmentations", "16 augmentations"]
NOTTT_G = [12.8, 14.4]                 # no-TTT: plain, +colour-perm
TTT_G = [17.6, 17.6]                   # TTT @ scale 1.25: plain, +colour-perm
BLEND = 16.28                          # best blended, full 120
BASE = 12.79                           # no-TTT AIRV, full 120

ABSTRACT = (
    "We ask how far the 2025 ARC-AGI-2 winning recipe, per-task test-time training (TTT) with "
    "augment-inference-reverse-vote (AIRV) decoding, can be pushed on a single 16 GB consumer GPU. "
    "The binding constraint turns out to be the serving engine, not the model: moving from eager "
    "decoding to llama.cpp raises throughput roughly 13x, from 13 to 170 tokens per second, which is "
    "what makes per-task TTT feasible at all. A no-TTT AIRV baseline scores 12.79% on the 120-task "
    "public evaluation. Training a small regularized adapter per task and tuning its scale lifts "
    "accuracy to 17.6% on the 84 tasks that fit in memory, against 12.8% without TTT, for a best "
    "blended result of 16.28% across the full 120. Colour-permutation augmentation lifts the baseline "
    "but does not stack with TTT, whose adapter is already colour-robust. The contribution is an "
    "honest single-GPU operating point and a map of the walls, the decode ceiling, the VRAM limit "
    "that excludes large grids, and adapter overfitting, reported as accuracy per GPU-hour."
)


def load_grid(path, where="test", idx=0, field="input"):
    d = json.load(open(path))
    return d[where][idx][field]


def draw_heatmap(ax, grid):
    ax.imshow(grid, cmap=ARC_CMAP, vmin=-0.5, vmax=9.5, interpolation="nearest")
    h, w = len(grid), len(grid[0])
    ax.set_xticks([x - 0.5 for x in range(w + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(h + 1)], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333333"); s.set_linewidth(0.8)


def draw_grid3d(ax, grid):
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
    # faint board on z=0
    for i in range(w + 1):
        ax.plot([i, i], [0, h], [0, 0], color="#d7d7d7", lw=0.5, zorder=0)
    for j in range(h + 1):
        ax.plot([0, w], [j, j], [0, 0], color="#d7d7d7", lw=0.5, zorder=0)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_zlim(0, 2.2)
    ax.set_box_aspect((w, h, 3.6))
    ax.view_init(elev=30, azim=-60)
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10]); ax.set_zticks([])
    ax.set_xlabel("column", fontsize=8.5, labelpad=1)
    ax.set_ylabel("row", fontsize=8.5, labelpad=1)
    ax.tick_params(labelsize=7.5, pad=0.5)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_alpha(1.0)
        pane.pane.set_edgecolor("#e6e6e6")
    ax.grid(False)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#444444",
    })
    ink = "#1b1b1b"
    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait

    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    def rule(y, x0=0.07, x1=0.93, lw=1.0):
        bg.add_line(Line2D([x0, x1], [y, y], color=ink, lw=lw))

    # ---- header
    bg.text(0.07, 0.972, "ARC Prize 2026", fontsize=9.5, color=ink)
    bg.text(0.50, 0.972, "Paper Track · Summary Sheet", fontsize=9.5,
            color=ink, ha="center")
    bg.text(0.93, 0.972, "rchen · 2026", fontsize=9.5, color=ink, ha="right")
    rule(0.962, lw=1.2)

    # ---- title
    bg.text(0.07, 0.930, "Test-time training on a single consumer GPU",
            fontsize=23, color=ink, weight="bold", va="top")
    bg.text(0.07, 0.900,
            "How far the 2025 ARC-AGI-2 winning recipe goes on one 16 GB RTX 5070 Ti",
            fontsize=12.5, color="#333333", style="italic", va="top")

    # ---- hero: 3D ARC grid + small 2D inset
    grid = load_grid(TASK, "test", 0, "input")
    ax3 = fig.add_axes([0.05, 0.55, 0.66, 0.33], projection="3d")
    draw_grid3d(ax3, grid)
    inset = fig.add_axes([0.745, 0.625, 0.175, 0.175 * 8.27 / 11.69])
    draw_heatmap(inset, grid)
    inset.set_title("test grid", fontsize=8.5, color="#333333", pad=3)

    bg.text(0.50, 0.553,
            "Figure 1.  ARC-AGI-2 evaluation task 28a6681f: the test grid (inset) rendered as "
            "3D cells. Background cells omitted.",
            fontsize=8.5, color="#444444", ha="center", style="italic")

    # ---- summary text
    bg.text(0.07, 0.525, "Summary", fontsize=12.5, color=ink, weight="bold", va="top")
    wrapped = textwrap.fill(ABSTRACT, width=104)
    bg.text(0.07, 0.503, wrapped, fontsize=10, color=ink, va="top",
            linespacing=1.5)

    # ---- results band
    axc = fig.add_axes([0.085, 0.13, 0.40, 0.185])
    xs = [p[0] for p in SCALE_PTS]; ys = [p[1] for p in SCALE_PTS]
    axc.plot(xs, ys, "o-", color=BLUE, lw=1.6, ms=5, zorder=3)
    axc.axhline(NO_TTT_84, ls="--", color="#888888", lw=1.2)
    axc.text(0.51, NO_TTT_84 + 0.12, "no-TTT 12.8%", fontsize=8, color="#666666")
    axc.set_xlabel("adapter scale", fontsize=9)
    axc.set_ylabel("accuracy on 84 tasks (%)", fontsize=9)
    axc.set_title("Adapter-scale sweep", fontsize=10)
    axc.tick_params(labelsize=8)
    axc.grid(True, color="#eeeeee", lw=0.7)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)

    axb = fig.add_axes([0.57, 0.13, 0.355, 0.185])
    import numpy as np
    x = np.arange(len(GROUPS)); bw = 0.38
    axb.bar(x - bw / 2, NOTTT_G, bw, color=GREY, label="no-TTT")
    axb.bar(x + bw / 2, TTT_G, bw, color=BLUE, label="TTT @ 1.25")
    for xi, v in zip(x - bw / 2, NOTTT_G):
        axb.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=7.5, color="#555555")
    for xi, v in zip(x + bw / 2, TTT_G):
        axb.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=7.5, color=BLUE)
    axb.set_xticks(x); axb.set_xticklabels(GROUPS, fontsize=8)
    axb.set_ylabel("accuracy (%)", fontsize=9)
    axb.set_ylim(0, 20)
    axb.set_title("TTT vs no-TTT (84 tasks)", fontsize=10)
    axb.tick_params(labelsize=8)
    axb.legend(fontsize=8, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=1.3)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    bg.text(0.50, 0.088,
            "Figure 2.  Left: accuracy rises with adapter scale, overtaking the no-TTT baseline.  "
            "Right: colour-perm lifts the baseline but does not stack with TTT.",
            fontsize=8.5, color="#444444", ha="center", style="italic")

    # ---- footer
    rule(0.055, lw=1.0)
    bg.text(0.07, 0.038,
            "Hardware: 1× RTX 5070 Ti, 16 GB, Blackwell  ·  Engine: llama.cpp, 13→170 tok/s  "
            "·  Baseline 12.79% no-TTT AIRV  ·  Best blended 16.28% (full 120)",
            fontsize=8.5, color="#333333")

    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=200)
    fig.savefig(OUT + ".pdf")
    print("saved", OUT + ".png", "and .pdf")


if __name__ == "__main__":
    main()
