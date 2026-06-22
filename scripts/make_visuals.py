"""Writeup figures for the ARC Prize 2026 Paper Track (high-res PNG, Kaggle-safe).

Shared style matches the card: Times serif, blue accent, ARC palette for grids.
All numbers are the measured values handed over from the runs.

-> paper/fig_pipeline.png   (method / serving bridge)
   paper/fig_results.png    (engine + accuracy + scale curve)
   paper/fig_solve.png      (a real task being solved, as colour grids)
   paper/fig_levers.png     (what worked / what didn't)
"""
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

OUT = r"D:\ARC_AGI-2,3\paper"

# Faithful-but-softened ARC palette (consistent with the card; not neon).
ARC = ["#1b1b1f", "#1E6FD0", "#E8453B", "#37B24D", "#E8C020",
       "#9AA0A6", "#C24BA0", "#E8820C", "#6FC8E8", "#8A1B2E"]
ACMAP = ListedColormap(ARC)
INK, BLUE, BLUEL, GREY, GREYL, RAIL, FAIL = (
    "#1b1b1b", "#1E5FA8", "#A9C2DD", "#6b6b6b", "#C2C7CC", "#ededed", "#B0473A")
GREEN = "#2E8B57"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# --- real solved task d35bdbdc (10x10) ---
D_TRAIN_IN = [[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,4,3],[0,5,0,0,0,0,0,3,3,3],
    [0,5,4,4,4,0,0,0,0,0],[5,0,4,2,4,0,0,6,6,6],[0,5,4,4,4,0,5,6,1,6],
    [0,5,5,5,5,5,0,6,6,6],[0,0,1,1,1,0,0,0,0,0],[0,0,1,3,1,0,0,0,0,0],[0,0,1,1,1,0,0,0,0,0]]
D_TRAIN_OUT = [[0,0,0,0,0,0,0,3,3,3],[0,0,5,5,5,5,5,3,2,3],[0,5,0,0,0,0,0,3,3,3],
    [0,5,0,0,0,0,0,0,0,0],[5,0,0,0,0,0,0,6,6,6],[0,5,0,0,0,0,5,6,3,6],
    [0,5,5,5,5,5,0,6,6,6],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
D_TEST_IN = [[0,0,0,0,0,0,0,2,0,0],[0,0,4,0,0,0,2,3,2,0],[0,4,2,4,0,0,0,2,0,0],
    [0,0,4,0,0,0,0,5,0,0],[0,0,0,0,0,0,5,0,0,0],[0,0,0,0,1,0,5,0,3,0],
    [0,7,0,1,9,1,5,3,3,3],[7,1,7,0,1,0,5,0,3,0],[0,7,0,5,0,5,0,0,0,0],[0,5,5,0,5,0,0,0,0,0]]
D_SOLVED = [[0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,2,3,2,0],[0,0,0,0,0,0,0,2,0,0],
    [0,0,0,0,0,0,0,5,0,0],[0,0,0,0,0,0,5,0,0,0],[0,0,0,0,0,0,5,0,0,0],
    [0,7,0,0,0,0,5,0,0,0],[7,9,7,0,0,0,5,0,0,0],[0,7,0,5,0,5,0,0,0,0],[0,5,5,0,5,0,0,0,0,0]]


# full adapter-scale sweep on the 84 trainable tasks (peak at 1.25, then declines)
SCALE_X = [0.5, 0.7, 0.85, 1.0, 1.25, 1.4, 1.5, 1.75]
SCALE_Y = [12.0, 12.8, 15.2, 16.8, 17.6, 16.0, 16.8, 16.8]


def render_grid(ax, grid):
    g = np.array(grid); h, w = g.shape
    ax.imshow(g, cmap=ACMAP, vmin=-0.5, vmax=9.5, interpolation="nearest")
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#333"); s.set_linewidth(1.1)


def despine(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(labelsize=9, colors=INK)
    a.set_axisbelow(True)


def footer(fig):
    fig.text(0.992, 0.012, "ARC Prize 2026  ·  Paper Track", ha="right", va="bottom",
             fontsize=8, color="#9a9a9a")


def save(fig, name, dpi=200):
    base = os.path.join(OUT, name)
    common = dict(facecolor="white", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(base + ".png", dpi=dpi, **common)   # raster: card, web, notebook
    fig.savefig(base + ".svg", **common)            # vector deliverable
    fig.savefig(base + ".pdf", **common)            # vector, embedded by the LaTeX build
    plt.close(fig)
    print("saved", name, "(png/svg/pdf)")


def draw_scale(ax, title, tfs=12.5):
    ax.plot(SCALE_X, SCALE_Y, "o-", color=BLUE, lw=2.1, ms=6, zorder=3)
    ax.axhline(12.8, ls="--", color=GREY, lw=1.3)
    ax.text(1.85, 12.95, "no-TTT 12.8%", ha="right", fontsize=8.6, color=GREY)
    ax.plot([1.25], [17.6], "o", ms=12, mfc="white", mec=BLUE, mew=2.3, zorder=4)
    ax.text(1.25, 18.2, "peak 17.6%", ha="center", fontsize=9, color=BLUE,
            fontweight="bold")
    ax.text(1.575, 14.7, "plateau ~16-17%\n(robust, no cliff)", ha="center",
            fontsize=8.2, color=GREY, style="italic")
    ax.set_xlabel("adapter scale", fontsize=10)
    ax.set_ylabel("accuracy on 84 tasks (%)", fontsize=10)
    ax.set_title(title, fontsize=tfs, fontweight="bold", color=INK)
    ax.set_xlim(0.42, 1.9); ax.set_ylim(11.3, 18.8)
    despine(ax); ax.grid(color=RAIL)


def draw_paired(ax):
    ax.set_xlim(-0.82, 2.06); ax.set_ylim(-0.6, 2.12)
    ax.set_aspect("equal"); ax.axis("off")
    GC, BT, RT = "#ECECEC", "#D4E2F1", "#F2DBD7"
    data = [(0, 1, 15, GC, INK, "#cccccc", 1.0, "both"),
            (1, 1, 6, BT, BLUE, BLUE, 1.8, "TTT only"),
            (0, 0, 1, RT, FAIL, FAIL, 1.8, "no-TTT only"),
            (1, 0, 103, GC, INK, "#cccccc", 1.0, "neither")]
    for c, r, val, fc, tc, ec, lw, desc in data:
        ax.add_patch(Rectangle((c, r), 0.96, 0.96, facecolor=fc, edgecolor=ec, lw=lw))
        ax.text(c + 0.48, r + 0.58, str(val), ha="center", va="center",
                fontsize=20, color=tc, fontweight="bold")
        ax.text(c + 0.48, r + 0.24, desc, ha="center", va="center", fontsize=8, color=tc)
    ax.text(0.48, 2.04, "no-TTT solved", ha="center", fontsize=8.6, color=INK)
    ax.text(1.48, 2.04, "no-TTT missed", ha="center", fontsize=8.6, color=INK)
    ax.text(-0.08, 1.48, "TTT\nsolved", ha="right", va="center", fontsize=8.6, color=INK)
    ax.text(-0.08, 0.48, "TTT\nmissed", ha="right", va="center", fontsize=8.6, color=INK)
    ax.set_title("Paired outcomes: TTT vs no-TTT (84 tasks)", fontsize=12.5,
                 fontweight="bold", color=INK, pad=12)
    ax.text(0.98, -0.46, "6 vs 1 discordant, net +5 for TTT  ·  McNemar p ≈ 0.13",
            ha="center", fontsize=9, color=GREY)


# ============================================================ RESULTS
def fig_results():
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.3))

    a = ax[0, 0]
    vals = [13, 102, 170, 244]
    cols = [GREY, BLUEL, BLUE, GREYL]
    a.bar(range(4), vals, color=cols, width=0.7, zorder=3)
    for i, v in enumerate(vals):
        a.text(i, v + 6, str(v), ha="center", va="bottom", fontsize=10.5,
               color=INK, fontweight="bold" if i == 2 else "normal")
    a.set_xticks(range(4))
    a.set_xticklabels(["HF\neager", "llama.cpp\nbf16", "llama.cpp\nQ8_0",
                       "llama.cpp\nQ4_K_M"], fontsize=8.6)
    a.set_ylabel("decode throughput (tok/s)", fontsize=10)
    a.set_title("Engine: 13 → 170 tok/s  (13×)", fontsize=12.5,
                fontweight="bold", color=INK)
    a.set_ylim(0, 292)
    a.text(2, 196, "operating point", ha="center", va="bottom", fontsize=8.6,
           color=BLUE)
    a.text(1.5, 288, "prefill ~3,400 tok/s  ·  with adapter ~115 tok/s",
           ha="center", va="top", fontsize=8.2, color=GREY)
    despine(a); a.grid(axis="y", color=RAIL)

    b = ax[0, 1]
    bv = [12.79, 16.28]
    b.bar([0, 1], bv, color=[GREY, BLUE], width=0.6, zorder=3)
    for i, v in enumerate(bv):
        b.text(i, v + 0.35, f"{v:.2f}%", ha="center", fontsize=11.5,
               fontweight="bold", color=INK)
    b.set_xticks([0, 1]); b.set_xticklabels(["no-TTT", "TTT blended"], fontsize=10)
    b.set_ylabel("accuracy (%)", fontsize=10); b.set_ylim(0, 20)
    b.set_title("Accuracy: 12.79 → 16.28%  (full 120)", fontsize=12.5,
                fontweight="bold", color=INK)
    b.annotate("", xy=(0.72, 16.2), xytext=(0.28, 12.95),
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    b.text(0.5, 15.5, "+3.5 pts", ha="center", fontsize=9.5, color=BLUE)
    b.text(0.5, 19.2, "22/172  →  28/172", ha="center", fontsize=8.4, color=GREY)
    despine(b); b.grid(axis="y", color=RAIL)

    draw_scale(ax[1, 0], "Test-time training: scale sweep")
    draw_paired(ax[1, 1])

    fig.tight_layout(h_pad=3.4, w_pad=3.0)
    footer(fig)
    save(fig, "fig_results")


# ============================================================ SOLVE STRIP
def fig_solve():
    fig = plt.figure(figsize=(13.6, 4.0))
    ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off"); ov.set_xlim(0, 1); ov.set_ylim(0, 1)

    fig.text(0.05, 0.95, "Solving ARC-AGI-2 task d35bdbdc", fontsize=16,
             fontweight="bold", color=INK)
    fig.text(0.05, 0.865, "The model learns the rule from the train pairs, then "
             "produces the test output. No hand-coded rules.", fontsize=10.5,
             color=GREY, style="italic")

    gy, gh = 0.22, 0.52
    gw = gh * 4.0 / 13.6
    gap = 0.05
    x1 = 0.05; x2 = x1 + gw + gap; x3 = 0.555; x4 = x3 + gw + gap
    xs = [x1, x2, x3, x4]
    grids = [D_TRAIN_IN, D_TRAIN_OUT, D_TEST_IN, D_SOLVED]
    labels = ["train input", "train output", "test input", "model output"]
    for x, g, lab in zip(xs, grids, labels):
        axg = fig.add_axes([x, gy, gw, gh]); render_grid(axg, g)
        ov.text(x + gw / 2, gy - 0.10, lab, ha="center", fontsize=10.5, color=INK)

    yc = gy + gh / 2

    def harrow(xa, xb, lab, color):
        ov.add_patch(FancyArrowPatch((xa, yc), (xb, yc), arrowstyle="-|>",
                     mutation_scale=18, color=color, lw=2.0))
        ov.text((xa + xb) / 2, yc + 0.055, lab, ha="center", fontsize=9.5,
                color=color, style="italic")

    harrow(x1 + gw + 0.008, x2 - 0.008, "rule", GREY)
    harrow(x3 + gw + 0.008, x4 - 0.008, "our model", BLUE)
    ov.plot([0.503, 0.503], [gy - 0.05, gy + gh + 0.05], ls=(0, (4, 4)),
            color="#cccccc", lw=1.2)
    ov.text((x1 + x2 + gw) / 2, gy + gh + 0.085, "worked example (1 of N pairs)",
            ha="center", fontsize=10.5, color=INK)
    ov.text((x3 + x4 + gw) / 2, gy + gh + 0.085, "test → solved",
            ha="center", fontsize=10.5, color=INK)
    ov.text(x4 + gw + 0.006, yc, "✓", fontfamily="DejaVu Sans", fontsize=19,
            color="#2E8B57", va="center", ha="left", fontweight="bold")
    footer(fig)
    save(fig, "fig_solve")


# ============================================================ LEVERS
def fig_levers():
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.6))
    fig.suptitle("Three levers, characterised: what worked and what didn't",
                 fontsize=15, fontweight="bold", color=INK, x=0.045, ha="left", y=0.975)

    draw_scale(ax[0, 0], "Adapter scale: a clear optimum at 1.25", tfs=11.5)

    p = ax[0, 1]
    names = ["no-TTT", "naive\n40 st, r32", "aggressive\n8 st, r8", "regularized\n20 st, r16, p0.1"]
    vv = [13.0, 13.0, 13.0, 17.4]
    cl = [GREY, GREYL, GREYL, BLUE]
    p.bar(range(4), vv, color=cl, width=0.66, zorder=3)
    for i, v in enumerate(vv):
        p.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=10,
               fontweight="bold" if i == 3 else "normal", color=INK)
    p.set_xticks(range(4)); p.set_xticklabels(names, fontsize=8.2)
    p.set_ylabel("accuracy, 15 tasks (%)", fontsize=9.5); p.set_ylim(0, 20)
    p.set_title("Regularization makes TTT win (naive overfits, ties)",
                fontsize=11.5, fontweight="bold", color=INK)
    despine(p); p.grid(axis="y", color=RAIL)

    p = ax[1, 0]
    g = np.arange(2); bw = 0.36
    p.bar(g - bw / 2, [12.8, 17.6], bw, color=BLUEL, label="8 augmentations", zorder=3)
    p.bar(g + bw / 2, [14.4, 17.6], bw, color=BLUE, label="16 (+ colour-perm)", zorder=3)
    for xi, v in zip(g - bw / 2, [12.8, 17.6]):
        p.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=8.6, color=INK)
    for xi, v in zip(g + bw / 2, [14.4, 17.6]):
        p.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=8.6, color=INK)
    p.set_xticks(g); p.set_xticklabels(["no-TTT", "TTT @ 1.25"], fontsize=9.5)
    p.set_ylabel("accuracy, 84 tasks (%)", fontsize=9.5); p.set_ylim(0, 20)
    p.set_title("Colour-perm helps the baseline (+1.6), not TTT",
                fontsize=11.5, fontweight="bold", color=INK)
    p.legend(fontsize=8.6, frameon=False, loc="upper left")
    despine(p); p.grid(axis="y", color=RAIL)

    p = ax[1, 1]
    vv = [12.77, 10.64]
    p.bar([0, 1], vv, color=[GREY, FAIL], width=0.6, zorder=3)
    for i, v in enumerate(vv):
        p.text(i, v + 0.25, f"{v:.2f}%", ha="center", fontsize=10.5,
               fontweight="bold", color=INK)
    p.set_xticks([0, 1]); p.set_xticklabels(["no-TTT", "forced TTT"], fontsize=9.5)
    p.set_ylabel("accuracy, 36 big tasks (%)", fontsize=9.5); p.set_ylim(0, 15)
    p.set_title("On grids that need >16 GB, forcing TTT hurts",
                fontsize=11.5, fontweight="bold", color=INK)
    p.text(1, 6.0, "keep\nno-TTT", ha="center", fontsize=9, color=FAIL)
    despine(p); p.grid(axis="y", color=RAIL)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0, w_pad=2.5)
    footer(fig)
    save(fig, "fig_levers")


# ============================================================ PIPELINE
def _box(ax, x, y, w, h, title, sub=None, fc="white", ec=BLUE, tc=INK,
         tfs=11.5, sfs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                 linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=tfs, color=tc, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center",
                fontsize=sfs, color=GREY, zorder=4, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, color=tc, fontweight="bold", zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=INK, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=15, color=color, lw=lw,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def fig_pipeline():
    fig = plt.figure(figsize=(15, 7.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 74)
    ax.set_aspect("equal"); ax.axis("off")

    ax.text(5, 69.5, "Per-task test-time training, served on one 16 GB GPU",
            fontsize=17.5, fontweight="bold", color=INK)
    ax.text(5, 64.6, "Train a LoRA in PyTorch, bridge it to GGUF, hot-swap it on "
            "llama.cpp, then vote across augmentations.", fontsize=11, color=GREY,
            style="italic")

    # input thumbnail
    gi = ax.inset_axes([4, 38, 15, 15], transform=ax.transData)
    render_grid(gi, D_TEST_IN)
    ax.text(11.5, 35.5, "ARC task", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(11.5, 32.6, "train pairs + test input", ha="center", fontsize=8.4, color=GREY)

    # bridge band
    ax.add_patch(FancyBboxPatch((24, 24), 80, 30, boxstyle="round,pad=0,rounding_size=2",
                 facecolor="#EAF1F8", edgecolor=BLUE, linewidth=1.3, zorder=1))
    ax.text(64, 50.6, "Per-task TTT: the serving bridge", ha="center",
            fontsize=12.5, fontweight="bold", color=BLUE, zorder=4)

    _box(ax, 26.5, 30, 22, 16, "Fit LoRA",
         "PyTorch · rank-16\nleave-one-out · ~20 steps @ 2s\ntrains with random recolour")
    _box(ax, 53, 30, 22, 16, "Convert → GGUF", "convert_lora_to_gguf.py")
    _box(ax, 79.5, 30, 22, 16, "Serve · llama.cpp",
         "base Q8_0 + hot-swap\nPOST /lora-adapters\n~115 tok/s decode")
    _arrow(ax, 48.5, 38, 53, 38); _arrow(ax, 75, 38, 79.5, 38)
    ax.text(50.75, 39.6, "adapter", ha="center", fontsize=7.6, color=GREY, style="italic")
    ax.text(77.25, 39.6, "GGUF", ha="center", fontsize=7.6, color=GREY, style="italic")

    _arrow(ax, 19, 45, 26.5, 41)            # input -> bridge
    _arrow(ax, 104, 38, 108, 38)            # bridge -> AIRV

    _box(ax, 108, 29.5, 25, 17, "AIRV decoding",
         "8 D4 augmentations\ntransform → generate → invert\nvote → top-2",
         ec=INK, tc=INK)
    _arrow(ax, 133, 38, 137, 38)            # AIRV -> answer

    ga = ax.inset_axes([137, 32, 12, 12], transform=ax.transData)
    render_grid(ga, D_SOLVED)
    ax.text(143, 29.5, "answer", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(143, 26.8, "2 attempts", ha="center", fontsize=8.4, color=GREY)

    # callouts
    ax.text(64, 18.4, "Why the bridge is the contribution", ha="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(64, 13.2,
            "PyTorch can train the adapter but decodes at only 13 tok/s, far too slow to run per-task TTT across the eval set;\n"
            "llama.cpp serves at ~115 tok/s with the adapter but cannot train it. Converting to GGUF and hot-swapping links the two.\n"
            "Train, convert, and serve run as three phases so they never share the 16 GB at once.",
            ha="center", fontsize=9.2, color=GREY, linespacing=1.6)
    ax.add_patch(FancyArrowPatch((64, 24), (64, 21), arrowstyle="-", color="#bbbbbb",
                 lw=1.0, zorder=1))

    ax.text(5, 3.0, "Model: NVARC Qwen3-4B (16-token grid vocab), Q8_0 GGUF 3.6 GB     "
            "Hardware: 1× RTX 5070 Ti, 16 GB, Blackwell, Windows", fontsize=8.6,
            color="#888888")
    footer(fig)
    save(fig, "fig_pipeline", dpi=180)


# ============================================================ A1 ROOFLINE
def fig_roofline():
    # measured tok/s vs the 896 GB/s memory roof (896 / weight GB; weights are re-read per token).
    paths = ["HF eager\n(PyTorch fp16)", "llama.cpp\nbf16", "llama.cpp\nQ8_0", "llama.cpp\nQ4_K_M"]
    measured = [13, 102, 170, 244]
    ceiling = [132, 132, 249, 444]          # 896 / (6.77, 6.77, 3.60, 2.02 GB)
    pct = [10, 77, 68, 55]
    cols = [GREY, BLUEL, BLUE, GREYL]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.bar(x, [c - m for c, m in zip(ceiling, measured)], bottom=measured, width=0.6,
           color="#f2f2f2", edgecolor="#dadada", lw=0.6, zorder=2)   # lost throughput
    ax.bar(x, measured, width=0.6, color=cols, zorder=3)             # achieved
    for xi, c in zip(x, ceiling):
        ax.hlines(c, xi - 0.32, xi + 0.32, color=INK, lw=2.0, zorder=4)
    for xi, m, c, p in zip(x, measured, ceiling, pct):
        ax.text(xi, m + 7, str(m), ha="center", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold", zorder=5)
        ax.text(xi, c + 8, "%d%%\nof roof" % p, ha="center", va="bottom", fontsize=8.8,
                color=BLUE if xi == 2 else GREY, fontweight="bold" if xi == 2 else "normal",
                linespacing=1.0)
    ax.text(2, 196, "operating point", ha="center", fontsize=8.4, color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(paths, fontsize=9)
    ax.set_ylabel("decode throughput (tok/s)", fontsize=10)
    ax.set_ylim(0, 512)
    ax.set_title("Decode is memory-bandwidth-bound", fontsize=13.5, fontweight="bold",
                 color=INK, pad=26)
    ax.text(0.5, 1.05, "measured throughput vs the 896 GB/s roof; caps mark each precision's ceiling",
            transform=ax.transAxes, ha="center", fontsize=9.6, color=GREY, style="italic")
    ax.text(0.5, -0.22,
            "eager and llama.cpp bf16 run the same weights at the same roof, so their ~8x gap is pure "
            "engine overhead (no CUDA graphs), not the card.\nRoof = 896 GB/s / weight bytes per token "
            "(bf16 6.77, Q8_0 3.60, Q4_K_M 2.02 GB); a first-order model, KV-cache reads add a few percent.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.1, color=GREY, linespacing=1.5)
    despine(ax); ax.grid(axis="y", color=RAIL)
    footer(fig)
    save(fig, "fig_roofline")


def _load_csv(name):
    with open(os.path.join(OUT, "data", name)) as f:
        return list(csv.DictReader(f))


# ============================================================ A2 VRAM WALL (histogram)
def fig_vram_wall():
    rows = _load_csv("task_dims.csv")
    mp = np.array([int(r["maxpair"]) for r in rows])
    tr = np.array([int(r["trainable"]) for r in rows]) == 1
    CUT = 1000
    binw = 100
    top = (int(mp.max() // binw) + 2) * binw
    edges = np.arange(0, top + 1, binw)
    maxc = int(np.histogram(mp, bins=edges)[0].max())
    ntr, nex = int(tr.sum()), int((~tr).sum())

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.axvspan(0, CUT, color=GREEN, alpha=0.05, zorder=0)
    ax.axvspan(CUT, top, color=FAIL, alpha=0.06, zorder=0)
    ax.hist([mp[tr], mp[~tr]], bins=edges, stacked=True, color=[GREEN, FAIL],
            edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(CUT, ls="--", color=FAIL, lw=2.0, zorder=4)

    ax.set_xlabel("largest train pair (tokens), the peak-VRAM driver", fontsize=10)
    ax.set_ylabel("number of tasks", fontsize=10)
    ax.set_xlim(0, top); ax.set_ylim(0, maxc * 1.34)
    ax.set_xticks(np.arange(0, top + 1, 250))

    ax.text(CUT * 0.52, maxc * 1.17, "%d tasks fit in 16 GB\nper-task TTT" % ntr, ha="center",
            va="center", fontsize=11, color=GREEN, fontweight="bold")
    ax.text((CUT + top) / 2, maxc * 1.17, "%d exceed it\nserved no-TTT" % nex, ha="center",
            va="center", fontsize=11, color=FAIL, fontweight="bold")
    ax.annotate("16 GB budget = 1000 tokens", xy=(CUT, maxc * 0.42),
                xytext=(CUT * 1.05, maxc * 0.66), ha="left", fontsize=8.8, color=FAIL,
                arrowprops=dict(arrowstyle="->", color=FAIL, lw=1.2))
    despine(ax); ax.grid(axis="y", color=RAIL)

    ax.set_title("The VRAM wall: which tasks fit per-task TTT in 16 GB", fontsize=13.5,
                 fontweight="bold", color=INK, pad=26)
    ax.text(0.5, 1.05, "peak VRAM under leave-one-out TTT is set by the longest single training sequence",
            transform=ax.transAxes, ha="center", fontsize=9.6, color=GREY, style="italic")
    ax.text(0.5, -0.205, "Tasks past the 1000-token cutoff exceed 16 GB to train and are served no-TTT; "
            "forcing TTT on them scores 10.6% vs 12.8%.", transform=ax.transAxes, ha="center",
            va="top", fontsize=8.6, color=GREY)
    footer(fig)
    save(fig, "fig_vram_wall")


# ============================================================ A3 SCALE FLIP
def fig_scale_flip():
    rows = _load_csv("scale_flip.csv")
    cols = ["no_ttt", "s050", "s070", "s085", "s100", "s125", "s140", "s150", "s175"]
    lab = ["no-TTT", "0.5", "0.7", "0.85", "1.0", "1.25", "1.4", "1.5", "1.75"]
    solved = [r for r in rows if any(int(r[c]) for c in cols)]
    # drop outputs solved at every scale (they add a constant and carry no scale signal)
    disc = [r for r in solved if sum(int(r[c]) for c in cols) < len(cols)]
    n_always = len(solved) - len(disc)

    def firstsolve(r):
        for i, c in enumerate(cols):
            if int(r[c]):
                return i
        return len(cols)

    disc.sort(key=lambda r: (firstsolve(r), -sum(int(r[c]) for c in cols)))
    M = np.array([[int(r[c]) for c in cols] for r in disc])
    n = len(disc)
    counts = M.sum(0)

    fig = plt.figure(figsize=(9.0, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4.4], hspace=0.07, top=0.83, bottom=0.12,
                          left=0.10, right=0.95)
    axb = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=axb)
    x = np.arange(9)
    bc = [BLUEL] * 9
    bc[0] = GREY
    bc[5] = BLUE
    axb.bar(x, counts, color=bc, width=0.9, zorder=3)
    for i, c in enumerate(counts):
        axb.text(i, c + 0.45, str(c), ha="center", va="bottom", fontsize=8.8, color=INK,
                 fontweight="bold" if i == 5 else "normal")
    axb.set_ylim(0, int(counts.max()) + 4)
    axb.axis("off")

    axh.imshow(M, aspect="auto", cmap=ListedColormap(["#ECECEC", BLUE]), vmin=0, vmax=1,
               interpolation="nearest")
    axh.set_xticks(np.arange(-.5, 9, 1), minor=True)
    axh.set_yticks(np.arange(-.5, n, 1), minor=True)
    axh.grid(which="minor", color="white", linewidth=0.8)
    axh.set_xticks(x)
    axh.set_xticklabels(lab, fontsize=9.4)
    axh.set_yticks([]); axh.set_xlim(-0.5, 8.5)
    axh.set_ylabel("outputs that scale flips (n=%d)" % n, fontsize=9.5)
    axh.tick_params(length=0)
    axh.add_patch(Rectangle((4.5, -0.5), 1, n, fill=False, edgecolor=BLUE, lw=2.2, zorder=5))
    for yi, r in enumerate(disc):
        if int(r["s125"]) and not all(int(r[c]) for c in ("s140", "s150", "s175")):
            axh.add_patch(Rectangle((4.5, yi - 0.5), 1, 1, fill=False, edgecolor=FAIL, lw=1.8, zorder=6))
    fig.text(0.5, 0.945, "Which adapter scale solves which tasks", ha="center", fontsize=13.5,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.90, "rows are the outputs scale flips on or off; bars count solves per column, "
             "peaking at 1.25", ha="center", fontsize=9.6, color=GREY, style="italic")
    fig.text(0.5, 0.045, "Blue = solved (exact match, either attempt); grey = unsolved. A further %d "
             "outputs are solved at every scale and omitted (a constant offset).\nRed outlines mark "
             "outputs solved at 1.25 but lost at a higher scale, the plateau's churn." % n_always,
             ha="center", fontsize=8.4, color=GREY, linespacing=1.5)
    footer(fig)
    save(fig, "fig_scale_flip")


# ============================================================ B5 COST FRONTIER
def fig_cost_frontier():
    rows = _load_csv("timings.csv")
    pts = {r["config"]: (float(r["gpu_minutes"]), float(r["accuracy"]), r["kind"]) for r in rows}
    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    gx, gy, _ = pts["no-TTT eager"]
    ax.scatter([gx], [gy], s=95, facecolor="none", edgecolor=GREYL, linewidth=1.6, zorder=3)
    ax.annotate("the same no-TTT run on eager\n(13 tok/s): ~17 hours", (gx, gy), xytext=(gx, gy + 0.95),
                ha="center", fontsize=8, color=GREY)

    for k, (x, y, kind) in pts.items():
        if kind == "measured" and k.startswith("TTT"):
            ax.scatter([x], [y], s=42, color=BLUEL, zorder=4)

    nx, ny, _ = pts["no-TTT"]
    ax.scatter([nx], [ny], s=140, color=GREY, zorder=5)
    ax.annotate("no-TTT\n12.79%", (nx, ny), xytext=(nx * 0.90, ny - 1.5), ha="center", fontsize=9.2, color=INK)

    cx, cy, _ = pts["colour-perm"]
    ax.scatter([cx], [cy], s=66, facecolor="white", edgecolor=BLUE, linewidth=1.4, zorder=4)
    ax.annotate("colour-perm\n13.95%  (est. time)", (cx, cy), xytext=(cx * 1.02, cy - 1.55),
                ha="left", fontsize=8, color=GREY)

    ox, oy, _ = pts["TTT 1.25"]
    ax.scatter([ox], [oy], s=155, color=BLUE, zorder=6, edgecolor="white", linewidth=0.9)
    ax.annotate("TTT blended, scale 1.25\n16.28%  (operating point)", (ox, oy), xytext=(ox * 1.045, oy + 0.05),
                ha="left", fontsize=9.2, color=BLUE, fontweight="bold")

    ax.plot([nx, ox], [ny, oy], "--", color=BLUE, lw=1.5, zorder=2)
    ax.annotate("+3.5 pts\nfor +17 min", ((nx + ox) / 2 * 0.985, (ny + oy) / 2 - 0.15), fontsize=8.4,
                color=BLUE, ha="right", style="italic")

    ax.set_xscale("log")
    ax.set_xlim(64, 1600)
    ax.set_ylim(11, 18)
    ax.set_xticks([70, 100, 200, 500, 1000])
    ax.set_xticklabels(["70", "100", "200", "500", "1000"], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("GPU-minutes on one 16 GB card (log scale)", fontsize=10)
    ax.set_ylabel("accuracy, full 120 (%)", fontsize=10)
    ax.set_title("Accuracy per GPU-hour on one 16 GB card", fontsize=13.5, fontweight="bold",
                 color=INK, pad=24)
    ax.text(0.5, 1.05, "each accuracy gain has a compute price; the 13x engine keeps the whole "
            "frontier in hours, not days", transform=ax.transAxes, ha="center", fontsize=9.4,
            color=GREY, style="italic")
    despine(ax)
    ax.grid(color=RAIL, which="major")
    ax.text(0.5, -0.21, "Compute is wall-clock on the single RTX 5070 Ti. The eight TTT points share "
            "one training pass (adapter scale is set at serve time), so they stack at one cost and 1.25 "
            "is the best of them.\nThe eager ghost is a 13x decode projection; the colour-perm time is "
            "estimated (2x augmentations, accuracy measured).", transform=ax.transAxes, ha="center",
            va="top", fontsize=8.0, color=GREY, linespacing=1.5)
    footer(fig)
    save(fig, "fig_cost_frontier")


# ============================================================ B4 AIRV VOTES
def fig_airv_votes():
    rows = _load_csv("airv_votes.csv")
    mcols = [k for k in rows[0].keys() if k.startswith("m_")]
    labels = [c[2:].replace("_", " ") for c in mcols]
    A = np.array([[int(r[c]) for c in mcols] for r in rows])
    votes = np.array([int(r["top1_votes"]) for r in rows])
    correct = np.array([int(r["correct"]) for r in rows])
    order = np.lexsort((correct, A.sum(1)))[::-1]
    A2, c2 = A[order], correct[order]
    n = len(rows)

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.07, 1.05], wspace=0.05, top=0.83, bottom=0.14,
                          left=0.07, right=0.96)
    axh = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1])
    axb = fig.add_subplot(gs[2])

    axh.imshow(A2, aspect="auto", cmap=ListedColormap(["#ECECEC", BLUE]), vmin=0, vmax=1,
               interpolation="nearest")
    axh.set_xticks(range(len(labels)))
    axh.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axh.set_yticks([])
    axh.set_ylabel("test outputs, sorted by agreement (n=%d)" % n, fontsize=9.5)
    axh.set_title("Augmentation agreement with the voted answer", fontsize=11.5, fontweight="bold",
                  color=INK)

    axs.imshow(c2.reshape(-1, 1), aspect="auto", cmap=ListedColormap([FAIL, GREEN]), vmin=0, vmax=1)
    axs.set_xticks([0])
    axs.set_xticklabels(["correct"], rotation=40, ha="right", fontsize=7.5)
    axs.set_yticks([])

    bins = list(range(1, 9))
    frac, ns = [], []
    for v in bins:
        mask = votes == v
        ns.append(int(mask.sum()))
        frac.append(100.0 * correct[mask].mean() if mask.sum() else 0.0)
    axb.bar(bins, frac, color=BLUE, width=0.72, zorder=3)
    for v, f, nn in zip(bins, frac, ns):
        axb.text(v, f + 1.5, "n=%d" % nn, ha="center", fontsize=7.6, color=GREY)
    axb.set_xlabel("top-1 vote count, of 8 augmentations", fontsize=9.5)
    axb.set_ylabel("fraction correct (%)", fontsize=9.5)
    axb.set_xticks(bins)
    axb.set_ylim(0, 105)
    axb.set_title("Agreement predicts correctness", fontsize=11.5, fontweight="bold", color=INK)
    despine(axb)
    axb.grid(axis="y", color=RAIL)

    fig.text(0.5, 0.95, "AIRV: augmentation agreement and correctness", ha="center", fontsize=14,
             fontweight="bold", color=INK)
    fig.text(0.5, 0.04, "Left: for each test output, which of the 8 dihedral (D4) transforms produced "
             "the voted top-1 grid after inversion (blue = agree); the strip marks whether top-1 was "
             "correct.\nRight: the more transforms agree on one answer, the more often it is right, so "
             "the vote count is a usable confidence signal.", ha="center", fontsize=8.2, color=GREY,
             linespacing=1.5)
    footer(fig)
    save(fig, "fig_airv_votes")


# ============================================================ B6 PHASE TIMELINE
def fig_phases():
    rows = _load_csv("vram_trace.csv")
    t = np.array([float(r["t_seconds"]) for r in rows])
    m = np.array([float(r["mem_used_gb"]) for r in rows])
    ph = np.array([r["phase"] for r in rows])
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    cmap = {"idle": GREYL, "train": GREEN, "convert": "#E8820C", "serve": BLUE}
    for name in ["idle", "train", "convert", "serve"]:
        mask = ph == name
        if mask.any():
            ax.fill_between(t, 0, m, where=mask, color=cmap[name], alpha=0.5, step="mid", linewidth=0)
    ax.plot(t, m, color=INK, lw=0.9, zorder=4)
    ax.axhline(16, ls="--", color=FAIL, lw=1.5, zorder=3)
    ax.text(t.max() * 0.995, 16.35, "16 GB ceiling", ha="right", fontsize=9, color=FAIL)
    for name in ["train", "convert", "serve"]:
        mask = ph == name
        if not mask.any():
            continue
        ts, ms = t[mask], m[mask]
        mid = (ts.min() + ts.max()) / 2
        peak = ms.max()
        ax.annotate("%s\n%.0fs · peak %.1f GB" % (name, ts.max() - ts.min(), peak), (mid, peak),
                    xytext=(mid, max(peak + 1.5, 2.2)), ha="center", fontsize=8.4, color=cmap[name],
                    fontweight="bold")
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("VRAM used (GB)", fontsize=10)
    ax.set_ylim(0, 17.5)
    ax.set_xlim(0, t.max())
    ax.set_title("Train, convert, serve: three phases sharing 16 GB in sequence", fontsize=13,
                 fontweight="bold", color=INK, pad=10)
    despine(ax)
    ax.grid(axis="y", color=RAIL)
    fig.text(0.5, -0.02, "One task through the bridge on the single 16 GB card. PyTorch trains the "
             "adapter, then exits to free VRAM; the adapter converts to GGUF; then llama.cpp serves it. "
             "The trainer and engine never co-reside.", ha="center", fontsize=8.2, color=GREY,
             linespacing=1.5)
    footer(fig)
    save(fig, "fig_phases")


if __name__ == "__main__":
    fig_roofline()
    fig_vram_wall()
    fig_scale_flip()
    fig_cost_frontier()
    fig_airv_votes()
    fig_phases()
    fig_results()
    fig_solve()
    fig_levers()
    fig_pipeline()
