"""The two chosen hero designs, built out.

  paper/hero_triptych.png   three findings, three panels, one hue each
  paper/hero_field.png      the live leaderboard with our bars inside it and the gap decomposed

Fixes over the option drafts: consistent two-decimal values, the replicate band drawn wherever a
blue bar appears, panel two rebuilt as shown-run pairs per notebook, the inversion panel carrying
its endpoint values and z, the field's displayed-to-run drops drawn inside the leaderboard itself,
and the 5.8 rounded honestly to its 5.4 to 5.8 range.

Palette is the validated five-hue set: #2166AC, #C25E11, #22957A, #9B59B6, #D6425A on white.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"D:\ARC_AGI-2,3\paper"
W, H, DPI = 11.2, 5.6, 100

BLUE, AMBER, TEAL, VIOLET, ROSE = "#2166AC", "#C25E11", "#22957A", "#9B59B6", "#D6425A"
INK, MUTE, RAIL, SURF, GREY = "#141414", "#6b6b66", "#e9e9e6", "#ffffff", "#b9b9b4"
SANS = ["Helvetica Neue", "Arial", "DejaVu Sans"]
LO, HI = 30.56, 30.97

# the ARC cell palette, for the drawn task motif that carries the benchmark's visual identity
ARCPAL = ["#101014", "#1E6FD0", "#E23B36", "#37B24D", "#F0C419",
          "#9AA0A6", "#D6489B", "#E8820C", "#5FC6E8", "#8A1B2E"]
MOTIF_IN = [[0, 0, 2, 0, 0], [0, 0, 2, 0, 0], [2, 2, 2, 2, 2],
            [0, 0, 2, 0, 0], [0, 0, 2, 0, 0]]
MOTIF_OUT = [[4, 0, 3, 0, 4], [0, 0, 3, 0, 0], [3, 3, 3, 3, 3],
             [0, 0, 3, 0, 0], [4, 0, 3, 0, 4]]


def arc_motif(fig, x, y, cell=0.0125):
    """A small drawn ARC task, input to output, in figure coordinates. Our own grids, not a logo."""
    import matplotlib.patches as mp
    aspect = H / W
    cw, ch = cell, cell / aspect
    for gx, grid in ((x, MOTIF_IN), (x + 6.6 * cw, MOTIF_OUT)):
        n = len(grid)
        for r in range(n):
            for c in range(n):
                fig.patches.append(mp.Rectangle(
                    (gx + c * cw, y + (n - 1 - r) * ch), cw * 0.92, ch * 0.92,
                    transform=fig.transFigure, facecolor=ARCPAL[grid[r][c]],
                    edgecolor="white", linewidth=0.4, zorder=5))
    ax_, ay_ = x + 5.55 * cw, y + 2.5 * ch
    fig.patches.append(mp.FancyArrow(ax_ - 0.35 * cw, ay_, 0.62 * cw, 0,
                                     transform=fig.transFigure, width=0.16 * ch,
                                     head_width=0.55 * ch, head_length=0.30 * cw,
                                     color="#8e8e89", zorder=5))


def styled(ax, ylab):
    ax.set_ylabel(ylab, fontsize=9.5, color=MUTE)
    ax.tick_params(labelsize=9, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RAIL)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, facecolor=SURF)
    plt.close(fig)
    print("saved", name)


def hero_triptych():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS})
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURF)
    fig.text(0.055, 0.895, "The 16GB Ceiling", fontsize=31, color=INK, fontweight="bold")
    fig.text(0.055, 0.828, "one card priced, one leaderboard audited, one checkpoint caught",
             fontsize=13, color=MUTE)
    arc_motif(fig, 0.795, 0.825)

    # panel 1: the constraint, gap drawn as a dimension line clear of both bars
    a1 = fig.add_axes([0.055, 0.235, 0.245, 0.45], facecolor=SURF)
    a1.axhspan(LO, HI, color=BLUE, alpha=0.15, zorder=1)
    a1.bar([0, 1], [18.19, 30.97], 0.55, color=[GREY, BLUE], zorder=2)
    a1.text(0, 19.1, "18.19", ha="center", fontsize=14, color="#8e8e89", fontweight="bold")
    a1.text(1, 31.9, "30.97", ha="center", fontsize=14, color=BLUE, fontweight="bold")
    a1.plot([0.28, 1.55], [18.19, 18.19], color="#cfcfca", lw=1.0, ls=":", zorder=1)
    a1.annotate("", xy=(1.52, 30.75), xytext=(1.52, 18.4),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.0))
    a1.text(1.73, 24.6, "12.78", fontsize=13, color=AMBER, fontweight="bold",
            va="center", ha="center", rotation=90)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["16 GB", "96 GB"], fontsize=10.5)
    a1.set_ylim(0, 37)
    a1.set_xlim(-0.55, 1.95)
    styled(a1, "hidden score (%)")

    # panel 2: shown against run, one pair per notebook
    a2 = fig.add_axes([0.385, 0.235, 0.245, 0.45], facecolor=SURF)
    xs = [0, 0.8, 2.2, 3.0]
    vals = [33.89, 28.06, 33.89, 28.47]
    cols = ["#d8d8d3", VIOLET, "#d8d8d3", VIOLET]
    a2.bar(xs, vals, 0.62, color=cols, zorder=2)
    for x, v, c in zip(xs, vals, cols):
        if c == VIOLET:
            a2.text(x, v - 1.1, "%.2f" % v, ha="center", va="top", fontsize=10.5,
                    color="white", fontweight="bold")
        else:
            a2.text(x, v + 0.8, "%.2f" % v, ha="center", fontsize=10.5,
                    color="#8e8e89", fontweight="bold")
    for x0, x1, v in ((0, 0.8, 28.06), (2.2, 3.0, 28.47)):
        a2.annotate("", xy=(x1, v + 1.0), xytext=(x0, 33.3),
                    arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.6,
                                    connectionstyle="arc3,rad=-0.32"))
    a2.set_xticks([0.4, 2.6])
    a2.set_xticklabels(["notebook A", "notebook B"], fontsize=10.5)
    a2.set_ylim(0, 37)
    a2.set_xlim(-0.6, 3.6)
    styled(a2, "hidden score (%)")

    # panel 3: the ordering inversion
    a3 = fig.add_axes([0.715, 0.235, 0.245, 0.45], facecolor=SURF)
    a3.plot([0, 1], [10, 78.3], "-o", color=TEAL, lw=2.6, ms=9, zorder=3)
    a3.plot([0, 1], [80, 65], "-o", color=ROSE, lw=2.6, ms=9, zorder=2)
    a3.text(-0.07, 10, "10", ha="right", va="center", fontsize=11, color=TEAL,
            fontweight="bold")
    a3.text(-0.07, 80, "80", ha="right", va="center", fontsize=11, color=ROSE,
            fontweight="bold")
    a3.text(1.07, 78.3, "78  4B", ha="left", va="center", fontsize=11, color=TEAL,
            fontweight="bold")
    a3.text(1.07, 65, "65  2B", ha="left", va="center", fontsize=11, color=ROSE,
            fontweight="bold")
    a3.text(0.5, 90, "interaction z = 8.0", ha="center", fontsize=9.5, color=INK,
            style="italic")
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["published\neval set", "fresh\ngenerated"], fontsize=9.5)
    a3.set_xlim(-0.42, 1.55)
    a3.set_ylim(0, 100)
    styled(a3, "exact match, no TTT (%)")

    heads = [("What one card costs", 0.055, BLUE),
             ("What a displayed score means", 0.385, VIOLET),
             ("What a public checkpoint knows", 0.715, ROSE)]
    for lab, x, c in heads:
        fig.text(x, 0.74, lab, fontsize=12.5, color=c, fontweight="bold")
    subs = [("same method, twice the memory; band = replicate spread", 0.055),
            ("two public notebooks, forked unmodified and scored", 0.385),
            ("capability order inverts on data it cannot have seen", 0.715)]
    for lab, x in subs:
        fig.text(x, 0.14, lab, fontsize=9, color=MUTE)
    fig.text(0.955, 0.045, "ARC Prize 2026 Paper Track  ·  submission 54974874",
             fontsize=8.5, color="#9a9a95", ha="right")
    save(fig, "hero_triptych.png")


def hero_field():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS})
    field = [67.50, 44.86, 37.22, 34.44, 33.89, 33.89, 33.47, 33.47, 33.47,
             33.19, 33.19, 33.06, 33.06, 33.06, 33.06]
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURF)
    fig.text(0.055, 0.895, "The 16GB Ceiling", fontsize=31, color=INK, fontweight="bold")
    fig.text(0.055, 0.828, "where one consumer card puts you, and what the gaps are made of",
             fontsize=13, color=MUTE)
    arc_motif(fig, 0.795, 0.825)

    ax = fig.add_axes([0.055, 0.19, 0.60, 0.56], facecolor=SURF)
    xs = np.arange(len(field))
    ax.bar(xs, field, 0.68, color="#d3d3ce", zorder=2)
    ax.text(0, 68.6, "67.50", fontsize=10.5, color="#8e8e89", ha="center")
    ax.text(8.2, 61, "public leaderboard top 15\n2026-08-04", fontsize=10, color="#8e8e89",
            ha="center", linespacing=1.4)

    # the displayed-to-run drops, drawn where the 33.89 pair sits
    for i, run in ((4, 28.06), (5, 28.47)):
        ax.annotate("", xy=(i, run + 0.3), xytext=(i, 33.6),
                    arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.8))
        ax.plot([i - 0.30, i + 0.30], [run, run], color=VIOLET, lw=2.2, zorder=4)

    bx, cx = len(field) + 1.0, len(field) + 2.15
    ax.axhspan(LO, HI, xmin=(bx - 0.55 + 0.8) / (len(field) + 3.3 + 0.8), xmax=1.0,
               color=BLUE, alpha=0.15, zorder=1)
    ax.bar([bx], [30.97], 0.68, color=BLUE, zorder=3)
    ax.bar([cx], [18.19], 0.68, color=AMBER, zorder=3)
    ax.text(bx, 31.9, "30.97", ha="center", fontsize=11.5, color=BLUE, fontweight="bold")
    ax.text(cx, 19.1, "18.19", ha="center", fontsize=11.5, color=AMBER, fontweight="bold")
    ax.text(bx, -3.2, "ours\n96 GB", ha="center", va="top", fontsize=9.5, color=BLUE,
            linespacing=1.3)
    ax.text(cx, -3.2, "ours\n16 GB", ha="center", va="top", fontsize=9.5, color=AMBER,
            linespacing=1.3)
    ax.set_xticks([])
    ax.set_ylim(0, 74)
    ax.set_xlim(-0.8, len(field) + 3.3)
    styled(ax, "ARC-AGI-2 hidden score (%)")
    ax.set_ylabel("ARC-AGI-2 hidden score (%)", fontsize=10, color=INK)

    col = [("12.78 points", AMBER, "what the card costs,\nmeasured on the same method"),
           ("0.41 points", BLUE, "the run-to-run noise,\nmeasured by replicate"),
           ("5.4 to 5.8 points", VIOLET, "what two public notebooks lose\nbetween displayed and run")]
    y = 0.665
    for num, c, txt in col:
        fig.text(0.705, y, num, fontsize=18, color=c, fontweight="bold")
        fig.text(0.705, y - 0.088, txt, fontsize=10.5, color=INK, linespacing=1.45)
        y -= 0.205
    fig.text(0.955, 0.045, "ARC Prize 2026 Paper Track  ·  submission 54974874",
             fontsize=8.5, color="#9a9a95", ha="right")
    save(fig, "hero_field.png")


if __name__ == "__main__":
    hero_triptych()
    hero_field()
