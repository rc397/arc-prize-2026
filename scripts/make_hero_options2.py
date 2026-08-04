"""Second round of hero designs, colour-carrying.

Round one failed because two bars cannot carry colour. These put more of the season on the page so
the palette encodes something, rather than decorating a two-value comparison.

  paper/opt_d_season.png     the whole submission arc, coloured by what each build was
  paper/opt_e_triptych.png   three panels, one per finding, each with its own hue
  paper/opt_f_field.png      our band against the live leaderboard, the constraint in context

Palette validated light-surface, five slots, all checks pass:
#2166AC blue, #C25E11 amber, #22957A teal, #9B59B6 violet, #D6425A rose.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"D:\ARC_AGI-2,3\paper"
W, H, DPI = 11.2, 5.6, 100

BLUE, AMBER, TEAL, VIOLET, ROSE = "#2166AC", "#C25E11", "#22957A", "#9B59B6", "#D6425A"
INK, MUTE, RAIL, SURF = "#141414", "#6b6b66", "#e9e9e6", "#ffffff"

SANS = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------- D: the season

def option_d():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS})
    # (tag, score, category)
    runs = [("v1", 6.53, "climb"), ("v2", 11.94, "climb"), ("v2.1", 13.61, "climb"),
            ("v3.1", 12.78, "climb"), ("v4", 13.19, "climb"), ("v5", 12.78, "climb"),
            ("v6", 12.36, "climb"), ("v6.1", 18.19, "card"), ("v7", 16.81, "climb"),
            ("v7.1", 30.56, "repro"), ("v7.2", 30.97, "repro"), ("v7.3", 28.61, "ours"),
            ("v7.4", 29.44, "ours"), ("v7.5a", 27.92, "ours"), ("v7.2r", 30.56, "repro")]
    col = {"climb": "#b9b9b4", "card": AMBER, "repro": BLUE, "ours": ROSE}

    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURF)
    fig.text(0.055, 0.885, "The 16GB Ceiling", fontsize=33, color=INK, fontweight="bold")
    fig.text(0.055, 0.815, "fifteen scored submissions, and the twelve points a single card costs",
             fontsize=13, color=MUTE)

    ax = fig.add_axes([0.055, 0.20, 0.90, 0.55], facecolor=SURF)
    xs = np.arange(len(runs))
    ax.axhspan(30.56, 30.97, color=BLUE, alpha=0.13, zorder=1)
    for i, (tag, v, cat) in enumerate(runs):
        ax.bar(i, v, 0.66, color=col[cat], zorder=2)
    for i, (tag, v, cat) in enumerate(runs):
        if cat in ("card", "repro") or v == min(r[1] for r in runs if r[2] == "ours"):
            ax.text(i, v + 0.7, "%.2f" % v, ha="center", fontsize=11.5,
                    color=col[cat], fontweight="bold")
    ax.annotate("", xy=(10, 30.4), xytext=(7, 18.6),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(8.1, 25.6, "12.78 points", fontsize=13.5, color=AMBER, fontweight="bold")
    ax.text(8.1, 23.9, "the price of one card", fontsize=10, color=AMBER)

    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in runs], fontsize=9, color=INK)
    ax.set_ylim(0, 36)
    ax.set_xlim(-0.7, len(runs) - 0.3)
    ax.set_ylabel("hidden score (%)", fontsize=10.5, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RAIL)
    ax.set_axisbelow(True)

    keys = [("the climb, 16 GB", "#b9b9b4"), ("best on one card", AMBER),
            ("public method, 96 GB", BLUE), ("our own additions", ROSE)]
    for j, (lab, c) in enumerate(keys):
        fig.patches.append(plt.Rectangle((0.055 + j * 0.175, 0.085), 0.016, 0.028,
                                         transform=fig.transFigure, color=c, zorder=3))
        fig.text(0.077 + j * 0.175, 0.088, lab, fontsize=10, color=INK)
    fig.text(0.955, 0.030, "ARC Prize 2026 Paper Track  ·  submission 54974874",
             fontsize=8.5, color="#9a9a95", ha="right")
    save(fig, "opt_d_season.png")


# ----------------------------------------------------------------------------- E: triptych

def option_e():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS})
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURF)
    fig.text(0.055, 0.885, "The 16GB Ceiling", fontsize=32, color=INK, fontweight="bold")
    fig.text(0.055, 0.818, "one card priced, one leaderboard audited, one checkpoint caught",
             fontsize=13, color=MUTE)

    # panel 1, the constraint
    a1 = fig.add_axes([0.055, 0.235, 0.245, 0.46], facecolor=SURF)
    a1.bar([0, 1], [18.19, 30.97], 0.55, color=["#b9b9b4", BLUE], zorder=2)
    for x, v, c in ((0, 18.19, "#8e8e89"), (1, 30.97, BLUE)):
        a1.text(x, v + 0.9, "%.1f" % v, ha="center", fontsize=15, color=c, fontweight="bold")
    a1.annotate("", xy=(0.5, 30.2), xytext=(0.5, 18.9),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.9))
    a1.text(0.58, 24.3, "12.78", fontsize=13, color=AMBER, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["16 GB", "96 GB"], fontsize=10.5)
    a1.set_ylim(0, 37)

    # panel 2, displayed against scored
    a2 = fig.add_axes([0.385, 0.235, 0.245, 0.46], facecolor=SURF)
    a2.bar([0, 1, 2], [33.89, 28.06, 28.47], 0.55,
           color=["#d8d8d3", VIOLET, VIOLET], zorder=2)
    a2.text(0, 34.8, "33.89", ha="center", fontsize=13, color="#8e8e89", fontweight="bold")
    for x, v in ((1, 28.06), (2, 28.47)):
        a2.text(x, v + 0.9, "%.2f" % v, ha="center", fontsize=12.5, color=VIOLET,
                fontweight="bold")
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels(["shown", "run", "run"], fontsize=10.5)
    a2.set_ylim(0, 37)

    # panel 3, the inversion
    a3 = fig.add_axes([0.715, 0.235, 0.245, 0.46], facecolor=SURF)
    for y0, y1, c, lab in ((10, 78.3, TEAL, "4B"), (80, 65, ROSE, "2B")):
        a3.plot([0, 1], [y0, y1], "-o", color=c, lw=2.6, ms=9, zorder=2)
        a3.text(1.06, y1, lab, va="center", fontsize=11.5, color=c, fontweight="bold")
    a3.text(0, 8, "10", ha="center", va="top", fontsize=11, color=TEAL, fontweight="bold")
    a3.text(0, 84, "80", ha="center", fontsize=11, color=ROSE, fontweight="bold")
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["published", "fresh"], fontsize=10.5)
    a3.set_xlim(-0.25, 1.3)
    a3.set_ylim(0, 100)

    for ax, ylab in ((a1, "hidden score (%)"), (a2, "hidden score (%)"), (a3, "exact match (%)")):
        ax.set_ylabel(ylab, fontsize=9.5, color=MUTE)
        ax.tick_params(labelsize=9, colors=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=RAIL)
        ax.set_axisbelow(True)

    heads = [("What one card costs", 0.055, BLUE),
             ("What a displayed score means", 0.385, VIOLET),
             ("What a public checkpoint knows", 0.715, ROSE)]
    for lab, x, c in heads:
        fig.text(x, 0.735, lab, fontsize=12, color=c, fontweight="bold")
    subs = [("the same method, twice the memory", 0.055),
            ("two notebooks, forked unmodified", 0.385),
            ("ordering inverts on fresh data", 0.715)]
    for lab, x in subs:
        fig.text(x, 0.145, lab, fontsize=9.5, color=MUTE)
    fig.text(0.955, 0.045, "ARC Prize 2026 Paper Track  ·  submission 54974874",
             fontsize=8.5, color="#9a9a95", ha="right")
    save(fig, "opt_e_triptych.png")


# ----------------------------------------------------------------------------- F: the field

def option_f():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": SANS})
    # live public leaderboard top, 2026-08-04
    field = [67.50, 44.86, 37.22, 34.44, 33.89, 33.89, 33.47, 33.47, 33.47,
             33.19, 33.19, 33.06, 33.06, 33.06, 33.06]
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURF)
    fig.text(0.055, 0.885, "The 16GB Ceiling", fontsize=33, color=INK, fontweight="bold")
    fig.text(0.055, 0.815, "where one consumer card puts you, and what that gap is made of",
             fontsize=13, color=MUTE)

    ax = fig.add_axes([0.055, 0.20, 0.62, 0.55], facecolor=SURF)
    xs = np.arange(len(field))
    ax.bar(xs, field, 0.68, color="#d3d3ce", zorder=2)
    ax.bar([len(field) + 0.8], [30.97], 0.68, color=BLUE, zorder=3)
    ax.bar([len(field) + 1.8], [18.19], 0.68, color=AMBER, zorder=3)
    ax.text(len(field) + 0.8, 31.9, "30.97", ha="center", fontsize=12, color=BLUE,
            fontweight="bold")
    ax.text(len(field) + 1.8, 19.1, "18.19", ha="center", fontsize=12, color=AMBER,
            fontweight="bold")
    ax.text(len(field) + 0.8, -4.4, "ours\n96 GB", ha="center", va="top", fontsize=10, color=BLUE)
    ax.text(len(field) + 1.8, -4.4, "ours\n16 GB", ha="center", va="top", fontsize=10, color=AMBER)
    ax.text(7, 60, "the public leaderboard, top fifteen", fontsize=11.5, color="#8e8e89")
    ax.text(0, 69.5, "67.50", fontsize=11, color="#8e8e89", ha="center")
    ax.set_xticks([])
    ax.set_ylim(0, 76)
    ax.set_xlim(-0.8, len(field) + 2.5)
    ax.set_ylabel("ARC-AGI-2 hidden score (%)", fontsize=10.5, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=RAIL)
    ax.set_axisbelow(True)

    fig.text(0.71, 0.66, "12.78 points", fontsize=19, color=AMBER, fontweight="bold")
    fig.text(0.71, 0.605, "is what the card costs,\nmeasured on the same method.",
             fontsize=11, color=INK, linespacing=1.5)
    fig.text(0.71, 0.455, "0.41 points", fontsize=19, color=BLUE, fontweight="bold")
    fig.text(0.71, 0.40, "is the run-to-run noise,\nmeasured by replicate.",
             fontsize=11, color=INK, linespacing=1.5)
    fig.text(0.71, 0.25, "5.8 points", fontsize=19, color=VIOLET, fontweight="bold")
    fig.text(0.71, 0.195, "is what two public notebooks\nlose when actually run.",
             fontsize=11, color=INK, linespacing=1.5)
    fig.text(0.955, 0.045, "ARC Prize 2026 Paper Track  ·  submission 54974874",
             fontsize=8.5, color="#9a9a95", ha="right")
    save(fig, "opt_f_field.png")


if __name__ == "__main__":
    option_d()
    option_e()
    option_f()
