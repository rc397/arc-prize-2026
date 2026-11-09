"""Three hero designs for the same claim, so the gallery cover can be chosen rather than defaulted.

  paper/opt_a_editorial.png    bold editorial bars, light surface
  paper/opt_b_dumbbell.png     one axis, two endpoints, the gap as the subject
  paper/opt_c_dark.png         dark surface, luminous accents

All three carry identical numbers. Palettes are light #0B4F9E/#D1495B and dark #4C93E8/#E8616F,
picked to stay distinct under colour-blindness and to keep contrast against their own surface.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = r"D:\ARC_AGI-2,3\paper"
W, H, DPI = 11.2, 5.6, 100          # 1120x560, Kaggle cover geometry

CARD = 18.19
NODE = 30.97
LO, HI = 30.56, 30.97


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI)
    plt.close(fig)
    print("saved", name)


# ----------------------------------------------------------------------------- A: editorial

def option_a():
    ink, blue, red, rail = "#141414", "#0B4F9E", "#D1495B", "#e9e9e6"
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    fig.text(0.055, 0.855, "The 16GB Ceiling", fontsize=40, color=ink, fontweight="bold")
    fig.text(0.055, 0.775, "One consumer GPU costs 12.78 points on ARC-AGI-2",
             fontsize=15.5, color="#4a4a4a")

    ax = fig.add_axes([0.055, 0.16, 0.60, 0.52])
    ax.set_facecolor("white")
    for i, (v, c, lab) in enumerate(((CARD, "#9a9a95", "RTX 5070 Ti, 16 GB\none consumer card"),
                                     (NODE, blue, "four L4s, 96 GB\nthe contest node"))):
        ax.barh(i, v, 0.44, color=c, zorder=2)
        ax.text(v + 0.6, i, "%.2f" % v, va="center", fontsize=27, color=c, fontweight="bold")
        ax.text(0.5, i + 0.36, lab, va="bottom", fontsize=10.5, color="white"
                if False else "#3a3a3a")
    ax.set_xlim(0, 38)
    ax.set_ylim(-0.6, 1.85)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # the gap as the subject, drawn between the bar ends
    ax.annotate("", xy=(NODE, 1.42), xytext=(CARD, 1.42),
                arrowprops=dict(arrowstyle="<->", color=red, lw=2.4))
    ax.text((CARD + NODE) / 2, 1.52, "12.78 points", ha="center", fontsize=15,
            color=red, fontweight="bold")

    fig.text(0.71, 0.60, "Same method.\nSame base model.\nSame hidden set.\nSame twelve hours.",
             fontsize=14, color=ink, linespacing=1.55)
    fig.text(0.71, 0.30, "Three runs of the unchanged\nconfiguration span %.2f to %.2f,\n"
                         "so the gap is 30x the noise." % (LO, HI),
             fontsize=10.5, color="#6b6b6b", linespacing=1.5)
    fig.text(0.055, 0.055, "ARC Prize 2026 Paper Track   ·   submission 54974874",
             fontsize=9, color="#9a9a95")
    save(fig, "opt_a_editorial.png")


# ----------------------------------------------------------------------------- B: dumbbell

def option_b():
    ink, blue, red = "#141414", "#0B4F9E", "#D1495B"
    grey = "#9a9a95"
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    fig.text(0.5, 0.88, "The 16GB Ceiling", fontsize=38, color=ink,
             fontweight="bold", ha="center")
    fig.text(0.5, 0.805, "what one consumer GPU costs on ARC-AGI-2, measured",
             fontsize=14, color="#4a4a4a", ha="center", style="italic")

    ax = fig.add_axes([0.09, 0.30, 0.82, 0.34])
    ax.set_xlim(14, 35)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # the measured noise band, drawn first so it reads as the tolerance on the right endpoint
    ax.axvspan(LO, HI, ymin=0.36, ymax=0.64, color=blue, alpha=0.16, zorder=1)

    ax.plot([CARD, NODE], [0, 0], color=grey, lw=3.5, solid_capstyle="round", zorder=2)
    ax.plot([CARD], [0], "o", ms=26, color=grey, zorder=3)
    ax.plot([NODE], [0], "o", ms=26, color=blue, zorder=3)
    ax.text(CARD, 0.34, "%.2f" % CARD, ha="center", fontsize=25, color=grey, fontweight="bold")
    ax.text(NODE, 0.34, "%.2f" % NODE, ha="center", fontsize=25, color=blue, fontweight="bold")
    ax.text(CARD, -0.44, "one consumer card\nRTX 5070 Ti, 16 GB", ha="center", va="top",
            fontsize=11.5, color=ink)
    ax.text(NODE, -0.44, "the contest node\nfour L4s, 96 GB", ha="center", va="top",
            fontsize=11.5, color=ink)

    arrow = FancyArrowPatch((CARD, 0.86), (NODE, 0.86), arrowstyle="<->",
                            mutation_scale=18, color=red, lw=2.2)
    ax.add_patch(arrow)
    ax.text((CARD + NODE) / 2, 0.96, "12.78 points", ha="center", fontsize=16,
            color=red, fontweight="bold")

    fig.text(0.5, 0.145, "Same method, same base model, same hidden set, same twelve hours. "
                         "The shaded band is the\nmeasured spread of three runs of the unchanged "
                         "configuration, 30.56 to 30.97.",
             fontsize=11, color="#5a5a5a", ha="center", linespacing=1.6)
    fig.text(0.5, 0.045, "ARC Prize 2026 Paper Track   ·   submission 54974874",
             fontsize=9, color="#9a9a95", ha="center")
    save(fig, "opt_b_dumbbell.png")


# ----------------------------------------------------------------------------- C: dark

def option_c():
    surf, ink, blue, red, mute = "#1a1a19", "#f2f2ef", "#4C93E8", "#E8616F", "#8d8d88"
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=surf)
    fig.text(0.06, 0.845, "The 16GB Ceiling", fontsize=40, color=ink, fontweight="bold")
    fig.text(0.06, 0.765, "One consumer GPU costs 12.78 points on ARC-AGI-2",
             fontsize=15, color=mute)

    ax = fig.add_axes([0.06, 0.17, 0.55, 0.50], facecolor=surf)
    for i, (v, c, lab) in enumerate(((CARD, mute, "16 GB"), (NODE, blue, "96 GB"))):
        ax.bar(i, v, 0.46, color=c, zorder=2)
        ax.text(i, v + 1.1, "%.2f" % v, ha="center", fontsize=28, color=c, fontweight="bold")
        ax.text(i, -2.6, lab, ha="center", fontsize=13, color=ink)
    ax.axhspan(LO, HI, color=blue, alpha=0.22, zorder=1)
    ax.annotate("", xy=(0.5, NODE - 0.4), xytext=(0.5, CARD + 0.4),
                arrowprops=dict(arrowstyle="<->", color=red, lw=2.2))
    ax.text(0.5, (CARD + NODE) / 2, "  12.78", ha="left", va="center", fontsize=17,
            color=red, fontweight="bold")
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylim(0, 37)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.665, 0.575, "Same method.\nSame base model.\nSame hidden set.\nSame twelve hours.",
             fontsize=14.5, color=ink, linespacing=1.55)
    fig.text(0.665, 0.275, "Three runs of the unchanged\nconfiguration span 30.56 to 30.97.\n"
                           "The gap is thirty times that noise.",
             fontsize=10.5, color=mute, linespacing=1.5)
    fig.text(0.06, 0.055, "ARC Prize 2026 Paper Track   ·   submission 54974874",
             fontsize=9, color="#5f5f5b")
    save(fig, "opt_c_dark.png")


if __name__ == "__main__":
    option_a()
    option_b()
    option_c()
