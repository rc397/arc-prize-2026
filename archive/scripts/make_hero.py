"""The cover figure: the paper's central claim as one image.

  paper/fig_ceiling.{png,svg,pdf}      the 16GB-versus-96GB pair with the measured noise band
  paper/cover_ceiling_1120x560.png     the same at Kaggle cover-image geometry

Same author, same base model, same hidden set, same twelve hours. v6.1 is the best configuration
that fits one 16GB card, v7.2 the same lineage on the contest's four L4s, and the band is the
three-run same-configuration spread measured by the replicate.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"D:\ARC_AGI-2,3\paper"
INK, BLUE, GREY, RAIL, FAIL = "#1b1b1b", "#1E5FA8", "#6b6b6b", "#ededed", "#B0473A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


def draw(ax, big):
    f = 1.35 if big else 1.0
    bars = [("one consumer card\nRTX 5070 Ti, 16 GB", 18.19, GREY),
            ("the contest node\nfour L4s, 96 GB", 30.97, BLUE)]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, 0.52, color=c, zorder=2)
        ax.annotate("%.2f" % v, (i, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=21 * f, color=c, fontweight="bold")
    ax.axhspan(30.56, 30.97, color=BLUE, alpha=0.14, zorder=1)
    ax.text(1.30, 30.05, "three runs of the same\nconfiguration span\n30.56 to 30.97",
            fontsize=8.6 * f, color=BLUE, va="top")

    ax.annotate("", xy=(0.5, 30.4), xytext=(0.5, 18.7),
                arrowprops=dict(arrowstyle="<->", color=FAIL, lw=1.8))
    ax.text(0.5, 34.6, "12.78 points", ha="center", fontsize=13 * f, color=FAIL,
            fontweight="bold")
    ax.text(0.5, 32.9, "the price of the card", ha="center", fontsize=9.5 * f, color=FAIL)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([b[0] for b in bars], fontsize=10.5 * f)
    ax.set_ylabel("ARC-AGI-2 hidden score (%)", fontsize=10.5 * f)
    ax.set_ylim(0, 36)
    ax.set_xlim(-0.55, 1.72)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=9 * f, colors=INK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=RAIL)


def main():
    # paper form
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    draw(ax, big=False)
    ax.set_title("The same method, the same model, the same twelve hours",
                 fontsize=12.5, fontweight="bold", color=INK, pad=10)
    fig.text(0.992, 0.012, "ARC Prize 2026  ·  Paper Track", ha="right", va="bottom",
             fontsize=8, color="#9a9a9a")
    common = dict(facecolor="white", bbox_inches="tight", pad_inches=0.18)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUT, "fig_ceiling." + ext),
                    **(dict(common, dpi=200) if ext == "png" else common))
    plt.close(fig)
    print("saved fig_ceiling (png/svg/pdf)")

    # cover form, 1120x560 exactly
    fig = plt.figure(figsize=(11.2, 5.6), dpi=100)
    ax = fig.add_axes([0.42, 0.13, 0.55, 0.72])
    draw(ax, big=False)
    fig.text(0.045, 0.78, "The 16GB Ceiling", fontsize=34, color=INK, fontweight="bold")
    fig.text(0.045, 0.60, "Test-time training for ARC-AGI-2\non one consumer GPU",
             fontsize=16, color=INK)
    fig.text(0.045, 0.40, "and what it takes to trust a number\nin this competition",
             fontsize=12.5, color=GREY, style="italic")
    fig.text(0.045, 0.16, "measured, replicated, and audited\nsubmission 54974874, hidden score 30.97",
             fontsize=10, color=GREY)
    fig.savefig(os.path.join(OUT, "cover_ceiling_1120x560.png"), facecolor="white", dpi=100)
    plt.close(fig)
    print("saved cover_ceiling_1120x560.png")


if __name__ == "__main__":
    main()
