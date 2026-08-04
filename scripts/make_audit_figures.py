"""Figures for the two August results: the contamination audit and the phantom-score pattern.

  paper/fig_contamination.{png,svg,pdf}  the 2x2 that convicts by ordering inversion, plus the
                                         truth-likelihood distributions that motivated it
  paper/fig_phantom.{png,svg,pdf}        scores displayed beside public notebooks against what the
                                         same notebooks score when forked unmodified

Data comes from paper/data/contamination_g{2,3,5}.json, written by scripts/rig_contamination_audit.py
on 2026-08-01. The phantom numbers are submission-history facts (submission ids 55053664, 55160403)
and the same-config band is the three-run replicate measurement of 2026-08-03.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"D:\ARC_AGI-2,3\paper"
DATA = os.path.join(OUT, "data")

INK, BLUE, GREY, RAIL, FAIL = "#1b1b1b", "#1E5FA8", "#6b6b6b", "#ededed", "#B0473A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

M4, M2 = "4B sft139", "2B sft141"
L4, L2 = "4B sft139 (our base)", "2B sft141"


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
    fig.savefig(base + ".png", dpi=dpi, **common)
    fig.savefig(base + ".svg", **common)
    fig.savefig(base + ".pdf", **common)
    plt.close(fig)
    print("saved", name, "(png/svg/pdf)")


def fig_contamination():
    g2 = json.load(open(os.path.join(DATA, "contamination_g2.json")))
    g3 = json.load(open(os.path.join(DATA, "contamination_g3.json")))
    g5 = json.load(open(os.path.join(DATA, "contamination_g5.json")))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Left: ECDF of per-output truth NLL on the public evaluation answers. The whole audit
    # started from the gap between these two curves.
    for key, col, lab in ((M4, BLUE, L4), (M2, FAIL, L2)):
        v = np.sort(np.maximum(np.array(g2[key], dtype=float), 1e-6))
        ax1.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=2.0, label=lab)
    ax1.axvline(0.01, color=GREY, ls=":", lw=1.2)
    ax1.text(0.0115, 0.60, "0.01 nats", fontsize=8.5, color=GREY)
    f2 = np.mean(np.array(g2[M2]) < 0.01)
    f4 = np.mean(np.array(g2[M4]) < 0.01)
    ax1.text(2.2e-4, 0.965, "%d%% of answers below\n0.01 nats per token" % round(100 * f2),
             fontsize=9, color=FAIL, va="top")
    ax1.text(0.0135, 0.155, "%d%%" % round(100 * f4), fontsize=9, color=BLUE)
    ax1.set_xscale("log")
    ax1.set_xlabel("per-token NLL of the true answer (nats, log scale)", fontsize=10)
    ax1.set_ylabel("fraction of the 167 evaluation outputs", fontsize=10)
    ax1.set_title("Truth-likelihood on public evaluation answers", fontsize=11.5,
                  fontweight="bold", color=INK, pad=8)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    despine(ax1)
    ax1.grid(axis="both", color=RAIL)

    # Right: the 2x2 as a slopegraph. Capability orders the models on data minted after both
    # were published; the suspect set inverts it.
    cols = ["public eval", "fresh RE-ARC"]
    for key, col, lab in ((M4, BLUE, L4), (M2, FAIL, L2)):
        e = 100.0 * g3[key]["hit"] / g3[key]["n"]
        f = 100.0 * g5[key]["hit"] / g5[key]["n"]
        ax2.plot([0, 1], [e, f], "-o", color=col, lw=2.2, ms=8)
        ax2.annotate("%.0f%%" % e, (0, e), textcoords="offset points", xytext=(-30, -3),
                     fontsize=10.5, color=col, fontweight="bold")
        ax2.annotate("%.0f%%" % f, (1, f), textcoords="offset points", xytext=(14, -3),
                     fontsize=10.5, color=col, fontweight="bold")
        ax2.annotate(lab, (1, f), textcoords="offset points", xytext=(14, 11),
                     fontsize=8.6, color=col)
    ax2.text(0.5, 88, "interaction +83.3 points\nz = 8.0", ha="center", fontsize=9.5,
             color=INK, style="italic")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["ARC-AGI-2 public eval\n(published 2025)",
                         "fresh RE-ARC\n(minted 2026-08-01, verified)"], fontsize=9.5)
    ax2.set_xlim(-0.35, 1.55)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("greedy exact-match, no TTT (60 outputs per cell)", fontsize=10)
    ax2.set_title("The ordering inversion", fontsize=11.5, fontweight="bold", color=INK, pad=8)
    despine(ax2)
    ax2.grid(axis="y", color=RAIL)

    fig.suptitle("A 2B beats a 4B from the same author only where the answers were published",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    footer(fig)
    save(fig, "fig_contamination")


def fig_phantom():
    # (label, displayed beside the notebook, scored when forked unmodified)
    rows = [("koushikrudra/failed-in-aimo\nforked unmodified, sub 55053664", 33.89, 28.06),
            ("mikelou1/arc-agi2-lb33-89-minimal-perfpatch\nforked unmodified, sub 55160403",
             33.89, 28.47)]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    w = 0.34
    xs = np.arange(len(rows))

    # our unchanged pipeline, three scored runs, drawn as the reference band
    ax.axhspan(30.56, 30.97, color=BLUE, alpha=0.12, zorder=1)
    ax.text(2.02, 30.76, "our unchanged pipeline,\nthree runs: 30.56 to 30.97",
            fontsize=8.8, color=BLUE, va="center", ha="right")

    for i, (lab, shown, scored) in enumerate(rows):
        ax.bar(i - w / 2, shown, w, color=RAIL, edgecolor=GREY, lw=1.0, zorder=2)
        ax.bar(i + w / 2, scored, w, color=FAIL, zorder=2)
        ax.annotate("displayed\n%.2f" % shown, (i - w / 2, shown), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9.5, color=GREY)
        ax.annotate("scored\n%.2f" % scored, (i + w / 2, scored), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9.5, color=FAIL,
                    fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("ARC-AGI-2 hidden score (%)", fontsize=10.5)
    ax.set_ylim(0, 38.5)
    ax.set_xlim(-0.6, 2.05)
    ax.set_title("A score displayed beside a public notebook is not a measurement of that notebook",
                 fontsize=12, fontweight="bold", color=INK, pad=10)
    despine(ax)
    ax.grid(axis="y", color=RAIL)
    footer(fig)
    save(fig, "fig_phantom")


if __name__ == "__main__":
    fig_contamination()
    fig_phantom()
