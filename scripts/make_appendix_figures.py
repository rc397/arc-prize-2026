"""Appendix figures for three measurements that had numbers in prose but no picture.

  paper/fig_budget.{png,svg,pdf}       where the 40-task measurement run's seconds actually went
  paper/fig_restart.{png,svg,pdf}      the seed-decorrelation probe and its negative control
  paper/fig_pool_census.{png,svg,pdf}  the selection funnel and the near-miss structure of the pool

fig_budget reads paper/data/task_timing.csv and fig_restart reads
paper/data/restart_probe_summary.json, both written by the rig runs they depict. fig_pool_census
carries its numbers inline with provenance comments, matching how make_version_progress.py carries
scores: they were measured on 2026-08-01 by scripts/pool_census.py and scripts/pool_selector_fit.py
against paper/data/pool_dump.json, and the solutions file those scripts need is competition data we
do not redistribute in this repository.
"""
import csv
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


def fig_budget():
    rows = list(csv.DictReader(open(os.path.join(DATA, "task_timing.csv"))))
    rows.sort(key=lambda r: -float(r["wall_seconds"]))
    secs = np.array([float(r["wall_seconds"]) for r in rows])
    yields_ = np.array([r["outcome"] in ("solved", "in-pool") for r in rows])
    tot = secs.sum()

    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    xs = np.arange(len(rows))
    for cond, col in ((yields_, BLUE), (~yields_, FAIL)):
        ax.bar(xs[cond], secs[cond], 0.8, color=col, zorder=2)
    ax.axhline(1200, ls="--", color=GREY, lw=1.4)
    ax.text(len(rows) - 0.6, 1245, "1200 s cap candidate", ha="right", fontsize=8.8, color=GREY)

    zy = secs[~yields_].sum() / tot
    over = secs[secs > 1200].sum() / tot
    ax.text(10.5, 1660,
            "tasks yielding nothing (no solve, no correct grid in pool):\n"
            "%d of 40 tasks, %.0f%% of all task-seconds" % ((~yields_).sum(), 100 * zy),
            fontsize=9.3, color=FAIL)
    ax.text(10.5, 1420, "the four tasks past 1200 s: %.1f%% of the budget, all misses" % (100 * over),
            fontsize=9.3, color=INK)

    ax.set_xticks([])
    ax.set_xlabel("the 40 measured tasks, longest first "
                  "(blue: truth entered the pool, red: it never did)", fontsize=10)
    ax.set_ylabel("wall seconds per task", fontsize=10.5)
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_title("Where the measurement run's 24822 task-seconds went",
                 fontsize=12.5, fontweight="bold", color=INK, pad=10)
    despine(ax)
    ax.grid(axis="y", color=RAIL)
    footer(fig)
    save(fig, "fig_budget")


def fig_restart():
    S = json.load(open(os.path.join(DATA, "restart_probe_summary.json")))
    seeds = [S["single_weighted"][k] for k in ("1", "2", "3")]
    names = ["seed 1", "seed 2", "seed 3", "union of\nall three", "intersection"]
    vals = seeds + [S["union_weighted"], S["intersection_weighted"]]
    counts = ["11", "17", "15", "17", "11"]     # outputs of 30, from the probe log

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    xs = np.arange(len(vals))
    cols = [BLUE, BLUE, BLUE, "white", GREY]
    for i, (v, c) in enumerate(zip(vals, cols)):
        if i == 3:
            ax.bar(i, 100 * v, 0.62, color="white", edgecolor=BLUE, lw=2.0, hatch="//", zorder=2)
        else:
            ax.bar(i, 100 * v, 0.62, color=c, zorder=2)
        ax.annotate("%.1f%%\n(%s/30)" % (100 * v, counts[i]), (i, 100 * v),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9.5, color=INK)
    ax.axhline(100 * max(seeds), ls=":", color=GREY, lw=1.2)
    ax.text(4.35, 100 * max(seeds) + 0.7, "best single seed", ha="right", fontsize=8.8, color=GREY)

    ax.text(0.02, 0.97,
            "union equals the best single seed exactly\n"
            "same-seed replicate disagreement 12.5%, cross-seed 13.3%, net +0.008\n"
            "48 of 65 decodes budget-truncated, so the design favours finding an effect",
            transform=ax.transAxes, fontsize=9, color=INK, va="top")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("weighted truth-in-pool rate (%)", fontsize=10.5)
    ax.set_ylim(0, 40)
    ax.set_title("Colour-permutation restarts do not decorrelate the misses",
                 fontsize=12.5, fontweight="bold", color=INK, pad=10)
    despine(ax)
    ax.grid(axis="y", color=RAIL)
    footer(fig)
    save(fig, "fig_restart")


def fig_pool_census():
    # Measured 2026-08-01 on paper/data/pool_dump.json (60 deployed-condition outputs) by
    # scripts/pool_census.py and scripts/pool_selector_fit.py. The funnel: truth entered the pool
    # on 19 outputs; at most one candidate Pareto-dominates the truth on 18 of them, so 18 is the
    # exact attainable ceiling for ANY monotone selector over the recorded statistics; the deployed
    # selector attains 15; refitting the weights under leave-one-task-out attains 14, one WORSE,
    # because with only four leaked outputs the weights memorise them.
    funnel = [("outputs scored", 60, GREY),
              ("pool contains the truth", 19, BLUE),
              ("monotone bound over summary statistics", 18, BLUE),
              ("attained by the deployed selector", 15, BLUE),
              ("attained by LOO-refit weights", 14, FAIL)]
    # Near-miss census over the 41 truth-absent outputs: cell distance from the nearest same-shape
    # candidate to the truth. Cell-wise voting across candidates reconstructs 0 of the 41 under
    # uniform, multiplicity and exp(-augNLL) weightings (scripts/pool_vote.py).
    near = [("1 cell", 1), ("2 to 3", 6), ("4 to 10", 7), ("over 10", 21), ("no same-shape\ncandidate", 6)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    ys = np.arange(len(funnel))[::-1]
    for y, (lab, v, c) in zip(ys, funnel):
        ax1.barh(y, v, 0.62, color=c, zorder=2)
        ax1.annotate(str(v), (v, y), textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax1.text(-1.2, y, lab, ha="right", va="center", fontsize=9.5, color=INK)
    ax1.set_yticks([])
    ax1.set_xlim(0, 66)
    ax1.set_xlabel("test outputs", fontsize=10)
    ax1.set_title("The selection funnel: two to three outputs of headroom,\nnone of it learnable from 60 outputs",
                  fontsize=11, fontweight="bold", color=INK, pad=8)
    despine(ax1)
    ax1.spines["left"].set_visible(False)
    ax1.grid(axis="x", color=RAIL)

    xs = np.arange(len(near))
    ax2.bar(xs, [v for _, v in near], 0.62, color=BLUE, zorder=2)
    for i, (_, v) in enumerate(near):
        ax2.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=10, color=INK, fontweight="bold")
    ax2.text(0.03, 0.95, "cell-wise voting across candidates\nreconstructs 0 of the 41:\nerrors are correlated through\nthe model's beliefs",
             transform=ax2.transAxes, fontsize=9, color=FAIL, va="top")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([n for n, _ in near], fontsize=9)
    ax2.set_ylabel("truth-absent outputs", fontsize=10)
    ax2.set_ylim(0, 25)
    ax2.set_title("How far the pool lands from the 41 truths it misses\n(cells changed, nearest same-shape candidate)",
                  fontsize=11, fontweight="bold", color=INK, pad=8)
    despine(ax2)
    ax2.grid(axis="y", color=RAIL)

    footer(fig)
    save(fig, "fig_pool_census")


if __name__ == "__main__":
    fig_budget()
    fig_restart()
    fig_pool_census()
