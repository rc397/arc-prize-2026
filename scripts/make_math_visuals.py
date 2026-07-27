"""Figures for Appendix F, the mathematics of the measurement.

Every panel is drawn from data committed in the repo, nothing is illustrative. The dominance panels and
the margin histogram are measurements. The resolution curve is exact arithmetic. The cap curve is a
bootstrap projection from measured task timings and says so on its face.

-> paper/fig_dominance.{png,svg,pdf}   truth Pareto-dominated inside its own pool, four missed outputs
   paper/fig_resolution.{png,svg,pdf}  the McNemar bound with the project's episodes placed on it
   paper/fig_delta_margins.{png,svg,pdf}  round-trip gate margins, winning and losing overlap
   paper/fig_cap.{png,svg,pdf}         bootstrap expected score against the per-task cap
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

INK, BLUE, GREY, GREYL, FAIL = "#1b1b1b", "#1E5FA8", "#6b6b6b", "#C2C7CC", "#B0473A"
GREEN = "#2E8B57"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})


def save(fig, name):
    base = os.path.join(OUT, name)
    common = dict(bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".png", dpi=180, **common)
    fig.savefig(base + ".svg", **common)
    fig.savefig(base + ".pdf", **common)
    plt.close(fig)
    print("wrote", name)


def mean_aug(c):
    return float(np.mean([np.mean(a) for a in c["aug"]]))


def kgmon(c):
    return c["count"] - mean_aug(c)


def fig_dominance():
    pool = json.load(open(os.path.join(DATA, "pool_dump.json")))
    leaked = ["142ca369_0", "3a25b0d8_0", "3dc255db_0", "4c7dc4dd_1"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for ax, bk in zip(axes.flat, leaked):
        cs = pool[bk]
        truth = next(c for c in cs if c["correct"])
        winner = max((c for c in cs), key=kgmon)
        tx, ty = mean_aug(truth), truth["count"]
        # the region that dominates the truth on these two axes, better likelihood and more votes
        ax.fill_between([0, tx], ty, 20, color="#F3E8E6", zorder=0)
        for c in cs:
            if c is truth or c is winner:
                continue
            ax.scatter(mean_aug(c), c["count"], s=26, color=GREYL, edgecolor=GREY,
                       linewidth=0.5, zorder=2)
        ax.scatter(mean_aug(winner), winner["count"], s=64, marker="D", color=BLUE,
                   zorder=3, label="selected")
        ax.scatter(tx, ty, s=140, marker="*", color=FAIL, zorder=4, label="truth")
        rank = sorted(range(len(cs)), key=lambda i: kgmon(cs[i]), reverse=True)
        trank = next(r for r, i in enumerate(rank) if cs[i]["correct"]) + 1
        xs = [mean_aug(c) for c in cs]
        ax.set_xlim(0, max(xs) * 1.18)
        ax.set_ylim(0, max(c["count"] for c in cs) + 1.2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title("%s,  truth ranked %d of %d" % (bk, trank, len(cs)), fontsize=10.5)
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.supxlabel("mean augmentation NLL (lower is better)", fontsize=11)
    fig.supylabel("multiplicity, decodes producing the grid", fontsize=11)
    fig.suptitle("The truth is Pareto-dominated inside its own candidate pool", fontsize=12.5, y=0.99)
    fig.text(0.5, 0.935, "shaded region dominates the truth on both axes, every point in it is a wrong grid",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout(rect=(0.01, 0.0, 1, 0.96))
    save(fig, "fig_dominance")


def fig_resolution():
    n = np.arange(1, 11)
    p = 2.0 * 0.5 ** n
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.axhspan(1e-3, 0.05, color="#E9F1E9", zorder=0)
    ax.axhline(0.05, color=GREEN, lw=1.2, ls="--")
    ax.text(9.9, 0.055, "p = 0.05", ha="right", fontsize=9.5, color=GREEN)
    ax.plot(n, p, "-o", color=BLUE, ms=5, lw=1.6, zorder=3,
            label=r"unanimous flips, $p = 2^{\,1-n}$")
    ax.axvline(6, color=GREY, lw=0.9, ls=":")
    ax.text(6.08, 0.6, "minimum resolvable effect", rotation=90, fontsize=9, color=GREY, va="top")
    # the project's episodes, exact p-values
    ax.scatter([3], [0.25], s=70, marker="s", color=FAIL, zorder=4)
    ax.annotate("selector switch, 3 flips\nshipped, reversed on the full sample",
                (3, 0.25), (3.35, 0.42), fontsize=9, color=FAIL,
                arrowprops=dict(arrowstyle="-", color=FAIL, lw=0.8))
    ax.scatter([7], [0.125], s=70, marker="^", color=INK, zorder=4)
    ax.annotate("regularized TTT, 6 flips to 1\nexact p = 0.125, held on submission",
                (7, 0.125), (7.2, 0.24), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate("repeat-run noise floor, one credit", (1, 1.0), (1.5, 1.25), fontsize=9,
                color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.scatter([1], [1.0], s=55, marker="o", facecolor="white", edgecolor=GREY, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1.2e-3, 2.6)
    ax.set_xlim(0.6, 10.4)
    ax.set_xticks(n)
    ax.set_xlabel("discordant tasks between two variants", fontsize=11)
    ax.set_ylabel("exact two-sided p-value", fontsize=11)
    ax.legend(loc="lower left", fontsize=9.5, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_resolution")


def fig_delta_margins():
    gate = json.load(open(os.path.join(DATA, "roundtrip_gate.json")))
    margins = []
    for recs in gate.values():
        d = {r["kind"]: r["delta"] for r in recs}
        if "true" not in d or len(d) < 2:
            continue
        others = [v for k, v in d.items() if k != "true"]
        margins.append(min(others) - d["true"])   # positive means the truth won
    margins = np.array(margins)
    win_med = float(np.median(margins[margins > 0]))
    loss_med = float(np.median(-margins[margins <= 0]))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    bins = np.linspace(-0.06, 0.06, 49)
    ax.hist(np.clip(margins, -0.06, 0.06), bins=bins, color="#A9C2DD", edgecolor=BLUE, lw=0.5)
    ax.axvline(0, color=INK, lw=1.1)
    ax.axvline(win_med, color=GREEN, lw=1.2, ls="--")
    ax.axvline(-loss_med, color=FAIL, lw=1.2, ls="--")
    ax.text(win_med + 0.001, ax.get_ylim()[1] * 0.92, "median win\n%.4f nats" % win_med,
            fontsize=9, color=GREEN)
    ax.text(-loss_med - 0.001, ax.get_ylim()[1] * 0.92, "median loss\n%.4f" % loss_med,
            fontsize=9, color=FAIL, ha="right")
    ax.set_xlabel("round-trip margin of the true grid over its best rival (nats, clipped)", fontsize=11)
    ax.set_ylabel("test outputs", fontsize=11)
    ax.set_title("A signal indistinguishable from its own noise, %d of %d wins but the margins overlap zero"
                 % (int((margins > 0).sum()), len(margins)), fontsize=11.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_delta_margins")


def fig_cap():
    rows = list(csv.DictReader(open(os.path.join(DATA, "task_timing.csv"))))
    # per-task harvested credits in the measurement run, from the scored submission
    credit = {"135a2760": 1.0, "1818057f": 1.0, "2ba387bc": 1.0, "36a08778": 1.0, "38007db0": 1.0,
              "45a5af55": 1.0, "4c3d4a41": 1.0, "53fb4810": 1.0, "2d0172a1": 1.0,
              "1ae2feb7": 1 / 3, "3a25b0d8": 0.5}
    tau = np.array([float(r["wall_seconds"]) for r in rows])
    val = np.array([credit.get(r["task"], 0.0) for r in rows])
    T = 12 * 3600 - 600 - 200          # wall minus safety margin minus startup, per GPU
    rng = np.random.default_rng(0)
    caps = np.arange(600, 3601, 100)
    reps = 2000
    mean_pts, lo, hi, reach = [], [], [], []
    for C in caps:
        pts = np.empty(reps)
        kk = np.empty(reps)
        for r in range(reps):
            total, reached = 0.0, 0
            for _ in range(4):                     # four workers, sixty tasks each
                idx = rng.integers(0, len(tau), 60)
                t = 0.0
                for i in idx:
                    step = min(tau[i], C)
                    if t + step > T:
                        break
                    t += step
                    reached += 1
                    if tau[i] <= C:
                        total += val[i]
            pts[r] = total * 100.0 / 240.0
            kk[r] = reached / 240.0
        mean_pts.append(pts.mean())
        lo.append(np.percentile(pts, 25))
        hi.append(np.percentile(pts, 75))
        reach.append(np.mean(kk))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.fill_between(caps, lo, hi, color="#A9C2DD", alpha=0.55, label="interquartile band")
    ax.plot(caps, mean_pts, color=BLUE, lw=1.8, label="expected score, bootstrap mean")
    for C, lab, yoff in ((1200, "C = 1200, public 33.89 build\nand the v7.4 control", 0.15),
                         (1800, "C = 1800, every submission\nof ours through the fault", 1.35)):
        ax.axvline(C, color=GREY, lw=0.9, ls=":")
        ax.text(C + 25, min(lo) + yoff, lab, fontsize=8.8, color=GREY, va="bottom")
    ax2 = ax.twinx()
    ax2.plot(caps, reach, color=GREEN, lw=1.2, ls="--", label="fraction of queue reached")
    ax2.set_ylabel("fraction of 240 tasks reached", fontsize=10.5, color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN, labelsize=9)
    ax2.set_ylim(0.5, 1.02)
    ax.set_xlabel("per-task cap C (seconds)", fontsize=11)
    ax.set_ylabel("projected score (points on 240 tasks)", fontsize=11)
    ax.tick_params(labelsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9, frameon=False)
    ax.set_title("The sequential-deadline model of the cap, a projection the control submission tests",
                 fontsize=11.5)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_cap")


if __name__ == "__main__":
    fig_dominance()
    fig_resolution()
    fig_delta_margins()
    fig_cap()
