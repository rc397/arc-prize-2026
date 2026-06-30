"""Version-progression figures for the ARC Prize 2026 Paper Track.

Two line graphs across the leaderboard submission versions (v1, v2, v2.1, ...):
  paper/fig_version_score.{png,svg,pdf}  hidden 240-task main-track score by version
  paper/fig_version_time.{png,svg,pdf}   leaderboard run wall-clock by version, vs the 12 h budget

The VERSIONS table below is the single source of truth; append one row per new submission.
Values are measured leaderboard runs except where marked est (v1's hidden run was not instrumented,
so its time is the public-eval decode rate scaled to 240 tasks and P100 bandwidth).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"D:\ARC_AGI-2,3\paper"

INK, BLUE, GREY, RAIL, FAIL, GREEN = (
    "#1b1b1b", "#1E5FA8", "#6b6b6b", "#ededed", "#B0473A", "#2E8B57")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# One row per leaderboard submission. Append as the entry climbs (v3, v4, ...).
#   score    : hidden 240-task main-track score (%), or None if not yet scored
#   hours    : leaderboard-run wall-clock (h)
#   time_est : True if hours is an estimate (the run was not instrumented)
#   accel    : the scored-run accelerator
#   ttt      : number of tasks given per-task test-time training
VERSIONS = [
    dict(tag="v1",   score=6.53,  hours=3.2,  time_est=True,  accel="P100 x1",
         ttt=0,   note="no-TTT AIRV"),
    dict(tag="v2",   score=11.94, hours=2.45, time_est=False, accel="L4 x4",
         ttt=160, note="TTT + colour-perm"),
    dict(tag="v2.1", score=13.61, hours=3.90, time_est=False, accel="L4 x4",
         ttt=232, note="TTT, maxpair 1500"),
    dict(tag="v3.1", score=12.78, hours=5.50, time_est=False, accel="L4 x4",
         ttt=232, note="+ multi-scale,\ncolour-perm"),
    dict(tag="v4",   score=None,  hours=5.50, time_est=True,  accel="L4 x4",
         ttt=232, note="more-candidates\nsampling"),
]

BUDGET_H = 12.0
SOTA_2025 = 24.03  # top 2025 Kaggle ARC-AGI-2 score (NVARC), the climb target


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


def fig_version_score():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    xs = list(range(len(VERSIONS)))

    # 2025 Kaggle SOTA reference (the climb target)
    ax.axhline(SOTA_2025, ls="--", color=GREY, lw=1.3)
    ax.text(len(VERSIONS) - 1, SOTA_2025 + 0.5, "2025 Kaggle SOTA 24.03%",
            ha="right", fontsize=8.8, color=GREY)

    # solid spine through scored points only
    sx = [i for i, v in enumerate(VERSIONS) if v["score"] is not None]
    sy = [VERSIONS[i]["score"] for i in sx]
    ax.plot(sx, sy, "-", color=BLUE, lw=2.2, zorder=2)
    ax.plot(sx, sy, "o", color=BLUE, ms=9, zorder=3)

    for i, v in enumerate(VERSIONS):
        if v["score"] is None:
            ax.text(i, 2.4, "score\npending", ha="center", va="center",
                    fontsize=9.5, color=GREY, style="italic")
        else:
            ax.annotate(f"{v['score']:.2f}", (i, v["score"]),
                        textcoords="offset points", xytext=(0, 11), ha="center",
                        fontsize=11.5, color=BLUE, fontweight="bold")

    # relative lifts between successive versions
    ax.text(0.5, 10.6, "+83%", ha="center", fontsize=9.5, color=GREEN, style="italic")
    ax.text(1.5, 14.4, "+14%", ha="center", fontsize=9.5, color=GREEN, style="italic")
    ax.text(2.5, 14.3, "-6%", ha="center", fontsize=9.5, color=FAIL, style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v['tag']}\n{v['note']}" for v in VERSIONS], fontsize=9.5)
    ax.set_ylabel("hidden 240-task main-track score (%)", fontsize=10.5)
    ax.set_ylim(0, 26)
    ax.set_xlim(-0.45, len(VERSIONS) - 0.55)
    ax.set_title("ARC-AGI-2 hidden score by submission version",
                 fontsize=13, fontweight="bold", color=INK, pad=10)
    despine(ax)
    ax.grid(axis="y", color=RAIL)
    footer(fig)
    save(fig, "fig_version_score")


def fig_version_time():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    xs = list(range(len(VERSIONS)))

    # the 12 h offline budget (the headroom story)
    ax.axhline(BUDGET_H, ls="--", color=FAIL, lw=1.6)
    ax.text(len(VERSIONS) - 1, BUDGET_H - 0.6, "12 h offline budget",
            ha="right", fontsize=9.2, color=FAIL, fontweight="bold")

    # connect successive versions; dash any segment touching an estimated time
    for i in range(len(VERSIONS) - 1):
        a, b = VERSIONS[i], VERSIONS[i + 1]
        ls = "--" if (a["time_est"] or b["time_est"]) else "-"
        ax.plot([i, i + 1], [a["hours"], b["hours"]], ls, color=BLUE, lw=2.0, zorder=2)

    for i, v in enumerate(VERSIONS):
        if v["time_est"]:
            ax.plot([i], [v["hours"]], "o", ms=10, mfc="white", mec=BLUE, mew=2.0, zorder=3)
            tag = f"~{v['hours']:.1f} h est.\n{v['accel']}"
        else:
            ax.plot([i], [v["hours"]], "o", color=BLUE, ms=9, zorder=3)
            tag = f"{v['hours']:.2f} h\n{v['accel']}"
        ax.annotate(tag, (i, v["hours"]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9.4, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v['tag']}\n{v['ttt']} TTT tasks" for v in VERSIONS], fontsize=9.5)
    ax.set_ylabel("leaderboard run wall-clock (hours)", fontsize=10.5)
    ax.set_ylim(0, 13)
    ax.set_xlim(-0.45, len(VERSIONS) - 0.55)
    ax.set_title("Run time by submission version, against the 12 h budget",
                 fontsize=13, fontweight="bold", color=INK, pad=10)
    despine(ax)
    ax.grid(axis="y", color=RAIL)
    footer(fig)
    save(fig, "fig_version_time")


if __name__ == "__main__":
    fig_version_score()
    fig_version_time()
