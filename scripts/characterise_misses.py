"""Ask whether the tasks the system misses share structure we could generate synthetic data against.

The base model has already consumed the entire published synthetic corpus, so the only base-side lever
left is data that corpus does not contain. Targeted generation is only well-posed if the misses cluster:
if they share measurable structure, a few thousand puzzles aimed at that structure is a real experiment,
and if they look like the solved tasks on every axis we can measure then the difference is semantic and
no achievable volume of structurally-targeted data will cover them.

This is deliberately a cheap kill-test. It runs offline in seconds against data we already have, and a
null here saves three months.

The labels come from the deployed measurement run (paper/data/task_timing.csv): solved and in-pool mean
the correct grid entered the candidate pool at least once, miss and empty mean it never did. The
features are structural only, computed from the task definition without a model, which is exactly the
class of property a generator can be told to produce. That is also the honest limit of the test:
structural separability is sufficient to justify targeting, not necessary for the tasks to differ.

Usage: python characterise_misses.py
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
REPO = r"D:\ARC_AGI-2,3"


def find(name, *dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise SystemExit("cannot find %s" % name)


def components(g):
    """Count 4-connected non-zero regions, a coarse proxy for object count."""
    g = np.asarray(g)
    seen = np.zeros(g.shape, bool)
    n = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if g[i, j] == 0 or seen[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < g.shape[0] and 0 <= y < g.shape[1] and not seen[x, y] \
                            and g[x, y] == g[a, b]:
                        seen[x, y] = True
                        stack.append((x, y))
    return n


def features(task):
    tr = task["train"]
    ins = [np.array(p["input"]) for p in tr]
    outs = [np.array(p["output"]) for p in tr]
    te = np.array(task["test"][0]["input"])
    f = {
        "n_train_pairs": len(tr),
        "in_area_mean": float(np.mean([a.size for a in ins])),
        "in_area_max": float(max(a.size for a in ins)),
        "out_area_mean": float(np.mean([a.size for a in outs])),
        "test_area": float(te.size),
        "max_dim": float(max(max(a.shape) for a in ins + outs + [te])),
        "size_preserved": float(np.mean([i.shape == o.shape for i, o in zip(ins, outs)])),
        "area_ratio": float(np.mean([o.size / max(i.size, 1) for i, o in zip(ins, outs)])),
        "n_colours_in": float(np.mean([len(set(a.ravel().tolist())) for a in ins])),
        "n_colours_out": float(np.mean([len(set(a.ravel().tolist())) for a in outs])),
        "palette_changes": float(np.mean(
            [set(i.ravel().tolist()) != set(o.ravel().tolist()) for i, o in zip(ins, outs)])),
        "objects_in": float(np.mean([components(a) for a in ins])),
        "objects_out": float(np.mean([components(a) for a in outs])),
        "n_test_outputs": len(task["test"]),
    }
    f["object_ratio"] = f["objects_out"] / max(f["objects_in"], 1e-9)
    return f


def auc(pos, neg):
    """Mann-Whitney AUC: probability a random positive exceeds a random negative."""
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / float(len(pos) * len(neg))


def main():
    dd = os.path.join(BASE, "data", "kaggle")
    ch = find("arc-agi_evaluation_challenges.json", dd, os.path.join(BASE, "data"))
    tim = find("task_timing.csv", os.path.join(BASE, "data"),
               os.path.join(REPO, "paper", "data"))
    tasks = json.load(open(ch))
    rows = list(csv.DictReader(open(tim)))

    reached, missed = [], []
    for r in rows:
        t = tasks.get(r["task"])
        if t is None:
            continue
        (reached if r["outcome"] in ("solved", "in-pool") else missed).append((r["task"], features(t)))
    print("labelled tasks: %d reached (truth entered the pool), %d missed"
          % (len(reached), len(missed)))
    print("outcome mix: %s\n" % dict(Counter(r["outcome"] for r in rows)))

    keys = sorted(reached[0][1])
    print("%-18s %10s %10s %7s  %s" % ("feature", "reached", "missed", "AUC", "separation"))
    print("-" * 72)
    scored = []
    for k in keys:
        p = [f[k] for _, f in reached]
        n = [f[k] for _, f in missed]
        a = auc(p, n)
        scored.append((abs(a - 0.5), a, k, float(np.mean(p)), float(np.mean(n))))
    for gap, a, k, mp, mn in sorted(scored, reverse=True):
        bar = "#" * int(round(gap * 40))
        print("%-18s %10.2f %10.2f %7.3f  %s" % (k, mp, mn, a, bar))

    best = sorted(scored, reverse=True)[0]
    print("\nstrongest single structural feature: %s at AUC %.3f" % (best[2], best[1]))
    print("(0.50 is chance. Above about 0.75 on this sample size is a real, targetable difference;")
    print(" 0.50 to 0.65 means structure does not distinguish what we miss.)")

    # Coverage question: even a perfect structural rule is only useful if the missed tasks are
    # CONCENTRATED. Report how much of the miss set the best feature's tail actually contains.
    gap, a, k, _, _ = best
    allv = [(f[k], lab) for lab, grp in (("R", reached), ("M", missed)) for _, f in grp]
    allv.sort(reverse=(a < 0.5))
    top = allv[:len(missed)]
    hit = sum(1 for _, lab in top if lab == "M")
    print("\nif we targeted the %d tasks most extreme in %s, %d of them would be true misses (%.0f%%),"
          % (len(missed), k, hit, 100.0 * hit / max(len(missed), 1)))
    print("against a base rate of %.0f%% if we picked at random."
          % (100.0 * len(missed) / (len(missed) + len(reached))))

    # The verdict must use the criterion stated above (0.75) and must respect coverage. An AUC
    # can look non-chance while the corresponding selection is worse than picking at random,
    # because these features are heavily tied, and it is coverage that decides whether targeting
    # is worth doing.
    base_rate = 100.0 * len(missed) / (len(missed) + len(reached))
    cover = 100.0 * hit / max(len(missed), 1)
    separable = abs(best[1] - 0.5) >= 0.25
    enriches = cover > base_rate
    print("\nVERDICT  (separable=%s at AUC %.3f, enriches=%s at %.0f%% vs %.0f%% base)"
          % (separable, best[1], enriches, cover, base_rate))
    if not (separable and enriches):
        print("  No structural axis separates reached from missed. The tasks we fail are not")
        print("  distinguishable by grid size, colour count, object count, size preservation or")
        print("  palette change, which are precisely the properties a synthetic generator can be")
        print("  instructed to produce. Targeted structural generation is NOT well posed here, and")
        print("  the difference is in the rule rather than in the grids.")
    else:
        print("  A structural axis separates the two groups. Targeted generation is well posed and")
        print("  the feature above is what to aim at. Check the coverage line before committing:")
        print("  a high AUC on a thinly populated tail still buys little.")


if __name__ == "__main__":
    main()
