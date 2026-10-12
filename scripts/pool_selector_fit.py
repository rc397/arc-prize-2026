"""A1: can a re-weighted monotone selector close any of the 3-output gap under honest validation?

The census (pool_census.py) showed kgmon attains 15 of 60 outputs while the Appendix-F bound over
the three recorded statistics is 18, so up to 3 outputs are recoverable by SOME monotone rule. With
only 4 leaked outputs the entire risk is overfitting, so the protocol is leave-one-TASK-out: fit the
weights on 29 tasks, score the held-out task, sum. The training-set optimum is also reported to show
the overfit gap rather than hide it.

Features per candidate: multiplicity, mean augmentation NLL, mean beam NLL, and min augmentation
NLL. The bound was computed over the first three; adding a fourth can only loosen it, so a win here
is still inside the theorem, just over a richer summary.

Kill condition from archive/notes/score-levers.txt: LOO gain < 2 outputs kills the lever.

Usage: python pool_selector_fit.py [POOL_JSON]
"""
import itertools
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = json.load(open(sys.argv[1] if len(sys.argv) > 1
                      else os.path.join(REPO, "paper", "data", "pool_dump.json")))

GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def feats(c):
    aug = [float(np.mean(a)) for a in c["aug"]]
    return (float(len(c["beam"])), float(np.mean(aug)), float(np.mean(c["beam"])),
            float(np.min(aug)))


def hits(rows, w):
    wa, wb, wc = w
    n = 0
    for cands in rows:
        scored = sorted(cands, key=lambda c: (
            -(c["_f"][0] - wa * c["_f"][1] - wb * c["_f"][2] - wc * c["_f"][3]),
            -(c["_f"][0] - c["_f"][1])))          # ties break toward kgmon's order
        n += int(any(c["correct"] for c in scored[:2]))
    return n


def main():
    by_task = {}
    for bk, cands in sorted(POOL.items()):
        if not cands:
            continue
        for c in cands:
            c["_f"] = feats(c)
        by_task.setdefault(bk.rsplit("_", 1)[0], []).append(cands)
    tasks = sorted(by_task)
    allrows = [r for t in tasks for r in by_task[t]]

    kg = hits(allrows, (1.0, 0.0, 0.0))
    print("outputs %d over %d tasks; kgmon baseline %d" % (len(allrows), len(tasks), kg))

    combos = list(itertools.product(GRID, GRID, GRID))
    train_best = max(combos, key=lambda w: hits(allrows, w))
    print("training-set optimum %d at weights %s (the OVERFIT number, not the claim)"
          % (hits(allrows, train_best), train_best))

    loo = 0
    picks = []
    for t in tasks:
        rest = [r for u in tasks if u != t for r in by_task[u]]
        best = max(combos, key=lambda w: (hits(rest, w), w == (1.0, 0.0, 0.0)))
        loo += hits(by_task[t], best)
        picks.append(best)
    stab = len(set(picks))
    print("LOO attained %d (kgmon %d), gain %+d; %d distinct weight vectors across folds"
          % (loo, kg, loo - kg, stab))
    common = max(set(picks), key=picks.count)
    print("modal fold choice %s picked in %d/%d folds" % (common, picks.count(common), len(tasks)))

    print("\nVERDICT")
    if loo - kg >= 2:
        print("  A re-weighted selector survives leave-one-task-out with +%d outputs. Freeze the" % (loo - kg))
        print("  modal weights, validate once on a disjoint slice from the next rig run, and only")
        print("  then consider deploying. Do not tune further on this dump; it is spent.")
    else:
        print("  LOO gain %+d is below the pre-registered bar of +2. The 3-output gap is real but" % (loo - kg))
        print("  not learnable from 60 outputs without memorising them. The lever dies here, and")
        print("  the selection headroom belongs to the task-evidence route (A2) instead.")


if __name__ == "__main__":
    main()
