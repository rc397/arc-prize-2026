"""Does per-cell voting across pool candidates reconstruct truths the pool does not contain?

The near-miss census found 7 truth-absent outputs within 3 cells of a pool candidate, with 6-13
candidates each. If different candidates are wrong in different cells, a per-cell vote over the
modal-shape candidates could synthesise the exact truth from an all-wrong pool. That would be new
pool coverage for zero GPU cost, and it sits outside the selection ceiling because it creates a
candidate rather than ranking the existing ones.

Three weightings are tried and reported separately: picking the best afterwards would be an oracle,
so deployment gets whichever wins on this slice only if it wins clearly and is then validated on a
different slice.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "paper", "data", "pool_dump.json")))
SOLS = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "..", "arc-agi_evaluation_solutions.json")))


def weights(c, mode):
    if mode == "uniform":
        return 1.0
    if mode == "multiplicity":
        return float(len(c["beam"]))
    if mode == "exp-aug":
        return float(np.exp(-np.mean([np.mean(a) for a in c["aug"]])))
    raise ValueError(mode)


def vote(cands, mode):
    """Weighted per-cell vote over candidates of the weight-modal shape."""
    by_shape = defaultdict(list)
    for c in cands:
        g = np.array(c["grid"])
        by_shape[g.shape].append((g, weights(c, mode)))
    shape = max(by_shape, key=lambda s: sum(w for _, w in by_shape[s]))
    grids = by_shape[shape]
    out = np.zeros(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            tally = Counter()
            for g, w in grids:
                tally[int(g[i, j])] += w
            out[i, j] = tally.most_common(1)[0][0]
    return out, len(grids)


def main():
    modes = ["uniform", "multiplicity", "exp-aug"]
    rec = {m: [] for m in modes}
    already = {m: 0 for m in modes}
    n_absent = 0
    for bk, cands in sorted(POOL.items()):
        if not cands:
            continue
        task, idx = bk.rsplit("_", 1)
        truth = np.array(SOLS[task][int(idx)])
        absent = not any(c["correct"] for c in cands)
        n_absent += int(absent)
        for m in modes:
            v, n = vote(cands, m)
            hit = v.shape == truth.shape and (v == truth).all()
            if hit and absent:
                rec[m].append((bk, n))
            elif hit:
                already[m] += 1
    print("truth-absent outputs: %d\n" % n_absent)
    for m in modes:
        print("%-13s reconstructs %d absent truths %s   (also matches %d present ones)"
              % (m, len(rec[m]), [bk for bk, _ in rec[m]], already[m]))


if __name__ == "__main__":
    main()
