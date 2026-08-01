"""Three offline measurements on the dumped candidate pool, each deciding whether an idea lives.

1. The exact Appendix-F ceiling on this slice: a monotone selector over the recorded statistics can
   place the truth in its top two iff at most one candidate Pareto-dominates it. Counting dominators
   gives the attainable bound; the gap between that bound and what kgmon attains says whether tuning
   selector weights inside the theorem is worth anything at all.
2. Near-miss census against the actual solutions: when the truth is absent from the pool, how far in
   cells is the nearest same-shape candidate? Mass at distance 1-3 licenses a local-repair stage;
   none kills it. Also: how often is the truth's SHAPE even present?
3. Attempt-2 diversity: how often kgmon's top two picks differ by so few cells that the second
   attempt is nearly wasted.
"""
import json
import os

import numpy as np

import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "paper", "data", "pool_dump.json")
SOLS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "..", "arc-agi_evaluation_solutions.json")


def kgmon(c):
    return len(c["beam"]) - float(np.mean([np.mean(a) for a in c["aug"]]))


def stats(c):
    return (len(c["beam"]),
            float(np.mean([np.mean(a) for a in c["aug"]])),
            float(np.mean(c["beam"])))


def dominates(a, b):
    am, aa, ab = stats(a)
    bm, ba, bb = stats(b)
    ge = am >= bm and aa <= ba and ab <= bb
    strict = am > bm or aa < ba or ab < bb
    return ge and strict


def hamming(g1, g2):
    a, b = np.array(g1), np.array(g2)
    if a.shape != b.shape:
        return None
    return int((a != b).sum())


def main():
    pool = json.load(open(POOL))
    sols = json.load(open(SOLS))
    n_with_truth = attained = bound = 0
    blocked, near, dup2 = [], [], []
    for bk, cands in sorted(pool.items()):
        if not cands:
            task, idx = bk.rsplit("_", 1)
            near.append((bk, "empty-pool", None, 0))
            continue
        order = sorted(cands, key=kgmon, reverse=True)
        attained += int(any(c["correct"] for c in order[:2]))
        truth = next((c for c in cands if c["correct"]), None)
        if truth is not None:
            n_with_truth += 1
            nd = sum(1 for c in cands if not c["correct"] and dominates(c, truth))
            bound += int(nd <= 1)
            if nd >= 2:
                blocked.append((bk, nd, len(cands)))
        else:
            task, idx = bk.rsplit("_", 1)
            tg = sols[task][int(idx)]
            ds = [h for c in cands if (h := hamming(c["grid"], tg)) is not None]
            frac = (min(ds) / float(np.array(tg).size)) if ds else None
            near.append((bk, min(ds) if ds else "no-shape-match", frac, len(cands)))
        if len(order) >= 2:
            dup2.append((bk, hamming(order[0]["grid"], order[1]["grid"])))

    print("outputs: %d   truth in pool: %d   kgmon top-2 attained: %d" %
          (len(pool), n_with_truth, attained))
    print("\n[1] SELECTION CEILING")
    print("Appendix-F attainable bound on this slice: %d of %d" % (bound, len(pool)))
    print("gap (bound - attained): %d outputs" % (bound - attained))
    print("truth blocked by >=2 dominators on %d outputs:" % len(blocked))
    for bk, nd, nc in blocked:
        print("   %-14s dominators %2d of %3d candidates" % (bk, nd, nc))

    print("\n[2] NEAR-MISS CENSUS (truth absent, distance to nearest same-shape candidate)")
    hist = {}
    for bk, d, frac, nc in sorted(near, key=lambda r: (isinstance(r[1], str), r[1])):
        label = d if isinstance(d, str) else ("d=%d (%.1f%% of cells)" % (d, 100 * frac))
        print("   %-14s %-24s (%d candidates)" % (bk, label, nc))
        key = d if isinstance(d, str) else (1 if d <= 1 else 3 if d <= 3 else 10 if d <= 10 else 999)
        hist[key] = hist.get(key, 0) + 1
    print("summary:", {("d<=%d" % k if isinstance(k, int) else k): v
                       for k, v in sorted(hist.items(), key=str)})

    print("\n[3] ATTEMPT-2 DIVERSITY (kgmon top-2)")
    hs = [h for _, h in dup2 if h is not None]
    print("top-2 same shape on %d of %d outputs; Hamming<=2 on %d; median Hamming %s"
          % (len(hs), len(dup2), sum(1 for h in hs if h <= 2),
             int(np.median(hs)) if hs else "n/a"))


if __name__ == "__main__":
    main()
