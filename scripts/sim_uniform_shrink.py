"""Idea 4, in the only form the 16-token vocabulary allows, plus its safety gate.

A true run-length encoding needs a marker token. The vocabulary is 16 tokens with no spare, and
overloading a digit would collide with a colour. So the implementable version is LOSSLESS UNIFORM
DELETION: drop every row and column that is entirely background in EVERY grid of the task, which
leaves the serialization alphabet untouched and simply makes the grids smaller.

The safety gate is the point of this script. The deletion must be decided from the demonstration
grids and the test input, because the answer is unknown at solve time. If the true answer has
content in a row we deleted, that task is unrecoverable. Measure that failure rate before building.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P

dd = P.data_dir()
ch = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
sol = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
keys = sorted(ch)

def keep_rows(grids, n):
    """Row indices that are NOT background in every grid tall enough to have them."""
    keep = set()
    for g in grids:
        for i, r in enumerate(g):
            if any(v for v in r):
                keep.add(i)
    return sorted(set(range(n)) & keep) or list(range(n))

def keep_cols(grids, n):
    keep = set()
    for g in grids:
        for j in range(len(g[0])):
            if any(r[j] for r in g if j < len(r)):
                keep.add(j)
    return sorted(set(range(n)) & keep) or list(range(n))

tot_b = tot_a = 0
unsafe = 0
shrunk = 0
ratios = []
for k in keys:
    vis = [p["input"] for p in ch[k]["train"]] + [p["output"] for p in ch[k]["train"]] \
          + [t["input"] for t in ch[k]["test"]]
    H = max(len(g) for g in vis); W = max(len(g[0]) for g in vis)
    kr = keep_rows(vis, H); kc = keep_cols(vis, W)
    before = sum(len(g) * len(g[0]) for g in vis)
    after = sum(len([i for i in kr if i < len(g)]) * len([j for j in kc if j < len(g[0])]) for g in vis)
    tot_b += before; tot_a += after
    ratios.append(after / before if before else 1.0)
    if after < before:
        shrunk += 1
    # safety: does any true answer carry content in a row or column we would delete?
    bad = False
    for ans in sol[k]:
        for i, r in enumerate(ans):
            for j, v in enumerate(r):
                if v and (i not in kr or j not in kc):
                    bad = True
    unsafe += int(bad)

ratios.sort()
q = lambda p: ratios[int(p * (len(ratios) - 1))]
print("%d tasks" % len(keys))
print("cells before %d, after %d, overall ratio %.3f" % (tot_b, tot_a, tot_a / tot_b))
print("per-task retained: min %.3f p25 %.3f median %.3f p75 %.3f" % (ratios[0], q(.25), q(.5), q(.75)))
print("tasks that shrink at all      : %d (%.1f%%)" % (shrunk, 100.0 * shrunk / len(keys)))
print("tasks where the ANSWER would be truncated: %d (%.1f%%)  <-- unrecoverable" % (unsafe, 100.0 * unsafe / len(keys)))
