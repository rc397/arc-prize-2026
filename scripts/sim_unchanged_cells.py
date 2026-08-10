"""How much of an ARC output is just a copy of its input? Gate 4 for unchanged-cell loss masking.

The proposal: mask the training loss on output cells that are identical to the corresponding input
cell, so the gradient goes to what the rule CHANGES rather than to reproducing what it left alone.

It is a better-targeted idea than background weighting, because it does not assume anything about
which colour is background, and because copying is genuinely free signal: the model spends capacity
learning to echo cells no rule ever touched.

The quantity that decides it is the unchanged fraction. If outputs are 90 percent copies of their
inputs, masking those redirects almost all the gradient. If outputs mostly differ, there is little to
mask and the idea is small.

Only same-shape pairs admit a cell-to-cell comparison at all, so the share of tasks where the
transformation preserves shape is reported first, since it bounds where the idea can apply.

Usage: python sim_unchanged_cells.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P


def main():
    dd = P.data_dir()
    ch = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sol = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))

    same_shape_pairs = 0
    total_pairs = 0
    unchanged = []
    tasks_all_same_shape = 0

    for tid, task in ch.items():
        pairs = [(p["input"], p["output"]) for p in task["train"]]
        for t, ans in zip(task.get("test", []), sol.get(tid, [])):
            pairs.append((t["input"], ans))
        ok = True
        for gi, go in pairs:
            total_pairs += 1
            if len(gi) == len(go) and len(gi[0]) == len(go[0]):
                same_shape_pairs += 1
                n = sum(len(r) for r in go)
                same = sum(1 for ri, ro in zip(gi, go) for a, b in zip(ri, ro) if a == b)
                unchanged.append(same / n)
            else:
                ok = False
        tasks_all_same_shape += int(ok)

    def q(a, p):
        a = sorted(a)
        return a[int(p * (len(a) - 1))] if a else 0.0

    print("%d pairs across %d tasks" % (total_pairs, len(ch)))
    print("pairs where input and output SHAPES match: %d (%.1f%%)"
          % (same_shape_pairs, 100.0 * same_shape_pairs / total_pairs))
    print("tasks where EVERY pair matches shape    : %d of %d (%.1f%%)"
          % (tasks_all_same_shape, len(ch), 100.0 * tasks_all_same_shape / len(ch)))
    print("")
    if unchanged:
        print("among shape-matched pairs, fraction of output cells IDENTICAL to the input:")
        print("  p10 %.2f  p25 %.2f  median %.2f  p75 %.2f  p90 %.2f"
              % (q(unchanged, .1), q(unchanged, .25), q(unchanged, .5),
                 q(unchanged, .75), q(unchanged, .9)))
        for thr in (0.7, 0.8, 0.9):
            k = sum(1 for u in unchanged if u >= thr)
            print("  pairs at least %.0f%% unchanged: %d of %d (%.1f%%)"
                  % (100 * thr, k, len(unchanged), 100.0 * k / len(unchanged)))

    print("")
    print("VERDICT")
    cov = 100.0 * tasks_all_same_shape / len(ch)
    med = q(unchanged, .5)
    if not unchanged:
        print("  No shape-matched pairs at all; the idea has nothing to act on.")
    elif cov < 40:
        print("  The idea applies cleanly to only %.0f%% of tasks, since the rest change shape and" % cov)
        print("  admit no cell-to-cell comparison. Within those, outputs are %.0f%% copies at the"
              % (100 * med))
        print("  median, so the redirected gradient is real but confined to a minority of tasks.")
    elif med >= 0.8:
        print("  Outputs are %.0f%% copies of their inputs at the median, on %.0f%% of tasks."
              % (100 * med, cov))
        print("  Masking unchanged cells would redirect the large majority of the gradient, which")
        print("  is the largest structural room any loss-shaping idea in this batch has.")
    else:
        print("  Outputs are %.0f%% copies at the median on %.0f%% of tasks. Meaningful but not"
              % (100 * med, cov))
        print("  overwhelming: masking would redirect roughly half the gradient.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
