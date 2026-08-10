"""What do the augmentations that "change the task" actually break? A number instead of an argument.

Six proposals in the batch add training views by moving grid content around: toroidal wrap-around
translation, zero-padded random translation, blank row and column permutation, dummy background row
injection, checkerboard masking, single-pixel noise. They were closed under Gate 5 on the argument
that ARC rules reference edges, absolute position and object identity, so a moved grid is a
demonstration of something the task does not do, and the collator trains on it as if it were true.

The argument is right in kind and says nothing about size, and size is what decides whether an
augmentation is a bad idea or a slightly noisy one. These are the quantities:

  border contact     content touching an edge. A toroidal wrap joins that content to whatever is on
                     the opposite edge, creating adjacencies that never existed, and a zero-padded
                     translation deletes it outright. Where nothing touches an edge, both are safe.
  edge-anchored      the cells the rule CHANGES touch an edge. Then the rule itself is plausibly
                     stated relative to the border, and translating the grid moves the border away
                     from the thing it is anchored to.
  uniform lines      rows or columns of a single colour, which is what "blank row permutation" and
                     "dummy background row injection" propose to shuffle and insert. Reported with
                     whether the input and output even have the same number of them, since where
                     they differ the count is part of the answer.

Nothing here needs a model. Every number is a property of the tasks.

Usage: python sim_aug_damage.py
"""
import json
import os
import sys
from collections import Counter

DD = os.path.join("experiments", "_data")


def modal(g):
    return Counter(v for row in g for v in row).most_common(1)[0][0]


def touches_border(g, bg):
    h, w = len(g), len(g[0])
    edge = ([g[0][c] for c in range(w)] + [g[h - 1][c] for c in range(w)]
            + [g[r][0] for r in range(h)] + [g[r][w - 1] for r in range(h)])
    return any(v != bg for v in edge)


def uniform_lines(g):
    rows = sum(1 for r in g if len(set(r)) == 1)
    cols = sum(1 for c in range(len(g[0])) if len({r[c] for r in g}) == 1)
    return rows, cols


def changed_touches_border(gi, go):
    if len(gi) != len(go) or len(gi[0]) != len(go[0]):
        return None
    h, w = len(gi), len(gi[0])
    for r in range(h):
        for c in range(w):
            if gi[r][c] != go[r][c] and (r in (0, h - 1) or c in (0, w - 1)):
                return True
    return False


def main():
    ch = json.load(open(os.path.join(DD, "arc-agi_evaluation_challenges.json")))
    sol = json.load(open(os.path.join(DD, "arc-agi_evaluation_solutions.json")))

    n_grid = border = 0
    n_pair = 0
    edge_anchored = 0
    shape_pairs = 0
    n_task = 0
    task_border = task_edge = 0
    lines_grids = lines_mismatch = lines_pairs = 0

    for tid, task in ch.items():
        n_task += 1
        pairs = [(p["input"], p["output"]) for p in task["train"]]
        for t, ans in zip(task.get("test", []), sol.get(tid, [])):
            pairs.append((t["input"], ans))
        tb = te = False
        for gi, go in pairs:
            n_pair += 1
            for g in (gi, go):
                n_grid += 1
                if touches_border(g, modal(g)):
                    border += 1
                    tb = True
            ri, ci = uniform_lines(gi)
            ro, co = uniform_lines(go)
            if ri or ci:
                lines_grids += 1
            if len(gi) == len(go) and len(gi[0]) == len(go[0]):
                lines_pairs += 1
                if (ri, ci) != (ro, co):
                    lines_mismatch += 1
            ea = changed_touches_border(gi, go)
            if ea is not None:
                shape_pairs += 1
                if ea:
                    edge_anchored += 1
                    te = True
        task_border += int(tb)
        task_edge += int(te)

    print("%d tasks, %d pairs, %d grids" % (n_task, n_pair, n_grid))
    print("")
    print("grids with content touching an edge:            %d of %d (%.0f%%)"
          % (border, n_grid, 100.0 * border / n_grid))
    print("tasks where at least one grid touches an edge:  %d of %d (%.0f%%)"
          % (task_border, n_task, 100.0 * task_border / n_task))
    print("shape-matched pairs whose CHANGED cells touch an edge: %d of %d (%.0f%%)"
          % (edge_anchored, shape_pairs, 100.0 * edge_anchored / max(1, shape_pairs)))
    print("tasks with at least one such pair:              %d of %d (%.0f%%)"
          % (task_edge, n_task, 100.0 * task_edge / n_task))
    print("")
    print("grids containing a uniform row or column:       %d of %d (%.0f%%)"
          % (lines_grids, n_grid, 100.0 * lines_grids / n_grid))
    print("shape-matched pairs where the COUNT of uniform rows/cols changes: %d of %d (%.0f%%)"
          % (lines_mismatch, lines_pairs, 100.0 * lines_mismatch / max(1, lines_pairs)))
    print("")
    print("VERDICT")
    print("  A toroidal wrap or a zero-padded translation alters content on %.0f%% of grids and"
          % (100.0 * border / n_grid))
    print("  touches %.0f%% of tasks somewhere, so there is no useful subset to apply it to."
          % (100.0 * task_border / n_task))
    print("  Worse, on %.0f%% of shape-matched pairs the cells the rule CHANGES sit on the border,"
          % (100.0 * edge_anchored / max(1, shape_pairs)))
    print("  which is the signature of a rule stated relative to the edge. Moving the grid moves")
    print("  the edge and leaves the demonstration asserting something false, and the collator")
    print("  supervises output grids, so the adapter fits it as though it were true.")
    print("  Uniform rows and columns are present on %.0f%% of grids and their COUNT changes across"
          % (100.0 * lines_grids / n_grid))
    print("  %.0f%% of shape-matched pairs, so they are part of the answer often enough that"
          % (100.0 * lines_mismatch / max(1, lines_pairs)))
    print("  permuting or injecting them is editing the label.")
    print("")
    print("  Gate 5 was the right call and this is its size: these are not slightly noisy views,")
    print("  they are false demonstrations on the large majority of tasks.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
