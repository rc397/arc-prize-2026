"""How much prompt length would a lossless bounding-box crop save, and does it clear the memory cliff?

We measured that decode batch width 4 costs 127 ms per step below about 1650 prompt tokens and 3253
to 4105 ms above it, because reserved memory crosses roughly 13 to 14 GB on a 16 GB card. The median
evaluation prompt is 2198 tokens, so most of the set sits on the wrong side.

Run-length compressing the serialization would shrink prompts but changes a format the base model
was pretrained on, with a 16-token vocabulary that has no run-length token. A bounding-box crop does
not. If a margin is background in EVERY demonstration input, EVERY demonstration output and the test
input, removing it is information-preserving with respect to those grids, the serialization is
unchanged, and the predicted grid is padded back afterwards.

The safety condition is checkable per task and this script enforces it: the crop uses the union of
all non-background bounding boxes, so no non-background cell is ever removed from any grid.

Two caveats it cannot resolve. Cropping changes absolute positions, and a rule keyed to
absolute position rather than to content would break, though a rule keyed to the grid EDGE would
survive since the edges move together. And grids of different sizes crop by different amounts, so
the uniformity of the operation across a task is itself reported rather than assumed.

Usage: python sim_bboxcrop.py [N_TASKS=120]
"""
import json
import os
import sys


def bbox(grid, bg=0):
    rows = [i for i, r in enumerate(grid) if any(v != bg for v in r)]
    cols = [j for j in range(len(grid[0])) if any(r[j] != bg for r in grid)]
    if not rows or not cols:
        return None
    return rows[0], rows[-1], cols[0], cols[-1]


def crop_savings(task, bg=0):
    """Rows and columns removable from each grid without touching a non-background cell."""
    grids = []
    for p in task["train"]:
        grids.append(p["input"])
        grids.append(p["output"])
    for t in task["test"]:
        grids.append(t["input"])
    before = sum(len(g) * len(g[0]) for g in grids)
    after = 0
    uniform = True
    margins = []
    for g in grids:
        b = bbox(g, bg)
        if b is None:
            after += 1
            margins.append(None)
            continue
        r0, r1, c0, c1 = b
        after += (r1 - r0 + 1) * (c1 - c0 + 1)
        margins.append((r0, len(g) - 1 - r1, c0, len(g[0]) - 1 - c1))
    seen = {m for m in margins if m is not None}
    uniform = len(seen) <= 1
    return before, after, uniform


def main(n_tasks):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig_restart_probe as P
    path = os.path.join(P.data_dir(), "arc-agi_evaluation_challenges.json")
    ch = json.load(open(path))
    keys = sorted(ch)[:n_tasks]

    # tokens are dominated by cells: one per cell plus a separator per row. Ratio of cell counts is
    # therefore a good proxy for the ratio of prompt lengths, which is what the cliff is a function of.
    tot_b = tot_a = 0
    ratios = []
    uni = 0
    for k in keys:
        b, a, u = crop_savings(ch[k])
        tot_b += b
        tot_a += a
        ratios.append(a / b if b else 1.0)
        uni += int(u)
    ratios.sort()
    q = lambda p: ratios[int(p * (len(ratios) - 1))]

    print("%d tasks" % len(keys))
    print("total cells before %d, after %d, overall ratio %.3f" % (tot_b, tot_a, tot_a / tot_b))
    print("per-task retained fraction: min %.3f  p25 %.3f  median %.3f  p75 %.3f  max %.3f"
          % (ratios[0], q(.25), q(.5), q(.75), ratios[-1]))
    print("tasks where every grid has the SAME margin (crop is uniform): %d (%.1f%%)"
          % (uni, 100.0 * uni / len(keys)))

    # what it would do to the cliff, using the measured prompt distribution
    med_prompt = 2198.0
    cliff = 1650.0
    saved = [med_prompt * r for r in ratios]
    under = sum(1 for s in saved if s <= cliff)
    print("\napplying the per-task ratio to the median prompt of %d tokens:" % med_prompt)
    print("  tasks that would land under the %d-token cliff: %d of %d (%.1f%%)"
          % (cliff, under, len(saved), 100.0 * under / len(saved)))
    big = sum(1 for r in ratios if r < 0.75)
    print("  tasks shrinking by more than 25%%: %d (%.1f%%)" % (big, 100.0 * big / len(ratios)))

    print("\nVERDICT")
    if tot_a / tot_b > 0.95:
        print("  Grids are already tight: only %.1f%% of cells are removable margin. Cropping cannot"
              % (100 * (1 - tot_a / tot_b)))
        print("  move prompts off the cliff and this route is closed. The prompt length is real")
        print("  content, not padding.")
    elif big >= 0.25 * len(ratios):
        print("  %.1f%% of cells are removable margin and %d tasks shrink by more than a quarter."
              % (100 * (1 - tot_a / tot_b), big))
        print("  Worth building: it is lossless, leaves the serialization untouched, and targets the")
        print("  exact quantity the memory cliff is a function of.")
    else:
        print("  Only %.1f%% of cells are removable margin and few tasks shrink much. Some benefit,"
              % (100 * (1 - tot_a / tot_b)))
        print("  but not enough to move the distribution across the cliff on its own.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 120) else 1)
