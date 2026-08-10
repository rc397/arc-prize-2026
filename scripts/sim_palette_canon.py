"""Canonical palette pre-sorting: is it well defined, and is it even a change?

The proposal is a deterministic colour remapping sorted by total pixel count per demonstration.
Two things have to be true for it to be an experiment at all, and both are checkable for nothing.

FIRST, the deployed augmentation already applies a colour permutation drawn uniformly from all ten
factorial bijections (arc_loader.permute_rnd_all_ is np.random.permutation(10)). Composing a fixed
canonical map with a uniform random one gives a uniform random one. So canonicalising the task and
then augmenting it as usual leaves the training distribution EXACTLY unchanged, not approximately.
That is a proof, not a measurement, and it kills the idea in its stated form.

What survives is a different proposal: canonicalise INSTEAD of permuting, which removes the colour
axis from the augmentation. That is a real change, and it is an ablation of colour diversity rather
than a pre-sorting, so it should be argued and measured as one.

SECOND, "per demonstration" is load bearing and probably fatal on its own. If each demonstration is
sorted by its own colour counts, two demonstrations of the same task can send the same colour to
different places, and for the many ARC tasks whose rule is about a specific colour, that destroys
the correspondence the task is built on. This measures how often that happens.

Also reported: how often the sort is ambiguous at all, because equal colour counts make "sorted by
count" undefined without a tiebreak, and an undefined canonicalisation is not deterministic.

Usage: python sim_palette_canon.py
"""
import json
import os
import sys
from collections import Counter

DD = os.path.join("experiments", "_data")


def canon_map(grids):
    """Colours by descending total count, ties broken by first appearance in raster order."""
    cnt, first = Counter(), {}
    for g in grids:
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                cnt[v] += 1
                first.setdefault(v, (r, c))
    order = sorted(cnt, key=lambda v: (-cnt[v], first[v]))
    ties = sum(1 for i in range(1, len(order)) if cnt[order[i]] == cnt[order[i - 1]])
    return {v: i for i, v in enumerate(order)}, ties


def main():
    ch = json.load(open(os.path.join(DD, "arc-agi_evaluation_challenges.json")))

    tied_grids = n_grids = 0
    disagree_tasks = n_tasks = 0
    disagree_pairs = n_pairs = 0
    colour_tasks = 0

    for tid, task in ch.items():
        n_tasks += 1
        maps = []
        for p in task["train"]:
            m, ties = canon_map([p["input"], p["output"]])
            n_grids += 1
            tied_grids += int(ties > 0)
            maps.append(m)
        # a colour is "shared" if it appears in more than one demonstration; the question is whether
        # the per-demonstration maps agree about where it goes
        bad = 0
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                n_pairs += 1
                shared = set(maps[i]) & set(maps[j])
                if any(maps[i][c] != maps[j][c] for c in shared):
                    bad += 1
                    disagree_pairs += 1
        disagree_tasks += int(bad > 0)
        # does the task's rule plausibly depend on specific colours? proxy: some colour present in
        # every input is absent from some output, or vice versa, so colour identity carries meaning
        ins = [set(v for row in p["input"] for v in row) for p in task["train"]]
        outs = [set(v for row in p["output"] for v in row) for p in task["train"]]
        colour_tasks += int(any(i != o for i, o in zip(ins, outs)))

    print("%d tasks, %d demonstration pairs, %d demonstration-pair comparisons"
          % (n_tasks, n_grids, n_pairs))
    print("")
    print("demonstrations whose colour counts TIE, so the sort needs a tiebreak: %d of %d (%.0f%%)"
          % (tied_grids, n_grids, 100.0 * tied_grids / n_grids))
    print("demonstration pairs whose per-demonstration maps DISAGREE on a shared colour: %d of %d "
          "(%.0f%%)" % (disagree_pairs, n_pairs, 100.0 * disagree_pairs / max(1, n_pairs)))
    print("tasks with at least one such disagreement: %d of %d (%.0f%%)"
          % (disagree_tasks, n_tasks, 100.0 * disagree_tasks / n_tasks))
    print("tasks whose input and output palettes differ, so colour identity carries meaning: "
          "%d of %d (%.0f%%)" % (colour_tasks, n_tasks, 100.0 * colour_tasks / n_tasks))
    print("")
    print("VERDICT")
    print("  Sorting per demonstration sends the same colour to different places in %.0f%% of tasks."
          % (100.0 * disagree_tasks / n_tasks))
    print("  On the %.0f%% of tasks where the palette itself changes between input and output, that"
          % (100.0 * colour_tasks / n_tasks))
    print("  is not a relabelling, it is a different task per demonstration.")
    print("")
    print("  And the version that sorts per TASK is provably a no-op: the deployed augmentation")
    print("  already composes a uniform random permutation of all ten colours on top of whatever")
    print("  map it is given (arc_loader.py:33, np.random.permutation(10)), and a fixed bijection")
    print("  composed with a uniform one is uniform. The training distribution does not move.")
    print("")
    print("  The only version left that changes anything is canonicalising INSTEAD of permuting,")
    print("  which deletes the colour axis of the augmentation. That is an ablation of colour")
    print("  diversity, it costs the eval views their vote diversity too, and it should be run as")
    print("  that experiment rather than as a pre-sorting.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
