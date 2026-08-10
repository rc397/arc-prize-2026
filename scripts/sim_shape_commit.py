"""Does the decoder already agree with itself about the output's shape? Gate 4 for the dim prefix.

The forced dimension prefix is queued as a deliberate Gate 3 violation, and its whole argument is
that the decoder commits to a shape implicitly, one row at a time, with no way to check itself. A
prefix makes the commitment explicit and lets the parser reject a candidate that contradicts its own
header.

That argument is only worth the run if the decoder actually disagrees with itself. If every
candidate in a pool already has the same shape, there is nothing for the header to catch and the
five extra tokens buy nothing but risk. This counts the disagreement.

Usage: python sim_shape_commit.py [POOL]
"""
import json
import os
import sys
from collections import Counter

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join("paper", "data", "pool_dump.json")


def main():
    pool = json.load(open(POOL))
    single = multi = 0
    off_modal = total = 0
    truth_not_modal = truth_pools = 0
    spread = []
    for key, cands in pool.items():
        if not cands:
            continue
        shapes = Counter((len(c["grid"]), len(c["grid"][0])) for c in cands)
        spread.append(len(shapes))
        if len(shapes) == 1:
            single += 1
        else:
            multi += 1
        top = shapes.most_common(1)[0][0]
        total += len(cands)
        off_modal += sum(n for s, n in shapes.items() if s != top)
        truth = next((c for c in cands if c["correct"]), None)
        if truth is not None:
            truth_pools += 1
            if (len(truth["grid"]), len(truth["grid"][0])) != top:
                truth_not_modal += 1

    n = single + multi
    print("%d non-empty pools, %d candidates" % (n, total))
    print("pools where every candidate has the SAME shape: %d of %d (%.0f%%)"
          % (single, n, 100.0 * single / n))
    print("pools carrying more than one shape:             %d of %d (%.0f%%)"
          % (multi, n, 100.0 * multi / n))
    print("candidates whose shape is not the pool's modal shape: %d of %d (%.0f%%)"
          % (off_modal, total, 100.0 * off_modal / total))
    print("pools holding the truth where the truth is NOT the modal shape: %d of %d"
          % (truth_not_modal, truth_pools))
    print("mean distinct shapes per pool: %.2f" % (sum(spread) / len(spread)))
    print("")
    print("VERDICT")
    if 100.0 * single / n >= 80:
        print("  The decoder already agrees with itself about shape on %.0f%% of pools, so a header"
              % (100.0 * single / n))
        print("  has almost nothing to reject. What it would still change is the CONDITIONING: the")
        print("  model choosing a shape before drawing rather than discovering one. That is the")
        print("  only claim the queued run can test, and it is a weaker one than the filter story.")
    else:
        print("  %.0f%% of pools carry more than one shape and %.0f%% of candidates sit off the"
              % (100.0 * multi / n, 100.0 * off_modal / total))
        print("  modal shape, so a self-consistency header has real mass to act on.")
    if truth_not_modal:
        print("")
        print("  Note the cost side: on %d pools the truth is NOT the modal shape, so any rule that"
              % truth_not_modal)
        print("  prefers the majority shape would discard it.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
