"""How much of an ARC-AGI-2 grid is actually background? Gate 4 for a whole cluster of loss ideas.

Several proposals rest on the claim that ARC grids are 80 percent or more background, so that a
uniform cross-entropy over output tokens is dominated by background and dilutes the signal. If that
is true, background down-weighting, focal loss, spatial-distance weighting and background decay all
have a large quantity to act on. If it is false, they are all acting on a small one.

Measures two things, because "background" is ambiguous:
  colour 0 share   the literal zero colour, which the serialization treats as any other digit
  modal share      the most common colour in each grid, which is what "background" usually means

Reports the distribution over OUTPUT grids specifically, since that is what the collator supervises.

Usage: python sim_background_share.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P


def main():
    dd = P.data_dir()
    ch = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sol = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))

    zero, modal, sizes = [], [], []
    for tid, task in ch.items():
        outs = [p["output"] for p in task["train"]] + list(sol.get(tid, []))
        for g in outs:
            cells = [v for row in g for v in row]
            if not cells:
                continue
            c = Counter(cells)
            zero.append(c.get(0, 0) / len(cells))
            modal.append(c.most_common(1)[0][1] / len(cells))
            sizes.append(len(cells))

    def q(a, p):
        a = sorted(a)
        return a[int(p * (len(a) - 1))]

    n = len(zero)
    print("%d output grids across %d evaluation tasks" % (n, len(ch)))
    print("")
    print("%-16s %6s %6s %6s %6s %6s" % ("", "p10", "p25", "median", "p75", "p90"))
    print("%-16s %6.2f %6.2f %6.2f %6.2f %6.2f"
          % ("colour-0 share", q(zero, .1), q(zero, .25), q(zero, .5), q(zero, .75), q(zero, .9)))
    print("%-16s %6.2f %6.2f %6.2f %6.2f %6.2f"
          % ("modal share", q(modal, .1), q(modal, .25), q(modal, .5), q(modal, .75), q(modal, .9)))
    print("")
    for thr in (0.5, 0.6, 0.7, 0.8):
        print("  output grids that are >= %.0f%% modal colour: %d of %d (%.1f%%)"
              % (100 * thr, sum(1 for m in modal if m >= thr), n,
                 100.0 * sum(1 for m in modal if m >= thr) / n))
    print("")
    print("mean cells per output grid: %.0f" % (sum(sizes) / len(sizes)))
    print("")
    print("VERDICT")
    med = q(modal, .5)
    if med >= 0.75:
        print("  Background dominates at %.0f%% median. Down-weighting it acts on a large quantity"
              % (100 * med))
        print("  and the premise behind that cluster of loss ideas holds.")
    elif med >= 0.55:
        print("  The modal colour is %.0f%% of a median output grid. Real but not dominant: a"
              % (100 * med))
        print("  background re-weighting changes the loss by roughly a factor of two, not ten.")
    else:
        print("  The modal colour is only %.0f%% of a median output grid, and the literal zero"
              % (100 * med))
        print("  colour %.0f%%. The premise that background swamps the loss does not hold here, so"
              % (100 * q(zero, .5)))
        print("  down-weighting it, focal loss over it and distance-weighting from it are all")
        print("  acting on a much smaller quantity than they assume.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
