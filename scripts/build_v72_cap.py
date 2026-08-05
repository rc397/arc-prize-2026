"""Build the cap-only variant of the replicate: one literal changed, nothing else.

The per-task cap has never been tested cleanly. v7.3 declared 1200 and ran 1800, because the
builder's replace targeted a `spend_time` form that exists only inside a commented-out cell. v7.4
really ran 1200 but also fired 24 unbudgeted greedy fills through a gate that read the string "0" as
true, so two factors moved and the result is void rather than adverse.

This build changes exactly one thing against the artifact with the best-measured baseline. The
source is notebooks/arc_submission_v7_2_replicate.ipynb, the immutable kernel version that has now
scored 30.56 and 32.64, so its baseline is three runs rather than one. The live per-task cap is a
hardcoded literal in the solver cell and we set it to 1200. The config banner's default moves with
it, which is display-only, because a banner that prints a cap the code does not run is the exact
failure that voided two earlier builds.

Deliberately NOT changed: the 12-hour global wall, the 540-second DFS-internal budget inside
turbo_dfs, and the dead symbolic-arm computation. That last one matters. Removing it also frees
per-task time, so a build that removed it AND capped would confound two timing levers, and its own
run scored 30.00 which is inside noise rather than a clean baseline.

Prediction, recorded before the run. The measurement slice says the four tasks that ran past 1200
seconds were all misses and consumed 26.3 percent of the budget, and the slowest solve of the entire
run arrived at 1118 seconds. So the cap should cost zero solves and return roughly a fifth of the
wall to the queue tail. Against a family mean of 30.95 with sd 1.01, only an effect above about two
points is detectable in one run, so a single score here is weak evidence either way and is worth
reporting mainly if it lands outside the band.

Usage: python build_v72_cap.py [OUT=notebooks/arc_submission_v7_2_cap1200.ipynb]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "notebooks", "arc_submission_v7_2_replicate.ipynb")
DEFAULT_OUT = os.path.join(HERE, "notebooks", "arc_submission_v7_2_cap1200.ipynb")

LIVE_CAP = "if time.time() - start_time > 1800 or time.time() > end_time:"
LIVE_CAP_NEW = "if time.time() - start_time > 1200 or time.time() > end_time:"
BANNER = 'os.environ.get("ARC_TASK_TIMEOUT", "1800")'
BANNER_NEW = 'os.environ.get("ARC_TASK_TIMEOUT", "1200")'


def main(out):
    nb = json.load(open(SRC, encoding="utf-8"))
    hits = {"cap": 0, "banner": 0}
    for c in nb["cells"]:
        src = c.get("source", [])
        for i, line in enumerate(src):
            # the commented-out legacy cell carries the same strings; never touch a comment
            if line.lstrip().startswith("#"):
                continue
            if LIVE_CAP in line:
                src[i] = line.replace(LIVE_CAP, LIVE_CAP_NEW)
                hits["cap"] += 1
            if BANNER in line:
                src[i] = src[i].replace(BANNER, BANNER_NEW)
                hits["banner"] += 1

    # A replace that silently fails to apply is the whole reason this build exists.
    assert hits["cap"] == 1, "expected exactly one live per-task cap, patched %d" % hits["cap"]
    assert hits["banner"] == 1, "expected exactly one config banner, patched %d" % hits["banner"]

    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)

    # Verify against the source by comparing rendered sources cell by cell, so a stray edit anywhere
    # else in 2788 lines fails here rather than on the leaderboard.
    a = json.load(open(SRC, encoding="utf-8"))["cells"]
    b = json.load(open(out, encoding="utf-8"))["cells"]
    assert len(a) == len(b), "cell count changed"
    diffs = []
    for ci, (x, y) in enumerate(zip(a, b)):
        xs, ys = "".join(x.get("source", [])), "".join(y.get("source", []))
        if xs != ys:
            for la, lb in zip(xs.split("\n"), ys.split("\n")):
                if la != lb:
                    diffs.append((ci, la.strip(), lb.strip()))
    print("wrote %s" % out)
    print("changed lines: %d" % len(diffs))
    for ci, la, lb in diffs:
        print("  cell %d" % ci)
        print("    - %s" % la[:130])
        print("    + %s" % lb[:130])
    assert len(diffs) == 2, "expected exactly two changed lines, got %d" % len(diffs)
    print("\nOK: one live cap literal and one display banner, nothing else.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT) else 1)
