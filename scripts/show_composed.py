"""Print a couple of composed tasks as grids, so a human can judge whether they are actually tasks.

Everything so far says the compositions are consistent, non-degenerate and well formed. None of that
says they are SOLVABLE. ARC tasks are built so a person can infer the rule from a few examples; an
arbitrary chain of two DSL operations is guaranteed well defined and not guaranteed to admit any
description shorter than the chain itself. Before training on a corpus of them, look at some.

Picks the smallest-gridded compositions so the output is readable, not the most flattering ones.

Usage: python show_composed.py [N=2] [MAXDIM=12]
"""
import json
import os
import sys

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
REARC = os.path.join(BASE, "re-arc")
POOLS = os.path.join(BASE, "experiments", "rearc_pools", "tasks")
SURVEY = os.path.join(BASE, "experiments", "compose_survey.json")

PAL = ".123456789"          # 0 renders as a dot so structure is visible


def render(g):
    return [("".join(PAL[v] for v in row)) for row in g]


def side_by_side(gi, go, gap="   ->   "):
    a, b = render(gi), render(go)
    h = max(len(a), len(b))
    wa = max(len(r) for r in a)
    out = []
    for i in range(h):
        left = a[i] if i < len(a) else " " * wa
        right = b[i] if i < len(b) else ""
        out.append("  %-*s%s%s" % (wa, left, gap if i == 0 else " " * len(gap), right))
    return out


def load_verifiers():
    cwd = os.getcwd()
    sys.path.insert(0, REARC)
    os.chdir(REARC)
    try:
        import verifiers
    finally:
        os.chdir(cwd)
    return {n[len("verify_"):]: getattr(verifiers, n)
            for n in dir(verifiers) if n.startswith("verify_")}


def main(n_show, maxdim):
    vs = load_verifiers()
    survey = json.load(open(SURVEY, encoding="utf-8"))
    kept = survey.get("kept") or survey.get("examples") or []

    cands = []
    for k in kept:
        a, b = k["a"], k["b"]
        p = os.path.join(POOLS, "%s.json" % a)
        if not os.path.isfile(p):
            continue
        src = json.load(open(p, encoding="utf-8"))
        inst = []
        for e in src[:60]:
            x = [list(r) for r in e["input"]]
            if len(x) > maxdim or len(x[0]) > maxdim:
                continue
            try:
                y = [list(r) for r in vs[b](vs[a](x))]
            except Exception:
                continue
            # a kept pair was validated on four instances only. Others can degenerate to an empty
            # grid, so every instance is re-checked here rather than assumed.
            if not y or not y[0] or len(y) > maxdim or len(y[0]) > maxdim:
                continue
            if len(x) < 1 or len(x[0]) < 1:
                continue
            inst.append((x, y))
            if len(inst) == 4:
                break
        if len(inst) == 4:
            size = max(len(g) * len(g[0]) for g, _ in inst)
            cands.append((size, a, b, inst))
    cands.sort(key=lambda t: t[0])
    print("%d composed rules render inside %dx%d with 4 instances" % (len(cands), maxdim, maxdim))

    for size, a, b, inst in cands[:n_show]:
        print("")
        print("=" * 66)
        print("COMPOSED TASK   first stage %s   then   second stage %s" % (a, b))
        print("=" * 66)
        for i, (x, y) in enumerate(inst[:3]):
            print("")
            print("  demonstration %d       input -> output" % (i + 1))
            for line in side_by_side(x, y):
                print(line)
        x, y = inst[3]
        print("")
        print("  TEST                  input -> (answer)")
        for line in side_by_side(x, y):
            print(line)
        # what each stage does on its own, so the composition can be read apart
        print("")
        print("  stage-by-stage on the test input:")
        mid = [list(r) for r in vs[a](x)]
        for lbl, g in (("input", x), ("after A", mid), ("after B", y)):
            print("    %-8s %s" % (lbl, " | ".join(render(g))))
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 2,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 12) else 1)
