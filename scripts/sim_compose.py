"""Can we build ARC-2-style COMPOSITIONAL tasks from RE-ARC's parts? Measured, not asserted.

I claimed building generators for ARC-2-style composition was a larger project than our remaining
calendar. That was an assertion where our own triage doc calls for a CPU script, so here is the
script.

The route is direct. RE-ARC ships 400 verifiers, each a pure Grid -> Grid function implementing one
ARC-1 task's rule. Transformations compose: verify_B(verify_A(x)) is a new deterministic rule, and
400 verifiers give 159,600 ordered pairs. ARC-AGI-2's stated difficulty is multi-rule and multi-step
composition, which is exactly what a two-stage chain is.

Validity is free. Both stages are deterministic functions, so the composition is a well-defined rule
and every instance of it is consistent by construction. That is the part that usually makes synthetic
task generation hard, and here it is automatic.

What is NOT free is non-degeneracy, so the filters are the substance of this script:

  runs        verify_B must not raise on verify_A's output. The two were written for different input
              distributions, so most pairs will simply fail, and the failure rate is the headline.
  B acts      B(A(x)) != A(x) on at least one instance, or the second stage is a no-op.
  A matters   B(A(x)) != B(x) on at least one instance, or the first stage is invisible in the
              result and the "composition" is just B.
  non-trivial B(A(x)) != x, or the pair is an identity in disguise.
  well formed output within ARC's 1..30 dimensions and 0..9 colours.

A pair surviving all five is a genuinely two-step rule that neither stage produces alone. The number
of such pairs, and how many distinct instances each admits, is what decides whether a compositional
corpus is a weekend or a research project.

Usage: python sim_compose.py [N_PAIRS=3000] [N_INSTANCES=4]
Writes experiments/compose_survey.json.
"""
import json
import os
import random
import sys

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
REARC = os.path.join(BASE, "re-arc")
POOLS = os.path.join(BASE, "experiments", "rearc_pools", "tasks")
OUT = os.path.join(BASE, "experiments", "compose_survey.json")
MAXD = 30


def well_formed(g):
    if not isinstance(g, (list, tuple)) or not g:
        return False
    try:
        h = len(g)
        w = len(g[0])
    except Exception:
        return False
    if not (1 <= h <= MAXD and 1 <= w <= MAXD):
        return False
    for r in g:
        if len(r) != w:
            return False
        for v in r:
            if not isinstance(v, int) or not (0 <= v <= 9):
                return False
    return True


def as_list(g):
    return [list(r) for r in g] if g is not None else None


def main(n_pairs, n_inst):
    cwd = os.getcwd()
    sys.path.insert(0, REARC)
    os.chdir(REARC)
    try:
        import verifiers
    finally:
        os.chdir(cwd)
    vs = {n[len("verify_"):]: getattr(verifiers, n)
          for n in dir(verifiers) if n.startswith("verify_")}
    keys = sorted(vs)
    print("%d verifiers loaded" % len(keys), flush=True)

    # input grids come from the pools we already generated, one bank per task
    bank = {}
    for k in keys:
        p = os.path.join(POOLS, "%s.json" % k)
        if os.path.isfile(p):
            ex = json.load(open(p, encoding="utf-8"))
            if len(ex) >= n_inst:
                bank[k] = [e["input"] for e in ex[:n_inst]]
    print("%d tasks have an input bank of >= %d grids" % (len(bank), n_inst), flush=True)
    pool_keys = sorted(bank)

    rng = random.Random(5)
    stats = {"tried": 0, "ran": 0, "b_acts": 0, "a_matters": 0, "nontrivial": 0, "kept": 0}
    kept = []

    # n_pairs <= 0 means exhaust every ordered pair, which is the only way to get the exact count
    # rather than an extrapolation with sampling error on top of it.
    if n_pairs <= 0:
        cand = [(a, b) for a in pool_keys for b in keys if a != b]
        print("exhaustive: %d ordered pairs" % len(cand), flush=True)
    else:
        cand = []
        seen = set()
        while len(cand) < n_pairs:
            a, b = rng.choice(pool_keys), rng.choice(keys)
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                cand.append((a, b))

    for pi, (a, b) in enumerate(cand):
        if n_pairs <= 0 and pi and pi % 20000 == 0:
            print("  %d/%d pairs, %d kept" % (pi, len(cand), stats["kept"]), flush=True)
        stats["tried"] += 1

        pairs, ok = [], True
        b_acts = a_matters = nontrivial = False
        for x in bank[a]:
            try:
                ax = as_list(vs[a](x))
                bax = as_list(vs[b](ax))
            except Exception:
                ok = False
                break
            if not well_formed(bax):
                ok = False
                break
            try:
                bx = as_list(vs[b](x))
            except Exception:
                bx = None
            if bax != ax:
                b_acts = True
            if bx is None or bax != bx:
                a_matters = True
            if bax != as_list(x):
                nontrivial = True
            pairs.append({"input": as_list(x), "output": bax})
        if not ok:
            continue
        stats["ran"] += 1
        stats["b_acts"] += int(b_acts)
        stats["a_matters"] += int(a_matters)
        stats["nontrivial"] += int(nontrivial)
        if b_acts and a_matters and nontrivial:
            stats["kept"] += 1
            # every kept pair, not the first forty. The loop is A-major, so a truncated list shows
            # one first stage repeatedly and hides whether the yield is concentrated in a handful of
            # A's, which is the difference between 1440 rules and 1440 variations on a few.
            kept.append({"a": a, "b": b, "n": len(pairs)})

    def pct(x):
        return 100.0 * x / stats["tried"] if stats["tried"] else 0.0

    print("\ncomposition survey over %d ordered pairs, %d instances each" % (stats["tried"], n_inst))
    print("  B runs on A's output              : %5d (%.1f%%)" % (stats["ran"], pct(stats["ran"])))
    print("  ... and B actually acts           : %5d (%.1f%%)" % (stats["b_acts"], pct(stats["b_acts"])))
    print("  ... and A is visible in the result: %5d (%.1f%%)" % (stats["a_matters"], pct(stats["a_matters"])))
    print("  ... and the pair is non-trivial   : %5d (%.1f%%)" % (stats["nontrivial"], pct(stats["nontrivial"])))
    print("  KEPT, all filters passed          : %5d (%.1f%%)" % (stats["kept"], pct(stats["kept"])))
    total_possible = len(bank) * (len(keys) - 1)
    print("\nordered pairs available in principle: %d" % total_possible)
    print("extrapolated usable compositions    : %d" % int(total_possible * stats["kept"] / max(stats["tried"], 1)))
    from collections import Counter
    das = sorted({k["a"] for k in kept})
    dbs = sorted({k["b"] for k in kept})
    ca = Counter(k["a"] for k in kept)
    print("")
    print("distinct first stages  : %d of %d tasks with a bank" % (len(das), len(bank)))
    print("distinct second stages : %d of %d verifiers" % (len(dbs), len(keys)))
    if ca:
        top = ca.most_common(5)
        share = 100.0 * sum(n for _, n in top) / max(len(kept), 1)
        print("most prolific first stages: %s" % ", ".join("%s x%d" % t for t in top))
        print("share of all compositions from the top 5 first stages: %.1f%%" % share)
    json.dump({"stats": stats, "kept": kept, "n_inst": n_inst,
               "distinct_a": len(das), "distinct_b": len(dbs),
               "total_possible": total_possible}, open(OUT, "w"), indent=1)

    print("\nVERDICT")
    if stats["kept"] == 0:
        print("  No pair survives. Verifiers are written for their own input distributions and do not")
        print("  chain, so composition needs generators built for it rather than reused parts.")
    elif stats["kept"] < 0.02 * stats["tried"]:
        print("  Only %.1f%% of pairs compose. That still extrapolates to a few thousand tasks, which" % pct(stats["kept"]))
        print("  is a corpus, but the yield is low enough that curation dominates the work.")
    else:
        print("  %.1f%% of pairs compose cleanly, extrapolating to roughly %d two-step tasks that"
              % (pct(stats["kept"]), int(total_possible * stats["kept"] / stats["tried"])))
        print("  neither stage produces alone. Building a compositional corpus is an afternoon of")
        print("  compute, not a research project, and the claim that it was out of reach was wrong.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 4) else 1)
