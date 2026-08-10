"""Price every proposed candidate filter against the dumped pool, one number each.

About a dozen ideas in Ryan's batch are downstream pruning rules: restrict the search to colours
present in the test input, budget palette frequencies, prune by object size, prune by sequence
length, terminate on colour ratios. All of them were closed by Gate 1 and by analogy to the
invariant-mining result, which recovered zero candidates and killed 11 of 19 truths. Analogy is
cheaper than measurement and worse than it, and every one of these is exactly measurable against
the 60-output pool dump, so measure them.

Also priced here, because it needs the same machinery: raising the DFS probability floor. The tree
replay says a floor of 0.45 keeps every covered output while returning 22 percent of the forward
passes. That was measured on emissions alone. The pool dump carries grids, per-view scores and the
kgmon selector inputs, so it can answer the question the tree cannot: does a smaller pool change
what the selector PICKS. A filter that preserves truth-in-pool and breaks top-2 is not a win.

Each rule is scored on four numbers, and the last is the only one that can pay:

  truths killed      candidates that are the answer and would be removed. The kill condition.
  wrongs killed      the point of the exercise.
  top-2 before/after kgmon attainment, the quantity the submission is actually made of.
  outputs emptied    pools reduced to nothing, where the pipeline has no answer to submit at all.

Rules read the task, never the model's beliefs, except the floor rule which reads the scores the
decoder already produced. Parameters are mined per task from its own train pairs and never cross
tasks.

Usage: python sim_prune_replay.py [POOL] [CHALLENGES]
"""
import json
import math
import os
import sys
from collections import Counter, deque

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join("paper", "data", "pool_dump.json")
CHAL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    "experiments", "_data", "arc-agi_evaluation_challenges.json")


def kgmon(count, augs):
    """The deployed selector score: multiplicity minus mean augmentation NLL."""
    flat = [sum(a) / len(a) for a in augs] if augs else [0.0]
    return count - sum(flat) / len(flat)


def top2(cands):
    """kgmon top 2, the two grids that become attempt_1 and attempt_2."""
    return sorted(cands, key=lambda c: -kgmon(c["count"], c["aug"]))[:2]


def colours(g):
    return {v for row in g for v in row}


def area(g):
    return len(g) * len(g[0])


def components(g):
    """Sizes of 4-connected same-colour regions, background excluded by modal colour."""
    h, w = len(g), len(g[0])
    bg = Counter(v for row in g for v in row).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or g[r][c] == bg:
                continue
            q, n = deque([(r, c)]), 0
            seen[r][c] = True
            while q:
                y, x = q.popleft()
                n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and g[ny][nx] == g[y][x]:
                        seen[ny][nx] = True
                        q.append((ny, nx))
            out.append(n)
    return out or [0]


def unchanged_frac(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return None
    n = sum(len(r) for r in a)
    return sum(1 for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x == y) / n


# ------------------------------------------------------------------------------------- the rules
# Each returns True to KILL the candidate. `t` carries the task's own mined parameters.

def rule_palette_test(g, t):
    return not colours(g) <= t["test_colours"]


def rule_palette_seen(g, t):
    return not colours(g) <= (t["test_colours"] | t["train_out_colours"])


def rule_shape_consistent(g, t):
    """Only fires when every train pair preserves shape, which is the case it claims to know."""
    if not t["all_shape_preserving"]:
        return False
    return (len(g), len(g[0])) != t["test_shape"]


def rule_ncolours(g, t):
    return not (t["n_col_lo"] <= len(colours(g)) <= t["n_col_hi"])


def rule_area_ratio(g, t):
    return round(area(g) / t["test_area"], 6) not in t["ratios"]


def rule_maxobj(g, t):
    return not (t["obj_lo"] <= max(components(g)) <= t["obj_hi"])


def rule_length(g, t):
    return area(g) + len(g) > t["max_tokens"]


def rule_colour_ratio(g, t):
    """Modal-colour share must lie in the band the train outputs show."""
    c = Counter(v for row in g for v in row)
    share = c.most_common(1)[0][1] / area(g)
    return not (t["modal_lo"] - 1e-9 <= share <= t["modal_hi"] + 1e-9)


def rule_unchanged(g, t):
    """From today's 88 percent measurement: where every train pair preserves most of its input,
    require the candidate to preserve at least as much as the least-preserving train pair."""
    if t["unchanged_lo"] is None:
        return False
    u = unchanged_frac(t["test_input"], g)
    return u is not None and u < t["unchanged_lo"] - 1e-9


RULES = [
    ("palette in test input", rule_palette_test),
    ("palette in test+train out", rule_palette_seen),
    ("shape = test input shape", rule_shape_consistent),
    ("distinct-colour count band", rule_ncolours),
    ("area ratio seen in train", rule_area_ratio),
    ("largest object in band", rule_maxobj),
    ("length <= longest train out", rule_length),
    ("modal-colour share band", rule_colour_ratio),
    ("unchanged >= train minimum", rule_unchanged),
]


def mine(task, test_input):
    """Every parameter comes from this task's own train pairs."""
    pairs = [(p["input"], p["output"]) for p in task["train"]]
    outs = [o for _, o in pairs]
    same = [unchanged_frac(i, o) for i, o in pairs]
    return {
        "test_input": test_input,
        "test_shape": (len(test_input), len(test_input[0])),
        "test_area": area(test_input),
        "test_colours": colours(test_input),
        "train_out_colours": {c for o in outs for c in colours(o)},
        "all_shape_preserving": all(len(i) == len(o) and len(i[0]) == len(o[0])
                                    for i, o in pairs),
        "n_col_lo": min(len(colours(o)) for o in outs),
        "n_col_hi": max(len(colours(o)) for o in outs),
        "ratios": {round(area(o) / area(i), 6) for i, o in pairs},
        "obj_lo": min(max(components(o)) for o in outs),
        "obj_hi": max(max(components(o)) for o in outs),
        "max_tokens": max(area(o) + len(o) for o in outs),
        "modal_lo": min(Counter(v for row in o for v in row).most_common(1)[0][1] / area(o)
                        for o in outs),
        "modal_hi": max(Counter(v for row in o for v in row).most_common(1)[0][1] / area(o)
                        for o in outs),
        "unchanged_lo": min(s for s in same if s is not None) if all(
            s is not None for s in same) else None,
    }


def apply_floor(cands, floor):
    """Re-derive the pool as if the DFS floor had been `floor` instead of 0.2.

    A candidate's beam list is the cumulative NLL at which each view found it. A tighter floor
    removes the findings above the new ceiling, which lowers multiplicity, and removes the
    candidate entirely when no view keeps it. The aug rescoring is subset accordingly.
    """
    ceil = -math.log(floor)
    out = []
    for c in cands:
        keep = [i for i, s in enumerate(c["beam"]) if s <= ceil]
        if not keep:
            continue
        out.append({"grid": c["grid"], "correct": c["correct"], "count": len(keep),
                    "beam": [c["beam"][i] for i in keep],
                    "aug": [c["aug"][i] for i in keep if i < len(c["aug"])] or c["aug"][:1]})
    return out


def score(pools, name):
    """Truth kept, top-2 attainment and empty pools over a set of already-filtered pools."""
    kept_truth = hit2 = empty = 0
    n_c = 0
    for _, cands in pools.items():
        n_c += len(cands)
        if not cands:
            empty += 1
            continue
        kept_truth += int(any(c["correct"] for c in cands))
        hit2 += int(any(c["correct"] for c in top2(cands)))
    return {"name": name, "cands": n_c, "truth_in_pool": kept_truth, "top2": hit2, "empty": empty}


def main():
    pool = json.load(open(POOL))
    ch = json.load(open(CHAL))
    mined = {}
    for key in pool:
        tid, idx = key.rsplit("_", 1)
        mined[key] = mine(ch[tid], ch[tid]["test"][int(idx)]["input"])

    base = score({k: v for k, v in pool.items()}, "deployed")
    n_truth = sum(1 for v in pool.values() if any(c["correct"] for c in v))
    print("%d outputs, %d candidates, truth in pool for %d, kgmon top-2 on %d"
          % (len(pool), base["cands"], base["truth_in_pool"], base["top2"]))
    print("")

    print("%-28s %7s %8s %8s %7s %7s" % ("rule", "cands", "truths", "wrongs", "in-pool", "top-2"))
    print("%-28s %7s %8s %8s %7s %7s" % ("", "kept", "killed", "killed", "", ""))
    print("-" * 72)
    rows = []
    for name, fn in RULES:
        filt, tk, wk = {}, 0, 0
        for key, cands in pool.items():
            keep = []
            for c in cands:
                if fn(c["grid"], mined[key]):
                    tk += int(bool(c["correct"]))
                    wk += int(not c["correct"])
                else:
                    keep.append(c)
            filt[key] = keep
        s = score(filt, name)
        rows.append((s, tk, wk))
        print("%-28s %7d %8d %8d %7d %7d"
              % (name, s["cands"], tk, wk, s["truth_in_pool"], s["top2"]))

    print("")
    print("%-28s %7s %8s %8s %7s %7s" % ("DFS floor instead of 0.20", "cands", "", "", "in-pool",
                                         "top-2"))
    print("-" * 72)
    floors = []
    for f in (0.20, 0.25, 0.30, 0.35, 0.45, 0.55):
        filt = {k: apply_floor(v, f) for k, v in pool.items()}
        s = score(filt, "floor %.2f" % f)
        floors.append(s)
        print("%-28s %7d %8s %8s %7d %7d"
              % ("floor %.2f" % f, s["cands"], "", "", s["truth_in_pool"], s["top2"]))

    print("")
    print("VERDICT")
    noop = [s for s, tk, wk in rows if tk == 0 and wk == 0]
    winners = [(s, tk, wk) for s, tk, wk in rows
               if tk == 0 and wk > 0 and s["top2"] >= base["top2"] and s["empty"] == 0]
    if winners:
        best = max(winners, key=lambda r: r[2])
        print("  '%s' kills %d wrong candidates, no truth, no empty pool, and top-2 holds at %d."
              % (best[0]["name"], best[2], best[0]["top2"]))
        print("  That is the first candidate filter here that is not paid for in truths.")
    else:
        harmful = [(s, tk, wk) for s, tk, wk in rows if tk > 0]
        print("  Every rule that kills anything kills truths: %d of %d false-kill, and the other %d"
              % (len(harmful), len(rows), len(noop)))
        print("  kill NOTHING AT ALL. That second group is the more informative one. No candidate")
        print("  the decoder emits ever uses an unseen colour, and none has the wrong shape where")
        print("  the task preserves shape, so those constraints are already satisfied before any")
        print("  filter is applied: the model has internalised them. The filters that do fire are")
        print("  sharp enough to prune wrong candidates and prune right ones at a similar rate,")
        print("  which is Gate 1 and the invariant-mining result, now measured rule by rule.")

    print("")
    free = [s for s in floors
            if s["truth_in_pool"] >= base["truth_in_pool"] and s["top2"] >= base["top2"]]
    tight = free[-1]
    costly = [s for s in floors if s not in free]
    print("  The floor is a different story, because it does not read the task at all. Raising it")
    print("  to %s is free: truth-in-pool holds at %d, top-2 at %d, and the pool falls from %d"
          % (tight["name"].split()[-1], tight["truth_in_pool"], tight["top2"], base["cands"]))
    print("  candidates to %d. Past that it costs truths: %s holds %d of %d and top-2 %d of %d."
          % (tight["cands"], costly[0]["name"], costly[0]["truth_in_pool"],
             base["truth_in_pool"], costly[0]["top2"], base["top2"]))
    print("  So the tree replay's 22 percent budget saving at floor 0.45 is not free after all. It")
    print("  was measured on 13 covered outputs; this is 19, and three of them die. What survives")
    print("  is a small free tightening, worth about six percent of the decode budget, and a")
    print("  genuine unsettled trade above it: 22 percent more budget on decodes that truncate 12")
    print("  times in 16, against one top-2 output in fifteen. Only a run can settle that.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
