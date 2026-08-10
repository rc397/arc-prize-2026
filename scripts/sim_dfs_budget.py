"""Where does the decoder's budget actually go, and what would a depth-decayed threshold buy?

Ryan's proposal: decay the cumulative log-probability threshold as tree depth increases, so the
search admits deep subtrees a flat floor prunes. It is one of the few decode ideas that changes
WHAT IS IN THE POOL rather than the order candidates are found in, which is why best-first ordering
was null and this might not be.

Half of it is measurable offline and half is not, and the difference is the whole point:

  TIGHTENING with depth is exact. The logged tree contains every node the flat floor admitted, and
  cumulative NLL only grows along a path, so for any non-increasing schedule tau(d) a node survives
  iff its own score passes at its own depth: every ancestor has smaller score and larger tau, so
  ancestors pass automatically. The surviving tree, the surviving candidates and the surviving
  forward passes are all computable exactly.

  LOOSENING with depth is not measurable at all. Nodes the flat floor rejected were never expanded
  and never logged, and a node record carries only the token taken, not the distribution it came
  from, so there is no way to know what sat just outside. That direction needs GPU.

Which leaves a question worth more than the original one. Truncation is real: the logged runs record
whether the decode hit its wall clock, and a truncated decode is candidates never generated. If the
threshold can be tightened with no truth lost, the freed forward passes turn directly into coverage.
So this measures the exchange rate: forward passes saved against truths lost, over a family of
schedules that includes the flat floor itself and the pure depth cap that "branch caps" and
"sequence-length pruning" amount to.

Cost accounting is per FORWARD PASS, not per node. turbo_dfs is batched: one pass advances every
lane at once, so pruning one lane's subtree saves nothing unless every lane at that step is pruned.
Lane exploration is independent, so the surviving pass count is exactly the number of logged steps
at which at least one lane survives. Both numbers are reported, because the per-lane count is what
an unbatched decoder would save and it is the more flattering of the two.

Usage: python sim_dfs_budget.py [LOGDIR=experiments/dfslog]
"""
import json
import math
import os
import sys

PAD_SCORE = 1000.0                 # the sentinel turbo_dfs assigns to an exhausted lane
DFS_FLOOR = 0.2                    # deployed cumulative probability floor
MAX_SCORE = -math.log(DFS_FLOOR)   # 1.6094 nats


def schedules():
    """(name, tau(depth)). Every one is non-increasing in depth, so the replay stays exact."""
    out = [("flat 0.20 (deployed)", lambda d: MAX_SCORE)]
    for floor in (0.25, 0.30, 0.35, 0.45):
        out.append(("flat %.2f" % floor, (lambda f: lambda d: -math.log(f))(floor)))
    for half in (200, 100, 50):
        out.append(("halve every %d tok" % half,
                    (lambda h: lambda d: MAX_SCORE * 0.5 ** (d / h))(half)))
    for cap in (600, 400, 200):
        out.append(("depth cap %d" % cap,
                    (lambda c: lambda d: MAX_SCORE if d <= c else -1.0)(cap)))
    return out


def load(logdir):
    """One entry per logged batch, plus the per-task meta."""
    batches, metas = [], {}
    for name in sorted(os.listdir(logdir)):
        if not name.endswith(".jsonl"):
            continue
        for line in open(os.path.join(logdir, name), encoding="utf-8"):
            j = json.loads(line)
            if j.get("meta"):
                metas[j["task"]] = j
            else:
                batches.append(j)
    return batches, metas


def emission_depths(b):
    """Depth and correctness of every completed candidate in a batch.

    Emissions are matched to graded candidates by (lane, score) in emission order, which is how the
    original replay did it and is the only key the log carries.
    """
    open_c = {}
    for c in b["cands"]:
        open_c.setdefault((c["lane"], round(c["score"], 6)), []).append(c)
    out = []
    for rec in b["records"]:
        if rec[0] != "E":
            continue
        _, pos, lane, score = rec
        hits = open_c.get((lane, round(score, 6)))
        c = hits.pop(0) if hits else None
        out.append({"lane": lane, "depth": pos - b["prompt_lens"][lane], "score": score,
                    "correct": c["correct"] if c else None, "bk": c["bk"] if c else None})
    return out


def waste(b):
    """Lane-nodes that are not an ancestor of any emission: search that produced nothing.

    Parents come from the same stack rule the deployed replay uses. A node at position p in a lane
    pops that lane's stack until the top is shallower, and the top is then its parent.
    """
    nodes, stacks = [], {}
    productive = set()
    for rec in b["records"]:
        if rec[0] == "N":
            _, pos, toks, scores = rec
            for lane, s in enumerate(scores):
                if s >= PAD_SCORE:
                    continue
                st = stacks.setdefault(lane, [])
                while st and nodes[st[-1]]["pos"] >= pos:
                    st.pop()
                nodes.append({"pos": pos, "parent": st[-1] if st else None})
                st.append(len(nodes) - 1)
        else:
            _, pos, lane, _score = rec
            st = stacks.get(lane)
            i = st[-1] if st else None
            while i is not None:
                if i in productive:
                    break
                productive.add(i)
                i = nodes[i]["parent"]
    return len(nodes), len(nodes) - len(productive)


def main(logdir):
    batches, metas = load(logdir)
    if not batches:
        print("no logs under %s" % logdir)
        return False

    tasks = sorted(metas)
    truncated = [t for t in tasks if metas[t].get("truncated")]

    lane_nodes = passes = 0
    dead_nodes = 0
    truths, wrongs = [], []
    for b in batches:
        passes += sum(1 for r in b["records"] if r[0] == "N")
        n, w = waste(b)
        lane_nodes += n
        dead_nodes += w
        for e in emission_depths(b):
            (truths if e["correct"] else wrongs).append(e)

    print("%d tasks, %d batches, %d forward passes, %d lane-nodes"
          % (len(tasks), len(batches), passes, lane_nodes))
    print("decodes that hit the wall clock: %d of %d" % (len(truncated), len(tasks)))
    print("lane-nodes in subtrees that emitted nothing: %d (%.1f%%)"
          % (dead_nodes, 100.0 * dead_nodes / max(1, lane_nodes)))
    print("")

    print("%d correct emissions, %d incorrect" % (len(truths), len(wrongs)))
    if truths:
        def q(a, p):
            a = sorted(a)
            return a[int(p * (len(a) - 1))]
        head = [MAX_SCORE - t["score"] for t in truths]
        hw = [MAX_SCORE - w["score"] for w in wrongs]
        print("headroom under the 1.609 ceiling, nats  (larger = further from being pruned)")
        print("  truths   p10 %.3f  median %.3f  p90 %.3f" % (q(head, .1), q(head, .5), q(head, .9)))
        print("  wrongs   p10 %.3f  median %.3f  p90 %.3f" % (q(hw, .1), q(hw, .5), q(hw, .9)))
        print("depth at emission")
        print("  truths   p10 %d  median %d  p90 %d"
              % (q([t["depth"] for t in truths], .1), q([t["depth"] for t in truths], .5),
                 q([t["depth"] for t in truths], .9)))
        print("  wrongs   p10 %d  median %d  p90 %d"
              % (q([w["depth"] for w in wrongs], .1), q([w["depth"] for w in wrongs], .5),
                 q([w["depth"] for w in wrongs], .9)))
    print("")

    # coverage unit: an OUTPUT is covered if any one of its correct emissions survives, since the
    # task scores on the pool containing the truth at all, not on how many views found it
    covered_outputs = {t["bk"] for t in truths if t["bk"]}
    print("outputs with the truth in the pool: %d" % len(covered_outputs))
    print("")

    print("%-22s %8s %8s %9s %8s %8s" % ("schedule", "passes", "nodes", "outputs", "truths",
                                         "wrongs"))
    print("-" * 68)
    rows = []
    for name, tau in schedules():
        kept_passes = kept_nodes = 0
        for b in batches:
            for rec in b["records"]:
                if rec[0] != "N":
                    continue
                _, pos, _toks, scores = rec
                alive = 0
                for lane, s in enumerate(scores):
                    if s >= PAD_SCORE:
                        continue
                    if s <= tau(pos - b["prompt_lens"][lane]):
                        alive += 1
                kept_nodes += alive
                kept_passes += int(alive > 0)
        surviving = [t for t in truths if t["score"] <= tau(t["depth"])]
        kt = len(surviving)
        ko = len({t["bk"] for t in surviving if t["bk"]})
        kw = sum(1 for w in wrongs if w["score"] <= tau(w["depth"]))
        rows.append((name, kept_passes, kept_nodes, ko, kt, kw))
        print("%-22s %7d%% %7d%% %5d/%-3d %8d %8d"
              % (name, round(100.0 * kept_passes / passes), round(100.0 * kept_nodes / lane_nodes),
                 ko, len(covered_outputs), kt, kw))

    print("")
    print("VERDICT")
    base = rows[0]
    free = [r for r in rows[1:] if r[3] == base[3] and r[1] < base[1]]
    if not truths:
        print("  No correct emission is present in these logs, so nothing here can price a")
        print("  threshold against truth loss. The budget numbers still stand on their own.")
    elif free:
        best = min(free, key=lambda r: r[1])
        saved = 100.0 * (1 - best[1] / passes)
        print("  '%s' keeps the truth for all %d covered outputs and costs %d%% of the"
              % (best[0], base[3], round(100.0 * best[1] / passes)))
        print("  forward passes, so %.0f%% of the decode budget comes back. It also drops %d of the"
              % (saved, base[5] - best[5]))
        print("  %d wrong candidates, which shrinks the pool the selector has to rank." % base[5])
        print("  %d of %d decodes here hit the wall clock, so the freed passes are candidates that"
              % (len(truncated), len(tasks)))
        print("  were never generated. That makes this a coverage lever, not only a speed one.")
    else:
        cheap = [r for r in rows[1:] if r[1] <= 0.85 * base[1]]
        if cheap:
            best = max(cheap, key=lambda r: (r[3], -r[1]))
            print("  Every schedule that saves real budget also costs coverage. The best of them,")
            print("  '%s', keeps %d of %d covered outputs for %d%% of the passes."
                  % (best[0], best[3], base[3], round(100.0 * best[1] / passes)))
        else:
            print("  Nothing in this family separates cost from coverage. The flat floor is not")
            print("  obviously mis-set in the tightening direction.")
    print("")
    print("  Not measurable here: any schedule that LOOSENS with depth. Those nodes were never")
    print("  expanded, so they are not in the log. That arm needs the GPU.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1
                       else os.path.join("experiments", "dfslog")) else 1)
