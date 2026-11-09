"""Gate for the evolution route: does a program-writing baseline exist for evolution to multiply?

Evolution does not create capability, it amplifies it. The published result reports 12.1 to 34.0
percent on an open-weights model and 34.0 to 61.4 on a stronger one, so the multiplier is roughly
1.8 to 2.8 on whatever the baseline already is. If our offline-feasible coder's baseline is zero,
evolution has nothing to multiply and the route is closed for one card regardless of loop design.

Our existing negative does NOT answer this, and the reason matters, because it was
almost read as a closure. That probe ran 1616 programs and found 0 of 101 solved with 100 of 101
tasks failing to reproduce even one demonstration pair. But it differs from the question here in
three ways:

  1. it ran ONLY on the 101 tasks the 4B misses, the hardest slice by construction, whereas the
     published baselines are over a whole evaluation set. "0 on the hardest third" and "12 percent
     overall" are not in conflict.
  2. it scored EXACT reproduction of a demonstration pair. A program reproducing 85 percent of cells
     scores zero by that measure. Evolution climbs a partial-credit surface, so the relevant question
     is whether that surface has gradient, which nobody has measured.
  3. it drew K independent samples with no loop. No mutation of the best candidate, no execution
     feedback. The published claim is that the loop supplies the multiplier, not the sampling.

So this probe measures two things on the FULL public evaluation set, and neither has been measured:

  BASELINE   fraction of tasks where some sampled program reproduces every demonstration pair AND
             the held-out test output. This is the number evolution would multiply.
  GRADIENT   the per-cell fitness surface, scored against the identity function as the floor. If
             programs cluster at or below identity, there is nothing to climb and the loop cannot
             help however well it is engineered.

Kill condition, stated before the run. Baseline of zero AND a best-of-K per-cell lift over identity
that is not clearly positive on a meaningful share of tasks closes the single-card evolution route,
and we report it as a measured budget negative against the published positive. That is publishable:
it prices the route rather than gesturing at it.

Usage: python rig_codegate.py [N_TASKS=120] [K=8] [OUT]
Writes experiments/codegate.jsonl, one record per task.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_BASE", r"D:\arc")
# pyexec and coder_probe live at the rig's project root, not beside this file. Put both candidate
# locations on the path so this runs the same whether invoked from the root or from scripts/.
for _p in (HERE, BASE, os.path.dirname(HERE)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import pyexec
import coder_probe as C          # reuse the server, prompt and extraction that already work

OUT = os.path.join(BASE, "experiments", "codegate.jsonl")


def per_cell(pred, target):
    """Fraction of cells correct, and whether the shape matched.

    Shape mismatch scores zero rather than attempting an alignment, because any alignment rule
    invents a similarity the task never defined. Shape agreement is reported separately, since
    getting the output shape right is itself a rung on the ladder evolution would climb.
    """
    if pred is None:
        return 0.0, False
    if len(pred) != len(target) or any(len(a) != len(b) for a, b in zip(pred, target)):
        return 0.0, False
    tot = sum(len(r) for r in target)
    if not tot:
        return 0.0, True
    hit = sum(1 for a, b in zip(pred, target) for x, y in zip(a, b) if x == y)
    return hit / tot, True


def score_program(src, train_pairs):
    """Per-cell fitness over the demonstration pairs, plus exact-pair count."""
    ins = [p["input"] for p in train_pairs]
    outs = [p["output"] for p in train_pairs]
    got = pyexec.run_candidate(src, ins)
    if got is None:
        return {"ran": False, "cell": 0.0, "exact": 0, "shape": 0}
    cells, exact, shapes = [], 0, 0
    for g, t in zip(got, outs):
        c, sh = per_cell(g, t)
        cells.append(c)
        shapes += int(sh)
        exact += int(c == 1.0 and sh)
    return {"ran": True, "cell": sum(cells) / len(cells) if cells else 0.0,
            "exact": exact, "shape": shapes}


def matches_all(got, want):
    """Every produced grid equals its target exactly. A None anywhere fails the whole thing."""
    if got is None or len(got) != len(want):
        return False
    for a, b in zip(got, want):
        if a is None or len(a) != len(b):
            return False
        if [list(map(int, r)) for r in a] != [list(map(int, r)) for r in b]:
            return False
    return True


def identity_floor(train_pairs):
    """What the trivial copy-the-input program scores. The floor any real signal must clear."""
    return score_program("def solve(I):\n    return I\n", train_pairs)["cell"]


def main(n_tasks, k):
    challenges = json.load(open(C.CH))
    solutions = json.load(open(C.SOL))
    keys = sorted(challenges)[:n_tasks]

    done = set()
    if os.path.isfile(OUT):
        done = {json.loads(l)["task"] for l in open(OUT, encoding="utf-8")}
    C.ensure_server()
    print("code gate over %d tasks, K=%d, model %s" % (len(keys), k, os.path.basename(C.MODEL)),
          flush=True)

    for i, tid in enumerate(keys):
        if tid in done:
            continue
        task = challenges[tid]
        train_pairs = task["train"]
        floor = identity_floor(train_pairs)
        prompt, ctx_trimmed = C.prompt_for(task)   # returns (text, was_trimmed), not a bare string
        assert len(prompt) > 200, "prompt for %s is %d chars, the builder returned nothing usable" % (
            tid, len(prompt))

        best, progs, solved = None, 0, False
        for _ in range(k):
            text, _timings = C.chat(prompt)   # chat returns (text, timings), not a bare string
            for src in C.extract_programs(text or ""):
                progs += 1
                sc = score_program(src, train_pairs)
                sc["lift"] = sc["cell"] - floor
                if best is None or sc["cell"] > best["cell"]:
                    best = sc
                # a program reproducing every demonstration is a survivor; check the real test
                if sc["ran"] and sc["exact"] == len(train_pairs):
                    got = pyexec.run_candidate(src, [t["input"] for t in task["test"]])
                    if matches_all(got, solutions[tid]):
                        solved = True
        if best is None:
            best = {"ran": False, "cell": 0.0, "exact": 0, "shape": 0, "lift": -floor}

        rec = {"task": tid, "n_train": len(train_pairs), "programs": progs,
               "ctx_trimmed": ctx_trimmed,
               "identity_floor": round(floor, 4), "best_cell": round(best["cell"], 4),
               "best_lift": round(best["lift"], 4), "best_exact": best["exact"],
               "best_shape": best["shape"], "solved": solved}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("%3d/%d %-10s progs %2d  floor %.3f  best %.3f  lift %+.3f  exact %d/%d%s"
              % (i + 1, len(keys), tid, progs, floor, best["cell"], best["lift"],
                 best["exact"], len(train_pairs), "  SOLVED" if solved else ""), flush=True)

    summarize()
    return True


def summarize():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    n = len(rows)
    solved = sum(r["solved"] for r in rows)
    any_exact = sum(1 for r in rows if r["best_exact"] > 0)
    all_exact = sum(1 for r in rows if r["best_exact"] == r["n_train"])
    lifts = sorted(r["best_lift"] for r in rows)
    pos = sum(1 for x in lifts if x > 0.05)
    shape_any = sum(1 for r in rows if r["best_shape"] > 0)

    def pct(a):
        return 100.0 * a / n if n else 0.0

    print("\n%d tasks, %d programs" % (n, sum(r["programs"] for r in rows)))
    print("BASELINE  solved end to end            : %d (%.1f%%)" % (solved, pct(solved)))
    print("          best program reproduces ALL demos: %d (%.1f%%)" % (all_exact, pct(all_exact)))
    print("          best program reproduces ANY demo : %d (%.1f%%)" % (any_exact, pct(any_exact)))
    print("GRADIENT  best-of-K per-cell lift over identity")
    if lifts:
        q = [lifts[int(p * (len(lifts) - 1))] for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
        print("          min %+.3f  p25 %+.3f  median %+.3f  p75 %+.3f  max %+.3f" % tuple(q))
    print("          tasks with lift > 0.05        : %d (%.1f%%)" % (pos, pct(pos)))
    print("          best program matched a shape  : %d (%.1f%%)" % (shape_any, pct(shape_any)))

    # A verdict on a run that generated nothing is a statement about the harness, not the model, and
    # this script printed exactly that once before the prompt builder's return type was fixed.
    total_progs = sum(r["programs"] for r in rows)
    if total_progs < n:
        print("\nNO VERDICT")
        print("  Only %d programs across %d tasks. Fewer than one per task means generation or" % (
            total_progs, n))
        print("  extraction is broken, not that the model cannot solve them. Fix the harness and")
        print("  re-run before reading anything into these numbers.")
        return

    print("\nVERDICT")
    if solved == 0 and pos < 0.15 * n:
        print("  Baseline zero and the surface is flat. Evolution amplifies a baseline and there is")
        print("  none here, so the single-card evolution route is closed. Report as a measured")
        print("  budget negative against the published positive, which used models we cannot run.")
    elif solved == 0 and pos >= 0.15 * n:
        print("  Baseline zero but the surface has gradient on %.0f%% of tasks. Evolution has" % pct(pos))
        print("  something to climb even though sampling alone solves nothing, which is exactly the")
        print("  regime the published loop claims to exploit. Worth building the loop.")
    else:
        print("  Non-zero baseline (%d tasks). Evolution multiplies it, so build the loop and" % solved)
        print("  measure the multiplier against this run as the control.")
    json.dump({"n": n, "solved": solved, "all_exact": all_exact, "any_exact": any_exact,
               "lift_positive": pos},
              open(os.path.join(BASE, "experiments", "codegate_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 120,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 8) else 1)
