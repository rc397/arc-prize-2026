"""Are the composed tasks LEARNABLE, or merely consistent? The gate before any training run.

We can now build 1440 two-step rules by composing RE-ARC verifiers, each instantiable without limit.
Consistency is free, because both stages are deterministic functions. Learnability is not. A
composition is guaranteed well defined and is not guaranteed to admit a short description, and a
corpus that is consistent but arbitrary would teach a model noise rather than composition.

So run the deployed pipeline on them and see. Two arms, both drawn from RE-ARC so the grid
distribution is held fixed and only the rule structure differs:

  single    a task from one verifier, four demonstrations and one test. This is the control, and we
            already know roughly what it scores: the dual-expert run put the truth in the pool for
            18 of 23 fresh single-rule outputs.
  composed  a task from verify_B(verify_A(x)), same shape, same budget, sampled so that first stages
            differ across tasks rather than repeating the most prolific one.

Comparing composed tasks against ARC-AGI-2 evaluation tasks instead would confound rule structure
with grid distribution, which is why the control is RE-ARC and not ARC.

The reading, decided in advance:

  composed near single      the compositions are no harder than one rule, so they add instances but
                            not compositional difficulty, and are weak training data for the thing
                            we want taught.
  composed near zero        the compositions are arbitrary rather than hard. A rule the adapter
                            cannot fit from four demonstrations is not a lesson, it is noise, and
                            training on it would be worse than not.
  composed clearly between  the interesting case: harder than a single rule and still learnable,
                            which is what a corpus meant to teach composition should look like.

Usage: python rig_composelearn.py [N_TASKS=14] [DECODE_S=600]
Writes experiments/composelearn.jsonl, one record per (task, arm).
"""
import gc
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
REARC = os.path.join(BASE, "re-arc")
POOLS = os.path.join(BASE, "experiments", "rearc_pools", "tasks")
SURVEY = os.path.join(BASE, "experiments", "compose_survey.json")
OUT = os.path.join(BASE, "experiments", "composelearn.jsonl")
N_DEMO = 4


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


def as_list(g):
    return [list(r) for r in g]


def ok_grid(g):
    try:
        h, w = len(g), len(g[0])
    except Exception:
        return False
    return 1 <= h <= 30 and 1 <= w <= 30 and all(
        len(r) == w and all(isinstance(v, int) and 0 <= v <= 9 for v in r) for r in g)


def build_composed(vs, a, b, n):
    """n instances of verify_b(verify_a(x)) drawn from a's input pool."""
    p = os.path.join(POOLS, "%s.json" % a)
    if not os.path.isfile(p):
        return None
    ex = json.load(open(p, encoding="utf-8"))
    out = []
    for e in ex:
        try:
            y = as_list(vs[b](vs[a](e["input"])))
        except Exception:
            continue
        if not ok_grid(y):
            continue
        out.append({"input": as_list(e["input"]), "output": y})
        if len(out) == n:
            return out
    return None


def build_single(a, n):
    p = os.path.join(POOLS, "%s.json" % a)
    if not os.path.isfile(p):
        return None
    ex = json.load(open(p, encoding="utf-8"))
    if len(ex) < n:
        return None
    return [{"input": as_list(e["input"]), "output": as_list(e["output"])} for e in ex[:n]]


def main(n_tasks, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    vs = load_verifiers()
    survey = json.load(open(SURVEY, encoding="utf-8"))
    kept = survey.get("kept") or survey.get("examples") or []
    assert kept, "no kept compositions in %s" % SURVEY

    # one composition per distinct first stage, so the sample is not dominated by the prolific ones
    rng = random.Random(17)
    by_a = {}
    for k in kept:
        by_a.setdefault(k["a"], []).append(k)
    firsts = sorted(by_a)
    rng.shuffle(firsts)
    picks = [rng.choice(by_a[a]) for a in firsts[:n_tasks]]
    print("%d compositions available over %d distinct first stages; sampling %d"
          % (len(kept), len(by_a), len(picks)), flush=True)

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    tmp = os.path.join(BASE, "_compose")
    os.makedirs(tmp, exist_ok=True)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8")}

    def run(name, pairs, arm, meta):
        if (name, arm) in done:
            print("skip %s|%s" % (name, arm), flush=True)
            return
        ch = {name: {"train": pairs[:N_DEMO], "test": [{"input": pairs[N_DEMO]["input"]}]}}
        path = os.path.join(tmp, "%s_%s.json" % (name, arm))
        json.dump(ch, open(path, "w"))
        arc = ArcDataset.from_file(path)
        pz = arc.change_keys([name])
        labels = {"%s_0" % name: np.array(pairs[N_DEMO]["output"])}

        m = P.build_model()
        t0 = time.time()
        tr = P.train_one(m, tok, collator, fmt,
                         pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
                             formatter=fmt, name="text", max_len=P.MAXSEQ),
                         deadline=t0 + P.TRAIN_BUDGET)
        train_s = time.time() - t0
        base = m.get_base_model()
        base.config.use_cache = True
        base.gradient_checkpointing_disable()
        m.eval()
        pool, trunc, _ = P.decode_pool(m, tok, fmt, lifted, pz, time.time() + decode_s)
        graded = P.grade(pool, labels, sorted(labels))
        rec = {"task": name, "arm": arm, "meta": meta,
               "final_loss": round(float(tr.get("train_loss_last10", 0.0)), 5),
               "cands": sum(len(v) for v in pool.values()),
               "hits": sum(1 for v in graded.values() if v["in_pool"]),
               "outputs": len(graded), "truncated": trunc,
               "train_s": round(train_s, 1)}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("%-22s %-9s loss %.5f  %4d cands  hit %d/%d%s"
              % (name, arm, rec["final_loss"], rec["cands"], rec["hits"], rec["outputs"],
                 "  TRUNC" if trunc else ""), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    for k in picks:
        a, b = k["a"], k["b"]
        comp = build_composed(vs, a, b, N_DEMO + 1)
        sing = build_single(a, N_DEMO + 1)
        if comp is None or sing is None:
            print("skip %s->%s, could not build %d instances" % (a, b, N_DEMO + 1), flush=True)
            continue
        run("single_%s" % a, sing, "single", {"a": a})
        run("comp_%s_%s" % (a, b), comp, "composed", {"a": a, "b": b})

    print("ALL DONE", flush=True)
    summarize()
    return True


def summarize():
    import statistics as st
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    if not rows:
        print("no rows")
        return
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    print("")
    print("%-10s %6s %8s %8s %10s %9s" % ("arm", "tasks", "hits", "outputs", "final_loss", "cands"))
    for a in ("single", "composed"):
        rs = arms.get(a, [])
        if not rs:
            continue
        print("%-10s %6d %8d %8d %10.5f %9d"
              % (a, len(rs), sum(r["hits"] for r in rs), sum(r["outputs"] for r in rs),
                 st.mean(r["final_loss"] for r in rs), sum(r["cands"] for r in rs)))

    s, c = arms.get("single", []), arms.get("composed", [])
    if len(s) < 6 or len(c) < 6:
        print("")
        print("NO VERDICT")
        print("  %d single and %d composed tasks. Too few to compare rates." % (len(s), len(c)))
        return
    if sum(r["cands"] for r in rows) == 0:
        print("")
        print("NO VERDICT")
        print("  Every arm decoded zero candidates. Budget or harness failure, not a result.")
        return
    sr = sum(r["hits"] for r in s) / max(sum(r["outputs"] for r in s), 1)
    cr = sum(r["hits"] for r in c) / max(sum(r["outputs"] for r in c), 1)
    print("")
    print("truth-in-pool rate: single %.2f, composed %.2f" % (sr, cr))
    print("")
    print("VERDICT")
    if cr >= sr - 0.05:
        print("  Composed tasks are about as easy as single rules (%.2f against %.2f). They add" % (cr, sr))
        print("  instances but not compositional difficulty, so as training data they mostly")
        print("  teach what one-rule data already teaches.")
    elif cr <= 0.05:
        print("  Composed tasks are essentially unsolvable (%.2f against %.2f). Consistent but" % (cr, sr))
        print("  arbitrary: a rule the adapter cannot fit from four demonstrations is noise, not a")
        print("  lesson, and training on this corpus would likely hurt.")
    else:
        print("  Composed tasks are harder than single rules and still learnable, %.2f against" % cr)
        print("  %.2f. That is what a corpus meant to teach composition should look like, and it" % sr)
        print("  is the case in which a training run is worth its compute.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 14,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 600) else 1)
