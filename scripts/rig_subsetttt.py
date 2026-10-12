"""Subset-disagreement TTT: how many ARC tasks are underdetermined by their own demonstrations?

POSITIONING, decided after the prior-art check in archive/notes/subset-ttt-priorart.md, and it changes what
this script is for. The ensembling mechanism is old: query-by-bagging named committee-from-subsets
in 1998, and Lakshminarayanan et al. found bagging deteriorated deep ensembles, which is the default
prior any reviewer holds. ironbar already ablated single ARC adapters at n-1 and n-2 demonstrations
and concluded it is "just removing input information". Claiming a new ensembling trick would be
claiming something thick with prior art and with a published negative pointing the wrong way.

What is genuinely unclaimed is the MEASUREMENT. No published work quantifies what fraction of ARC
tasks admit more than one rule consistent with their demonstrations. ARC Prize's pass@2 exists
partly to absorb it, ARC-TGI names it precisely and designs it away in generated tasks, human
divergence data is consistent with it, and nobody has put a number on the official sets. So the
headline output here is that number, and the coverage union is the secondary test.


The measurement that motivates this is in the paper and is not ambiguous. Final training loss does
not separate solved tasks from unsolved ones, the ranges overlap across almost their whole extent,
so on the 65 percent of tasks that never produce a correct grid the adapter has still fitted the
demonstrations. It is not failing to learn. It is learning a rule consistent with two to five
examples that is not the rule the test pair wants. The failure is hypothesis underdetermination at
training time, and no part of the deployed pipeline treats it as such.

So perturb the evidence rather than the cosmetics. For a task with n demonstrations, adapter i is
trained with demonstration i removed from the training set entirely, not merely held out of the
few-shot context, which the deployed loss already does. On this data volume that is a 20 to 50
percent perturbation, unlike bagging where subsets are near-identical, and each adapter should
settle on a different consistent hypothesis.

Two things are measured, and they are independent.

  COVERAGE. Does the union of the subset pools contain truths that the full-data adapter's pool
  misses? This attacks the 65 percent, which is where the score actually lives. The comparison that
  matters is union-over-subsets against the full adapter, not against the best subset.

  DISAGREEMENT. Do the subset adapters agree with each other on the test input? Agreement is a
  statement about whether the demonstrations determine the rule. It is derived from training on
  different evidence rather than from one adapted model's likelihoods, which is why Appendix F's
  ceiling theorem does not bound it.

The design copies the restart probe's discipline because that probe's null is the reason this one
exists. It runs the FULL adapter as arm zero on the same task in the same process, so the coverage
comparison is paired. It records per-arm pools so union, intersection and pairwise agreement are all
recoverable offline. And it carries the same replicate control: arm zero is run twice on a subset of
tasks, because a hard probability floor over bf16 logits means two runs of identical training need
not produce identical pools, and without subtracting that noise "union exceeds single" would look
positive whether or not the subsets decorrelate. That is exactly the error the colour-permutation
probe was built to avoid, and its null licensed only "colour permutation restarts do not help".

The objection to answer, and it comes from inside our own method: the deployed augmentation set
already includes example reordering, so the pipeline already ensembles over permutations of the
demonstration set. Reordering preserves the sufficient statistics of the evidence and therefore
probes decoder sensitivity. Deletion changes what is identifiable and therefore probes the
specification. That is an argument until the union shows LOO adapters producing correct grids the
full-evidence adapter never produces, which is exactly what the coverage arm measures.

Usage: python rig_subsetttt.py [N_TASKS=12] [REPLICATE_EVERY=4]
Writes experiments/subsetttt.jsonl, one record per (task, arm).
"""
import gc
import json
import os
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
OUT = os.path.join(BASE, "experiments", "subsetttt.jsonl")
MAX_ARMS = int(os.environ.get("ARC_SUBSET_MAX_ARMS", "4"))
DECODE_BUDGET = int(os.environ.get("ARC_SUBSET_DECODE_BUDGET", "420"))


def main(n_tasks, replicate_every):
    import numpy as np
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    challenges = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"])
                for l in open(OUT, encoding="utf-8")}
    print("subset-disagreement TTT on %d tasks, arms capped at %d, decode %ds"
          % (len(keys), MAX_ARMS, DECODE_BUDGET), flush=True)

    def run_arm(task, arm, drop_idx, replicate=False):
        """arm 0 is the full-data control. arm i>0 drops demonstration i-1 from training."""
        tag = "%s|%d%s" % (task, arm, "r" if replicate else "")
        if (task, tag) in done:
            print("skip %s" % tag, flush=True)
            return
        q = challenges[task]
        train_pairs = q["train"]
        if drop_idx is not None:
            train_pairs = [p for j, p in enumerate(q["train"]) if j != drop_idx]
        # the TEST pair is untouched; only the training evidence changes
        sub = {task: {"train": train_pairs, "test": q["test"]}}
        tmp = os.path.join(BASE, "_subset_%s.json" % task)
        json.dump(sub, open(tmp, "w"))
        arc = ArcDataset.from_file(tmp)
        pz = arc.change_keys([task])
        labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}

        m = P.build_model()
        ds = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        t0 = time.time()
        stats = P.train_one(m, tok, collator, fmt, ds, deadline=t0 + P.TRAIN_BUDGET)
        train_s = time.time() - t0

        t1 = time.time()
        pool, trunc, overflow = P.decode_pool(m, tok, fmt, lifted, pz, t1 + DECODE_BUDGET)
        graded = P.grade(pool, labels, sorted(labels))
        # keep the grids so union, intersection and agreement are all recoverable offline
        dump = {bk: [np.asarray(c["solution"]).tolist() for c in cands]
                for bk, cands in pool.items()}
        rec = {"task": task, "arm": tag, "arm_idx": arm, "dropped": drop_idx,
               "replicate": replicate, "n_demos": len(q["train"]),
               "n_train_used": len(train_pairs), "stratum": strat[task],
               "train_s": round(train_s, 1), "decode_s": round(time.time() - t1, 1),
               "truncated": trunc, "overflow": overflow,
               "train_loss": stats if isinstance(stats, (int, float)) else None,
               "graded": {k: {"in_pool": v["in_pool"], "n_cands": v["n_cands"]}
                          for k, v in graded.items()},
               "pool": dump}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        hit = sum(1 for v in graded.values() if v["in_pool"])
        print("%-14s demos %d/%d  %d cands  truth-in-pool %d/%d  (%.0fs train, %.0fs decode)%s"
              % (tag, len(train_pairs), len(q["train"]),
                 sum(len(v) for v in dump.values()), hit, len(graded),
                 train_s, time.time() - t1, " TRUNC" if trunc else ""), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        os.remove(tmp)

    for ti, task in enumerate(keys):
        n_demos = len(challenges[task]["train"])
        run_arm(task, 0, None)
        for i in range(min(n_demos, MAX_ARMS)):
            run_arm(task, i + 1, i)
        if replicate_every and ti % replicate_every == 0:
            run_arm(task, 0, None, replicate=True)
    print("ALL ARMS DONE", flush=True)
    summarize()
    return True


def summarize():
    import numpy as np
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    def pool_set(rec, bk):
        return {tuple(map(tuple, g)) for g in rec["pool"].get(bk, [])}

    full_hits = subset_union_hits = combined_hits = 0
    n_outputs = 0
    agree_rows, rep_rows = [], []
    for task, recs in sorted(by_task.items()):
        full = next((r for r in recs if r["arm_idx"] == 0 and not r["replicate"]), None)
        rep = next((r for r in recs if r["arm_idx"] == 0 and r["replicate"]), None)
        subs = [r for r in recs if r["arm_idx"] > 0]
        if full is None or not subs:
            continue
        for bk, g in full["graded"].items():
            n_outputs += 1
            full_hits += int(g["in_pool"])
            su = any(r["graded"].get(bk, {}).get("in_pool") for r in subs)
            subset_union_hits += int(su)
            combined_hits += int(g["in_pool"] or su)
            # pairwise Jaccard between subset pools, the disagreement signal
            sets = [pool_set(r, bk) for r in subs]
            js = []
            for a in range(len(sets)):
                for b in range(a + 1, len(sets)):
                    u = len(sets[a] | sets[b])
                    if u:
                        js.append(len(sets[a] & sets[b]) / u)
            if js:
                agree_rows.append((bk, float(np.mean(js)), bool(g["in_pool"])))
            if rep is not None:
                fs, rs = pool_set(full, bk), pool_set(rep, bk)
                u = len(fs | rs)
                if u:
                    rep_rows.append(len(fs & rs) / u)

    print("\noutputs compared: %d" % n_outputs)
    print("truth in pool, FULL adapter        : %d" % full_hits)
    print("truth in pool, UNION of subsets    : %d" % subset_union_hits)
    print("truth in pool, full OR subsets     : %d" % combined_hits)
    gain = combined_hits - full_hits
    print("coverage gain from subsets         : %+d" % gain)

    if rep_rows:
        print("\nreplicate control: same-config pool Jaccard %.3f over %d outputs"
              % (float(np.mean(rep_rows)), len(rep_rows)))
    if agree_rows:
        hi = [a for _, a, t in agree_rows if t]
        lo = [a for _, a, t in agree_rows if not t]
        print("subset agreement (mean pairwise Jaccard):")
        print("  where the full adapter FOUND the truth : %.3f (n=%d)"
              % (float(np.mean(hi)) if hi else float("nan"), len(hi)))
        print("  where it did NOT                       : %.3f (n=%d)"
              % (float(np.mean(lo)) if lo else float("nan"), len(lo)))

    print("\nVERDICT")
    if gain >= 2:
        print("  Subset adapters recover %d output(s) the full adapter misses. Hypothesis" % gain)
        print("  perturbation reaches truths that cosmetic augmentation restarts did not, which is")
        print("  a coverage lever and the first thing in this project to attack the 65 percent.")
    elif gain == 1:
        print("  One output recovered. That is the noise floor, so it is a hint and not a result.")
        print("  Widen the task count before spending a submission on it.")
    else:
        print("  No coverage gain. Removing a demonstration does not move the adapter to a")
        print("  different reachable hypothesis, so underdetermination is not repairable by")
        print("  re-training on subsets, and the 65 percent stays a property of the base model.")
    if agree_rows and hi and lo:
        d = float(np.mean(lo)) - float(np.mean(hi))
        print("  Agreement is %.3f %s on outputs the full adapter misses, so subset disagreement %s"
              % (abs(d), "lower" if d < 0 else "higher",
                 "does flag" if abs(d) > 0.08 else "does not flag"))
        print("  the tasks that fail, and %s a routing signal."
              % ("is" if abs(d) > 0.08 else "is not"))
    json.dump({"n_outputs": n_outputs, "full": full_hits, "subset_union": subset_union_hits,
               "combined": combined_hits, "gain": gain},
              open(os.path.join(BASE, "experiments", "subsetttt_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 4) else 1)
