"""Perceptual pre-fitting: train the adapter on the test input grid before it learns the rule.

The gap this exploits is verified rather than assumed. Our TTT training text is
fmt_train(queries['train'], last_is_challenge=True), which is demonstration pairs only. The test
input grid never appears as a loss target anywhere in training; it enters solely as a decode-time
prompt. It is the largest piece of task-specific information we hold and we discard all of it
before a single gradient step.

CompressARC is the reason to think that is expensive. It reaches roughly 20 percent on ARC-1 by
gradient-fitting a randomly initialised network per task at test time to compress the task's own
grids, with no pretraining at all. That is a floor on how much signal lives in per-task grid
structure alone, extracted with no transformation supervision whatsoever. ARC Prize's own analyses
put perception rather than reasoning among the live bottlenecks, and VARC reached about 60 percent
on ARC-1 by treating the problem as vision. Three independent results say the grids carry signal we
throw away.

The intervention is a self-supervised objective on the test input, expressed inside the deployed
prompt format so the existing collator supervises it without modification. The collator masks
everything except assistant spans, so a sample is only trained on if the thing to be learned sits in
an assistant turn. Each pre-fit sample is therefore a user turn holding a corrupted copy of the test
input and an assistant turn holding the clean one, which makes the adapter denoise this instance's
grid grammar, its palette, its object sizes, its background, its symmetries.

Corruption is by cell dropout at a fixed rate to a sentinel colour drawn from the task's own palette
complement where one exists, so the model is never asked to invent a colour the task does not use.

Three arms, and the third is not optional.

  base        the deployed configuration, untouched.
  prefit      denoise THIS task's test input for k steps, then the identical demonstration objective.
  prefit_ctrl denoise ANOTHER task's test input for the same k steps, same objective, same data
              volume, then the identical demonstration objective.

Without the third arm the experiment cannot distinguish instance-specific grid fluency from the mere
fact that the prefit arm took k more gradient steps than the control did. That confound would make a
positive result unattributable, and an unattributable positive is how three of this project's
retracted explanations began. The reading is: prefit above base with prefit_ctrl at base means the
effect is fluency in the grid we are actually being asked about. prefit and prefit_ctrl both above
base means it is optimisation, not perception, and the honest lever is simply more steps. Neither
above base means the route is closed.

All arms decode with the same budget and the same deployed decoder.

Honest note on what this cannot show: a single-card probe with a truncated decode measures pool
coverage, not the hidden score. A positive here is a reason to spend a submission, not a result.

Usage: python rig_prefit.py [N_TASKS=10] [PREFIT_STEPS=40]
Writes experiments/prefit.jsonl, one record per (task, arm).
"""
import gc
import json
import os
import random
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
OUT = os.path.join(BASE, "experiments", "prefit.jsonl")
DECODE_BUDGET = int(os.environ.get("ARC_PREFIT_DECODE_BUDGET", "420"))
DROP_RATE = float(os.environ.get("ARC_PREFIT_DROP", "0.25"))
N_VIEWS = int(os.environ.get("ARC_PREFIT_VIEWS", "16"))


def corrupt(grid, rate, rng, fill):
    out = [row[:] for row in grid]
    for r in range(len(out)):
        for c in range(len(out[0])):
            if rng.random() < rate:
                out[r][c] = fill
    return out


def prefit_texts(test_input, n, rate, seed):
    """Denoising samples in the deployed prompt format, corrupted in, clean out.

    The clean grid sits in the assistant span, which is the only region the collator supervises, so
    these produce real gradients without touching the collator.
    """
    from arc_loader import convert_grid_to_string
    rng = random.Random(seed)
    used = {v for row in test_input for v in row}
    fill = next((c for c in range(10) if c not in used), 0)
    out = []
    for _ in range(n):
        bad = corrupt(test_input, rate, rng, fill)
        out.append("<|im_start|>user\n" + convert_grid_to_string(bad)
                   + "<|im_end|><|im_start|>assistant\n"
                   + convert_grid_to_string(test_input) + "<|im_end|>")
    return out


def main(n_tasks, prefit_steps):
    import numpy as np
    import torch
    from datasets import Dataset

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    challenges = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8")}
    print("perceptual pre-fitting on %d tasks, %d pre-fit steps, drop %.2f, decode %ds"
          % (len(keys), prefit_steps, DROP_RATE, DECODE_BUDGET), flush=True)

    def run(task, arm):
        if (task, arm) in done:
            print("skip %s|%s" % (task, arm), flush=True)
            return
        pz = arc.change_keys([task])
        labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
        m = P.build_model()
        prefit_s = 0.0

        if arm in ("prefit", "prefit_ctrl"):
            if arm == "prefit":
                grid = challenges[task]["test"][0]["input"]
            else:
                # a different task's test input: same objective, same volume, wrong instance
                other = donor_for(task)
                grid = challenges[other]["test"][0]["input"]
            texts = prefit_texts(grid, N_VIEWS, DROP_RATE,
                                 seed=abs(hash(task)) % 10 ** 6)
            t0 = time.time()
            P.train_one(m, tok, collator, fmt, Dataset.from_list([{"text": t} for t in texts]),
                        deadline=t0 + 600, max_batches=prefit_steps)
            prefit_s = time.time() - t0
            print("  %s %s: %d steps on a %dx%d grid in %.0fs"
                  % (task, arm, prefit_steps, len(grid), len(grid[0]), prefit_s), flush=True)

        ds = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        t1 = time.time()
        P.train_one(m, tok, collator, fmt, ds, deadline=t1 + P.TRAIN_BUDGET)
        train_s = time.time() - t1

        t2 = time.time()
        pool, trunc, _ = P.decode_pool(m, tok, fmt, lifted, pz, t2 + DECODE_BUDGET)
        graded = P.grade(pool, labels, sorted(labels))
        rec = {"task": task, "arm": arm, "stratum": strat[task],
               "prefit_s": round(prefit_s, 1), "train_s": round(train_s, 1),
               "decode_s": round(time.time() - t2, 1), "truncated": trunc,
               "graded": {k: {"in_pool": v["in_pool"], "n_cands": v["n_cands"],
                              "selected_proxy": v["selected_proxy"]} for k, v in graded.items()},
               "pool": {bk: [np.asarray(c["solution"]).tolist() for c in cs]
                        for bk, cs in pool.items()}}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        hit = sum(1 for v in graded.values() if v["in_pool"])
        print("%-12s %-7s %4d cands  truth-in-pool %d/%d  (%.0fs)%s"
              % (task, arm, sum(len(v) for v in rec["pool"].values()), hit, len(graded),
                 prefit_s + train_s, " TRUNC" if trunc else ""), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    def donor_for(t):
        """A different task's test input, chosen deterministically and never the task itself."""
        pool = [x for x in sorted(challenges) if x != t]
        return pool[abs(hash("donor" + t)) % len(pool)]

    for task in keys:
        run(task, "base")
        run(task, "prefit")
        run(task, "prefit_ctrl")
    print("ALL ARMS DONE", flush=True)
    summarize()
    return True


def summarize():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["task"], {})[r["arm"]] = r
    hits = {"base": 0, "prefit": 0, "prefit_ctrl": 0}
    n = 0
    gained, lost = [], []
    for task, arms in sorted(by.items()):
        if not {"base", "prefit"} <= set(arms):
            continue
        for bk in arms["base"]["graded"]:
            b = arms["base"]["graded"][bk]["in_pool"]
            p = arms["prefit"]["graded"].get(bk, {}).get("in_pool", False)
            n += 1
            hits["base"] += b
            hits["prefit"] += p
            if "prefit_ctrl" in arms:
                hits["prefit_ctrl"] += arms["prefit_ctrl"]["graded"].get(bk, {}).get("in_pool", False)
            if p and not b:
                gained.append(bk)
            if b and not p:
                lost.append(bk)
    have_ctrl = any("prefit_ctrl" in a for a in by.values())
    print("\noutputs compared: %d" % n)
    print("truth in pool, base (deployed)          : %d" % hits["base"])
    print("truth in pool, prefit (this task's grid): %d" % hits["prefit"])
    if have_ctrl:
        print("truth in pool, prefit_ctrl (other grid) : %d" % hits["prefit_ctrl"])
    print("gained by pre-fitting: %d %s" % (len(gained), gained))
    print("lost   by pre-fitting: %d %s" % (len(lost), lost))

    net = hits["prefit"] - hits["base"]
    ctrl_net = hits["prefit_ctrl"] - hits["base"] if have_ctrl else None
    print("\nVERDICT")
    if not have_ctrl:
        print("  No control arm present, so any difference is unattributable between grid fluency")
        print("  and the extra gradient steps. Run prefit_ctrl before reading anything into this.")
    elif net >= 2 and ctrl_net <= 0:
        print("  Pre-fitting adds %d output(s) while the control arm adds %d. The gain follows the"
              % (net, ctrl_net))
        print("  GRID, not the step count: an adapter fluent in the instance it is about to be asked")
        print("  about reaches truths the deployed configuration never produces. Worth a slot.")
    elif net >= 2 and ctrl_net >= 2:
        print("  Both pre-fit arms gain (%d and %d), so the effect is extra optimisation rather than"
              % (net, ctrl_net))
        print("  perception. That is still a lever, and a cheaper one, but the interesting claim is")
        print("  dead and the honest write-up is 'more steps help', not 'grid fluency helps'.")
    elif net <= 0:
        print("  No coverage gain. Per-task grid fluency does not transfer into the transformation")
        print("  objective for this base model, so CompressARC's per-task signal is either already")
        print("  present from pretraining or not reachable by this route.")
    elif len(gained) and len(lost):
        print("  Pre-fitting trades outputs rather than adding them (%d gained, %d lost), the"
              % (len(gained), len(lost)))
        print("  signature of a perturbation rather than an improvement.")
    else:
        print("  One output gained, which is the noise floor. Widen the task count before believing it.")
    json.dump({"n": n, "hits": hits, "gained": gained, "lost": lost},
              open(os.path.join(BASE, "experiments", "prefit_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 10,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 40) else 1)
