"""Symmetric two-expert selection, validated on fresh RE-ARC where neither model can be contaminated.

The ceiling theorem names a second independently trained model as one of two escapes from the
selection bound, and the novelty audit found nobody in the Kaggle track selects with two: NVARC's
published single-scorer arrangement missed the second generator's unique solves, which is the
baseline this design exists to beat. Our earlier validation of the 2B died because the public
evaluation set is memorised by that checkpoint, so this runs on fresh RE-ARC instances minted after
both checkpoints were published, where the audit showed the 2B is genuinely capable (65 percent)
and clean.

Per task: train the deployed per-task adapter on the 4B and decode its candidate pool, score every
unique candidate under a fixed augmentation set with the adapted 4B, then train the 2B on the same
task under the same seed and score the same candidates with it. Selectors compared on truth-in-top-2
against the pool's oracle:

  kgmon-4B        multiplicity minus mean 4B augmentation NLL (the deployed selector)
  kgmon-2B        the same statistic under the second expert alone
  sum-experts     multiplicity minus the mean of both experts' augmentation NLLs
  rank-fusion     sum of the candidate's ranks under kgmon-4B and kgmon-2B

Usage: python rig_dualexpert.py [N_TASKS=24]   writes experiments/dualexpert.jsonl and a summary
"""
import gc
import json
import os
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
FRESH = os.path.join(BASE, "experiments", "fresh_rearc.json")
OUT = os.path.join(BASE, "experiments", "dualexpert.jsonl")
MODEL_2B = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                        "qwen3_2b_grids15_sft141", "transformers", "bfloat16", "1")
N_SCORE_AUGS = 8
DECODE_BUDGET = int(os.environ.get("ARC_DUAL_DECODE_BUDGET", "600"))


def main(n_tasks):
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    fresh = json.load(open(FRESH))
    tasks = fresh["tasks"]
    keys = sorted(tasks)[:n_tasks]
    challenges = {k: {"train": tasks[k]["train"], "test": tasks[k]["test"]} for k in keys}
    solutions = {k: tasks[k]["solution"] for k in keys}
    tmp = os.path.join(BASE, "_dual_challenges.json")
    json.dump(challenges, open(tmp, "w"))
    arc = ArcDataset.from_file(tmp)

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    enc = P.encoder_for(tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)

    done = set()
    if os.path.isfile(OUT):
        done = {json.loads(l)["task"] for l in open(OUT, encoding="utf-8")}
    print("dual-expert on %d fresh tasks (%d already done), decode budget %ds"
          % (len(keys), len(done), DECODE_BUDGET), flush=True)

    def grid_str(g):
        return "\n".join("".join(str(int(v)) for v in row) for row in g)

    @torch.no_grad()
    def aug_nlls(model, pz_multi, bk, grid):
        """Mean answer NLL of grid over a fixed augmentation set of the task, one value per aug."""
        ds = ArcDataset(keys=[bk], queries={bk: pz_multi.queries.get(bk)},
                        replies={bk: [np.asarray(grid).tolist()]})
        ds = ds.augment(seed=abs(hash(bk)) % 1024 ** 2)
        ds = ds.cut_to_len(formatter=fmt, name="input", max_len=P.MAXSEQ - fmt.max_new_tokens())
        vals = []
        for s in ds.as_list(fmt)[:N_SCORE_AUGS]:
            pre, ans = enc(s["input"]), enc(s["reply"])
            ids = torch.tensor([pre + ans], device="cuda")
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                logits = model(input_ids=ids).logits[0].float()
            lp = torch.log_softmax(logits[len(pre) - 1:-1], dim=-1)
            tgt = torch.tensor(ans, device="cuda")
            vals.append(float(-lp[torch.arange(len(ans)), tgt].mean()))
        return vals

    def train_on(model_dir, pz):
        old = P.MODEL_DIR
        P.MODEL_DIR = model_dir
        try:
            m = P.build_model()
        finally:
            P.MODEL_DIR = old
        ds = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        t0 = time.time()
        P.train_one(m, tok, collator, fmt, ds, deadline=t0 + P.TRAIN_BUDGET)
        base = m.get_base_model()
        base.config.use_cache = True
        base.gradient_checkpointing_disable()
        m.eval()
        return m, time.time() - t0

    f = open(OUT, "a", encoding="utf-8")
    for key in keys:
        if key in done:
            continue
        pz = arc.change_keys([key])
        multi = pz.split_multi_replies()
        labels = {"%s_%d" % (key, i): np.array(g) for i, g in enumerate(solutions[key])}

        m4, t4 = train_on(P.MODEL_DIR, pz)
        pool, trunc, _ = P.decode_pool(m4, tok, fmt, lifted, pz, time.time() + DECODE_BUDGET)

        uniq = {}
        for bk, cands in pool.items():
            for c in cands:
                gkey = (bk, tuple(map(tuple, np.asarray(c["solution"]).tolist())))
                u = uniq.setdefault(gkey, {"bk": bk, "grid": np.asarray(c["solution"]).tolist(),
                                           "mult": 0, "beam": []})
                u["mult"] += 1
                u["beam"].append(float(c["beam_score"]))
        cands = list(uniq.values())
        with torch.inference_mode():
            for c in cands:
                c["aug4"] = aug_nlls(m4, multi, c["bk"], c["grid"])
        del m4
        gc.collect()
        torch.cuda.empty_cache()

        m2, t2 = train_on(MODEL_2B, pz)
        with torch.inference_mode():
            for c in cands:
                c["aug2"] = aug_nlls(m2, multi, c["bk"], c["grid"])
        del m2
        gc.collect()
        torch.cuda.empty_cache()

        for c in cands:
            lab = labels.get(c["bk"])
            c["correct"] = bool(lab is not None and np.array_equal(np.array(c["grid"]), lab))
        f.write(json.dumps({"task": key, "train4_s": round(t4, 1), "train2_s": round(t2, 1),
                            "truncated": trunc, "cands": cands}) + "\n")
        f.flush()
        n_true = sum(c["correct"] for c in cands)
        print("%s: %d unique cands, truth in pool %s (4B %.0fs, 2B %.0fs)%s"
              % (key, len(cands), "yes" if n_true else "no", t4, t2,
                 " truncated" if trunc else ""), flush=True)
    f.close()
    summarize()
    return True


def summarize():
    import numpy as np
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]

    def mean(v):
        return float(np.mean(v)) if v else 0.0

    sels = {
        "kgmon-4B": lambda c: c["mult"] - mean(c["aug4"]),
        "kgmon-2B": lambda c: c["mult"] - mean(c["aug2"]),
        "sum-experts": lambda c: c["mult"] - 0.5 * (mean(c["aug4"]) + mean(c["aug2"])),
    }
    per_output = {}
    for r in rows:
        by_bk = {}
        for c in r["cands"]:
            by_bk.setdefault(c["bk"], []).append(c)
        for bk, cs in by_bk.items():
            per_output[bk] = cs

    oracle = sum(1 for cs in per_output.values() if any(c["correct"] for c in cs))
    print("\n%d fresh outputs with a non-empty pool, oracle (truth in pool) = %d"
          % (len(per_output), oracle))
    print("%-14s %8s" % ("selector", "top-2"))
    print("-" * 26)
    results = {}
    for name, fsel in sels.items():
        hit = 0
        for cs in per_output.values():
            top2 = sorted(cs, key=fsel, reverse=True)[:2]
            hit += int(any(c["correct"] for c in top2))
        results[name] = hit
        print("%-14s %8d" % (name, hit))
    # rank fusion needs both orderings
    hit = 0
    for cs in per_output.values():
        r4 = {id(c): i for i, c in enumerate(sorted(cs, key=sels["kgmon-4B"], reverse=True))}
        r2 = {id(c): i for i, c in enumerate(sorted(cs, key=sels["kgmon-2B"], reverse=True))}
        top2 = sorted(cs, key=lambda c: r4[id(c)] + r2[id(c)])[:2]
        hit += int(any(c["correct"] for c in top2))
    results["rank-fusion"] = hit
    print("%-14s %8d" % ("rank-fusion", hit))

    base = results["kgmon-4B"]
    best = max(results.items(), key=lambda kv: kv[1])
    print("\nVERDICT")
    print("  oracle gap for the deployed selector: %d of %d" % (oracle - base, oracle))
    if best[1] > base:
        print("  %r closes %d of that gap on fresh, uncontaminated tasks. The escape route the"
              % (best[0], best[1] - base))
        print("  theorem names is live under a second expert, and this is the validated basis for")
        print("  a hidden-set submission (the hidden set is equally clean for both models).")
    else:
        print("  No fusion beats the deployed single-expert selector on fresh tasks. The second")
        print("  belief state agrees with the first where it matters, which closes the last named")
        print("  escape empirically and completes the ceiling story for the paper.")
    json.dump({"oracle": oracle, "n_outputs": len(per_output), "results": results},
              open(os.path.join(BASE, "experiments", "dualexpert_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 24) else 1)
