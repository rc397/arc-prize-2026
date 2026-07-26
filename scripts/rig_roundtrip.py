"""Score candidate grids by whether they behave like a real demonstration of the task's rule.

The selection leak measured on the v7.2 pool is not reachable by reweighting the pool's own statistics,
because on every missed output the correct grid carries a worse beam and augmentation score than the grid
the selector chose. Closing it needs a signal that is not in the pool at all.

This is that signal. A candidate output for the test input is either a correct instance of the task's rule
or it is not. If it is, appending the pair (test input, candidate) to the task's demonstrations gives the
model a fourth true example, and predicting a held-out real demonstration should get easier. If the
candidate is wrong, it contradicts what the other demonstrations imply and prediction should get harder.

So the score is a delta rather than a likelihood. For each held-out demonstration we measure the negative
log-likelihood of its true output twice, once from the remaining demonstrations alone and once with the
candidate injected, and average the difference. Negative means the candidate helped. The delta form matters
because it cancels per-task and per-fold difficulty, which a raw likelihood would confound with rule
consistency.

This first pass scores against the unadapted base model. The pool was produced by per-task adapters that
Kaggle discarded, so the question here is only whether the signal exists at all, not what it is worth in
deployment.

Usage: python rig_roundtrip.py [POOL_JSON] [OUT_JSON]
"""
import json
import os
import sys
import time

BASE = r"D:\arc"
sys.path.insert(0, BASE)

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "experiments", "pool_dump.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "experiments", "roundtrip.json")
STATUS = os.path.join(BASE, "eval_status_roundtrip.json")
MODEL_DIR = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                         "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
CHALLENGES = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
MAX_PROMPT = 12000

import torch                                                          # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                # noqa: E402
from transformers import AutoModelForCausalLM                         # noqa: E402

from src.arc.model import build_prompt_ids, grid_to_ids, IM_END       # noqa: E402


def heartbeat(phase, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "ts": time.time(), "extra": extra}, f)
    os.replace(tmp, STATUS)


@torch.inference_mode()
def target_nll(model, prompt_ids, target_grid):
    """Mean negative log-likelihood of the target grid's tokens, conditioned on the prompt."""
    tgt = grid_to_ids(target_grid) + [IM_END]
    ids = prompt_ids + tgt
    if len(ids) > MAX_PROMPT:
        return None
    x = torch.tensor([ids], device="cuda")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        logits = model(input_ids=x).logits
    # predict position i from position i-1, so the target's logits start one before it
    lp = torch.log_softmax(logits[0, len(prompt_ids) - 1:-1].float(), dim=-1)
    tt = torch.tensor(tgt, device="cuda")
    return float(-lp.gather(1, tt.unsqueeze(1)).mean())


def main():
    pool = json.load(open(POOL))
    tasks = json.load(open(CHALLENGES))
    heartbeat("load")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()

    out, t0 = {}, time.time()
    for n, (bk, cands) in enumerate(sorted(pool.items())):
        tid, sub = bk.split("_")
        task = tasks.get(tid)
        if task is None:
            continue
        train = task["train"]
        test_input = task["test"][int(sub)]["input"]
        heartbeat("scoring", "%s %d/%d" % (bk, n + 1, len(pool)))

        # baseline NLL per held-out demonstration, computed once and reused for every candidate
        folds, base_nll = [], {}
        for i in range(len(train)):
            others = [p for j, p in enumerate(train) if j != i]
            if not others:
                continue
            v = target_nll(model, build_prompt_ids(others, train[i]["input"]), train[i]["output"])
            if v is not None:
                base_nll[i] = v
                folds.append(i)

        recs = []
        for c in cands:
            deltas = []
            for i in folds:
                others = [p for j, p in enumerate(train) if j != i]
                injected = others + [{"input": test_input, "output": c["grid"]}]
                v = target_nll(model, build_prompt_ids(injected, train[i]["input"]), train[i]["output"])
                if v is not None:
                    deltas.append(v - base_nll[i])
            recs.append({"correct": c["correct"], "count": c["count"],
                         "delta": (sum(deltas) / len(deltas)) if deltas else None,
                         "folds": len(deltas)})
        out[bk] = recs
        with open(OUT, "w") as f:
            json.dump(out, f)
        done = sum(1 for r in recs if r["delta"] is not None)
        print("%-14s %2d cands, %d scored, %5.1f min elapsed" % (
            bk, len(recs), done, (time.time() - t0) / 60), flush=True)

    heartbeat("done")
    print("RESULTS", OUT, flush=True)


if __name__ == "__main__":
    main()
