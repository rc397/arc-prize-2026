"""DFS-decode vs greedy-AIRV A/B on a fixed TTT adapter (the ARChitects decoder-lever isolation).

Trains one LoRA per eval task with our validated rig config, then decodes that SAME in-memory adapter
two ways and scores both against the solution.
  Arm A (our baseline): greedy-decode each of the 8 D4 views, invert, majority vote (production AIRV).
  Arm B (ARChitects): DFS-enumerate every completion whose running probability stays above 0.2 per view,
                      invert, pool, then pick top-2 by the augmentation-product score (their score_kgmon).
Only the decoder differs, the adapter is identical, so a lift in B over A isolates the decode-and-select
lever from the heavy rank-256 / 8192-ctx TTT (which needs Unsloth and does not fit 16 GB). Reduced-rank
caveat, our adapter is rank 32 not their 256, so a NULL here is confounded by rank while a WIN is a cheap
real result we can build on.

Usage: python rig_dfs_ab.py [N_TASKS=16] [STEPS=60] [RANK=32] [MAXCTX=2000] [SEED=0]
"""
import copy
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

BASE = r"D:\arc"
sys.path.insert(0, BASE)
sys.setrecursionlimit(20000)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 32
MAXCTX = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
EXTEND = int(sys.argv[6]) if len(sys.argv) > 6 else 0   # 1 = also LoRA embed_tokens and lm_head (their config)

ALPHA = int(round(RANK * 1.25))   # match the 1.25 llama.cpp serve scale in torch (alpha/rank = 1.25)
MAXPAIR = 1000                    # rig 16 GB trainable envelope, same as the sweep
MAX_SCORE = -math.log(0.2)        # DFS keeps completions with running prob >= 0.2 (their threshold)
DFS_NODE_CAP = 400                # per-view forward-step budget, bounds the tree
DFS_TASK_TIME = 300               # per-task DFS wall budget in seconds, bounds worst case
K_AUG = 8                         # augmentation views for candidate scoring
MAX_NEW = 1000                    # a 30x30 grid is ~930 tokens

BASE_HF = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
SOL = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_solutions.json")
OUT = os.path.join(BASE, "experiments", "dfs_ab_s%d_r%d_e%d.json" % (SEED, RANK, EXTEND))
STATUS = os.path.join(BASE, "eval_status_dfsab.json")

from src.arc.model import (NL, IM_END, build_prompt_ids, grid_to_ids, ids_to_grid,   # noqa: E402
                           batched_generate, load_model)
from src.arc.data import load_consolidated                                            # noqa: E402
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES, random_augmentation  # noqa: E402
from src.arc.select import select_top2                                                # noqa: E402
from src.arc.ttt import test_time_train                                               # noqa: E402
from peft import LoraConfig, get_peft_model                                            # noqa: E402

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None

ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, NL, IM_END]
AUGS = [Augmentation(d4=n) for n in D4_NAMES]
DEV = "cuda"


def maxpair_tokens(task):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in task["train"])


def gkey(g):
    return tuple(tuple(int(v) for v in r) for r in g) if g else ()


def clone_cache(c):
    try:
        return copy.deepcopy(c)
    except Exception:
        leg = c.to_legacy_cache() if hasattr(c, "to_legacy_cache") else c
        leg2 = tuple(tuple(t.clone() for t in layer) for layer in leg)
        if DynamicCache is not None and hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(leg2)
        return leg2


def heartbeat(phase, i, n, extra=""):
    try:
        json.dump({"phase": phase, "i": i, "n": n, "ts": time.time(), "steps": STEPS,
                   "rank": RANK, "seed": SEED, "extra": extra}, open(STATUS, "w"))
    except Exception:
        pass


@torch.inference_mode()
def tf_nll(model, prompt_ids, answer_ids):
    """Teacher-forced negative log-likelihood of answer_ids following prompt_ids."""
    ids = torch.tensor([prompt_ids + answer_ids], device=DEV)
    out = model(input_ids=ids)
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)
    p = len(prompt_ids)
    s = 0.0
    for i, tok in enumerate(answer_ids):
        s += -float(logp[p - 1 + i, tok])
    return s


@torch.inference_mode()
def dfs_enumerate(model, prefix_ids, deadline, mode="clone"):
    """Enumerate completions whose running NLL stays below MAX_SCORE. Returns (list of
    (nll, token_seq_without_eos), truncated).

    `mode` selects how sibling branches share the KV cache, which is the whole point of the
    cache A/B. All three explore the identical tree; they differ only in what each sibling sees.

      clone   deepcopy the parent cache per sibling. Correct, and what this harness has always
              done, but it copies the whole cache at every branch.
      crop    truncate the shared cache back to `pos` before each sibling. Also correct, and
              cheap, because it drops rows rather than copying them.
      shared  pass the same cache with no truncation, which is what the DEPLOYED Kaggle solver
              does. On transformers 5.12 DynamicCache appends rather than overwriting, so each
              sibling attends over every branch explored before it and the cache grows for the
              whole search. This arm exists to measure what that costs.
    """
    out = model(input_ids=torch.tensor([prefix_ids], device=DEV), use_cache=True)
    root_logits = out.logits[0, -1]
    root_cache = out.past_key_values
    results = []
    nodes = [0]
    trunc = [False]

    def expand(seq, cum, logits, cache, pos):
        if nodes[0] >= DFS_NODE_CAP or time.time() > deadline:
            trunc[0] = True
            return
        logp = torch.log_softmax(logits.float(), dim=-1)
        kids = []
        for t in ARC_TOKENS:
            nll = cum - float(logp[t])
            if nll < MAX_SCORE:
                kids.append((nll, t))
        for nll, t in kids:
            if t == IM_END:
                results.append((nll, list(seq)))
        cont = [(nll, t) for nll, t in kids if t != IM_END and len(seq) < MAX_NEW]
        for idx, (nll, t) in enumerate(cont):
            if nodes[0] >= DFS_NODE_CAP or time.time() > deadline:
                trunc[0] = True
                return
            if mode == "clone":
                c = cache if idx == len(cont) - 1 else clone_cache(cache)
            elif mode == "crop":
                c = cache
                if hasattr(c, "crop"):
                    c.crop(pos)            # restore the parent state, dropping any sibling subtree
            else:
                c = cache                  # shared: the deployed pattern, deliberately uncorrected
            o = model(input_ids=torch.tensor([[t]], device=DEV), past_key_values=c, use_cache=True,
                      position_ids=torch.tensor([[pos]], device=DEV),
                      cache_position=torch.tensor([pos], device=DEV))
            nodes[0] += 1
            expand(seq + [t], nll, o.logits[0, -1], o.past_key_values, pos + 1)

    expand([], 0.0, root_logits, root_cache, len(prefix_ids))
    return results, trunc[0]


def greedy_airv(model, task, ti):
    """Arm A. Greedy-decode the 8 D4 views, invert, majority-vote top-2."""
    prompts = [build_prompt_ids(apply_task(a, task)["train"], apply_task(a, task)["test"][ti]["input"])
               for a in AUGS]
    gens = batched_generate(model, prompts, max_new_tokens=MAX_NEW, do_sample=False, chunk=4)
    cands = []
    for a, (gen, lp) in zip(AUGS, gens):
        cands.append((invert_grid(a, ids_to_grid(gen)), lp))
    a1, a2 = select_top2(cands)
    return [a1, a2], len(cands)


def calc_aug_scores(model, task, ti, cand_grid):
    """Teacher-forced NLL of the candidate under K fresh augmentations (lower means the model finds
    the candidate consistent across views)."""
    seed = int.from_bytes(hashlib.md5(str(gkey(cand_grid)).encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    local = {"train": task["train"], "test": [{"input": task["test"][ti]["input"], "output": cand_grid}]}
    scores = []
    for _ in range(K_AUG):
        aug = random_augmentation(rng, n_examples=len(task["train"]))
        at = apply_task(aug, local)
        prompt = build_prompt_ids(at["train"], at["test"][0]["input"])
        ans = grid_to_ids(at["test"][0]["output"]) + [IM_END]
        scores.append(tf_nll(model, prompt, ans))
    return scores


def getter_kgmon(beam, aug):
    return len(beam) - float(np.mean(aug))


def getter_probmul(beam, aug, baseline=3.0):
    return float(np.sum([baseline - b for b in beam])) + float(np.sum([baseline - s for s in aug]))


def dfs_select(model, task, ti, deadline, mode="clone"):
    """Arm B. DFS-enumerate candidates over the 8 views, score each by augmentation product,
    return top-2 for both getters plus stats."""
    beams = {}
    trunc_any = False
    for a in AUGS:
        if time.time() > deadline:
            trunc_any = True
            break
        at = apply_task(a, task)
        prefix = build_prompt_ids(at["train"], at["test"][ti]["input"])
        res, tr = dfs_enumerate(model, prefix, deadline, mode=mode)
        trunc_any = trunc_any or tr
        for nll, toks in res:
            g = invert_grid(a, ids_to_grid(toks))
            k = gkey(g)
            if k not in beams:
                beams[k] = {"grid": g, "beam": []}
            beams[k]["beam"].append(nll)
    for v in beams.values():
        v["aug"] = calc_aug_scores(model, task, ti, v["grid"])

    def top2(getter):
        if not beams:
            return [[[0]], [[0]]]
        ranked = sorted(beams.values(), key=lambda v: getter(v["beam"], v["aug"]), reverse=True)
        a1 = ranked[0]["grid"]
        a2 = ranked[1]["grid"] if len(ranked) > 1 else a1
        return [a1, a2]

    return top2(getter_kgmon), top2(getter_probmul), len(beams), trunc_any


def solved(attempts, sol_grid):
    return any(gkey(a) == gkey(sol_grid) for a in attempts)


def main():
    tasks, sols = load_consolidated(CH, SOL, n=None)
    tids = [t for t in tasks if maxpair_tokens(tasks[t]) <= MAXPAIR]
    tids.sort(key=lambda t: maxpair_tokens(tasks[t]))
    if N_TASKS:
        tids = tids[:N_TASKS]
    print("CONFIG dfs-ab n=%d steps=%d rank=%d alpha=%d maxctx=%d seed=%d extend=%d over %d tasks"
          % (len(tids), STEPS, RANK, ALPHA, MAXCTX, SEED, EXTEND, len(tids)), flush=True)

    done = {}
    if os.path.exists(OUT):
        try:
            done = json.load(open(OUT))
            print("resume: %d tasks already scored" % len(done), flush=True)
        except Exception:
            done = {}

    model, tok = load_model(BASE_HF, dtype=torch.bfloat16, device=DEV)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if EXTEND:
        targets = targets + ["embed_tokens", "lm_head"]
    cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.1, target_modules=targets, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    init = {k: v.detach().clone() for k, v in peft_model.named_parameters() if "lora_" in k}
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    for i, tid in enumerate(tids):
        if tid in done:
            continue
        heartbeat("task", i, len(tids), tid)
        ts = time.time()
        try:
            test_time_train(peft_model, init, tasks[tid], rng, steps=STEPS,
                            max_ctx_tokens=MAXCTX, time_budget=250)
        except Exception as e:
            print("[%d/%d] %s TRAIN-FAIL %r" % (i + 1, len(tids), tid, str(e)[:80]), flush=True)
            continue
        rec = {"a": 0.0, "b": 0.0, "bpm": 0.0, "ntest": len(tasks[tid]["test"]),
               "cands": [], "trunc": False}
        deadline = time.time() + DFS_TASK_TIME
        for ti in range(len(tasks[tid]["test"])):
            sol = sols[tid][ti]
            try:
                a_att, _ = greedy_airv(peft_model, tasks[tid], ti)
                b_att, bpm_att, ncand, tr = dfs_select(peft_model, tasks[tid], ti, deadline)
            except Exception as e:
                print("[%d/%d] %s DECODE-FAIL %r" % (i + 1, len(tids), tid, str(e)[:80]), flush=True)
                a_att, b_att, bpm_att, ncand, tr = [[[0]]], [[[0]]], [[[0]]], 0, False
            w = 1.0 / len(tasks[tid]["test"])
            rec["a"] += w * solved(a_att, sol)
            rec["b"] += w * solved(b_att, sol)
            rec["bpm"] += w * solved(bpm_att, sol)
            rec["cands"].append(ncand)
            rec["trunc"] = rec["trunc"] or tr
        done[tid] = rec
        json.dump(done, open(OUT, "w"))
        na = sum(v["a"] for v in done.values())
        nb = sum(v["b"] for v in done.values())
        nbpm = sum(v["bpm"] for v in done.values())
        print("[%d/%d] %s %.0fs cands=%s trunc=%s | running A=%.2f B=%.2f Bpm=%.2f of %d"
              % (i + 1, len(tids), tid, time.time() - ts, rec["cands"], rec["trunc"],
                 na, nb, nbpm, len(done)), flush=True)

    scored = [v for v in done.values()]
    na = sum(v["a"] for v in scored)
    nb = sum(v["b"] for v in scored)
    nbpm = sum(v["bpm"] for v in scored)
    n = len(scored)
    print("RESULT seed=%d tasks=%d greedy_A=%.3f (%.1f) dfs_kgmon_B=%.3f (%.1f) dfs_probmul=%.3f (%.1f) elapsed=%.2fh"
          % (SEED, n, na / max(n, 1), na, nb / max(n, 1), nb, nbpm / max(n, 1), nbpm,
             (time.time() - t0) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
