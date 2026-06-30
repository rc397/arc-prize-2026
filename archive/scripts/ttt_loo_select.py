"""v4 go/no-go: does leave-one-out adapter SELECTION beat a single adapter?

Per trainable task:
  baseline  = one adapter trained on all n train pairs, AIRV (D4) test prediction -> A0
  selection = K adapters, each leaving out a different pair; keep only those that reproduce their
              held-out pair; pool the kept adapters' AIRV candidates and vote -> A1 (fall back to the
              baseline when none pass).
Compare A0 vs A1 over a slice, read by the paired flips (recovered A0=miss->A1=hit vs lost A0=hit->A1=miss),
which is the sensitive test for a thin signal. If selection recovers more than it loses, leave-one-out is
actionable and v4 is worth building; otherwise drop it. Single 16 GB GPU, PyTorch throughout, no llama.cpp.

Usage: python ttt_loo_select.py [N_TASKS] [K] [STEPS] [N_AUG]
"""
import sys
import json
import time
import collections

import numpy as np
import torch

sys.path.insert(0, r"D:\arc")
from src.arc.model import load_model, build_prompt_ids, greedy_generate, ids_to_grid, grid_to_ids
from src.arc.ttt import make_adapter, test_time_train
from src.arc.data import load_consolidated
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
OUT = r"D:\arc\experiments\ttt_loo_select.json"
N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_AUG = int(sys.argv[4]) if len(sys.argv) > 4 else 4    # D4 augmentations for the AIRV test prediction
MAXPAIR = 1000
AUGS = [Augmentation(d4=name) for name in D4_NAMES[:N_AUG]]


def pair_tokens(p):
    return len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"]))


def airv_predict(model, train_pairs, test_inputs):
    """AIRV over the D4 augmentations; return per-test-input list of (grid, logprob) candidates."""
    cands = [[] for _ in test_inputs]
    for aug in AUGS:
        task = {"train": train_pairs, "test": [{"input": ti} for ti in test_inputs]}
        at = apply_task(aug, task)
        for j in range(len(test_inputs)):
            g = ids_to_grid(greedy_generate(model, build_prompt_ids(at["train"], at["test"][j]["input"]),
                                            max_new_tokens=1200))
            cands[j].append((invert_grid(aug, g), 0.0))
    return cands


def vote_correct(cands_per_ti, soln):
    ok = True
    for j, cands in enumerate(cands_per_ti):
        a1, a2 = select_top2(cands)
        if a1 != soln[j] and a2 != soln[j]:
            ok = False
    return ok


def main():
    device = "cuda"
    model, _ = load_model(MODEL, dtype=torch.bfloat16, device=device)
    peft_model, init = make_adapter(model, rank=16, alpha=16, dropout=0.1)
    tasks, sols = load_consolidated(CH, SOL, n=None)
    rng = np.random.default_rng(0)

    trainable = []
    for tid, task in tasks.items():
        if len(task["train"]) >= 3 and max(pair_tokens(p) for p in task["train"]) <= MAXPAIR:
            trainable.append(tid)
        if len(trainable) >= N_TASKS:
            break
    print("v4 selection test: %d tasks, K=%d, %d steps, AIRV over %d D4"
          % (len(trainable), K, STEPS, N_AUG), flush=True)

    rows = []
    t0 = time.time()
    for i, tid in enumerate(trainable):
        task = tasks[tid]
        n = len(task["train"])
        test_inputs = [t["input"] for t in task["test"]]
        soln = sols[tid]
        # baseline: one adapter on all n pairs
        test_time_train(peft_model, init, {"train": task["train"], "test": []}, rng, steps=STEPS, time_budget=200)
        A0 = vote_correct(airv_predict(peft_model, task["train"], test_inputs), soln)
        # selection: K leave-one-out adapters, keep those that reproduce their held-out pair
        kept = [[] for _ in test_inputs]
        n_pass = 0
        for h in range(min(K, n)):
            ctx = [task["train"][j] for j in range(n) if j != h]
            test_time_train(peft_model, init, {"train": ctx, "test": []}, rng, steps=STEPS, time_budget=200)
            hg = ids_to_grid(greedy_generate(peft_model, build_prompt_ids(ctx, task["train"][h]["input"]),
                                             max_new_tokens=1200))
            if hg == task["train"][h]["output"]:
                n_pass += 1
                lc = airv_predict(peft_model, ctx, test_inputs)
                for j in range(len(test_inputs)):
                    kept[j].extend(lc[j])
        A1 = vote_correct(kept, soln) if n_pass else A0
        rows.append({"tid": tid, "A0": bool(A0), "A1": bool(A1), "n_pass": n_pass})
        dt = time.time() - t0
        print("[%d/%d] %s A0=%s A1=%s pass=%d/%d t=%.0fs avg=%.0fs"
              % (i + 1, len(trainable), tid, A0, A1, n_pass, min(K, n), dt, dt / (i + 1)), flush=True)

    json.dump(rows, open(OUT, "w"))
    a0 = sum(r["A0"] for r in rows)
    a1 = sum(r["A1"] for r in rows)
    flips = collections.Counter((r["A0"], r["A1"]) for r in rows)
    recovered = flips.get((False, True), 0)
    lost = flips.get((True, False), 0)
    print("=== RESULT ===", flush=True)
    print("baseline single-adapter: %d/%d" % (a0, len(rows)), flush=True)
    print("LOO-selection ensemble:  %d/%d" % (a1, len(rows)), flush=True)
    print("paired flips: recovered (miss->hit) %d, lost (hit->miss) %d, net %+d"
          % (recovered, lost, recovered - lost), flush=True)
    verdict = "BEATS" if (a1 > a0 and recovered > lost) else ("LOSES TO" if a1 < a0 else "TIES")
    print("VERDICT: LOO-selection %s the single adapter" % verdict, flush=True)


if __name__ == "__main__":
    main()
