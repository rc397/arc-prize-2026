"""Stage 1 (leave-one-out TTT) correlation probe: does honest held-out accuracy predict test correctness?

For each trainable task: train a fresh adapter on n-1 train pairs (one held fully out), greedily predict
the held-out pair (the LOO hit), then train an adapter on all n pairs and greedily predict the test
output(s) against the known solution (the test hit). If LOO hit predicts test hit, held-out accuracy is a
usable gate / selector for v3; if the two are uncorrelated, leave-one-out is a dead end. Single 16 GB GPU,
PyTorch throughout (training and greedy generation), no llama.cpp.

This is the cheap gate before building the gated / adapter-ensemble pipeline. Usage:
  python ttt_loo.py [N_TASKS] [STEPS]
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

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
OUT = r"D:\arc\experiments\ttt_loo.json"
N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MAXPAIR = 1000   # the VRAM-safe trainable set (largest train pair token length)


def pair_tokens(p):
    return len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"]))


def predict(model, train_pairs, test_input):
    gen = greedy_generate(model, build_prompt_ids(train_pairs, test_input), max_new_tokens=1200)
    return ids_to_grid(gen)


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
    print("probing %d trainable tasks, %d steps each" % (len(trainable), STEPS), flush=True)

    rows = []
    t0 = time.time()
    for i, tid in enumerate(trainable):
        task = tasks[tid]
        h = len(task["train"]) - 1                       # hold out the last train pair
        # LOO: train on n-1 pairs, predict the held-out pair
        test_time_train(peft_model, init, {"train": task["train"][:h], "test": []},
                        rng, steps=STEPS, time_budget=200)
        loo_pred = predict(peft_model, task["train"][:h], task["train"][h]["input"])
        loo_hit = (loo_pred == task["train"][h]["output"])
        # TEST: train on all n pairs, predict the test output(s) vs the known solution
        test_time_train(peft_model, init, {"train": task["train"], "test": []},
                        rng, steps=STEPS, time_budget=200)
        test_hit = all(predict(peft_model, task["train"], task["test"][ti]["input"]) == sols[tid][ti]
                       for ti in range(len(task["test"])))
        rows.append({"tid": tid, "loo": bool(loo_hit), "test": bool(test_hit)})
        dt = time.time() - t0
        print("[%d/%d] %s loo=%s test=%s t=%.0fs avg=%.0fs"
              % (i + 1, len(trainable), tid, loo_hit, test_hit, dt, dt / (i + 1)), flush=True)

    json.dump(rows, open(OUT, "w"))
    n1 = sum(r["loo"] for r in rows)
    n0 = len(rows) - n1
    rate_loo1 = sum(r["test"] for r in rows if r["loo"]) / max(1, n1)
    rate_loo0 = sum(r["test"] for r in rows if not r["loo"]) / max(1, n0)
    print("=== RESULT ===", flush=True)
    print("LOO-hit tasks:  %d, test-correct rate %.2f" % (n1, rate_loo1), flush=True)
    print("LOO-miss tasks: %d, test-correct rate %.2f" % (n0, rate_loo0), flush=True)
    print("contingency (loo,test)->count:",
          dict(collections.Counter((r["loo"], r["test"]) for r in rows)), flush=True)
    print("VERDICT: leave-one-out %s predict test correctness"
          % ("DOES" if rate_loo1 - rate_loo0 >= 0.15 else "does NOT clearly"), flush=True)


if __name__ == "__main__":
    main()
