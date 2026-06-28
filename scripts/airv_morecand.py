"""More-candidates AIRV, a rig isolation test. Sample K completions per D4 augmentation and select by
consensus and NLL, to see if more candidates beats the 8-greedy AIRV before we cram it into a submission.
Optional per-task TTT (STEPS>0).

Usage: python airv_morecand.py [N_LIMIT] [K] [STEPS]   (N_LIMIT 0 = all tasks, STEPS 0 = no TTT)
The K=1 STEPS=0 case reproduces the 12.79 greedy no-TTT baseline.
"""
import sys
import time

sys.path.insert(0, r"D:\arc")
import numpy as np
import torch
from src.arc.model import load_model, build_prompt_ids, grid_to_ids, ids_to_grid, batched_generate
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
OUT = r"D:\arc\experiments\airv_morecand_eval.json"
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def airv_attempts(task):
    out_lens = [len(grid_to_ids(p["output"])) for p in task["train"]]
    cap = min(1024, max(96, 2 * max(out_lens) + 8))
    attempts = []
    for ti in range(len(task["test"])):
        rep = []
        for aug in AUGS:
            at = apply_task(aug, task)
            p = build_prompt_ids(at["train"], at["test"][ti]["input"])
            for _ in range(K):
                rep.append((aug, p))
        gens = batched_generate(peft, [p for _, p in rep], max_new_tokens=cap,
                                do_sample=(K > 1), temperature=0.8, top_p=0.95, chunk=8)
        cands = [(invert_grid(aug, ids_to_grid(gen)), lp) for (aug, _), (gen, lp) in zip(rep, gens)]
        attempts.append(dict(zip(("attempt_1", "attempt_2"), select_top2(cands))))
    return attempts


model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
submission = {}
tids = list(tasks.keys())
_wt = tasks[tids[0]]
batched_generate(peft, [build_prompt_ids(_wt["train"], _wt["test"][0]["input"])], max_new_tokens=8, chunk=1)
print("warmup done K=%d STEPS=%d n=%d" % (K, STEPS, len(tids)), flush=True)
t0 = time.time()
for i, tid in enumerate(tids):
    task = tasks[tid]
    rng = np.random.default_rng(1234 + i)
    try:
        if STEPS:
            test_time_train(peft, init, task, rng, steps=STEPS)
        submission[tid] = airv_attempts(task)
    except Exception as e:
        torch.cuda.empty_cache()
        submission[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]
        print("[%d/%d] %s ERROR %s" % (i + 1, len(tids), tid, repr(e)[:100]), flush=True)
        write_submission(submission, tasks, OUT)
        continue
    write_submission(submission, tasks, OUT)
    dt = time.time() - t0
    print("[%d/%d] %s t=%.0fs avg=%.0fs" % (i + 1, len(tids), tid, dt, dt / (i + 1)), flush=True)
if sols:
    acc, c, n = score_submission(submission, sols)
    print("SCORE acc=%.4f correct=%d total=%d K=%d STEPS=%d" % (acc, c, n, K, STEPS), flush=True)
