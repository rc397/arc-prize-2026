import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model, build_prompt_ids, ids_to_grid, greedy_generate, grid_to_ids
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train, reset_adapter
from src.arc.augment import Augmentation, apply_task, invert_grid

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
AUG = Augmentation(d4="identity")   # single-shot, no AIRV, just the TTT-vs-no-TTT signal

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=N)


def predict(task, ti):
    at = apply_task(AUG, task)
    prompt = build_prompt_ids(at["train"], at["test"][ti]["input"])
    out_lens = [len(grid_to_ids(p["output"])) for p in task["train"]]
    cap = min(1024, max(96, 2 * max(out_lens) + 8))
    gen = greedy_generate(peft, prompt, max_new_tokens=cap)
    return invert_grid(AUG, ids_to_grid(gen))


_t = tasks[list(tasks.keys())[0]]
greedy_generate(peft, build_prompt_ids(_t["train"], _t["test"][0]["input"]), max_new_tokens=8)  # warmup

no_ttt = ttt_c = total = 0
t0 = time.time()
for k, tid in enumerate(tasks):
    task = tasks[tid]
    ntest = len(task["test"])
    reset_adapter(peft, init)
    peft.eval()
    p0 = [predict(task, ti) for ti in range(ntest)]
    rng = np.random.default_rng(1234)
    test_time_train(peft, init, task, rng, steps=STEPS)
    p1 = [predict(task, ti) for ti in range(ntest)]
    for ti in range(ntest):
        total += 1
        sol = sols[tid][ti]
        no_ttt += int(p0[ti] == sol)
        ttt_c += int(p1[ti] == sol)
    print("[%d/%d %s] cum no_ttt=%d ttt=%d /%d  %.0fs" % (
        k + 1, N, tid, no_ttt, ttt_c, total, time.time() - t0), flush=True)
print("RESULT single-shot identity: no_ttt=%d/%d  ttt=%d/%d  steps=%d" % (
    no_ttt, total, ttt_c, total, STEPS), flush=True)
