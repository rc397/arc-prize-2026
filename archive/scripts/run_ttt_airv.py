import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model, build_prompt_ids, grid_to_ids, ids_to_grid, batched_generate
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
OUT = r"D:\arc\experiments\airv_ttt_eval.json"
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0 = all tasks
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def airv_attempts(task):
    # Bound generation by the task's own train-output size so a single runaway augmented view
    # cannot drag the whole batch to max_new_tokens.
    out_lens = [len(grid_to_ids(p["output"])) for p in task["train"]]
    cap = min(1024, max(96, 2 * max(out_lens) + 8))
    attempts = []
    for ti in range(len(task["test"])):
        prompts = []
        for aug in AUGS:
            at = apply_task(aug, task)
            prompts.append(build_prompt_ids(at["train"], at["test"][ti]["input"]))
        gens = batched_generate(peft, prompts, max_new_tokens=cap, chunk=4)
        cands = [(invert_grid(aug, ids_to_grid(gen)), lp) for aug, (gen, lp) in zip(AUGS, gens)]
        a1, a2 = select_top2(cands)
        attempts.append({"attempt_1": a1, "attempt_2": a2})
    return attempts


model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
submission = {}
tids = list(tasks.keys())
# Warm up the generate path once (Blackwell kernel JIT costs ~35s on the first call) so the
# first real task is not penalized.
_wt = tasks[tids[0]]
batched_generate(peft, [build_prompt_ids(_wt["train"], _wt["test"][0]["input"])], max_new_tokens=8, chunk=1)
print("warmup done", flush=True)
t0 = time.time()
for i, tid in enumerate(tids):
    task = tasks[tid]
    rng = np.random.default_rng(1234 + i)
    try:
        tt = time.time()
        loss = test_time_train(peft, init, task, rng, steps=STEPS)
        ttt_t = time.time() - tt
        gt = time.time()
        submission[tid] = airv_attempts(task)
        gen_t = time.time() - gt
    except Exception as e:
        torch.cuda.empty_cache()
        submission[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]
        print("[%d/%d] %s ERROR %s" % (i + 1, len(tids), tid, repr(e)[:120]), flush=True)
        write_submission(submission, tasks, OUT)
        continue
    write_submission(submission, tasks, OUT)
    dt = time.time() - t0
    print("[%d/%d] %s loss=%.3f ttt=%.0fs gen=%.0fs t=%.0fs avg=%.0fs" % (
        i + 1, len(tids), tid, loss, ttt_t, gen_t, dt, dt / (i + 1)), flush=True)
if sols:
    acc, c, n = score_submission(submission, sols)
    print("SCORE acc=%.4f correct=%d total=%d steps=%d" % (acc, c, n, STEPS), flush=True)
