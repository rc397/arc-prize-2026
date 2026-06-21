import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model, build_prompt_ids, greedy_generate
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=3)
task = tasks[list(tasks.keys())[0]]
prompt = build_prompt_ids(task["train"], task["test"][0]["input"])
print("prompt_len=%d" % len(prompt), flush=True)

peft.eval()
for trial in range(2):
    t = time.time(); out = greedy_generate(peft, prompt, max_new_tokens=36)
    print("FRESH-PEFT trial=%d out=%d %.1fs" % (trial, len(out), time.time() - t), flush=True)

base = peft.get_base_model()
t = time.time(); out = greedy_generate(base, prompt, max_new_tokens=36)
print("BASE out=%d %.1fs" % (len(out), time.time() - t), flush=True)

rng = np.random.default_rng(0)
test_time_train(peft, init, task, rng, steps=40)
peft.eval()
t = time.time(); greedy_generate(peft, prompt, max_new_tokens=1); t1 = time.time() - t
t = time.time(); greedy_generate(peft, prompt, max_new_tokens=36); t36 = time.time() - t
print("POST-TTT prefill(n=1)=%.1fs  n=36=%.1fs  decode_per_tok=%.2fs" % (t1, t36, (t36 - t1) / 35), flush=True)
