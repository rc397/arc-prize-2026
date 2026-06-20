import sys, os, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
OUTDIR = r"D:\arc\experiments\lora_gate1"

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=3)
tid = list(tasks.keys())[0]
task = tasks[tid]
rng = np.random.default_rng(0)
t0 = time.time()
loss = test_time_train(peft, init, task, rng, steps=40)
print("TTT tid=%s loss=%.3f %.0fs" % (tid, loss, time.time() - t0), flush=True)
os.makedirs(OUTDIR, exist_ok=True)
peft.save_pretrained(OUTDIR)
print("saved adapter ->", OUTDIR, flush=True)
print("files:", sorted(os.listdir(OUTDIR)), flush=True)
