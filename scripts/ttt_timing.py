import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

t0 = time.time()
model, tok = load_model(MODEL)
print("model_loaded %.1fs  vram_alloc_GB %.1f" % (time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

tasks, sols = load_consolidated(CH, SOL, n=3)
peft, init = make_adapter(model, rank=32)
rng = np.random.default_rng(0)
tid = list(tasks.keys())[0]
task = tasks[tid]
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
loss = test_time_train(peft, init, task, rng, steps=20, lr=1e-4)
dt = time.time() - t1
print("TTT tid=%s npairs=%d steps=20 loss=%.3f total=%.1fs per_step=%.2fs" % (
    tid, len(task["train"]), loss, dt, dt / 20), flush=True)
print("vram_peak_GB %.1f" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
