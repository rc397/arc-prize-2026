"""B6 phase 1: train one task's LoRA in PyTorch and save it, as a standalone process so all GPU
memory frees on exit (the bridge runs train, convert, serve as separate phases). args: tid adir [steps]."""
import sys, os
sys.path.insert(0, r"D:\arc")
import numpy as np
from src.arc.model import load_model
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

tid = sys.argv[1]
adir = sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=16, alpha=16, dropout=0.1)
allt, _ = load_consolidated(CH, SOL, n=None)
rng = np.random.default_rng(0)
loss = test_time_train(peft, init, allt[tid], rng, steps=STEPS, lr=1e-4, time_budget=150, max_ctx_tokens=1600)
os.makedirs(adir, exist_ok=True)
peft.save_pretrained(adir)
print("trained %s loss=%.3f" % (tid, loss), flush=True)
