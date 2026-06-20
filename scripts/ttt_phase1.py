"""Phase 1: per-task TTT in PyTorch, save each adapter. Training only (no generation, no
conversion), so neither the post-TTT decode bug nor a convert/VRAM collision can occur here.
Writes the list of saved task ids; phase 1b converts them from a clean process."""
import sys, os, json, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
ADAPTER_TMP = r"D:\arc\experiments\loras"
LIST = r"D:\arc\experiments\adapter_list.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=(N or None))
saved = []
t0 = time.time()
for i, tid in enumerate(tasks):
    rng = np.random.default_rng(1234 + i)
    adir = os.path.join(ADAPTER_TMP, tid)
    tt = time.time()
    if os.path.exists(os.path.join(adir, "adapter_model.safetensors")):
        loss = -1.0                                    # reuse an already-trained adapter (resume)
    else:
        loss = test_time_train(peft, init, tasks[tid], rng, steps=STEPS)
        os.makedirs(adir, exist_ok=True)
        peft.save_pretrained(adir)
    saved.append(tid)
    json.dump(saved, open(LIST, "w"))
    print("[%d/%d] %s loss=%.3f ttt=%.0fs t=%.0fs" % (
        i + 1, len(tasks), tid, loss, time.time() - tt, time.time() - t0), flush=True)
print("SAVED %d adapters -> %s" % (len(saved), LIST), flush=True)
