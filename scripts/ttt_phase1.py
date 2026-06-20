"""Phase 1: per-task TTT in PyTorch, save each adapter (training only). Config-driven for the
sweep. Selects the N smallest-context tasks (big-grid tasks are VRAM-bound and degrade TTT to a
crawl on a 16 GB card), and frees the CUDA cache between tasks so memory cannot creep."""
import sys, os, json, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model, build_prompt_ids
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 32
DROPOUT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
TAG = sys.argv[6] if len(sys.argv) > 6 else "base"
ADAPTER_TMP = r"D:\arc\experiments\loras_%s" % TAG
LIST = r"D:\arc\experiments\adapter_list_%s.json" % TAG

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=RANK, alpha=RANK, dropout=DROPOUT)
allt, sols = load_consolidated(CH, SOL, n=None)
order = sorted(allt.keys(),
               key=lambda t: len(build_prompt_ids(allt[t]["train"], allt[t]["test"][0]["input"])))
chosen = order[:N] if N else order
saved = []
t0 = time.time()
for i, tid in enumerate(chosen):
    rng = np.random.default_rng(1234 + i)
    adir = os.path.join(ADAPTER_TMP, tid)
    tt = time.time()
    if os.path.exists(os.path.join(adir, "adapter_model.safetensors")):
        loss = -1.0
    else:
        loss = test_time_train(peft, init, allt[tid], rng, steps=STEPS, lr=LR)
        os.makedirs(adir, exist_ok=True)
        peft.save_pretrained(adir)
    torch.cuda.empty_cache()
    saved.append(tid)
    json.dump(saved, open(LIST, "w"))
    print("[%d/%d] %s loss=%.3f ttt=%.0fs t=%.0fs" % (
        i + 1, len(chosen), tid, loss, time.time() - tt, time.time() - t0), flush=True)
print("SAVED %d adapters tag=%s (steps=%d rank=%d drop=%.2f lr=%g, smallest-context tasks)" % (
    len(saved), TAG, STEPS, RANK, DROPOUT, LR), flush=True)
