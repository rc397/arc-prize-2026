import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model, build_prompt_ids, greedy_generate
from src.arc.data import load_consolidated
from src.arc.augment import Augmentation, apply_task
from src.arc.ttt import make_adapter, test_time_train

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

model, tok = load_model(MODEL)
peft, init = make_adapter(model, rank=32)
tasks, sols = load_consolidated(CH, SOL, n=3)
task = tasks[list(tasks.keys())[0]]
rng = np.random.default_rng(0)
loss = test_time_train(peft, init, task, rng, steps=40)
print("ttt done loss=%.3f use_cache=%s" % (loss, model.config.use_cache), flush=True)
peft.eval()
for name in ["identity", "rot90", "flip_lr", "identity"]:
    aug = Augmentation(d4=name)
    at = apply_task(aug, task)
    prompt = build_prompt_ids(at["train"], at["test"][0]["input"])
    t = time.time()
    out = greedy_generate(peft, prompt, max_new_tokens=600)
    dt = time.time() - t
    print("aug=%-8s prompt=%d out=%d %.1fs %.2f tok/s" % (name, len(prompt), len(out), dt, len(out) / max(dt, 0.01)), flush=True)
