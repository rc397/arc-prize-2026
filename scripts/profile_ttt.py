"""One-task profiler: isolates TTT step cost from generation cost so we can size settings.
Run on the rig: python scripts/profile_ttt.py
"""
import sys
import time

sys.path.insert(0, "D:/arc/src")
import numpy as np   # noqa: E402
import torch         # noqa: E402

from arc import augment as A   # noqa: E402
from arc import data as D      # noqa: E402
from arc import model as M     # noqa: E402
from arc import ttt as T       # noqa: E402

MD = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"

ch, _ = D.load_consolidated(CH, None, n=1)
tid, task = next(iter(ch.items()))
rng = np.random.default_rng(0)
ids, _ = T._example(task, rng)
print(f"task {tid} n_train {len(task['train'])} train_seq_len {len(ids)}", flush=True)

t = time.time()
model, tok = M.load_model(MD)
peft, init = T.make_adapter(model)
print(f"load {time.time() - t:.1f}s attn {getattr(model.config, '_attn_implementation', '?')}", flush=True)

t = time.time()
T.test_time_train(peft, init, task, rng, steps=20)
torch.cuda.synchronize()
dt = time.time() - t
print(f"TTT_20steps {dt:.1f}s {dt / 20 * 1000:.0f}ms/step 150steps~{dt / 20 * 150:.0f}s", flush=True)

augs = [A.Augmentation(d4=n) for n in A.D4_NAMES]
prompts = []
for a in augs:
    ta = A.apply_task(a, {"train": task["train"], "test": [{"input": task["test"][0]["input"]}]})
    prompts.append(M.build_prompt_ids(ta["train"], ta["test"][0]["input"]))
print(f"gen_prompt_lens {[len(p) for p in prompts]}", flush=True)

t = time.time()
gens = M.batched_generate(peft, prompts, 1200)
torch.cuda.synchronize()
dt = time.time() - t
print(f"GEN_8augs {dt:.1f}s out_lens {[len(g) for g, _ in gens]}", flush=True)
