"""Kernel/decode/generation benchmark to locate the slowdown."""
import sys
import time

sys.path.insert(0, "D:/arc/src")
import torch                       # noqa: E402
from arc import augment as A      # noqa: E402
from arc import data as D         # noqa: E402
from arc import model as M        # noqa: E402
from arc import ttt as T          # noqa: E402

MD = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
model, tok = M.load_model(MD)
print(f"dtype {next(model.parameters()).dtype} attn {model.config._attn_implementation}", flush=True)

x = torch.randint(0, 10, (1, 2048), device="cuda")
with torch.inference_mode():
    for _ in range(2):
        model(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        model(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / 5
print(f"PREFILL 2048: {int(2048 / dt)} tok/s", flush=True)

with torch.inference_mode():
    torch.cuda.synchronize()
    t = time.time()
    out = model.generate(x, max_new_tokens=128, do_sample=False, use_cache=True,
                         eos_token_id=15, pad_token_id=13)
    torch.cuda.synchronize()
    dt = time.time() - t
print(f"DECODE 128: {128 / dt:.1f} tok/s", flush=True)

ch, _ = D.load_consolidated(r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json", None, n=1)
tid, task = next(iter(ch.items()))
peft, init = T.make_adapter(model)
augs = [A.Augmentation(d4=n) for n in A.D4_NAMES]
prompts = []
for a in augs:
    ta = A.apply_task(a, {"train": task["train"], "test": [{"input": task["test"][0]["input"]}]})
    prompts.append(M.build_prompt_ids(ta["train"], ta["test"][0]["input"]))
torch.cuda.synchronize()
t = time.time()
gens = M.batched_generate(peft, prompts, 256)
torch.cuda.synchronize()
dt = time.time() - t
print(f"GEN 8augs base: {dt:.1f}s out_lens {[len(g) for g, _ in gens]}", flush=True)
