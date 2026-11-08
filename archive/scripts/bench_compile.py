"""Test whether torch.compile + static KV cache fixes the slow decode on this stack."""
import sys
import time

sys.path.insert(0, "D:/arc/src")
import torch                       # noqa: E402
from arc import model as M        # noqa: E402

MD = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
model, tok = M.load_model(MD)
x = torch.randint(0, 10, (1, 2048), device="cuda")
GK = dict(max_new_tokens=128, do_sample=False, use_cache=True, eos_token_id=15, pad_token_id=13)

with torch.inference_mode():
    torch.cuda.synchronize()
    t = time.time()
    model.generate(x, **GK)
    torch.cuda.synchronize()
    base = time.time() - t
print(f"BASELINE decode 128: {128 / base:.1f} tok/s", flush=True)

try:
    model.generation_config.cache_implementation = "static"
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    with torch.inference_mode():
        print("compiling (first generate, slow)...", flush=True)
        model.generate(x, **GK)
        torch.cuda.synchronize()
        t = time.time()
        model.generate(x, **GK)
        torch.cuda.synchronize()
        comp = time.time() - t
    print(f"COMPILED decode 128: {128 / comp:.1f} tok/s", flush=True)
except Exception as e:
    print(f"COMPILE FAILED: {repr(e)[:400]}", flush=True)
