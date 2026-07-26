"""Ask whether our stack has a memory-efficient attention path, and how far it moves the context wall.

rig_fitprobe.py established that rank 256 fits one card at sequence 2048 and that every configuration
fails at 4096, with activations growing about 3.3x for a doubling. That is attention materializing its
score matrix, which is the signature of the math backend. The probe never forced a backend though, so the
wall it found may belong to the kernel selection rather than to the card.

This varies exactly one axis, the attention path, holding rank at 256 with bf16 adapters, the only
configuration of interest. Sequence is set exactly with synthetic batches for the same reason as before,
memory depends on shape rather than token values.

The comparison is not with Kaggle's kernel. That stack reaches rank 256 at 8192 context in about 13.0 GB
through unsloth and FA2, neither of which installs here, so this asks a different question. Does the torch
build we already have expose a kernel that buys the same thing. Backends that cannot run on this GPU raise
at the call, which is itself the measurement, so unsupported is recorded rather than treated as an error.

Usage: python rig_attnprobe.py [STEPS=3]
"""
import gc
import json
import os
import sys
import time

BASE = r"D:\arc"
sys.path.insert(0, BASE)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

MODEL_DIR = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                         "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
EXP = os.path.join(BASE, "experiments")
STATUS = os.path.join(BASE, "eval_status_attnprobe.json")
RESULTS = os.path.join(EXP, "attnprobe.json")

RANK = 256
SEQS = [2048, 4096, 8192]
LR = 1e-4
GRAD_CLIP = 1.0
THRASH_S = 8.0

import torch                                                          # noqa: E402
from peft import LoraConfig, get_peft_model                           # noqa: E402
from transformers import AutoModelForCausalLM                         # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                # noqa: E402

from src.arc.ttt import LORA_TARGETS                                  # noqa: E402

# name, attn_implementation at load, forced sdpa backend or None for whatever torch picks
PATHS = [
    ("eager",     "eager", None),
    ("sdpa-auto", "sdpa",  None),
    ("math",      "sdpa",  SDPBackend.MATH),
    ("efficient", "sdpa",  SDPBackend.EFFICIENT_ATTENTION),
    ("flash",     "sdpa",  SDPBackend.FLASH_ATTENTION),
    ("cudnn",     "sdpa",  SDPBackend.CUDNN_ATTENTION),
]


def heartbeat(phase, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "ts": time.time(), "extra": extra}, f)
    os.replace(tmp, STATUS)


def gb(x):
    return round(x / 1024 ** 3, 2)


def run_cell(name, attn_impl, backend, seq):
    tag = "%s seq%d" % (name, seq)
    heartbeat("cell", tag)
    rec = {"path": name, "attn_impl": attn_impl, "seq": seq}
    model = peft_model = opt = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                     attn_implementation=attn_impl)
        model.to("cuda").eval()
        w = torch.cuda.memory_allocated()
        rec["weights_gb"] = gb(w)

        cfg = LoraConfig(r=RANK, lora_alpha=RANK, lora_dropout=0.1,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        for n, p in peft_model.named_parameters():
            if "lora_" in n and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
        params = [p for p in peft_model.parameters() if p.requires_grad]
        rec["adapter_gb"] = gb(torch.cuda.memory_allocated() - w)

        base = peft_model.get_base_model()
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        peft_model.train()
        opt = torch.optim.AdamW(params, lr=LR)

        ids = torch.randint(0, 10, (1, seq), device="cuda")
        labels = ids.clone()
        labels[:, : seq // 2] = -100

        torch.cuda.reset_peak_memory_stats()

        def one_step():
            loss = peft_model(input_ids=ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad()

        t0 = time.time()
        for i in range(STEPS):
            if backend is None:
                one_step()
            else:
                with sdpa_kernel(backend):
                    one_step()
            # a step this slow is spilling into host RAM rather than computing. Windows overcommits
            # instead of raising, so stop on the clock rather than waiting for an error that never comes.
            if i == 0 and (time.time() - t0) > THRASH_S:
                rec["verdict"] = "THRASH"
                rec["s_step"] = round(time.time() - t0, 2)
                break
        else:
            rec["s_step"] = round((time.time() - t0) / STEPS, 2)
            rec["verdict"] = "OK"

        rec["peak_alloc_gb"] = gb(torch.cuda.max_memory_allocated())
        rec["peak_reserved_gb"] = gb(torch.cuda.max_memory_reserved())

    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        rec["verdict"] = "UNSUPPORTED" if ("kernel" in msg.lower() or "backend" in msg.lower()) else "ERROR"
        rec["error"] = "%s %s" % (type(e).__name__, msg)
    finally:
        del model, peft_model, opt
        gc.collect()
        torch.cuda.empty_cache()
    return rec


def main():
    os.makedirs(EXP, exist_ok=True)
    total = torch.cuda.get_device_properties(0).total_memory
    out = {"card_gb": gb(total), "rank": RANK, "steps": STEPS,
           "torch": torch.__version__, "cells": []}
    heartbeat("start")

    for name, attn_impl, backend in PATHS:
        for seq in SEQS:
            rec = run_cell(name, attn_impl, backend, seq)
            out["cells"].append(rec)
            print("%-10s seq%-5d %-12s peak %-6s step %-7s %s" % (
                name, seq, rec["verdict"], rec.get("peak_reserved_gb", "-"),
                rec.get("s_step", "-"), rec.get("error", "")), flush=True)
            with open(RESULTS, "w") as f:
                json.dump(out, f, indent=1)
            # a path that cannot hold this sequence will not hold a longer one
            if rec["verdict"] != "OK":
                break

    heartbeat("done")
    print("RESULTS", RESULTS, flush=True)


if __name__ == "__main__":
    main()
