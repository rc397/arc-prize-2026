"""Map what per-task TTT configurations fit on one 16 GB card, and where the memory actually goes.

Supersedes a first version whose design was wrong in two ways. It varied rank and context together so a
failure could not be attributed to either, and it drove sequence length by picking large real tasks, but
max_ctx_tokens only stops packing further train pairs and never truncates, so the sequence length was an
uncontrolled side effect of which task got chosen.

Here sequence length is set exactly, using synthetic token batches. Memory and step time depend on tensor
shapes rather than token values, so a synthetic sequence of length N costs the same as a real one, and it
decouples the measurement from task selection entirely. The training setup mirrors src/arc/ttt.py exactly,
gradient checkpointing on, cache off, input grads enabled, per-step clip and AdamW.

Each cell reports where the memory went, weights against adapter state against activations, because the
interesting question is not whether a config fits but which term is blocking it. Two modes exist for that
reason. std is our shipped path, peft's fp32 adapters with torch AdamW. lean casts adapters to bf16 and
uses an 8-bit optimizer when bitsandbytes is present, which is roughly what Unsloth does on the four-L4
runs that fit rank 256 into 13 GB.

Usage: python rig_fitprobe.py [MODEL=4b] [MODE=std|lean|both] [STEPS=3]
"""
import gc
import json
import os
import sys
import time

BASE = r"D:\arc"
sys.path.insert(0, BASE)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "4b"
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

MODELS = {
    "4b": os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1"),
    "2b": os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_2b_grids15_sft141", "transformers", "bfloat16", "1"),
}
MODEL_DIR = MODELS.get(MODEL, MODEL)

EXP = os.path.join(BASE, "experiments")
STATUS = os.path.join(BASE, "eval_status_fitprobe.json")
RESULTS = os.path.join(EXP, "fitprobe_%s.json" % MODEL)

RANKS = [32, 64, 128, 256]
SEQS = [2048, 4096, 8192]
ATTN = "sdpa"
LR = 1e-4
GRAD_CLIP = 1.0

# a cell slower than this is spilling into host RAM rather than computing, which is a failure even
# though nothing raised. Abort it immediately instead of paying minutes to confirm.
THRASH_S = 8.0
SPILL_FRAC = 0.95

import torch                                                          # noqa: E402
from peft import LoraConfig, get_peft_model                           # noqa: E402
from transformers import AutoModelForCausalLM                         # noqa: E402

from src.arc.ttt import LORA_TARGETS                                  # noqa: E402

TOTAL_GB = 0.0


def heartbeat(phase, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "ts": time.time(), "model": MODEL, "extra": extra}, f)
    os.replace(tmp, STATUS)


def gb(x):
    return round(x / 1024 ** 3, 2)


def load_base():
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                 attn_implementation=ATTN)
    model.to("cuda").eval()
    return model


def build_optimizer(params, lean):
    if lean:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=LR), "adamw8bit"
        except Exception:
            return torch.optim.AdamW(params, lr=LR), "adamw(no-bnb)"
    return torch.optim.AdamW(params, lr=LR), "adamw"


def run_cell(rank, seq, lean):
    tag = "rank%d seq%d %s" % (rank, seq, "lean" if lean else "std")
    heartbeat("cell", tag)
    rec = {"rank": rank, "seq": seq, "mode": "lean" if lean else "std"}
    model = peft_model = opt = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = load_base()
        w = torch.cuda.memory_allocated()
        rec["weights_gb"] = gb(w)

        cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.1,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        if lean:
            for n, p in peft_model.named_parameters():
                if "lora_" in n and p.dtype == torch.float32:
                    p.data = p.data.to(torch.bfloat16)
        params = [p for p in peft_model.parameters() if p.requires_grad]
        rec["trainable_m"] = round(sum(p.numel() for p in params) / 1e6, 1)
        rec["param_dtype"] = str(params[0].dtype).replace("torch.", "")
        rec["adapter_gb"] = gb(torch.cuda.memory_allocated() - w)

        # our TTT keeps a full clone of the adapter to reset between tasks. At high rank that copy is
        # not free, so price it separately rather than hiding it in the activation residual.
        snap = {k: v.detach().clone() for k, v in peft_model.named_parameters() if "lora_" in k}
        rec["init_snapshot_gb"] = gb(sum(v.numel() * v.element_size() for v in snap.values()))

        base = peft_model.get_base_model()
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        peft_model.train()
        opt, opt_name = build_optimizer(params, lean)
        rec["optimizer"] = opt_name

        ids = torch.randint(0, 10, (1, seq), device="cuda")
        labels = ids.clone()
        labels[:, : seq // 2] = -100

        pre_step = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i in range(STEPS):
            loss = peft_model(input_ids=ids, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad()
            if i == 0 and (time.time() - t0) > THRASH_S:
                rec["verdict"] = "THRASH"
                rec["s_step"] = round(time.time() - t0, 2)
                break
        else:
            wall = time.time() - t0
            rec["s_step"] = round(wall / STEPS, 2)
            rec["verdict"] = "THRASH" if rec["s_step"] > THRASH_S else "OK"

        rec["peak_alloc_gb"] = gb(torch.cuda.max_memory_allocated())
        rec["peak_reserved_gb"] = gb(torch.cuda.max_memory_reserved())
        rec["optim_state_gb"] = gb(torch.cuda.memory_allocated() - pre_step)
        rec["activations_gb"] = round(max(0.0, rec["peak_alloc_gb"] - gb(pre_step)
                                          - rec["optim_state_gb"]), 2)
        if rec["peak_reserved_gb"] > TOTAL_GB * SPILL_FRAC and rec["verdict"] == "OK":
            rec["verdict"] = "SPILL"
    except torch.cuda.OutOfMemoryError as e:
        rec["verdict"] = "OOM"
        rec["error"] = str(e).splitlines()[0][:160]
    except Exception as e:
        rec["verdict"] = "ERROR"
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    finally:
        del opt, peft_model, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return rec


def main():
    global TOTAL_GB
    assert os.path.isdir(MODEL_DIR), "model dir not found: %s" % MODEL_DIR
    os.makedirs(EXP, exist_ok=True)
    TOTAL_GB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    modes = [False, True] if MODE == "both" else [MODE == "lean"]
    print("model  %s" % MODEL_DIR)
    print("device %s  %.1f GB  attn=%s  steps=%d"
          % (torch.cuda.get_device_name(0), TOTAL_GB, ATTN, STEPS), flush=True)

    out = {"model": MODEL, "total_gb": round(TOTAL_GB, 1), "attn": ATTN,
           "steps": STEPS, "cells": []}
    for lean in modes:
        label = "lean (bf16 adapters, 8-bit optim)" if lean else "std (fp32 adapters, AdamW)"
        print("\n######## %s ########" % label, flush=True)
        for rank in RANKS:
            for seq in SEQS:
                rec = run_cell(rank, seq, lean)
                out["cells"].append(rec)
                with open(RESULTS, "w") as f:
                    json.dump(out, f, indent=1)
                print("  rank %-4d seq %-5d  %-6s peak_res %-6s s/step %-7s "
                      "[w %s + adapter %s + optim %s + act %s]"
                      % (rank, seq, rec["verdict"], rec.get("peak_reserved_gb", "-"),
                         rec.get("s_step", "-"), rec.get("weights_gb", "-"),
                         rec.get("adapter_gb", "-"), rec.get("optim_state_gb", "-"),
                         rec.get("activations_gb", "-")), flush=True)
                if rec["verdict"] in ("OOM", "THRASH", "SPILL"):
                    print("       longer sequences at this rank will not fit, skipping", flush=True)
                    break

    print("\n==== summary ====")
    print("%-6s %-6s %-6s %-10s %-8s %s" % ("mode", "rank", "seq", "peak_res", "s/step", "verdict"))
    for c in out["cells"]:
        print("%-6s %-6d %-6d %-10s %-8s %s"
              % (c["mode"], c["rank"], c["seq"], c.get("peak_reserved_gb", "-"),
                 c.get("s_step", "-"), c["verdict"]))

    for mode in ("std", "lean"):
        ok = [c for c in out["cells"] if c["mode"] == mode and c["verdict"] == "OK"]
        if not ok:
            continue
        best = max(ok, key=lambda c: (c["rank"], c["seq"]))
        print("\n%s best fitting cell: rank %d at seq %d, %.2f GB, %.2f s/step"
              % (mode, best["rank"], best["seq"], best["peak_reserved_gb"], best["s_step"]))
    heavy = [c for c in out["cells"] if c["rank"] == 256 and c["seq"] == 8192]
    for c in heavy:
        print("heavy config (rank 256, seq 8192) in %s mode: %s" % (c["mode"], c["verdict"]))
    heartbeat("done", "complete")
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
