"""Measure what per-task TTT configurations actually fit on one 16 GB card.

The paper asserts that the heavy public-pipeline config (rank 256 at an eight-thousand-token context)
does not fit 16 GB. The v7 Kaggle logs report 13011 MB allocated for exactly that config, which puts
the claim in doubt, so this measures it directly instead of arguing about it.

Runs a ladder of (rank, max_ctx) configurations from the shipped v6.1 point up to the heavy point,
training a few real steps on the largest evaluation tasks so the sequences are genuinely long. Each rung
reports peak reserved VRAM, seconds per step and a fit verdict. An out-of-memory rung is recorded and
the ladder continues, so one run produces the whole curve.

No Unsloth and no CPU offload. Plain peft on the native loader, which is what our stack already runs.

Usage: python rig_fitprobe.py [MODEL=4b] [STEPS=6] [N_TASKS=2]
  MODEL  4b, 2b, or an explicit path to a HF model directory.
"""
import gc
import json
import os
import sys
import time

BASE = r"D:\arc"
sys.path.insert(0, BASE)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "4b"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
N_TASKS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

MODELS = {
    "4b": os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1"),
    "2b": os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_2b_grids15_sft141", "transformers", "bfloat16", "1"),
}
MODEL_DIR = MODELS.get(MODEL, MODEL)

CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
EXP = os.path.join(BASE, "experiments")
STATUS = os.path.join(BASE, "eval_status_fitprobe.json")
RESULTS = os.path.join(EXP, "fitprobe_%s.json" % MODEL)

# rank, max_ctx, label. The first rung is the shipped v6.1 point and anchors the curve.
LADDER = [
    (32, 2000, "v6.1 shipped"),
    (64, 4096, "mid"),
    (128, 8192, "heavy rank-128"),
    (256, 8192, "v7.1 heavy"),
]

# treat this as the usable budget. The card is 16 GB and Windows holds back roughly a gigabyte.
FIT_GB = 14.5
TIME_BUDGET = 600

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from src.arc.model import load_model, grid_to_ids                     # noqa: E402
from src.arc.data import load_consolidated                            # noqa: E402
from src.arc.ttt import make_adapter, test_time_train, _example       # noqa: E402


def heartbeat(phase, i, n, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "i": i, "n": n, "ts": time.time(),
                   "model": MODEL, "extra": extra}, f)
    os.replace(tmp, STATUS)


def gpu_used_mb():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def maxpair_tokens(task):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in task["train"])


def pick_tasks():
    tasks, _ = load_consolidated(CH)
    ranked = sorted(tasks.items(), key=lambda kv: -maxpair_tokens(kv[1]))
    return ranked[:N_TASKS]


def free_all(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_rung(rank, maxctx, label, tasks):
    print("\n=== rank %d  ctx %d  (%s) ===" % (rank, maxctx, label), flush=True)
    heartbeat("rung", rank, maxctx, label)
    rec = {"rank": rank, "max_ctx": maxctx, "label": label}
    model = peft_model = None
    try:
        t0 = time.time()
        model = load_model(MODEL_DIR)
        rec["load_s"] = round(time.time() - t0, 1)
        rec["weights_gb"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)
        print("  weights %.2f GB loaded in %.0fs" % (rec["weights_gb"], rec["load_s"]), flush=True)

        peft_model, init = make_adapter(model, rank=rank, alpha=rank, dropout=0.1)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        rec["trainable_m"] = round(trainable / 1e6, 1)
        print("  trainable %.1fM params" % rec["trainable_m"], flush=True)

        per_task = []
        for tid, task in tasks:
            rng = np.random.default_rng([0, int(tid, 16) % (2 ** 31)])
            ids, _ = _example(task, rng, max_ctx_tokens=maxctx)
            seq = len(ids)
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            loss = test_time_train(peft_model, init, task, rng, steps=STEPS,
                                   max_ctx_tokens=maxctx, time_budget=TIME_BUDGET)
            wall = time.time() - t1
            row = {"task": tid, "maxpair": maxpair_tokens(task), "seq_len": seq,
                   "loss": round(float(loss), 4), "s_step": round(wall / max(STEPS, 1), 2),
                   "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                   "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
                   "smi_mb": gpu_used_mb()}
            per_task.append(row)
            print("  %s seq=%d  %.2fs/step  peak_res %.2f GB  smi %d MB  loss %.4f"
                  % (tid, seq, row["s_step"], row["peak_reserved_gb"], row["smi_mb"], row["loss"]),
                  flush=True)
        rec["tasks"] = per_task
        rec["peak_reserved_gb"] = max(r["peak_reserved_gb"] for r in per_task)
        rec["max_smi_mb"] = max(r["smi_mb"] for r in per_task)
        rec["s_step"] = round(sum(r["s_step"] for r in per_task) / len(per_task), 2)
        rec["fits"] = rec["peak_reserved_gb"] <= FIT_GB
        rec["verdict"] = ("FITS" if rec["peak_reserved_gb"] <= FIT_GB - 1.5 else
                          "TIGHT" if rec["fits"] else "OVER")
    except torch.cuda.OutOfMemoryError as e:
        rec["fits"] = False
        rec["verdict"] = "OOM"
        rec["error"] = str(e).splitlines()[0][:200]
        print("  OOM: %s" % rec["error"], flush=True)
    except Exception as e:
        rec["fits"] = False
        rec["verdict"] = "ERROR"
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        print("  ERROR: %s" % rec["error"], flush=True)
    finally:
        free_all(peft_model, model)
        rec["after_free_smi_mb"] = gpu_used_mb()
    return rec


def main():
    assert os.path.isdir(MODEL_DIR), "model dir not found: %s" % MODEL_DIR
    os.makedirs(EXP, exist_ok=True)
    baseline_mb = gpu_used_mb()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("model   %s" % MODEL_DIR)
    print("device  %s  %.1f GB total  %d MB already used"
          % (torch.cuda.get_device_name(0), total_gb, baseline_mb))
    print("steps %d per task, %d tasks, fit budget %.1f GB" % (STEPS, N_TASKS, FIT_GB), flush=True)

    tasks = pick_tasks()
    print("stress tasks: %s" % ", ".join("%s(%d)" % (t, maxpair_tokens(k)) for t, k in tasks), flush=True)

    out = {"model": MODEL, "model_dir": MODEL_DIR, "steps": STEPS,
           "total_gb": round(total_gb, 1), "baseline_mb": baseline_mb, "rungs": []}
    for rank, maxctx, label in LADDER:
        out["rungs"].append(run_rung(rank, maxctx, label, tasks))
        with open(RESULTS, "w") as f:
            json.dump(out, f, indent=1)

    print("\n==== ladder ====")
    print("%-6s %-6s %-16s %10s %9s %8s" % ("rank", "ctx", "label", "peak_res_GB", "s/step", "verdict"))
    for r in out["rungs"]:
        print("%-6d %-6d %-16s %10s %9s %8s"
              % (r["rank"], r["max_ctx"], r["label"],
                 r.get("peak_reserved_gb", "-"), r.get("s_step", "-"), r["verdict"]))

    heavy = [r for r in out["rungs"] if r["rank"] == 256]
    if heavy and heavy[0].get("fits"):
        h = heavy[0]
        print("\nThe heavy config FITS at %.2f GB reserved, %.2f s/step." % (h["peak_reserved_gb"], h["s_step"]))
        print("Projected 120-task eval train time at 128 steps: %.1f h"
              % (h["s_step"] * 128 * 120 / 3600.0))
        print("The paper's does-not-fit claim needs correcting.")
    elif heavy:
        print("\nThe heavy config does NOT fit (%s). The paper's claim stands." % heavy[0]["verdict"])
    heartbeat("done", len(out["rungs"]), len(LADDER), "complete")
    print("\nwrote %s" % RESULTS)


if __name__ == "__main__":
    main()
