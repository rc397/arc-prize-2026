"""Does the deployed DFS lose solutions by sharing one growing KV cache across sibling branches?

The local harness `rig_dfs_ab.py` clones the cache at every branch, and its docstring says why:
"so siblings do not corrupt the parent state". The DEPLOYED Kaggle solver
(`notebooks/v7_2_cells/08_arc_solver.py`) does no such thing. It threads one cache object through
the recursion and reuses it for every sibling in the while-loop, with `position_ids` pinned to the
branch position and no crop anywhere.

Measured on the rig (transformers 5.12.0): DynamicCache APPENDS rather than overwriting, so a
prefix of 10 becomes 11 then 12 across two siblings, and the second sibling's logits move. Two
consequences follow, and both match things we have already observed and failed to explain:

  - every sibling attends over the branches explored before it, so the tree is explored under a
    context that depends on visit order
  - nothing ever releases cache rows, so the cache grows for the whole 540-second search, which on
    a 16 GB card is the constraint this project is named for

That predicts deeper DFS should DEGRADE rather than plateau, and every depth and floor lever we
tried came back neutral or negative.

Three arms, decoding the SAME per-task adapter so the comparison is paired:

  shared   the deployed pattern, uncorrected
  crop     truncate the cache back to the branch position, correct and cheap
  clone    deepcopy per sibling, correct and expensive, the reference

If crop matches clone and both beat shared, the fix is one line and worth a submission slot.
If all three agree, the deployed pattern is harmless and the hypothesis dies for the cost of a
night on an idle card.

NOTE: this measures the RIG's transformers. Kaggle's image (unsloth 2025.9.7) may return a fresh
cache per forward, in which case the deployed code is already correct there and this result does not
transfer. That has to be checked separately before any submission.

Usage: python rig_dfscache_ab.py [N_TASKS=10] [STEPS=60] [RANK=32] [PER_ARM_SECONDS=180]
"""
import json
import os
import sys
import time

import numpy as np
import torch

BASE = r"D:\arc"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.setrecursionlimit(20000)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 32
PER_ARM = int(sys.argv[4]) if len(sys.argv) > 4 else 180
MODES = ("shared", "crop", "clone")

# rig_dfs_ab reads its constants from argv at import, so set them before importing it.
sys.argv = ["rig_dfs_ab.py", str(N_TASKS), str(STEPS), str(RANK), "2000", "0"]
import rig_dfs_ab as R                                                    # noqa: E402
from peft import LoraConfig, get_peft_model                               # noqa: E402
from src.arc.data import load_consolidated                                # noqa: E402
from src.arc.model import load_model                                      # noqa: E402
from src.arc.ttt import test_time_train                                   # noqa: E402

OUT = os.path.join(BASE, "experiments", "dfscache_ab.json")
DEV = R.DEV


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def main():
    tasks, sols = load_consolidated(R.CH, R.SOL, n=None)
    tids = [t for t in tasks if R.maxpair_tokens(tasks[t]) <= R.MAXPAIR]
    tids.sort(key=lambda t: R.maxpair_tokens(tasks[t]))
    tids = tids[:N_TASKS]
    say("CONFIG n=%d steps=%d rank=%d per_arm=%ds modes=%s"
        % (len(tids), STEPS, RANK, PER_ARM, ",".join(MODES)))

    done = {}
    if os.path.exists(OUT):
        try:
            done = json.load(open(OUT))
            say("resume: %d tasks already done" % len(done))
        except Exception:
            done = {}

    model, tok = load_model(R.BASE_HF, dtype=torch.bfloat16, device=DEV)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(r=RANK, lora_alpha=R.ALPHA, lora_dropout=0.1,
                     target_modules=targets, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    init = {k: v.detach().clone() for k, v in peft_model.named_parameters() if "lora_" in k}
    rng = np.random.default_rng(0)

    for i, tid in enumerate(tids):
        if tid in done:
            continue
        say("[%d/%d] %s training adapter" % (i + 1, len(tids), tid))
        try:
            test_time_train(peft_model, init, tasks[tid], rng, steps=STEPS,
                            max_ctx_tokens=2000, time_budget=250)
        except Exception as e:
            say("  TRAIN-FAIL %r" % str(e)[:100])
            continue

        rec = {}
        for mode in MODES:
            # One adapter, decoded three ways. Nothing between arms touches the weights, so any
            # difference is the cache pattern and nothing else.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            solved_k = solved_p = 0.0
            ncands = 0
            trunc = False
            for ti in range(len(tasks[tid]["test"])):
                sol = sols[tid][ti]
                deadline = time.time() + PER_ARM
                try:
                    k_att, p_att, nc, tr = R.dfs_select(peft_model, tasks[tid], ti,
                                                        deadline, mode=mode)
                except Exception as e:
                    say("  %-6s DECODE-FAIL %r" % (mode, str(e)[:90]))
                    k_att, p_att, nc, tr = [[[0]]], [[[0]]], 0, False
                w = 1.0 / len(tasks[tid]["test"])
                solved_k += w * R.solved(k_att, sol)
                solved_p += w * R.solved(p_att, sol)
                ncands += nc
                trunc = trunc or tr
            rec[mode] = {"kgmon": solved_k, "probmul": solved_p, "cands": ncands,
                         "trunc": trunc, "secs": round(time.time() - t0, 1),
                         "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            say("  %-6s kgmon %.2f  probmul %.2f  cands %3d  %5.0fs  peak %.2f GB%s"
                % (mode, solved_k, solved_p, ncands, rec[mode]["secs"],
                   rec[mode]["peak_gb"], "  TRUNC" if trunc else ""))
        done[tid] = rec
        json.dump(done, open(OUT, "w"), indent=1)

    say("")
    say("SUMMARY over %d tasks" % len(done))
    say("  %-6s %8s %8s %8s %8s %8s" % ("mode", "kgmon", "probmul", "cands", "secs", "peakGB"))
    for mode in MODES:
        rs = [r[mode] for r in done.values() if mode in r]
        if not rs:
            continue
        say("  %-6s %8.3f %8.3f %8.1f %8.0f %8.2f"
            % (mode, sum(r["kgmon"] for r in rs) / len(rs),
               sum(r["probmul"] for r in rs) / len(rs),
               sum(r["cands"] for r in rs) / len(rs),
               sum(r["secs"] for r in rs) / len(rs),
               max(r["peak_gb"] for r in rs)))
    say("")
    say("crop == clone > shared  =>  the deployed cache pattern is costing solutions, and the fix")
    say("                            is one line. Worth a submission slot.")
    say("all three equal         =>  the pattern is harmless here and the hypothesis is dead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
