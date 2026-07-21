"""Teacher-forced augmentation scoring of the select16 candidate pools under the BASE model.

Every aggregate selector over the 16-view pools stalls at 28 of 125 while the pools contain 38, and
the diagnostic shows votes and generation logprobs cannot separate the hidden answers from confident
wrong ones. This computes the signal the ARChitects actually select with, the teacher-forced NLL of
each candidate scored AS THE ANSWER under K fresh augmentations of the task, here under the plain base
model (adapter-free, forward passes only). Output feeds a second offline selector round.

Writes experiments/select16_aug_nll.json as {tid: {test_idx: {candidate_json: [K nlls]}}}.
Waits for the GPU behind the big-grid run. Resume-safe per output.

Usage: python rig_tfscore.py [K=8] [WAIT=1]
"""
import json
import os
import subprocess
import sys
import time

BASE = r"D:\arc"
sys.path.insert(0, BASE)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WAIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1

BASE_HF = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
VIEWS = os.path.join(BASE, "experiments", "select16_views.json")
OUT = os.path.join(BASE, "experiments", "select16_aug_nll.json")
STATUS = os.path.join(BASE, "eval_status_tfscore.json")

from src.arc.model import IM_END, build_prompt_ids, grid_to_ids                  # noqa: E402
from src.arc.augment import apply_task, random_augmentation                      # noqa: E402


def gkey(g):
    return tuple(tuple(int(v) for v in r) for r in g) if g else ()


def heartbeat(phase, i, n, extra=""):
    try:
        json.dump({"phase": phase, "i": i, "n": n, "ts": time.time(), "extra": extra}, open(STATUS, "w"))
    except Exception:
        pass


def gpu_mem_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def rival_jobs_running():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return True
    for line in out.splitlines():
        if any(s in line for s in ("rig_ttt_sweep.py", "rig_rankval.py", "rig_dfs_ab.py",
                                   "rig_biggrid_probe.py", "rig_select16.py")):
            return True
    return False


def wait_for_gpu_idle():
    ok = 0
    i = 0
    while ok < 3:
        mem = gpu_mem_used_mb()
        busy = rival_jobs_running()
        ok = ok + 1 if (0 <= mem < 1500 and not busy) else 0
        heartbeat("wait", i, 0, "mem=%dMB busy=%s ok=%d/3" % (mem, busy, ok))
        i += 1
        if ok < 3:
            time.sleep(120)
    print("gate open", flush=True)


def main():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    views = json.load(open(VIEWS))
    tasks = json.load(open(CH))
    done = {}
    if os.path.exists(OUT):
        try:
            done = json.load(open(OUT))
            print("resume: %d tasks scored" % len(done), flush=True)
        except Exception:
            done = {}

    if WAIT:
        wait_for_gpu_idle()

    model = AutoModelForCausalLM.from_pretrained(BASE_HF, dtype=torch.bfloat16)
    model.to("cuda").eval()

    @torch.inference_mode()
    def tf_nll(prompt_ids, answer_ids):
        ids = torch.tensor([prompt_ids + answer_ids], device="cuda")
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
        p = len(prompt_ids)
        s = 0.0
        for i, tok in enumerate(answer_ids):
            s += -float(logp[p - 1 + i, tok])
        return s

    tids = [t for t in views if "r32" in views[t] and "r64" in views[t]]
    t0 = time.time()
    for i, tid in enumerate(tids):
        if tid in done:
            continue
        heartbeat("score", i, len(tids), tid)
        task = tasks[tid]
        rec = {}
        try:
            for ti in range(len(views[tid]["r32"])):
                pool = {}
                for cfg in ("r32", "r64"):
                    for x in views[tid][cfg][ti]:
                        if x["grid"]:
                            pool[gkey(x["grid"])] = x["grid"]
                scores = {}
                for ck, grid in pool.items():
                    seed = abs(hash((tid, ti, ck))) % (2 ** 31)
                    rng = np.random.default_rng(seed)
                    local = {"train": task["train"],
                             "test": [{"input": task["test"][ti]["input"], "output": grid}]}
                    nlls = []
                    for _ in range(K):
                        aug = random_augmentation(rng, n_examples=len(task["train"]))
                        at = apply_task(aug, local)
                        prompt = build_prompt_ids(at["train"], at["test"][0]["input"])
                        ans = grid_to_ids(at["test"][0]["output"]) + [IM_END]
                        nlls.append(round(tf_nll(prompt, ans), 3))
                    scores[json.dumps(grid)] = nlls
                rec[str(ti)] = scores
            done[tid] = rec
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("[%d/%d] %s OOM, skipped" % (i + 1, len(tids), tid), flush=True)
        except Exception as e:
            print("[%d/%d] %s FAIL %r" % (i + 1, len(tids), tid, str(e)[:90]), flush=True)
        json.dump(done, open(OUT, "w"))
        if (i + 1) % 10 == 0 or i + 1 == len(tids):
            print("[%d/%d] %.2fh" % (i + 1, len(tids), (time.time() - t0) / 3600.0), flush=True)
    print("DONE %d tasks scored" % len(done), flush=True)
    heartbeat("done", 1, 1)


if __name__ == "__main__":
    main()
