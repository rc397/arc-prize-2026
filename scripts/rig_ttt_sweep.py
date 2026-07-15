"""Rig sweep of the per-task TTT configuration, the one lever with a proven mechanism.

The original sweep conflated depth with regularization. It tested 40 steps at rank 32 with NO dropout
(overfit, tied the baseline) against 20 steps at rank 16 WITH dropout (won), so "more steps" was only
ever measured with the regularizer removed. The 2025 winners ran 128 steps at rank 32 with a fresh
augmentation every step. This script tests the untested corner, deep AND regularized, against the
current 20/16/0.1 operating point.

One config per invocation. Trains a fresh LoRA per task, converts it to GGUF, serves the batch on
llama.cpp, runs AIRV with the adapter applied at SCALE, and scores against the solutions. Resume,
heartbeat, and per-task checkpoint so a long run survives interruption.

Usage: python rig_ttt_sweep.py [N_TASKS] [STEPS] [RANK] [DROPOUT] [SCALE] [TAG]
  N_TASKS 0 = all trainable (maxpair <= 1500). Baseline config: 0 20 16 0.1 1.25 base
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

BASE = r"D:\arc"
sys.path.insert(0, BASE)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 16
DROPOUT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
SCALE = float(sys.argv[5]) if len(sys.argv) > 5 else 1.25
TAG = sys.argv[6] if len(sys.argv) > 6 else "cfg"

MODEL_GGUF = os.path.join(BASE, "models", "arc-qwen4b-q8_0.gguf")
SERVER = os.path.join(BASE, "llamacpp", "llama-server.exe")
CONVERT = os.path.join(BASE, "llama.cpp", "convert_lora_to_gguf.py")
GGUF_PY = os.path.join(BASE, "llama.cpp", "gguf-py")
BASE_HF = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
SOL = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_solutions.json")
EXP = os.path.join(BASE, "experiments")
ADAPTERS = os.path.join(BASE, "models", "sweep_%s" % TAG)
# adapters are keyed by TAG (train once), the score file by TAG+SCALE, so re-running a tag at a new
# scale reuses the trained adapters and only re-serves. Scale is an inference-time knob.
OUT = os.path.join(EXP, "ttt_sweep_%s_s%s.json" % (TAG, str(SCALE).replace(".", "")))
STATUS = os.path.join(BASE, "eval_status_sweep.json")
PORT = 8080
# 1000, not the notebook's 1500. 1500 is the L4's 24GB budget. On the rig's 16GB it selects 111 tasks
# and the big ones OOM, and a rank-32 or deeper config would OOM more often than a shallow one, dropping
# different tasks per config and confounding the comparison. 1000 is the rig's validated limit and gives
# the same 84 trainable tasks the recorded 17.6 baseline was measured on.
MAXPAIR = 1000
BATCH = 24
CTX = "16384"

from src.arc.model import build_prompt_ids, grid_to_ids                      # noqa: E402
from src.arc.data import load_consolidated, write_submission                 # noqa: E402
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES  # noqa: E402
from src.arc.select import select_top2                                       # noqa: E402
from src.arc.eval import score_submission                                    # noqa: E402

AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def maxpair_tokens(task):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in task["train"])


def health():
    try:
        u = "http://127.0.0.1:%d/health" % PORT
        return json.loads(urllib.request.urlopen(u, timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def start_server(lora_paths=()):
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", CTX, "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--no-cache-prompt"]
    if lora_paths:
        args += ["--lora-init-without-apply"]
        for p in lora_paths:
            args += ["--lora", p]
    proc = subprocess.Popen(args, creationflags=0x08000000,
                            stdout=open(os.path.join(BASE, "srv_sweep.log"), "a"),
                            stderr=open(os.path.join(BASE, "srv_sweep.err"), "a"))
    for _ in range(150):
        time.sleep(2)
        if health():
            return proc
    return proc


def stop_server(proc):
    try:
        if proc:
            proc.terminate()
            proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    time.sleep(1)


def set_adapter(idx, n_loaded, scale):
    body = json.dumps([{"id": j, "scale": (scale if j == idx else 0.0)} for j in range(n_loaded)]).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/lora-adapters" % PORT, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def query(prompt_ids, n_predict=1536, retries=3):
    body = json.dumps({"prompt": prompt_ids, "n_predict": n_predict, "temperature": 0.0,
                       "top_p": 0.95, "cache_prompt": False}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception:
            if a < retries - 1:
                time.sleep(3)
    return None


def parse(text):
    rows = []
    for line in text.split("\n"):
        s = line.strip()
        if s and all(c in "0123456789" for c in s):
            rows.append([int(c) for c in s])
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None


def airv(task, adapter_idx=None, n_loaded=0):
    if adapter_idx is not None:
        set_adapter(adapter_idx, n_loaded, SCALE)
    out = []
    for ti in range(len(task["test"])):
        cands = []
        for aug in AUGS:
            at = apply_task(aug, task)
            r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
            if r is None:
                continue
            g = parse(r.get("content", ""))
            if g:
                cands.append((invert_grid(aug, g), 0.0))
        a1, a2 = select_top2(cands)
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out


def heartbeat(phase, i, n):
    try:
        json.dump({"phase": phase, "i": i, "n": n, "ts": time.time(), "tag": TAG,
                   "steps": STEPS, "rank": RANK, "dropout": DROPOUT, "scale": SCALE},
                  open(STATUS, "w"))
    except Exception:
        pass


def main():
    tasks, sols = load_consolidated(CH, SOL, n=None)
    tids = [t for t in tasks if maxpair_tokens(tasks[t]) <= MAXPAIR]
    tids.sort(key=lambda t: maxpair_tokens(tasks[t]))
    if N_TASKS:
        tids = tids[:N_TASKS]
    print("CONFIG tag=%s steps=%d rank=%d dropout=%s scale=%s over %d trainable tasks"
          % (TAG, STEPS, RANK, DROPOUT, SCALE, len(tids)), flush=True)

    os.makedirs(ADAPTERS, exist_ok=True)
    manifest = {}
    mpath = os.path.join(EXP, "sweep_manifest_%s.json" % TAG)
    if os.path.exists(mpath):
        try:
            manifest = {k: v for k, v in json.load(open(mpath)).items() if os.path.exists(v)}
            print("resume: %d adapters already built" % len(manifest), flush=True)
        except Exception:
            manifest = {}

    # Phase A, train + convert each adapter
    todo = [t for t in tids if t not in manifest]
    if todo:
        import torch
        from src.arc.model import load_model
        from src.arc.ttt import make_adapter, test_time_train
        import numpy as np
        cenv = dict(os.environ, PYTHONPATH=GGUF_PY + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    PYTHONIOENCODING="utf-8")
        model, tok = load_model(BASE_HF, dtype=torch.bfloat16, device="cuda")
        peft_model, init = make_adapter(model, rank=RANK, alpha=RANK, dropout=DROPOUT)
        rng = np.random.default_rng(0)
        t0 = time.time()
        for i, tid in enumerate(todo):
            heartbeat("train", i, len(todo))
            adir = os.path.join(ADAPTERS, tid)
            gguf = os.path.join(ADAPTERS, tid + ".gguf")
            try:
                test_time_train(peft_model, init, tasks[tid], rng, steps=STEPS,
                                max_ctx_tokens=2000, time_budget=400)
                peft_model.save_pretrained(adir)
                subprocess.run([sys.executable, CONVERT, adir, "--base", BASE_HF, "--outfile", gguf,
                                "--outtype", "f16"], env=cenv,
                               stdout=open(os.path.join(BASE, "convert_sweep.log"), "ab"),
                               stderr=subprocess.STDOUT, timeout=600)
                if os.path.exists(gguf):
                    manifest[tid] = gguf
                shutil.rmtree(adir, ignore_errors=True)
            except Exception as e:
                shutil.rmtree(adir, ignore_errors=True)
                print("[train %d/%d] %s FAIL %r" % (i + 1, len(todo), tid, str(e)[:80]), flush=True)
            json.dump(manifest, open(mpath, "w"))
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                print("[train %d/%d] %.2fh elapsed, %d adapters" % (i + 1, len(todo),
                      (time.time() - t0) / 3600.0, len(manifest)), flush=True)
        del peft_model, model
        torch.cuda.empty_cache()
    print("Phase A done, %d adapters ready" % len(manifest), flush=True)

    # Phase B, serve in batches and run AIRV with each task's adapter
    submission = {}
    if os.path.exists(OUT):
        try:
            raw = json.load(open(OUT))
            for t, att in raw.items():
                if att and att[0].get("attempt_1") != [[0]]:
                    submission[t] = att
            print("resume: %d scored tasks loaded" % len(submission), flush=True)
        except Exception:
            submission = {}

    have = [t for t in tids if t in manifest and t not in submission]
    proc = None
    t0 = time.time()
    for b in range(0, len(have), BATCH):
        batch = have[b:b + BATCH]
        stop_server(proc)
        proc = start_server([manifest[t] for t in batch])
        if not health():
            print("server failed to start, aborting", flush=True)
            break
        for k, tid in enumerate(batch):
            heartbeat("serve", b + k, len(have))
            try:
                submission[tid] = airv(tasks[tid], adapter_idx=k, n_loaded=len(batch))
            except Exception as e:
                print("  %s ERR %r" % (tid, str(e)[:70]), flush=True)
                submission[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[tid]["test"]]
            write_submission(submission, {t: tasks[t] for t in tids}, OUT)
            if (b + k + 1) % 10 == 0:
                print("[serve %d/%d] %.2fh" % (b + k + 1, len(have), (time.time() - t0) / 3600.0), flush=True)
    stop_server(proc)

    scored = {t: submission[t] for t in tids if t in submission}
    s = {t: sols[t] for t in tids if t in sols}
    acc, c, n = score_submission(scored, s)
    print("RESULT tag=%s steps=%d rank=%d dropout=%s scale=%s acc=%.4f correct=%d total=%d tasks=%d"
          % (TAG, STEPS, RANK, DROPOUT, SCALE, acc, c, n, len(scored)), flush=True)


if __name__ == "__main__":
    main()
