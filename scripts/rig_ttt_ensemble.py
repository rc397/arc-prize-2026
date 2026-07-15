"""Adapter ensembling, the genuinely untested diversity lever.

Per-task TTT is stochastic. It draws a fresh random augmentation every step, so two adapters trained on
the same task with different seeds descend to different local minima and become different solvers. This
votes over the union of the D4 candidates produced by each adapter.

This is NOT the more-candidates lever that failed. That one drew extra samples at temperature 0.8 from a
SINGLE adapter, which is sampling noise, and it correctly diluted the vote (13.61 -> 13.19 hidden).
Candidates from independently trained adapters are genuine model diversity, the classic ensembling win,
and are a different mechanism. If diverse solvers disagree, the vote should sharpen rather than dilute.

Takes N adapter manifests (one per seed), serves each task's adapters together, collects 8 greedy D4
candidates per adapter, and selects the top 2 over the union.

Usage: python rig_ttt_ensemble.py <scale> <out_tag> <manifest1> <manifest2> [manifest3...]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = r"D:\arc"
sys.path.insert(0, BASE)

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.25
TAG = sys.argv[2] if len(sys.argv) > 2 else "ens"
MANIFESTS = sys.argv[3:] if len(sys.argv) > 3 else []

MODEL_GGUF = os.path.join(BASE, "models", "arc-qwen4b-q8_0.gguf")
SERVER = os.path.join(BASE, "llamacpp", "llama-server.exe")
CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
SOL = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_solutions.json")
EXP = os.path.join(BASE, "experiments")
OUT = os.path.join(EXP, "ttt_ens_%s.json" % TAG)
STATUS = os.path.join(BASE, "eval_status_sweep.json")
PORT = 8080
BATCH = 8   # tasks per server start, times N seeds adapters loaded each

from src.arc.model import build_prompt_ids                                   # noqa: E402
from src.arc.data import load_consolidated, write_submission                 # noqa: E402
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES  # noqa: E402
from src.arc.select import select_top2                                       # noqa: E402
from src.arc.eval import score_submission                                    # noqa: E402

AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def health():
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT,
                                                 timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def start_server(lora_paths):
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", "16384", "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--no-cache-prompt",
            "--lora-init-without-apply"]
    for p in lora_paths:
        args += ["--lora", p]
    proc = subprocess.Popen(args, creationflags=0x08000000,
                            stdout=open(os.path.join(BASE, "srv_ens.log"), "a"),
                            stderr=open(os.path.join(BASE, "srv_ens.err"), "a"))
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


def query(prompt_ids, retries=3):
    body = json.dumps({"prompt": prompt_ids, "n_predict": 1536, "temperature": 0.0,
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


def main():
    mans = []
    for m in MANIFESTS:
        mans.append(json.load(open(m)))
    tids = sorted(set(mans[0]).intersection(*[set(m) for m in mans[1:]])) if len(mans) > 1 else sorted(mans[0])
    print("ENSEMBLE tag=%s seeds=%d scale=%s over %d tasks with an adapter from every seed"
          % (TAG, len(mans), SCALE, len(tids)), flush=True)

    tasks, sols = load_consolidated(CH, SOL, n=None)
    submission = {}
    if os.path.exists(OUT):
        try:
            raw = json.load(open(OUT))
            for t, att in raw.items():
                if att and att[0].get("attempt_1") != [[0]]:
                    submission[t] = att
            print("resume: %d tasks already scored" % len(submission), flush=True)
        except Exception:
            submission = {}

    have = [t for t in tids if t not in submission]
    proc = None
    t0 = time.time()
    for b in range(0, len(have), BATCH):
        batch = have[b:b + BATCH]
        # load every seed's adapter for every task in the batch, ordered task-major
        paths, index = [], {}
        for t in batch:
            index[t] = []
            for m in mans:
                index[t].append(len(paths))
                paths.append(m[t])
        stop_server(proc)
        proc = start_server(paths)
        if not health():
            print("server failed to start, aborting", flush=True)
            break
        for t in batch:
            try:
                json.dump({"phase": "ens", "i": b, "n": len(have), "ts": time.time(), "tag": TAG},
                          open(STATUS, "w"))
                atts = []
                for ti in range(len(tasks[t]["test"])):
                    cands = []
                    for slot in index[t]:                     # one adapter per seed = one solver
                        set_adapter(slot, len(paths), SCALE)
                        for aug in AUGS:
                            at = apply_task(aug, tasks[t])
                            r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
                            if r is None:
                                continue
                            g = parse(r.get("content", ""))
                            if g:
                                cands.append((invert_grid(aug, g), 0.0))
                    a1, a2 = select_top2(cands)
                    atts.append({"attempt_1": a1, "attempt_2": a2})
                submission[t] = atts
            except Exception as e:
                print("  %s ERR %r" % (t, str(e)[:70]), flush=True)
                submission[t] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[t]["test"]]
            write_submission(submission, {k: tasks[k] for k in tids}, OUT)
        print("[ens %d/%d] %.2fh" % (min(b + BATCH, len(have)), len(have), (time.time() - t0) / 3600.0),
              flush=True)
    stop_server(proc)

    scored = {t: submission[t] for t in tids if t in submission}
    s = {t: sols[t] for t in tids if t in sols}
    acc, c, n = score_submission(scored, s)
    print("RESULT ensemble tag=%s seeds=%d scale=%s acc=%.4f correct=%d total=%d tasks=%d"
          % (TAG, len(mans), SCALE, acc, c, n, len(scored)), flush=True)


if __name__ == "__main__":
    main()
