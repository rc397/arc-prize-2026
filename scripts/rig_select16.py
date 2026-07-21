"""Cross-config candidate pool with per-view logging, the selection side of the two-adapter ensemble.

The rank-32 and rank-64 sweeps solve overlapping-but-different tasks, union 32 of 125 against the best
single 28, but the naive first-attempt ensemble scores 26 because the vote's second choice carries real
solves. Whether the extra 4 outputs are reachable is a SELECTION question over a 16-view candidate pool
(8 D4 views from each config's adapter), which needs per-view grids and their sequence log-probs, data
the sweep discards. This re-serves both configs' existing GGUF adapters, logs every view's grid and
logprob, then evaluates selector rules offline against the union ceiling.

Selectors compared at the end, per test output, choose 2 distinct grids from the 16-view pool:
  single32 / single64: consensus-then-NLL within one config (sanity, should match the sweep scores)
  xvote: consensus across all 16 views, NLL tiebreak
  xvote_lp: consensus across 16, but rank by mean sequence logprob among tied
  best1x2: each config's top-1 as the two attempts (the naive ensemble, known 26)

Pure inference on existing adapters. Waits for the GPU behind the big-grid probe and any sweep. Port 8082.

Usage: python rig_select16.py [N_TASKS=0 all-84] [WAIT=1]
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

BASE = r"D:\arc"
sys.path.insert(0, BASE)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WAIT = int(sys.argv[2]) if len(sys.argv) > 2 else 1

CONFIGS = [
    ("r32", os.path.join(BASE, "models", "sweep_deep60r32"), 1.25),
    ("r64", os.path.join(BASE, "models", "sweep_rankval_r64"), 1.25),
]
MAXPAIR = 1000
PORT = 8082
BATCH = 12
CTX = "16384"
N_PREDICT = 1536

MODEL_GGUF = os.path.join(BASE, "models", "arc-qwen4b-q8_0.gguf")
SERVER = os.path.join(BASE, "llamacpp", "llama-server.exe")
CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
SOL = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_solutions.json")
OUT = os.path.join(BASE, "experiments", "select16_views.json")
STATUS = os.path.join(BASE, "eval_status_select16.json")

from src.arc.model import build_prompt_ids, grid_to_ids                     # noqa: E402
from src.arc.data import load_consolidated                                   # noqa: E402
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES  # noqa: E402

AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def maxpair_tokens(task):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in task["train"])


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
                                   "rig_biggrid_probe.py")):
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


def health():
    try:
        u = "http://127.0.0.1:%d/health" % PORT
        return json.loads(urllib.request.urlopen(u, timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def start_server(lora_paths=()):
    assert not health(), "port %d already serving, refusing to adopt" % PORT
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", CTX, "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--no-cache-prompt"]
    if lora_paths:
        args += ["--lora-init-without-apply"]
        for p in lora_paths:
            args += ["--lora", p]
    proc = subprocess.Popen(args, creationflags=0x08000000,
                            stdout=open(os.path.join(BASE, "srv_select16.log"), "a"),
                            stderr=open(os.path.join(BASE, "srv_select16.err"), "a"))
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
    body = json.dumps({"prompt": prompt_ids, "n_predict": N_PREDICT, "temperature": 0.0,
                       "top_p": 0.95, "cache_prompt": False, "n_probs": 1}).encode()
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


def seq_logprob(resp):
    """Sum log(prob of the chosen token) over generated tokens, from n_probs=1 output."""
    total = 0.0
    n = 0
    for entry in resp.get("completion_probabilities", []):
        # server variants name the chosen-token probability differently across versions
        p = None
        if "probs" in entry and entry["probs"]:
            p = entry["probs"][0].get("prob")
        elif "top_probs" in entry and entry["top_probs"]:
            p = entry["top_probs"][0].get("prob")
        elif "prob" in entry:
            p = entry.get("prob")
        if p and p > 0:
            total += math.log(p)
            n += 1
    return total if n else None


def serve_all(tasks, tids, views):
    for cname, adir, scale in CONFIGS:
        todo = [t for t in tids if cname not in views.get(t, {})]
        have = [t for t in todo if os.path.exists(os.path.join(adir, t + ".gguf"))]
        miss = len(todo) - len(have)
        if miss:
            print("%s: %d tasks missing adapters, skipped" % (cname, miss), flush=True)
        print("%s: serving %d tasks" % (cname, len(have)), flush=True)
        proc = None
        try:
            for b in range(0, len(have), BATCH):
                batch = have[b:b + BATCH]
                stop_server(proc)
                proc = start_server([os.path.join(adir, t + ".gguf") for t in batch])
                if not health():
                    print("server failed, aborting %s" % cname, flush=True)
                    break
                for k, tid in enumerate(batch):
                    heartbeat("serve-" + cname, b + k, len(have), tid)
                    set_adapter(k, len(batch), scale)
                    task = tasks[tid]
                    per_test = []
                    for ti in range(len(task["test"])):
                        pv = []
                        for aug in AUGS:
                            at = apply_task(aug, task)
                            r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
                            if r is None:
                                pv.append({"grid": None, "lp": None})
                                continue
                            g = parse(r.get("content", ""))
                            pv.append({"grid": invert_grid(aug, g) if g else None,
                                       "lp": seq_logprob(r)})
                        per_test.append(pv)
                    views.setdefault(tid, {})[cname] = per_test
                    json.dump(views, open(OUT, "w"))
                    if (b + k + 1) % 10 == 0:
                        print("[%s %d/%d]" % (cname, b + k + 1, len(have)), flush=True)
        finally:
            stop_server(proc)
    return views


def pick2(cands):
    """cands: list of (grid, lp). Consensus-vote then mean-lp rank, two distinct grids."""
    agg = {}
    for g, lp in cands:
        if not g:
            continue
        k = gkey(g)
        if k not in agg:
            agg[k] = [0, 0.0, 0, g]
        agg[k][0] += 1
        if lp is not None:
            agg[k][1] += lp
            agg[k][2] += 1
    ranked = sorted(agg.values(), key=lambda v: (v[0], (v[1] / v[2]) if v[2] else -1e9), reverse=True)
    a1 = ranked[0][3] if ranked else [[0]]
    a2 = ranked[1][3] if len(ranked) > 1 else a1
    return a1, a2


def pick2_lp(cands):
    """Consensus for attempt_1, then pure mean-lp among the rest for attempt_2."""
    agg = {}
    for g, lp in cands:
        if not g:
            continue
        k = gkey(g)
        if k not in agg:
            agg[k] = [0, 0.0, 0, g]
        agg[k][0] += 1
        if lp is not None:
            agg[k][1] += lp
            agg[k][2] += 1
    if not agg:
        return [[0]], [[0]]
    byvote = sorted(agg.values(), key=lambda v: (v[0], (v[1] / v[2]) if v[2] else -1e9), reverse=True)
    a1 = byvote[0][3]
    rest = [v for v in agg.values() if gkey(v[3]) != gkey(a1)]
    bylp = sorted(rest, key=lambda v: (v[1] / v[2]) if v[2] else -1e9, reverse=True)
    a2 = bylp[0][3] if bylp else a1
    return a1, a2


def evaluate(views, tids, sols):
    rules = {"single32": 0, "single64": 0, "xvote": 0, "xvote_lp": 0, "best1x2": 0, "ceiling": 0}
    n = 0
    for tid in tids:
        v = views.get(tid, {})
        if "r32" not in v or "r64" not in v:
            continue
        for ti in range(len(v["r32"])):
            sol = sols[tid][ti] if ti < len(sols.get(tid, [])) else None
            if sol is None:
                continue
            n += 1
            sk = gkey(sol)
            c32 = [(x["grid"], x["lp"]) for x in v["r32"][ti]]
            c64 = [(x["grid"], x["lp"]) for x in v["r64"][ti]]
            allc = c32 + c64

            def hit2(pair):
                return sk in (gkey(pair[0]), gkey(pair[1]))

            rules["single32"] += hit2(pick2(c32))
            rules["single64"] += hit2(pick2(c64))
            rules["xvote"] += hit2(pick2(allc))
            rules["xvote_lp"] += hit2(pick2_lp(allc))
            a32, _ = pick2(c32)
            a64, _ = pick2(c64)
            rules["best1x2"] += sk in (gkey(a32), gkey(a64))
            rules["ceiling"] += any(gkey(g) == sk for g, _ in allc)
    print("\n==== selector evaluation over %d outputs ====" % n, flush=True)
    for k in ("single32", "single64", "best1x2", "xvote", "xvote_lp", "ceiling"):
        print("%-9s %3d/%d (%.1f%%)" % (k, rules[k], n, 100.0 * rules[k] / max(n, 1)), flush=True)


def main():
    for _, adir, _ in CONFIGS:
        assert os.path.isdir(adir), "missing adapter dir %s" % adir
    tasks, sols = load_consolidated(CH, SOL, n=None)
    tids = sorted([t for t in tasks if maxpair_tokens(tasks[t]) <= MAXPAIR],
                  key=lambda t: maxpair_tokens(tasks[t]))
    if N_TASKS:
        tids = tids[:N_TASKS]
    print("select16 over %d tasks, port %d" % (len(tids), PORT), flush=True)
    views = {}
    if os.path.exists(OUT):
        try:
            views = json.load(open(OUT))
            print("resume: %d tasks have views" % len(views), flush=True)
        except Exception:
            views = {}
    if WAIT:
        wait_for_gpu_idle()
    views = serve_all(tasks, tids, views)
    evaluate(views, tids, sols)
    heartbeat("done", 1, 1)


if __name__ == "__main__":
    main()
