"""Full-context TTT probe on the 36 VRAM-excluded big-grid tasks (maxpair > 1000).

The recorded big-task negative (10.6% vs 12.8% no-TTT) trained at max_ctx 600, about one-shot
context for tasks whose single pair is >= 1010 tokens, so it is confounded by truncation. This
probe trains at FULL context (max_ctx 8000, above the 7471-token worst case) by keeping a GPU
weight budget and streaming the rest of the frozen bf16 base from RAM via accelerate device_map,
then serves both arms paired (adapter scale 1.25 vs 0.0) in one llama.cpp session per task and
scores them against the solutions.

Stages. smoke = 1 task (the largest, worst-case memory) x 5 steps, end-to-end train-convert-
serve-score with telemetry and a printed PASS/FAIL checklist. panel = 6 size-stratified tasks
x 60 steps at the validated deep config (rank 32, dropout 0.1, seed 0). N_TASKS=0 extends the
panel to all 36, reusing the panel adapters through the config-keyed manifest.

The wait gate runs inside this process (no pid-chained waiter), polling GPU memory and process
command lines until the rank sweep is done. Serving uses port 8081, the sweep owns 8080.

Usage: python rig_biggrid_probe.py smoke|panel [GPU_GIB=6] [N_TASKS] [STEPS] [WAIT=1] [TIME_BUDGET]
"""
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request

BASE = r"D:\arc"
sys.path.insert(0, BASE)

STAGE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
assert STAGE in ("smoke", "panel"), "stage must be smoke or panel"
GPU_GIB = int(sys.argv[2]) if len(sys.argv) > 2 else 6
N_TASKS = int(sys.argv[3]) if len(sys.argv) > 3 else (1 if STAGE == "smoke" else 6)
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else (5 if STAGE == "smoke" else 60)
WAIT = int(sys.argv[5]) if len(sys.argv) > 5 else 1
TIME_BUDGET = int(sys.argv[6]) if len(sys.argv) > 6 else (900 if STAGE == "smoke" else 2400)

RANK = 32
ALPHA = 32
DROPOUT = 0.1
SCALE = 1.25
SEED = 0
MAXCTX = 8000
CTX = "16384"
N_PREDICT = 1536
PORT = 8081
SERVE_BATCH = 12
MAXPAIR_CUT = 1000

BASE_HF = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                       "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
CH = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json")
SOL = os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_solutions.json")
MODEL_GGUF = os.path.join(BASE, "models", "arc-qwen4b-q8_0.gguf")
SERVER = os.path.join(BASE, "llamacpp", "llama-server.exe")
CONVERT = os.path.join(BASE, "llama.cpp", "convert_lora_to_gguf.py")
GGUF_PY = os.path.join(BASE, "llama.cpp", "gguf-py")
EXP = os.path.join(BASE, "experiments")

TAG = "biggrid_s%d_r%d" % (STEPS, RANK)
ADAPTERS = os.path.join(BASE, "models", TAG)
MANIFEST_PATH = os.path.join(EXP, TAG + "_manifest.json")
RESULTS_PATH = os.path.join(EXP, TAG + "_results.json")
STATUS = os.path.join(BASE, "eval_status_biggrid.json")

from src.arc.model import build_prompt_ids, grid_to_ids                       # noqa: E402
from src.arc.data import load_consolidated                                     # noqa: E402
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES    # noqa: E402
from src.arc.select import select_top2                                         # noqa: E402
from src.arc.ttt import make_adapter, test_time_train                          # noqa: E402

AUGS = [Augmentation(d4=n) for n in D4_NAMES]
SRV_PID = [0]


def maxpair_tokens(task):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in task["train"])


def gkey(g):
    return tuple(tuple(int(v) for v in r) for r in g) if g else ()


def heartbeat(phase, i, n, extra=""):
    try:
        json.dump({"phase": phase, "i": i, "n": n, "ts": time.time(), "stage": STAGE,
                   "gpu_gib": GPU_GIB, "steps": STEPS, "tag": TAG, "srv_pid": SRV_PID[0],
                   "extra": extra}, open(STATUS, "w"))
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
    """True if the rank sweep (or any other known GPU job) still runs. Command-line based, not pid."""
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
                                   "rig_select16.py")):
            return True
    return False


def wait_for_gpu_idle():
    """Gate inside this process. Proceed after 3 consecutive clean polls 120s apart."""
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
    print("wait gate open (gpu idle, no rival jobs)", flush=True)


def pick_tasks(tasks):
    exc = sorted(((maxpair_tokens(tasks[t]), t) for t in tasks
                  if maxpair_tokens(tasks[t]) > MAXPAIR_CUT))
    assert len(exc) == 36, "expected 36 excluded eval tasks, got %d" % len(exc)
    if STAGE == "smoke":
        chosen = [exc[-1]]
    elif N_TASKS == 0:
        chosen = exc
    else:
        bands = [(1010, 1250), (1250, 1500), (1500, 10 ** 9)]
        strat = []
        for lo, hi in bands:
            members = [x for x in exc if lo <= x[0] < hi]
            if members:
                strat.append(members[0])
                if members[-1] != members[0]:
                    strat.append(members[-1])
        rest = [x for x in exc if x not in strat]
        chosen = (strat + rest)[:N_TASKS]
    print("tasks (%d): %s" % (len(chosen), [(t, mp) for mp, t in chosen]), flush=True)
    return [t for _, t in chosen]


def load_offload_model():
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        BASE_HF, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "%dGiB" % GPU_GIB, "cpu": "48GiB"},
        attn_implementation="sdpa")
    dmap = getattr(model, "hf_device_map", {}) or {}
    n_gpu = sum(1 for v in dmap.values() if str(v) in ("0", "cuda", "cuda:0"))
    n_cpu = sum(1 for v in dmap.values() if str(v) == "cpu")
    embed_dev = str(dmap.get("model.embed_tokens", "?"))
    print("device map: %d entries on gpu, %d on cpu, embed_tokens on %s" % (n_gpu, n_cpu, embed_dev), flush=True)
    print("attn=%s sdp flash=%s mem_eff=%s math=%s" % (
        getattr(model.config, "_attn_implementation", "?"),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled()), flush=True)
    return model, dmap, n_cpu, embed_dev


def fix_lora_devices(peft_model, expected_layers):
    """Move every LoRA param to cuda (peft leaves adapters of cpu-mapped modules on cpu), assert
    none on meta, rebuild the init snapshot on the same device. Returns (init, n_params, n_meta)."""
    import torch
    n_meta = 0
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        for sub in (module.lora_A, module.lora_B):
            for key in sub:
                lin = sub[key]
                dev = next(lin.parameters()).device
                if dev.type == "meta":
                    n_meta += 1
                    print("META_LORA_MATERIALIZED %s" % name, flush=True)
                    lin.to_empty(device="cuda")
                    with torch.no_grad():
                        if sub is module.lora_A:
                            torch.nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
                        else:
                            lin.weight.zero_()
                elif dev.type != "cuda":
                    lin.to("cuda")
    lora = {k: v for k, v in peft_model.named_parameters() if "lora_" in k}
    bad = [k for k, v in lora.items() if v.device.type != "cuda"]
    assert not bad, "lora params not on cuda after fix: %s" % bad[:4]
    expected = expected_layers * 7 * 2
    assert len(lora) == expected, "lora param count %d != expected %d" % (len(lora), expected)
    init = {k: v.detach().clone() for k, v in lora.items()}
    return init, len(lora), n_meta


def train_one(peft_model, init, task, rng):
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss = test_time_train(peft_model, init, task, rng, steps=STEPS,
                           max_ctx_tokens=MAXCTX, time_budget=TIME_BUDGET)
    wall = time.time() - t0
    return {"loss": float(loss), "wall_s": round(wall, 1), "s_step": round(wall / max(STEPS, 1), 2),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
            "budget_hit": wall >= TIME_BUDGET - 5}


def verify_adapter_file(adir):
    from safetensors import safe_open
    import torch
    p = os.path.join(adir, "adapter_model.safetensors")
    total = 0.0
    n = 0
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            assert torch.isfinite(t).all(), "non-finite tensor %s" % k
            total += float(t.float().norm())
            n += 1
    assert total > 0, "adapter global norm is zero"
    return n, total


def convert_adapter(adir, gguf):
    cenv = dict(os.environ, PYTHONPATH=GGUF_PY + os.pathsep + os.environ.get("PYTHONPATH", ""),
                PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, CONVERT, adir, "--base", BASE_HF, "--outfile", gguf,
                    "--outtype", "f16"], env=cenv,
                   stdout=open(os.path.join(BASE, "convert_biggrid.log"), "ab"),
                   stderr=subprocess.STDOUT, timeout=600)
    return os.path.exists(gguf) and os.path.getsize(gguf) > 100 * 1024 * 1024


def health():
    try:
        u = "http://127.0.0.1:%d/health" % PORT
        return json.loads(urllib.request.urlopen(u, timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def start_server(lora_paths=()):
    assert not health(), "port %d already has a live server, refusing to adopt it" % PORT
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", CTX, "-np", "1",
            "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--no-cache-prompt"]
    if lora_paths:
        args += ["--lora-init-without-apply"]
        for p in lora_paths:
            args += ["--lora", p]
    proc = subprocess.Popen(args, creationflags=0x08000000,
                            stdout=open(os.path.join(BASE, "srv_biggrid.log"), "a"),
                            stderr=open(os.path.join(BASE, "srv_biggrid.err"), "a"))
    SRV_PID[0] = proc.pid
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
    SRV_PID[0] = 0
    time.sleep(1)


def set_scales(scales):
    body = json.dumps([{"id": j, "scale": s} for j, s in enumerate(scales)]).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/lora-adapters" % PORT, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def query(prompt_ids, retries=3):
    body = json.dumps({"prompt": prompt_ids, "n_predict": N_PREDICT, "temperature": 0.0,
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


def airv_arm(task, scales, n_loaded):
    if n_loaded:
        set_scales(scales)
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


def score_attempts(attempts, sol_grids):
    k = 0
    for ti, sol in enumerate(sol_grids):
        if ti < len(attempts):
            att = attempts[ti]
            if gkey(att["attempt_1"]) == gkey(sol) or gkey(att["attempt_2"]) == gkey(sol):
                k += 1
    return k, len(sol_grids)


def load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return default


def train_phase(tasks, tids, checks):
    import numpy as np
    import torch
    manifest = {k: v for k, v in load_json(MANIFEST_PATH, {}).items() if os.path.exists(v)}
    results = load_json(RESULTS_PATH, {})
    todo = [t for t in tids if t not in manifest]
    if not todo:
        print("train phase: all %d adapters already built" % len(tids), flush=True)
        return manifest, results
    print("train phase: %d to build (%d resumed)" % (len(todo), len(manifest)), flush=True)
    os.makedirs(ADAPTERS, exist_ok=True)

    model, dmap, n_cpu, embed_dev = load_offload_model()
    checks.append(("offload engaged (>=1 module on cpu)", n_cpu >= 1, "%d cpu entries" % n_cpu))
    checks.append(("embed_tokens on gpu", embed_dev in ("0", "cuda", "cuda:0"), embed_dev))
    peft_model, _ = make_adapter(model, rank=RANK, alpha=ALPHA, dropout=DROPOUT)
    init, n_lora, n_meta = fix_lora_devices(peft_model, model.config.num_hidden_layers)
    checks.append(("lora all-cuda, count %d, meta-materialized %d" % (n_lora, n_meta), True, ""))
    print("lora ready: %d params on cuda (%d were meta)" % (n_lora, n_meta), flush=True)

    n_oom = 0
    for i, tid in enumerate(todo):
        heartbeat("train", i, len(todo), tid)
        rng = np.random.default_rng([SEED, int(tid, 16)])
        adir = os.path.join(ADAPTERS, tid)
        gguf = os.path.join(ADAPTERS, tid + ".gguf")
        shutil.rmtree(adir, ignore_errors=True)
        try:
            tel = train_one(peft_model, init, tasks[tid], rng)
            peft_model.save_pretrained(adir)
            if STAGE == "smoke":
                n_t, norm = verify_adapter_file(adir)
                checks.append(("adapter file valid (%d tensors, norm %.1f)" % (n_t, norm), True, ""))
            ok = convert_adapter(adir, gguf)
            shutil.rmtree(adir, ignore_errors=True)
            if ok:
                manifest[tid] = gguf
            results.setdefault(tid, {})["telemetry"] = tel
            results[tid]["maxpair"] = maxpair_tokens(tasks[tid])
            print("[train %d/%d] %s %.0fs (%.1fs/step) peak %.1fGB loss %.3f%s%s"
                  % (i + 1, len(todo), tid, tel["wall_s"], tel["s_step"], tel["peak_alloc_gb"],
                     tel["loss"], " BUDGET_HIT" if tel["budget_hit"] else "",
                     "" if ok else " CONVERT_FAIL"), flush=True)
        except torch.cuda.OutOfMemoryError as e:
            n_oom += 1
            shutil.rmtree(adir, ignore_errors=True)
            torch.cuda.empty_cache()
            print("[train %d/%d] %s OOM %r" % (i + 1, len(todo), tid, str(e)[:80]), flush=True)
            results.setdefault(tid, {})["train_fail"] = "oom"
        except Exception as e:
            shutil.rmtree(adir, ignore_errors=True)
            torch.cuda.empty_cache()
            print("[train %d/%d] %s FAIL %r" % (i + 1, len(todo), tid, str(e)[:100]), flush=True)
            results.setdefault(tid, {})["train_fail"] = str(e)[:100]
        json.dump(manifest, open(MANIFEST_PATH, "w"))
        json.dump(results, open(RESULTS_PATH, "w"))
        torch.cuda.empty_cache()
    if n_oom >= 2:
        print("HINT: %d tasks OOMed, relaunch with GPU_GIB=2, manifest resumes the %d done"
              % (n_oom, len(manifest)), flush=True)
    del peft_model, model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest, results


def serve_phase(tasks, tids, manifest, sols, results, checks):
    port_was_clean = not health()
    checks.append(("port %d clean before serving" % PORT, port_was_clean, ""))
    done = [t for t in tids if "ttt" in results.get(t, {}) and "nottt" in results.get(t, {})]
    have = [t for t in tids if t in manifest and t not in done]
    no_adapter = [t for t in tids if t not in manifest and t not in done]
    print("serve phase: %d paired, %d no-adapter, %d already done" % (len(have), len(no_adapter), len(done)), flush=True)
    proc = None
    try:
        for b in range(0, len(have), SERVE_BATCH):
            batch = have[b:b + SERVE_BATCH]
            stop_server(proc)
            proc = start_server([manifest[t] for t in batch])
            if not health():
                print("server failed to start, aborting serve", flush=True)
                break
            for k, tid in enumerate(batch):
                heartbeat("serve", b + k, len(have), tid)
                sol = sols[tid]
                try:
                    zeros = [0.0] * len(batch)
                    nottt = airv_arm(tasks[tid], zeros, len(batch))
                    scales = list(zeros)
                    scales[k] = SCALE
                    ttt = airv_arm(tasks[tid], scales, len(batch))
                    kn, nn = score_attempts(nottt, sol)
                    kt, nt = score_attempts(ttt, sol)
                    results.setdefault(tid, {})["nottt"] = {"k": kn, "n": nn, "attempts": nottt}
                    results[tid]["ttt"] = {"k": kt, "n": nt, "attempts": ttt}
                    print("[serve %d/%d] %s noTTT %d/%d  TTT %d/%d" % (b + k + 1, len(have), tid, kn, nn, kt, nt), flush=True)
                except Exception as e:
                    print("[serve] %s ERR %r" % (tid, str(e)[:80]), flush=True)
                json.dump(results, open(RESULTS_PATH, "w"))
        if no_adapter:
            stop_server(proc)
            proc = start_server()
            if health():
                for j, tid in enumerate(no_adapter):
                    heartbeat("serve-noadapter", j, len(no_adapter), tid)
                    try:
                        nottt = airv_arm(tasks[tid], [], 0)
                        kn, nn = score_attempts(nottt, sols[tid])
                        results.setdefault(tid, {})["nottt"] = {"k": kn, "n": nn, "attempts": nottt}
                        print("[serve-na %d/%d] %s noTTT %d/%d (TRAIN_FAIL)" % (j + 1, len(no_adapter), tid, kn, nn), flush=True)
                    except Exception as e:
                        print("[serve-na] %s ERR %r" % (tid, str(e)[:80]), flush=True)
                    json.dump(results, open(RESULTS_PATH, "w"))
    finally:
        stop_server(proc)
    return results


def report(results, tids):
    print("\n==== paired report tag=%s gpu_gib=%d steps=%d ====" % (TAG, GPU_GIB, STEPS), flush=True)
    print("%-10s %7s %9s %7s %7s %5s %7s %7s %s"
          % ("task", "maxpair", "noTTT", "TTT", "flip", "s/st", "peakGB", "budget", "note"), flush=True)
    tk = tn = nk = nn2 = 0
    ttt_only = nottt_only = 0
    under = 0
    for tid in tids:
        r = results.get(tid, {})
        tel = r.get("telemetry", {})
        no = r.get("nottt", {})
        tt = r.get("ttt", {})
        flip = "="
        if no and tt:
            if tt["k"] > no["k"]:
                flip = "+TTT"
                ttt_only += 1
            elif no["k"] > tt["k"]:
                flip = "-TTT"
                nottt_only += 1
            nk += no["k"]
            nn2 += no["n"]
            tk += tt["k"]
            tn += tt["n"]
        bh = tel.get("budget_hit", False)
        under += bool(bh)
        note = r.get("train_fail", "")
        print("%-10s %7s %9s %7s %7s %5s %7s %7s %s"
              % (tid, r.get("maxpair", "?"),
                 ("%d/%d" % (no["k"], no["n"])) if no else "-",
                 ("%d/%d" % (tt["k"], tt["n"])) if tt else "-",
                 flip, tel.get("s_step", "-"), tel.get("peak_alloc_gb", "-"),
                 "HIT" if bh else "", ("UNDERTRAINED " if bh else "") + note), flush=True)
    print("---- totals: noTTT %d/%d  TTT %d/%d  tasks +TTT=%d -TTT=%d undertrained=%d"
          % (nk, nn2, tk, tn, ttt_only, nottt_only, under), flush=True)
    walls = [results[t]["telemetry"]["wall_s"] for t in tids
             if "telemetry" in results.get(t, {})]
    if walls:
        avg = sum(walls) / len(walls)
        print("---- cost: avg %.0fs/task, projected all-36 train %.1fh" % (avg, 36 * avg / 3600.0), flush=True)


def main():
    for lbl, p in [("base_hf", BASE_HF), ("challenges", CH), ("solutions", SOL),
                   ("gguf", MODEL_GGUF), ("server", SERVER), ("convert", CONVERT), ("gguf-py", GGUF_PY)]:
        assert os.path.exists(p), "missing %s: %s" % (lbl, p)
    os.makedirs(EXP, exist_ok=True)
    free_gb = shutil.disk_usage(BASE).free / 1024 ** 3
    assert free_gb > 5, "low disk: %.1f GB" % free_gb
    import torch
    assert torch.cuda.is_available(), "cuda unavailable"

    print("CONFIG stage=%s gpu_gib=%d n=%d steps=%d wait=%d budget=%ds tag=%s"
          % (STAGE, GPU_GIB, N_TASKS, STEPS, WAIT, TIME_BUDGET, TAG), flush=True)
    tasks, sols = load_consolidated(CH, SOL, n=None)
    tids = pick_tasks(tasks)

    if WAIT:
        wait_for_gpu_idle()

    checks = []
    manifest, results = train_phase(tasks, tids, checks)
    results = serve_phase(tasks, tids, manifest, sols, results, checks)
    report(results, tids)

    if STAGE == "smoke":
        tid = tids[0]
        r = results.get(tid, {})
        tel = r.get("telemetry", {})
        checks.append(("training completed (loss finite)", "telemetry" in r and math.isfinite(tel.get("loss", float("nan"))), str(tel.get("loss"))))
        checks.append(("peak alloc < 15 GB", tel.get("peak_alloc_gb", 99) < 15, "%s GB" % tel.get("peak_alloc_gb")))
        checks.append(("s/step < 60", tel.get("s_step", 999) < 60, "%s s" % tel.get("s_step")))
        checks.append(("gguf built", tid in manifest, manifest.get(tid, "")))
        checks.append(("noTTT arm returned grids", bool(r.get("nottt", {}).get("attempts")), ""))
        checks.append(("TTT arm returned grids", bool(r.get("ttt", {}).get("attempts")), ""))
        print("\n==== SMOKE CHECKLIST ====", flush=True)
        fails = []
        for name, ok, detail in checks:
            print("%s  %s %s" % ("PASS" if ok else "FAIL", name, ("(%s)" % detail) if detail else ""), flush=True)
            if not ok:
                fails.append(name)
        if tel.get("s_step"):
            per_task = 60.0 * tel["s_step"]
            print("projection: panel-6 train ~%.1fh, all-36 ~%.1fh"
                  % (6 * per_task / 3600.0, 36 * per_task / 3600.0), flush=True)
            if per_task > TIME_BUDGET:
                print("NOTE: 60 steps at this s/step exceeds TIME_BUDGET %ds, raise argv[6] for the panel"
                      % TIME_BUDGET, flush=True)
        print("SMOKE %s%s" % ("PASS" if not fails else "FAIL ", ", ".join(fails)), flush=True)
        heartbeat("done", 1, 1, "smoke " + ("pass" if not fails else "fail"))
        sys.exit(0 if not fails else 1)
    heartbeat("done", 1, 1, "panel complete")


if __name__ == "__main__":
    main()
