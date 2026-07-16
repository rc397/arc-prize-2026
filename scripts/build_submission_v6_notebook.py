"""Build notebooks/arc_submission_v6.ipynb: deep-regularized per-task TTT, training sharded across GPUs.

v6 changes exactly one thing over the v2.1 recipe, the per-task adapter is trained deeper and wider, 60
steps at rank 32 with dropout kept on, which a rig sweep measured at a blended 18.9% on the 120-task
public eval against v2.1's 17.44% (both seeds above baseline, confirmed across two seeds). Everything on
the candidate-and-voting side is the proven v2.1, eight greedy D4 views at a single scale 1.25, no
colour-perm, no filter, no extra sampling, since every elaboration there regressed on the hidden set.

The one structural change the deeper adapter forces is training throughput. A deep adapter trains about
5x slower than the shallow one, so 232 of them do not fit a single L4 in 12h. Phase A shards the
trainable tasks across all GPUs, one training worker process per GPU pinned with CUDA_VISIBLE_DEVICES,
each running the proven per-task training loop on its shard, so the new code is the launcher and the
per-adapter training is unchanged. Serving in Phase C is sharded exactly as v2.1.

All paper measurements are on the single consumer GPU. This notebook is the leaderboard artifact only.
"""
import json
import os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

# The per-GPU training worker, written to /kaggle/working and launched once per GPU. It reads a config
# json the notebook writes, takes its shard index on argv, trains its slice of the trainable tasks, and
# converts each adapter to GGUF. The heavy per-task work is src.arc.ttt.test_time_train, unchanged.
TRAIN_WORKER = r'''
import json, os, sys, shutil, subprocess
SHARD = int(sys.argv[1]); NSHARD = int(sys.argv[2])
cfg = json.load(open(sys.argv[3]))
sys.path.insert(0, cfg["REPO"])
import numpy as np, torch
from src.arc.model import load_model, grid_to_ids
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, test_time_train

def maxpair(task):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in task["train"])

tasks, _ = load_consolidated(cfg["CH"], None, n=(cfg["N_LIMIT"] or None))
tids = [t for t in tasks if maxpair(tasks[t]) <= cfg["MAXPAIR"]]
tids.sort(key=lambda t: maxpair(tasks[t]))
tids = tids[:cfg["MAX_TTT_TASKS"]][SHARD::NSHARD]

adir_root = cfg["ADAPTERS"]; os.makedirs(adir_root, exist_ok=True)
mpath = os.path.join(adir_root, "manifest_%d.json" % SHARD)
manifest = {}
if os.path.exists(mpath):
    try: manifest = {k: v for k, v in json.load(open(mpath)).items() if os.path.exists(v)}
    except Exception: manifest = {}
cenv = dict(os.environ, PYTHONPATH=cfg["GGUF_PY"] + os.pathsep + os.environ.get("PYTHONPATH", ""),
            PYTHONIOENCODING="utf-8")
model, tok = load_model(cfg["BASE_HF"], dtype=torch.bfloat16, device="cuda")
peft_model, init = make_adapter(model, rank=cfg["RANK"], alpha=cfg["RANK"], dropout=cfg["DROPOUT"])
rng = np.random.default_rng(0)
import time as _t; t0 = _t.time()
for i, tid in enumerate(tids):
    if tid in manifest: continue
    adir = os.path.join(adir_root, tid); gguf = os.path.join(adir_root, tid + ".gguf")
    try:
        test_time_train(peft_model, init, tasks[tid], rng, steps=cfg["STEPS"],
                        max_ctx_tokens=cfg["MAXCTX"], time_budget=cfg["TIME_BUDGET"])
        peft_model.save_pretrained(adir)
        subprocess.run([sys.executable, cfg["CONVERT"], adir, "--base", cfg["BASE_HF"],
                        "--outfile", gguf, "--outtype", "f16"], env=cenv,
                       stdout=open(os.path.join(adir_root, "convert_%d.log" % SHARD), "ab"),
                       stderr=subprocess.STDOUT, timeout=600)
        if os.path.exists(gguf): manifest[tid] = gguf
        shutil.rmtree(adir, ignore_errors=True)
    except Exception as e:
        shutil.rmtree(adir, ignore_errors=True)
        print("shard %d [%d/%d] %s FAIL %r" % (SHARD, i + 1, len(tids), tid, str(e)[:80]), flush=True)
    json.dump(manifest, open(mpath, "w"))
    if (i + 1) % 10 == 0 or i + 1 == len(tids):
        print("shard %d [%d/%d] %.2fh %d done" % (SHARD, i + 1, len(tids),
              (_t.time() - t0) / 3600.0, len(manifest)), flush=True)
print("shard %d DONE %d adapters" % (SHARD, len(manifest)), flush=True)
'''

cells = [
md(r"""
# ARC-AGI-2, v6 (deep-regularized per-task TTT, training sharded across GPUs)

The v2.1 recipe with one change, the per-task adapter is trained deeper and wider, 60 steps at rank 32
with dropout 0.1, which a rig sweep measured at a blended 18.9% on the 120-task public eval against
v2.1's 17.44%, confirmed across two seeds. The candidate and voting side is the unchanged v2.1, eight
greedy D4 views at adapter scale 1.25, no colour-perm, no filter, no extra sampling. Training is sharded
across the GPUs because a deep adapter trains about 5x slower than the shallow one, so serving stays
identical and only the training phase is parallelized.

All paper numbers are measured on a single RTX 5070 Ti. This notebook is the leaderboard submission only.
"""),

md(r"""
## Attach four inputs, smoke on one GPU, then run on L4 x4

Offline competition. Via **+ Add Input**:
1. **prep Output**: Q8_0 GGUF, Linux `llama-server` + libs, `convert_lora_to_gguf.py`, `gguf-py`.
2. **NVARC model** (`qwen3-4b-grids15-sft139`): HF weights, for training adapters and as the conversion base.
3. **This repo** as a dataset: `src.arc`.
4. **ARC Prize 2026 competition data**.

Smoke with `N_LIMIT = 3` on a single GPU to confirm a training worker trains and converts and the three
phases finish, then set `N_LIMIT = 0`, switch to **L4 x4**, confirm the training-phase projection is under
budget in `run.log`, and submit. Internet **OFF**.
"""),

md("## 0. Configuration"),
code(r"""
import os, sys, glob, json, time, subprocess, urllib.request, shutil, threading
from pathlib import Path

class _Tee:
    def __init__(self, *s): self._s = s
    def write(self, x):
        for st in self._s:
            try: st.write(x); st.flush()
            except Exception: pass
        return len(x)
    def flush(self):
        for st in self._s:
            try: st.flush()
            except Exception: pass
    def isatty(self): return False
    def __getattr__(self, n): return getattr(self._s[0], n)
if not isinstance(sys.stdout, _Tee):
    _log = open('run.log', 'a', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.stdout, _log); sys.stderr = _Tee(sys.stderr, _log)
print('run.log capturing at', time.strftime('%Y-%m-%d %H:%M:%S'))

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
ON_KAGGLE = os.path.isdir('/kaggle')

# ---- v6 knobs: the deep-regularized adapter, everything else is v2.1 ----
N_LIMIT       = 3 if ON_KAGGLE else 2   # 3 = smoke; 0 for all tasks
SHARDS        = 4
MAXPAIR       = 1500
MAX_TTT_TASKS = 240
TTT_STEPS     = 60      # deep (v2.1 was 20). The rig sweep confirmed this lifts, with dropout kept on
TTT_RANK      = 32      # wider (v2.1 was 16)
TTT_DROPOUT   = 0.1
SCALE         = 1.25    # unchanged v2.1 inference scale
MAXCTX        = 2000    # unchanged v2.1 training context
TIME_BUDGET   = 400     # per-task train cap (s), the VRAM-thrash safety net
BATCH         = 24
TRAIN_BUDGET_H = 9.0    # sharded training finishes well inside this; guard degrades extras to no-TTT
TOTAL_BUDGET_H = 11.0

def _find(pats):
    for p in pats:
        h = sorted(glob.glob(p, recursive=True))
        if h: return h[0]
    return None

if ON_KAGGLE:
    MODEL_GGUF = _find(['/kaggle/input/**/assets/arc-qwen4b-q8_0.gguf', '/kaggle/input/**/arc-qwen4b-q8_0.gguf'])
    server_src = _find(['/kaggle/input/**/assets/bin/llama-server', '/kaggle/input/**/bin/llama-server', '/kaggle/input/**/llama-server'])
    CONVERT = _find(['/kaggle/input/**/convert_lora_to_gguf.py'])
    gp = _find(['/kaggle/input/**/gguf-py/gguf/__init__.py']); GGUF_PY = os.path.dirname(os.path.dirname(gp)) if gp else None
    bc = _find(['/kaggle/input/**/transformers/**/config.json', '/kaggle/input/**/qwen3*/**/config.json', '/kaggle/input/**/config.json'])
    BASE_HF = os.path.dirname(bc) if bc else None
    rm = _find(['/kaggle/input/**/src/arc/ttt.py']); REPO = rm[:rm.index('/src/arc/ttt.py')] if rm else None
    CH = _find(['/kaggle/input/**/arc-agi_test_challenges.json', '/kaggle/input/**/*test*challenge*.json', '/kaggle/input/**/arc-agi_evaluation_challenges.json'])
    SOL = None; OUT = '/kaggle/working/submission.json'; WORK = '/kaggle/working'; SERVER = None
    if server_src:
        bd = '/kaggle/working/llamabin'; shutil.copytree(os.path.dirname(server_src), bd, dirs_exist_ok=True)
        SERVER = os.path.join(bd, 'llama-server'); os.chmod(SERVER, 0o755)
        os.environ['LD_LIBRARY_PATH'] = bd + ':' + os.environ.get('LD_LIBRARY_PATH', '')
else:
    REPO = r'D:\arc'; MODEL_GGUF = r'D:\arc\models\arc-qwen4b-q8_0.gguf'; SERVER = r'D:\arc\llamacpp\llama-server.exe'
    CONVERT = r'D:\arc\llama.cpp\convert_lora_to_gguf.py'; GGUF_PY = r'D:\arc\llama.cpp\gguf-py'
    BASE_HF = r'D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1'
    CH = r'D:\arc\data\kaggle\arc-agi_evaluation_challenges.json'; SOL = r'D:\arc\data\kaggle\arc-agi_evaluation_solutions.json'
    OUT = r'D:\arc\experiments\submission_v6.json'; WORK = r'D:\arc\experiments'

sys.path.insert(0, REPO or '')
ADAPTERS = os.path.join(WORK, 'loras_v6')
print('on_kaggle', ON_KAGGLE, 'n_limit', N_LIMIT, 'steps', TTT_STEPS, 'rank', TTT_RANK)
for lbl, p in [('model', MODEL_GGUF), ('server', SERVER), ('base_hf', BASE_HF), ('convert', CONVERT), ('gguf-py', GGUF_PY), ('repo', REPO), ('challenges', CH)]:
    print('%-11s %-5s %s' % (lbl, 'OK' if p and os.path.exists(p) else 'MISSING', p))
assert all([MODEL_GGUF, SERVER, BASE_HF, CONVERT, GGUF_PY, REPO, CH]), 'an input is missing'
"""),

md("## 1. Helpers (serving, augmentations, maxpair), unchanged from v2.1"),
code(r"""
import numpy as np
from src.arc.model import build_prompt_ids, grid_to_ids, IM_END
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2

try:
    import torch
    NGPU = max(1, min(SHARDS, torch.cuda.device_count()))
except Exception:
    NGPU = 1
PORTS = [8080 + i for i in range(NGPU)]
AUGS = [Augmentation(d4=n) for n in D4_NAMES]
print('NGPU', NGPU, 'ports', PORTS, 'augs', len(AUGS))

def maxpair_tokens(task):
    return max(len(grid_to_ids(p['input'])) + len(grid_to_ids(p['output'])) for p in task['train'])

def health(port):
    try:
        return json.loads(urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=3).read()).get('status') == 'ok'
    except Exception:
        return False

def start_server(gpu, port, lora_paths=()):
    args = [SERVER, '-m', MODEL_GGUF, '-ngl', '99', '-c', '16384', '-np', '1', '--host', '127.0.0.1',
            '--port', str(port), '--no-webui', '--no-cache-prompt', '--no-cache-idle-slots']
    if lora_paths:
        args += ['--lora-init-without-apply']
        for p in lora_paths: args += ['--lora', p]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    lf = open(os.path.join(WORK, 'llama-server-%d.log' % port), 'ab')
    flags = 0x08000000 if os.name == 'nt' else 0
    proc = (subprocess.Popen(args, env=env, stdout=lf, stderr=lf, creationflags=flags) if os.name == 'nt'
            else subprocess.Popen(args, env=env, stdout=lf, stderr=lf))
    for _ in range(150):
        time.sleep(2)
        if health(port): return proc
    return proc

def stop_server(proc):
    try:
        if proc: proc.terminate(); proc.wait(timeout=15)
    except Exception:
        try: proc.kill()
        except Exception: pass
    time.sleep(1)

def set_adapter(idx, n_loaded, port, scale):
    body = json.dumps([{'id': j, 'scale': (scale if j == idx else 0.0)} for j in range(n_loaded)]).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/lora-adapters' % port, data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(req, timeout=10).read()

def query(prompt, port, n_predict=1024, retries=3):
    body = json.dumps({'prompt': prompt, 'n_predict': n_predict, 'temperature': 0.0, 'top_p': 0.95, 'cache_prompt': False}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request('http://127.0.0.1:%d/completion' % port, data=body, headers={'Content-Type': 'application/json'})
            return json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception:
            if a < retries - 1: time.sleep(5)
    return None

def parse(text):
    rows = []
    for line in text.split('\n'):
        s = line.strip()
        if s and all(c in '0123456789' for c in s):
            rows.append([int(c) for c in s])
    if not rows: return None
    w = len(rows[0]); rows = [r for r in rows if len(r) == w]
    return rows or None

def airv(task, port, adapter_idx=None, n_loaded=0):
    # unchanged v2.1 selection: 8 greedy D4 views at a single scale, consensus + NLL top-2, no filter
    if adapter_idx is not None:
        set_adapter(adapter_idx, n_loaded, port, SCALE)
    out = []
    for ti in range(len(task['test'])):
        cands = []
        for aug in AUGS:
            at = apply_task(aug, task)
            r = query(build_prompt_ids(at['train'], at['test'][ti]['input']), port)
            if r is None: continue
            g = parse(r.get('content', ''))
            if g: cands.append((invert_grid(aug, g), 0.0))
        a1, a2 = select_top2(cands)
        out.append({'attempt_1': a1, 'attempt_2': a2})
    return out

def empty(task):
    return [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in task['test']]

T0 = time.time()
def elapsed_h(): return (time.time() - T0) / 3600.0
"""),

md("## 2. Phase A: train the deep adapters, sharded one worker per GPU"),
code(r"""
tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
tids = [t for t in tasks if maxpair_tokens(tasks[t]) <= MAXPAIR]
tids.sort(key=lambda t: maxpair_tokens(tasks[t]))
tids = tids[:MAX_TTT_TASKS]
print('Phase A: %d trainable tasks, deep TTT sharded across %d GPU(s)' % (len(tids), NGPU), flush=True)

os.makedirs(ADAPTERS, exist_ok=True)
cfg = dict(REPO=REPO, CH=CH, BASE_HF=BASE_HF, CONVERT=CONVERT, GGUF_PY=GGUF_PY, ADAPTERS=ADAPTERS,
           N_LIMIT=N_LIMIT, MAXPAIR=MAXPAIR, MAX_TTT_TASKS=MAX_TTT_TASKS, STEPS=TTT_STEPS, RANK=TTT_RANK,
           DROPOUT=TTT_DROPOUT, MAXCTX=MAXCTX, TIME_BUDGET=TIME_BUDGET)
cfg_path = os.path.join(WORK, 'train_cfg.json'); json.dump(cfg, open(cfg_path, 'w'))
worker_path = os.path.join(WORK, 'train_worker.py')
open(worker_path, 'w').write(TRAIN_WORKER_SRC)

MANIFEST = {}
try:
    procs = []
    for g in range(NGPU):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONIOENCODING='utf-8')
        lf = open(os.path.join(WORK, 'train_shard_%d.log' % g), 'ab')
        procs.append(subprocess.Popen([sys.executable, worker_path, str(g), str(NGPU), cfg_path],
                                      env=env, stdout=lf, stderr=subprocess.STDOUT))
    # wait, but never blow the training budget: if a worker overruns, stop it and serve its rest no-TTT
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        if elapsed_h() > TRAIN_BUDGET_H:
            print('Phase A: training budget reached at %.2fh, stopping workers' % elapsed_h(), flush=True)
            for p in procs:
                if p.poll() is None:
                    try: p.terminate()
                    except Exception: pass
            break
    for g in range(NGPU):
        mp = os.path.join(ADAPTERS, 'manifest_%d.json' % g)
        if os.path.exists(mp):
            try: MANIFEST.update({k: v for k, v in json.load(open(mp)).items() if os.path.exists(v)})
            except Exception: pass
except Exception as e:
    print('Phase A failed (%r), all tasks serve no-TTT' % (str(e)[:120],), flush=True); MANIFEST = {}
print('Phase A done: %d adapters ready, %.2fh' % (len(MANIFEST), elapsed_h()), flush=True)
"""),

md("## 3. Phase C: serve sharded and run AIRV, adapters hot-swapped"),
code(r"""
submission = {}
LOCK = threading.Lock()
def record(tid, att):
    with LOCK:
        submission[tid] = att
        write_submission(submission, tasks, OUT)

def chunks(xs, n): return [xs[i:i + n] for i in range(0, len(xs), n)]

def worker(gpu, port, my_no_ttt, my_ttt):
    proc = start_server(gpu, port)
    if not health(port):
        print('gpu%d server failed' % gpu, flush=True); return
    for tid in my_no_ttt:
        try: att = airv(tasks[tid], port)
        except Exception as e: att = empty(tasks[tid]); print('  gpu%d noTTT %s ERR %r' % (gpu, tid, str(e)[:70]), flush=True)
        record(tid, att); print('[gpu%d noTTT] %s %.2fh' % (gpu, tid, elapsed_h()), flush=True)
    for batch in chunks(my_ttt, BATCH):
        over = elapsed_h() > TOTAL_BUDGET_H
        stop_server(proc)
        proc = start_server(gpu, port) if over else start_server(gpu, port, [MANIFEST[t] for t in batch])
        ok = (not over) and health(port)
        if not health(port): proc = start_server(gpu, port)
        for k, tid in enumerate(batch):
            try: att = airv(tasks[tid], port, adapter_idx=(k if ok else None), n_loaded=(len(batch) if ok else 0))
            except Exception as e: att = empty(tasks[tid]); print('  gpu%d TTT %s ERR %r' % (gpu, tid, str(e)[:70]), flush=True)
            record(tid, att); print('[gpu%d %s] %s %.2fh' % (gpu, 'TTT' if ok else 'noTTT-fb', tid, elapsed_h()), flush=True)
    stop_server(proc)

no_ttt = [t for t in tids if t not in MANIFEST]
ttt_ids = [t for t in tids if t in MANIFEST]
sh_no = [no_ttt[i::NGPU] for i in range(NGPU)]
sh_ttt = [ttt_ids[i::NGPU] for i in range(NGPU)]
print('Phase C: %d no-TTT + %d TTT across %d GPU(s)' % (len(no_ttt), len(ttt_ids), NGPU), flush=True)
threads = [threading.Thread(target=worker, args=(g, PORTS[g], sh_no[g], sh_ttt[g])) for g in range(NGPU)]
for t in threads: t.start()
for t in threads: t.join()
write_submission(submission, tasks, OUT)
print('wrote', OUT, '| %d no-TTT %d TTT | %.2fh' % (len(no_ttt), len(ttt_ids), elapsed_h()), flush=True)

if sols:
    from src.arc.eval import score_submission
    acc, c, n = score_submission({t: submission[t] for t in tids if t in submission}, {t: sols[t] for t in tids if t in sols})
    print('SCORE acc=%.4f correct=%d total=%d' % (acc, c, n), flush=True)
"""),
]

# define the worker source as a constant in the cell just before Phase A writes it to disk
cells.insert(7, code("TRAIN_WORKER_SRC = %r" % TRAIN_WORKER))

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "arc_submission_v6.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", os.path.abspath(out), "cells:", len(cells))
