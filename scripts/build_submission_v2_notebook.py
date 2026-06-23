"""Build notebooks/arc_submission_v2.ipynb: layered TTT + colour-perm, multi-GPU sharded.

v2 over v1 (no-TTT AIRV): per-task test-time training on the small/trainable tasks (maxpair filter)
plus colour-permutation augmentation, with a wall-clock budget and per-task try/except so a run always
degrades to no-TTT rather than breaking. The serving phase (the dominant cost) shards across all
available GPUs: one llama-server per GPU pinned with CUDA_VISIBLE_DEVICES, driven by parallel worker
threads. NGPU auto-detects, so this runs single-GPU for the smoke test and 4-way on the L4 x4 node.

Thesis note: all paper measurements are on the single consumer GPU (rig 5070 Ti). This notebook is the
competition leaderboard artifact only, run on Kaggle's provided hardware; the two are reported
separately.

Imports src.arc from the attached repo. Attach FOUR inputs (see the markdown cell):
  1. prep output  (GGUF, llama-server + libs, convert_lora_to_gguf.py, gguf-py)
  2. the NVARC HF model  (sorokin/qwen3_4b_grids15_sft139) for training + as the conversion base
  3. this repo  (rc397/arc-prize-2026) as a dataset, for src.arc
  4. the ARC Prize 2026 competition data

SMOKE TEST on a single GPU (N_LIMIT=3) before a full L4 x4 run; read run.log for per-phase timing.
"""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# ARC-AGI-2, v2 (TTT + colour-perm), multi-GPU sharded

Layered on the v1 AIRV pipeline: per-task test-time training on the small/trainable tasks at adapter
scale 1.25, colour-perm on top of the 8 D4 augmentations, no-TTT + colour-perm on the rest. The
serving phase shards across every available GPU (one `llama-server` per GPU); training runs on one GPU
and fits the budget. A global wall-clock budget and per-task fallback keep a run from breaking.

All paper numbers are measured on the single consumer GPU (RTX 5070 Ti). This notebook is the
leaderboard submission only, on Kaggle's provided hardware.
"""),

md(r"""
## Attach four inputs, smoke-test on one GPU, then run on L4 x4

Offline competition. Via **+ Add Input**:
1. **prep Output** (`arc_prep_v1`+): Q8_0 GGUF, Linux `llama-server` + libs, `convert_lora_to_gguf.py`, `gguf-py`.
2. **NVARC model** (`sorokin/qwen3_4b_grids15_sft139`): HF weights, for training adapters and as the conversion base.
3. **This repo** (`rc397/arc-prize-2026`) as a dataset: `src.arc`.
4. **ARC Prize 2026 competition data**.

**Smoke test first:** leave `N_LIMIT = 3`, run on a single GPU, confirm all three phases finish and
`submission.json` is written, and read `run.log` + `llama-server-*.log` for timing and any error.
Then set `N_LIMIT = 0`, switch the accelerator to **L4 x4** (NGPU auto-detects to 4), confirm the
projected total is under 12h, and submit. Internet **OFF**.
"""),

md("## 0. Configuration"),

code(r"""
import os, sys, glob, json, time, subprocess, urllib.request, shutil, threading
from pathlib import Path

class _Tee:
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams:
            try: st.write(s); st.flush()
            except Exception: pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try: st.flush()
            except Exception: pass
    def isatty(self): return False
    def __getattr__(self, name): return getattr(self._streams[0], name)
if not isinstance(sys.stdout, _Tee):
    _run_log = open('run.log', 'a', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.stdout, _run_log)
    sys.stderr = _Tee(sys.stderr, _run_log)
print('run.log: capturing this run at', time.strftime('%Y-%m-%d %H:%M:%S'))

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ON_KAGGLE = os.path.isdir('/kaggle')

# ---- v2 knobs (tune from the smoke-test timing) ----
N_LIMIT       = 3 if ON_KAGGLE else 2     # 3 = smoke test; 0 for all tasks once timing is known
SHARDS        = 4                         # serve across up to this many GPUs
MAXPAIR       = 1500                      # train a task if its largest train pair <= this many tokens (L4 24GB fits more than the rig's 16GB)
MAX_TTT_TASKS = 240                       # effectively no cap; 12h budget has lots of headroom
TTT_STEPS     = 20                        # regularized sweet spot; more steps overfit (do not raise)
TTT_RANK      = 16
TTT_DROPOUT   = 0.1
SCALE         = 1.25
N_COLORPERM   = 8                         # colour-permuted augmentations on top of the 8 D4 (16 total votes)
BATCH         = 24                        # adapters loaded per server start
TRAIN_BUDGET_H = 7.0                      # stop training new adapters after this many hours
TOTAL_BUDGET_H = 11.0                     # leave headroom under the 12h cap

def _find(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None

if ON_KAGGLE:
    MODEL_GGUF = _find(['/kaggle/input/**/assets/arc-qwen4b-q8_0.gguf', '/kaggle/input/**/arc-qwen4b-q8_0.gguf'])
    server_src = _find(['/kaggle/input/**/assets/bin/llama-server', '/kaggle/input/**/bin/llama-server',
                        '/kaggle/input/**/llama-server'])
    CONVERT  = _find(['/kaggle/input/**/convert_lora_to_gguf.py'])
    gp = _find(['/kaggle/input/**/gguf-py/gguf/__init__.py'])
    GGUF_PY  = os.path.dirname(os.path.dirname(gp)) if gp else None
    base_cfg = _find(['/kaggle/input/**/transformers/**/config.json', '/kaggle/input/**/qwen3*/**/config.json',
                      '/kaggle/input/**/config.json'])
    BASE_HF  = os.path.dirname(base_cfg) if base_cfg else None
    rm = _find(['/kaggle/input/**/src/arc/ttt.py'])
    REPO = rm[:rm.index('/src/arc/ttt.py')] if rm else None
    CH = _find(['/kaggle/input/**/arc-agi_test_challenges.json', '/kaggle/input/**/*test*challenge*.json',
                '/kaggle/input/**/arc-agi_evaluation_challenges.json'])
    SOL = None
    OUT = '/kaggle/working/submission.json'
    WORK = '/kaggle/working'
    SERVER = None
    if server_src:
        bindst = '/kaggle/working/llamabin'
        shutil.copytree(os.path.dirname(server_src), bindst, dirs_exist_ok=True)
        SERVER = os.path.join(bindst, 'llama-server')
        os.chmod(SERVER, 0o755)
        os.environ['LD_LIBRARY_PATH'] = bindst + ':' + os.environ.get('LD_LIBRARY_PATH', '')
else:
    REPO = r'D:\arc'
    MODEL_GGUF = r'D:\arc\models\arc-qwen4b-q8_0.gguf'
    SERVER = r'D:\arc\llamacpp\llama-server.exe'
    CONVERT = r'D:\arc\llama.cpp\convert_lora_to_gguf.py'
    GGUF_PY = r'D:\arc\llama.cpp\gguf-py'
    BASE_HF = r'D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1'
    CH = r'D:\arc\data\kaggle\arc-agi_evaluation_challenges.json'
    SOL = r'D:\arc\data\kaggle\arc-agi_evaluation_solutions.json'
    OUT = r'D:\arc\experiments\submission_v2.json'
    WORK = r'D:\arc\experiments'

sys.path.insert(0, REPO or '')
print('on_kaggle =', ON_KAGGLE, ' n_limit =', N_LIMIT)
for label, path in [('model_gguf', MODEL_GGUF), ('server', SERVER), ('base_hf', BASE_HF),
                    ('convert', CONVERT), ('gguf-py', GGUF_PY), ('repo', REPO), ('challenges', CH)]:
    print('%-11s %-5s %s' % (label, 'OK' if path and os.path.exists(path) else 'MISSING', path))
assert all([MODEL_GGUF, SERVER, BASE_HF, CONVERT, GGUF_PY, REPO, CH]), 'an input is missing'
"""),

md("## 1. Helpers (per-GPU server management, augmentations, maxpair)"),

code(r"""
import numpy as np
from src.arc.model import build_prompt_ids, grid_to_ids, IM_END
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES, random_augmentation
from src.arc.select import select_top2

try:
    import torch
    NGPU = max(1, min(SHARDS, torch.cuda.device_count()))
except Exception:
    NGPU = 1
PORTS = [8080 + i for i in range(NGPU)]
print('NGPU =', NGPU, ' ports =', PORTS)

_rng = np.random.default_rng(20260620)
AUGS = [Augmentation(d4=name) for name in D4_NAMES] + [
    random_augmentation(_rng, use_d4=True, recolor=True, keep_background=True, reorder=False)
    for _ in range(N_COLORPERM)]
print('augmentations: %d D4 + %d colour-perm = %d' % (len(D4_NAMES), N_COLORPERM, len(AUGS)))

def maxpair_tokens(task):
    return max(len(grid_to_ids(p['input'])) + len(grid_to_ids(p['output'])) for p in task['train'])

def health(port):
    try:
        u = 'http://127.0.0.1:%d/health' % port
        return json.loads(urllib.request.urlopen(u, timeout=3).read()).get('status') == 'ok'
    except Exception:
        return False

def start_server(gpu, port, lora_paths=()):
    args = [SERVER, '-m', MODEL_GGUF, '-ngl', '99', '-c', '16384', '-np', '1', '--host', '127.0.0.1',
            '--port', str(port), '--no-webui', '--no-cache-prompt', '--no-cache-idle-slots']
    if lora_paths:
        args += ['--lora-init-without-apply']
        for p in lora_paths:
            args += ['--lora', p]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    lf = open(os.path.join(WORK, 'llama-server-%d.log' % port), 'ab')
    flags = 0x08000000 if os.name == 'nt' else 0
    proc = (subprocess.Popen(args, env=env, stdout=lf, stderr=lf, creationflags=flags) if os.name == 'nt'
            else subprocess.Popen(args, env=env, stdout=lf, stderr=lf))
    for _ in range(150):
        time.sleep(2)
        if health(port):
            return proc
    return proc  # caller treats unhealthy as failure

def stop_server(proc):
    try:
        if proc:
            proc.terminate(); proc.wait(timeout=15)
    except Exception:
        try: proc.kill()
        except Exception: pass
    time.sleep(1)

def set_adapter(idx, n_loaded, port):
    body = json.dumps([{'id': j, 'scale': (SCALE if j == idx else 0.0)} for j in range(n_loaded)]).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/lora-adapters' % port, data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(req, timeout=10).read()

def query(prompt, port, n_predict=1024, retries=3):
    body = json.dumps({'prompt': prompt, 'n_predict': n_predict, 'temperature': 0.0, 'cache_prompt': False}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request('http://127.0.0.1:%d/completion' % port, data=body,
                                         headers={'Content-Type': 'application/json'})
            return json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception:
            if a < retries - 1:
                time.sleep(5)
    return None

def parse(text):
    rows = []
    for line in text.split('\n'):
        s = line.strip()
        if s and all(c in '0123456789' for c in s):
            rows.append([int(c) for c in s])
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None

def airv(task, port, adapter_idx=None, n_loaded=0):
    if adapter_idx is not None:
        set_adapter(adapter_idx, n_loaded, port)
    attempts = []
    for ti in range(len(task['test'])):
        cands = []
        for aug in AUGS:
            at = apply_task(aug, task)
            r = query(build_prompt_ids(at['train'], at['test'][ti]['input']), port)
            if r is None:
                continue
            g = parse(r.get('content', ''))
            if g:
                cands.append((invert_grid(aug, g), 0.0))
        a1, a2 = select_top2(cands)
        attempts.append({'attempt_1': a1, 'attempt_2': a2})
    return attempts

def empty(task):
    return [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in task['test']]

T_START = time.time()
def elapsed_h():
    return (time.time() - T_START) / 3600.0
"""),

md("## 2. Phase A+B: train each adapter, convert to GGUF, delete the PEFT dir (bounds disk)"),

code(r"""
tasks, sols = load_consolidated(CH, SOL, n=None)
tids = list(tasks)
if N_LIMIT:
    tids = tids[:N_LIMIT]

trainable = sorted([t for t in tids if maxpair_tokens(tasks[t]) <= MAXPAIR],
                   key=lambda t: maxpair_tokens(tasks[t]))[:MAX_TTT_TASKS]
print('Phase A+B: %d/%d tasks trainable (maxpair <= %d, cap %d)' % (len(trainable), len(tids), MAXPAIR, MAX_TTT_TASKS))

ADAPTERS = os.path.join(WORK, 'loras_v2')
os.makedirs(ADAPTERS, exist_ok=True)
MANIFEST = {}
cenv = dict(os.environ, PYTHONPATH=GGUF_PY + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONIOENCODING='utf-8')
try:
    import torch, shutil as _sh
    from src.arc.model import load_model
    from src.arc.ttt import make_adapter, test_time_train
    model, tok = load_model(BASE_HF, dtype=torch.bfloat16, device='cuda')
    peft_model, init = make_adapter(model, rank=TTT_RANK, alpha=TTT_RANK, dropout=TTT_DROPOUT)
    rng = np.random.default_rng(0)
    for i, tid in enumerate(trainable):
        if elapsed_h() > TRAIN_BUDGET_H:
            print('Phase A+B: budget reached at %d/%d, rest served no-TTT' % (i, len(trainable)), flush=True)
            break
        adir = os.path.join(ADAPTERS, tid)
        gguf = os.path.join(ADAPTERS, tid + '.gguf')
        try:
            test_time_train(peft_model, init, tasks[tid], rng, steps=TTT_STEPS, max_ctx_tokens=2000, time_budget=200)
            peft_model.save_pretrained(adir)
            subprocess.run([sys.executable, CONVERT, adir, '--base', BASE_HF, '--outfile', gguf, '--outtype', 'f16'],
                           env=cenv, stdout=open(os.path.join(WORK, 'convert.log'), 'ab'),
                           stderr=subprocess.STDOUT, timeout=600)
            if os.path.exists(gguf):
                MANIFEST[tid] = gguf
            _sh.rmtree(adir, ignore_errors=True)   # keep only the small GGUF, not the fp32 PEFT dir (disk guard)
            print('[AB %d/%d] %s ok=%s %.2fh' % (i + 1, len(trainable), tid, tid in MANIFEST, elapsed_h()), flush=True)
        except Exception as e:
            _sh.rmtree(adir, ignore_errors=True)
            print('[AB %d/%d] %s FAIL %r' % (i + 1, len(trainable), tid, str(e)[:90]), flush=True)
    del peft_model, model
    torch.cuda.empty_cache()
except Exception as e:
    print('Phase A+B unavailable (%r); all tasks served no-TTT' % (str(e)[:120],), flush=True)
    MANIFEST = {}
print('Phase A+B done: %d adapters ready, %.2fh elapsed' % (len(MANIFEST), elapsed_h()), flush=True)
"""),

md("## 4. Phase C: serve and generate, sharded across all GPUs"),

code(r"""
submission = {}
LOCK = threading.Lock()
def record(tid, attempts):
    with LOCK:
        submission[tid] = attempts
        write_submission(submission, tasks, OUT)

def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]

def worker(gpu, port, my_no_ttt, my_ttt):
    proc = start_server(gpu, port)
    if not health(port):
        print('gpu%d: server failed to start' % gpu, flush=True); return
    for tid in my_no_ttt:
        try: att = airv(tasks[tid], port)
        except Exception as e: att = empty(tasks[tid]); print('  gpu%d no-TTT %s ERR %r' % (gpu, tid, str(e)[:80]), flush=True)
        record(tid, att)
        print('[gpu%d no-TTT] %s %.2fh' % (gpu, tid, elapsed_h()), flush=True)
    for batch in chunks(my_ttt, BATCH):
        over = elapsed_h() > TOTAL_BUDGET_H
        stop_server(proc)
        proc = start_server(gpu, port) if over else start_server(gpu, port, [MANIFEST[t] for t in batch])
        ttt_ok = (not over) and health(port)
        if not health(port):
            proc = start_server(gpu, port)  # last-ditch no-TTT server
        for k, tid in enumerate(batch):
            try:
                att = airv(tasks[tid], port, adapter_idx=(k if ttt_ok else None),
                           n_loaded=(len(batch) if ttt_ok else 0))
            except Exception as e:
                att = empty(tasks[tid]); print('  gpu%d TTT %s ERR %r' % (gpu, tid, str(e)[:80]), flush=True)
            record(tid, att)
            print('[gpu%d %s] %s %.2fh' % (gpu, 'TTT' if ttt_ok else 'noTTT-fb', tid, elapsed_h()), flush=True)
    stop_server(proc)

no_ttt = [t for t in tids if t not in MANIFEST]
ttt_ids = [t for t in tids if t in MANIFEST]
shards_no = [no_ttt[i::NGPU] for i in range(NGPU)]
shards_ttt = [ttt_ids[i::NGPU] for i in range(NGPU)]
print('Phase C: %d no-TTT + %d TTT tasks across %d GPU(s)' % (len(no_ttt), len(ttt_ids), NGPU), flush=True)

threads = [threading.Thread(target=worker, args=(g, PORTS[g], shards_no[g], shards_ttt[g])) for g in range(NGPU)]
for t in threads: t.start()
for t in threads: t.join()
write_submission(submission, tasks, OUT)
print('wrote', OUT, '| %d no-TTT, %d TTT | %.2fh total' % (len(no_ttt), len(ttt_ids), elapsed_h()), flush=True)

if sols:
    from src.arc.eval import score_submission
    s = {t: sols[t] for t in tids if t in sols}
    acc, c, n = score_submission(submission, s)
    print('SCORE acc=%.4f correct=%d total=%d' % (acc, c, n), flush=True)
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "arc_submission_v2.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", os.path.abspath(out), "cells:", len(cells))
