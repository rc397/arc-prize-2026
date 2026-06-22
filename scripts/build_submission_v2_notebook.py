"""Build notebooks/arc_submission_v2.ipynb: the layered TTT + colour-perm submission.

v2 over v1 (no-TTT AIRV): adds per-task test-time training on the small/trainable tasks (maxpair
filter) plus colour-permutation augmentation, with a global wall-clock budget and per-task try/except
so a run always degrades to no-TTT rather than breaking. Three phases keep PyTorch (train) and
llama.cpp (serve) from sharing VRAM, the same shape the rig validated.

Reuses the repo's proven code by importing src.arc from the attached repo dataset (rather than
inlining the whole training stack). Attach FOUR inputs (see the markdown cell):
  1. prep output  (GGUF, llama-server + libs, convert_lora_to_gguf.py, gguf-py)
  2. the NVARC HF model  (sorokin/qwen3_4b_grids15_sft139) for training + as the conversion base
  3. this repo  (rc397/arc-prize-2026) as a dataset, for src.arc
  4. the ARC Prize 2026 competition data

SMOKE TEST FIRST: set N_LIMIT to ~3 and confirm a clean submission.json before a full run, then read
run.log for the per-phase timing so the full 240-task run fits the 12h cap.
"""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# ARC-AGI-2 on a single 16 GB GPU, v2 (TTT + colour-perm)

Layered on the v1 AIRV pipeline: per-task test-time training on the small/trainable tasks at adapter
scale 1.25, colour-permutation augmentation on top of the 8 D4 augmentations, and no-TTT + colour-perm
on the tasks too large to train. A global wall-clock budget and per-task fallback keep a run from
breaking: anything that errors or runs long degrades to no-TTT and still writes `submission.json`.
"""),

md(r"""
## Attach four inputs, smoke-test, then Run All

Offline competition (no internet at submission). Via **+ Add Input**:

1. **The prep notebook Output** (`arc_prep_v1` or newer): the Q8_0 GGUF, the Linux `llama-server`
   with its shared libraries, and `convert_lora_to_gguf.py` + `gguf-py`.
2. **The NVARC model** (`sorokin/qwen3_4b_grids15_sft139`): the HF weights, needed to train the
   per-task adapters and as the base for the LoRA-to-GGUF conversion.
3. **This repo as a dataset** (`rc397/arc-prize-2026`): provides `src.arc` (model, augment, ttt,
   select, data). Create a Kaggle dataset from the repo, or add it as a utility.
4. **The ARC Prize 2026 competition data**.

**Smoke test before a real submission:** leave `N_LIMIT = 3` in the config cell, Run All, confirm the
three phases complete and `submission.json` is written, then open `run.log` in the Output and check
the per-phase timing. Only then set `N_LIMIT = 0` (all tasks), confirm the projected total is under
12h, and submit. Internet **OFF**, Accelerator = the GPU you ran prep on.
"""),

md("## 0. Configuration (auto-detect inputs, set the v2 knobs)"),

code(r"""
import os, sys, glob, json, time, subprocess, urllib.request, shutil
from pathlib import Path

# Tee stdout/stderr to run.log so the full per-phase log lands in the Output (downloadable).
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
N_LIMIT       = 3 if ON_KAGGLE else 2     # 3 = smoke test; set 0 for all tasks once timing is known
MAXPAIR       = 1000                      # train a task only if its largest train pair <= this many tokens
MAX_TTT_TASKS = 120                       # hard cap on how many tasks get TTT (timing guard)
TTT_STEPS     = 20                        # regularized: few steps
TTT_RANK      = 16
TTT_DROPOUT   = 0.1
SCALE         = 1.25                      # adapter scale at inference (rig optimum)
N_COLORPERM   = 4                         # colour-permuted augmentations on top of the 8 D4
TRAIN_BUDGET_H = 3.0                      # stop training new adapters after this many hours
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
    GGUF_PY  = _find(['/kaggle/input/**/gguf-py/gguf/__init__.py'])
    GGUF_PY  = os.path.dirname(os.path.dirname(GGUF_PY)) if GGUF_PY else None
    base_cfg = _find(['/kaggle/input/**/transformers/**/config.json', '/kaggle/input/**/qwen3*/**/config.json',
                      '/kaggle/input/**/config.json'])
    BASE_HF  = os.path.dirname(base_cfg) if base_cfg else None
    repo_mark = _find(['/kaggle/input/**/src/arc/ttt.py'])
    REPO = repo_mark[:repo_mark.index('/src/arc/ttt.py')] if repo_mark else None
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
assert all([MODEL_GGUF, SERVER, BASE_HF, CONVERT, GGUF_PY, REPO, CH]), 'an input is missing: attach prep output, NVARC model, the repo, and the competition data'
"""),

md("## 1. Helpers (imports from the attached repo; augmentations, server, maxpair)"),

code(r"""
import numpy as np
from src.arc.model import build_prompt_ids, grid_to_ids, IM_END
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES, random_augmentation
from src.arc.select import select_top2

_rng = np.random.default_rng(20260620)
AUGS = [Augmentation(d4=name) for name in D4_NAMES] + [
    random_augmentation(_rng, use_d4=True, recolor=True, keep_background=True, reorder=False)
    for _ in range(N_COLORPERM)]
print('augmentations: %d D4 + %d colour-perm = %d' % (len(D4_NAMES), N_COLORPERM, len(AUGS)))

def maxpair_tokens(task):
    return max(len(grid_to_ids(p['input'])) + len(grid_to_ids(p['output'])) for p in task['train'])

def health():
    try:
        return json.loads(urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).read()).get('status') == 'ok'
    except Exception:
        return False

def stop_server():
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True)
    else:
        subprocess.run(['pkill', '-f', 'llama-server'], capture_output=True)
    time.sleep(1)

def start_server(lora_paths=()):
    args = [SERVER, '-m', MODEL_GGUF, '-ngl', '99', '-c', '16384', '-np', '1', '--host', '127.0.0.1',
            '--port', '8080', '--no-webui', '--no-cache-prompt', '--no-cache-idle-slots']
    if lora_paths:
        args += ['--lora-init-without-apply']
        for p in lora_paths:
            args += ['--lora', p]
    flags = 0x08000000 if os.name == 'nt' else 0
    lf = open(os.path.join(WORK, 'llama-server.log'), 'ab')
    subprocess.Popen(args, stdout=lf, stderr=lf, creationflags=flags) if os.name == 'nt' else \
        subprocess.Popen(args, stdout=lf, stderr=lf)
    for _ in range(120):
        time.sleep(2)
        if health():
            return True
    return False

def set_adapter(idx, n_loaded):
    body = json.dumps([{'id': j, 'scale': (SCALE if j == idx else 0.0)} for j in range(n_loaded)]).encode()
    req = urllib.request.Request('http://127.0.0.1:8080/lora-adapters', data=body,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    urllib.request.urlopen(req, timeout=10).read()

def query(prompt, n_predict=1024, retries=3):
    body = json.dumps({'prompt': prompt, 'n_predict': n_predict, 'temperature': 0.0, 'cache_prompt': False}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request('http://127.0.0.1:8080/completion', data=body,
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

def airv(task, adapter_idx=None, n_loaded=0):
    if adapter_idx is not None:
        set_adapter(adapter_idx, n_loaded)
    attempts = []
    for ti in range(len(task['test'])):
        cands = []
        for aug in AUGS:
            at = apply_task(aug, task)
            r = query(build_prompt_ids(at['train'], at['test'][ti]['input']))
            if r is None:
                continue
            g = parse(r.get('content', ''))
            if g:
                cands.append((invert_grid(aug, g), 0.0))
        a1, a2 = select_top2(cands)
        attempts.append({'attempt_1': a1, 'attempt_2': a2})
    return attempts

T_START = time.time()
def elapsed_h():
    return (time.time() - T_START) / 3600.0
"""),

md("## 2. Phase A: train per-task adapters on the trainable tasks (PyTorch)"),

code(r"""
tasks, sols = load_consolidated(CH, SOL, n=None)
tids = list(tasks)
if N_LIMIT:
    tids = tids[:N_LIMIT]

# Select trainable tasks by largest train pair (the VRAM driver), smallest first, capped.
trainable = sorted([t for t in tids if maxpair_tokens(tasks[t]) <= MAXPAIR],
                   key=lambda t: maxpair_tokens(tasks[t]))[:MAX_TTT_TASKS]
print('Phase A: %d/%d tasks trainable (maxpair <= %d, cap %d)' % (len(trainable), len(tids), MAXPAIR, MAX_TTT_TASKS))

ADAPTERS = os.path.join(WORK, 'loras_v2')
os.makedirs(ADAPTERS, exist_ok=True)
trained = []
try:
    import torch
    from src.arc.model import load_model
    from src.arc.ttt import make_adapter, test_time_train
    model, tok = load_model(BASE_HF, dtype=torch.bfloat16, device='cuda')
    peft_model, init = make_adapter(model, rank=TTT_RANK, alpha=TTT_RANK, dropout=TTT_DROPOUT)
    rng = np.random.default_rng(0)
    for i, tid in enumerate(trainable):
        if elapsed_h() > TRAIN_BUDGET_H:
            print('Phase A: train budget reached at %d/%d, rest served no-TTT' % (i, len(trainable)), flush=True)
            break
        try:
            t1 = time.time()
            loss = test_time_train(peft_model, init, tasks[tid], rng, steps=TTT_STEPS,
                                   max_ctx_tokens=1600, time_budget=120)
            adir = os.path.join(ADAPTERS, tid)
            peft_model.save_pretrained(adir)
            trained.append(tid)
            print('[A %d/%d] %s loss=%.3f %.0fs' % (i + 1, len(trainable), tid, loss, time.time() - t1), flush=True)
        except Exception as e:
            print('[A %d/%d] %s TRAIN-FAIL %r' % (i + 1, len(trainable), tid, str(e)[:90]), flush=True)
    del peft_model, model
    torch.cuda.empty_cache()
except Exception as e:
    print('Phase A unavailable (%r); falling back to no-TTT for all tasks' % (str(e)[:120],), flush=True)
    trained = []
print('Phase A done: %d adapters trained, %.2fh elapsed' % (len(trained), elapsed_h()), flush=True)
"""),

md("## 3. Phase B: convert each adapter to GGUF (clean process, no model in VRAM)"),

code(r"""
MANIFEST = {}
env = dict(os.environ, PYTHONPATH=GGUF_PY + os.pathsep + os.environ.get('PYTHONPATH', ''),
           PYTHONIOENCODING='utf-8')
for i, tid in enumerate(trained):
    gguf = os.path.join(ADAPTERS, tid + '.gguf')
    if not os.path.exists(gguf):
        try:
            subprocess.run([sys.executable, CONVERT, os.path.join(ADAPTERS, tid), '--base', BASE_HF,
                            '--outfile', gguf, '--outtype', 'f16'], env=env,
                           stdout=open(os.path.join(WORK, 'convert.log'), 'ab'),
                           stderr=subprocess.STDOUT, timeout=600)
        except Exception as e:
            print('[B %d/%d] %s CONVERT-FAIL %r' % (i + 1, len(trained), tid, str(e)[:90]), flush=True)
    if os.path.exists(gguf):
        MANIFEST[tid] = gguf
print('Phase B done: %d/%d adapters converted, %.2fh elapsed' % (len(MANIFEST), len(trained), elapsed_h()), flush=True)
"""),

md("## 4. Phase C: serve and generate (TTT on trainable, no-TTT + colour-perm on the rest)"),

code(r"""
submission = {}

# Pass 1: no-TTT tasks (no adapter) under one server.
no_ttt = [t for t in tids if t not in MANIFEST]
stop_server(); start_server()
for i, tid in enumerate(no_ttt):
    try:
        submission[tid] = airv(tasks[tid])
    except Exception as e:
        submission[tid] = [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in tasks[tid]['test']]
        print('  no-TTT %s ERROR %r' % (tid, str(e)[:90]), flush=True)
    write_submission(submission, tasks, OUT)
    print('[C1 %d/%d] %s (no-TTT) %.2fh' % (i + 1, len(no_ttt), tid, elapsed_h()), flush=True)

# Pass 2: TTT tasks, batched so each server load holds a set of adapters. Bail to no-TTT if over budget.
ttt_ids = [t for t in tids if t in MANIFEST]
BATCH = 24
for b in range(0, len(ttt_ids), BATCH):
    batch = ttt_ids[b:b + BATCH]
    stop_server()
    if elapsed_h() > TOTAL_BUDGET_H or not start_server([MANIFEST[t] for t in batch]):
        start_server()  # fall back: serve these no-TTT
        for tid in batch:
            try: submission[tid] = airv(tasks[tid])
            except Exception: submission[tid] = [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in tasks[tid]['test']]
            write_submission(submission, tasks, OUT)
        print('[C2] batch at %d served no-TTT (over budget or server fail)' % b, flush=True)
        continue
    for k, tid in enumerate(batch):
        try:
            if not health():
                stop_server(); start_server([MANIFEST[t] for t in batch])
            submission[tid] = airv(tasks[tid], adapter_idx=k, n_loaded=len(batch))
        except Exception as e:
            submission[tid] = [{'attempt_1': [[0]], 'attempt_2': [[0]]} for _ in tasks[tid]['test']]
            print('  TTT %s ERROR %r' % (tid, str(e)[:90]), flush=True)
        write_submission(submission, tasks, OUT)
        print('[C2 %d/%d] %s (TTT s%.2f) %.2fh' % (b + k + 1, len(ttt_ids), tid, SCALE, elapsed_h()), flush=True)
stop_server()
print('wrote', OUT, '| tasks: %d no-TTT, %d TTT | %.2fh total' % (len(no_ttt), len(ttt_ids), elapsed_h()), flush=True)

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
