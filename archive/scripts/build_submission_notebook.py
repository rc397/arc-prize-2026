"""Build notebooks/arc_submission.ipynb: the runnable offline ARC-AGI-2 pipeline (v1 no-TTT AIRV).
Self-contained: the config cell auto-detects the GGUF + llama-server from the attached prep-notebook
output (and sets LD_LIBRARY_PATH for the server's shared libs), and the pipeline code from this
repo's src/arc is inlined, so the only things to attach are the prep output and the competition data."""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# ARC-AGI-2 on a single 16 GB GPU, runnable pipeline (v1)

AIRV (Augment-Infer-Reverse-Vote) over the NVARC Qwen3-4B grid model, served with llama.cpp. This
notebook runs end to end offline and writes `submission.json`. v1 is the no-TTT baseline (~12.79% on
the public eval); TTT layers on top later.
"""),

md(r"""
## Attach two inputs, then Run All

This is an offline competition (no internet at submission). Attach via **+ Add Input**:

1. **The prep notebook's Output** (`arc_prep_v1` or newer): provides the Q8_0 GGUF and the Linux
   `llama-server` with its shared libraries (under `assets/bin/`).
2. **The ARC Prize 2026 competition data** (Add Input -> Competitions -> ARC Prize 2026): provides the
   test challenges JSON.

Nothing else to attach: the config cell finds the GGUF, the server, and the challenges by searching
`/kaggle/input`, and the pipeline code is inlined below. Settings: **Internet OFF**, Accelerator =
the same GPU you ran prep on (the server is built for that arch). Then **Run All**.
"""),

md("## 0. Configuration (auto-detect attached inputs)"),

code(r"""
import os, sys, glob, json, time, subprocess, urllib.request, shutil
from pathlib import Path

# Tee stdout/stderr to run.log so the full pipeline log lands in the notebook Output (downloadable),
# not just the live cell log that Kaggle can truncate.
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

# Offline: never let HF/transformers phone home, never pip-install at runtime.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ON_KAGGLE = os.path.isdir('/kaggle')

def _find(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None

if ON_KAGGLE:
    MODEL_GGUF = _find(['/kaggle/input/**/assets/arc-qwen4b-q8_0.gguf',
                        '/kaggle/input/**/arc-qwen4b-q8_0.gguf'])
    server_src = _find(['/kaggle/input/**/assets/bin/llama-server',
                        '/kaggle/input/**/bin/llama-server',
                        '/kaggle/input/**/llama-server'])
    CH = _find(['/kaggle/input/**/arc-agi_test_challenges.json',
                '/kaggle/input/**/*test*challenge*.json',
                '/kaggle/input/**/arc-agi_evaluation_challenges.json'])
    SOL = None                                   # hidden at submission time
    OUT = '/kaggle/working/submission.json'
    # llama-server is dynamically linked: copy its directory (binary + .so libraries) to a writable
    # spot, mark it executable, and add it to the loader path so the libs resolve.
    SERVER = None
    if server_src:
        bindst = '/kaggle/working/llamabin'
        shutil.copytree(os.path.dirname(server_src), bindst, dirs_exist_ok=True)
        SERVER = os.path.join(bindst, 'llama-server')
        os.chmod(SERVER, 0o755)
        os.environ['LD_LIBRARY_PATH'] = bindst + ':' + os.environ.get('LD_LIBRARY_PATH', '')
else:
    sys.path.insert(0, r'D:\arc')
    MODEL_GGUF = r'D:\arc\models\arc-qwen4b-q8_0.gguf'
    SERVER = r'D:\arc\llamacpp\llama-server.exe'
    CH = r'D:\arc\data\kaggle\arc-agi_evaluation_challenges.json'
    SOL = r'D:\arc\data\kaggle\arc-agi_evaluation_solutions.json'
    OUT = r'D:\arc\experiments\submission_nb.json'

N_LIMIT = 2 if not ON_KAGGLE else 0              # small smoke run locally, all tasks on Kaggle
print('on_kaggle =', ON_KAGGLE, ' n_limit =', N_LIMIT)
for label, path in [('model', MODEL_GGUF), ('server', SERVER), ('challenges', CH)]:
    print('%-11s %-5s %s' % (label, 'OK' if path and os.path.exists(path) else 'MISSING', path))
assert MODEL_GGUF and SERVER and CH, 'an input is missing: attach the prep output and the competition data'
"""),

md("## 1. Pipeline code (inlined from this repo's src/arc, no repo attach needed)"),

code(r"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# --- model.py: 16-token grid vocab, id-level ChatML prompt (a color value equals its token id) ---
NL, USER, ASSISTANT, PAD, IM_START, IM_END = 10, 11, 12, 13, 14, 15

def grid_to_ids(grid):
    ids = []
    for i, row in enumerate(grid):
        if i:
            ids.append(NL)
        ids.extend(int(v) for v in row)
    return ids

def _turn(role_id, content_ids):
    return [IM_START, role_id, NL] + content_ids + [IM_END]

def build_prompt_ids(train_pairs, test_input):
    seq = []
    for p in train_pairs:
        seq += _turn(USER, grid_to_ids(p['input'])) + [NL]
        seq += _turn(ASSISTANT, grid_to_ids(p['output'])) + [NL]
    seq += _turn(USER, grid_to_ids(test_input)) + [NL]
    seq += [IM_START, ASSISTANT, NL]
    return seq

# --- augment.py: invertible D4 augmentations ---
def _id(a): return a
def _rot90(a): return np.rot90(a, 1)
def _rot180(a): return np.rot90(a, 2)
def _rot270(a): return np.rot90(a, 3)
def _flip_lr(a): return np.fliplr(a)
def _flip_ud(a): return np.flipud(a)
def _transpose(a): return a.T
def _anti(a): return np.fliplr(np.fliplr(a).T)
D4_FWD = {'identity': _id, 'rot90': _rot90, 'rot180': _rot180, 'rot270': _rot270,
          'flip_lr': _flip_lr, 'flip_ud': _flip_ud, 'transpose': _transpose, 'anti': _anti}
D4_INV = {'identity': _id, 'rot90': _rot270, 'rot180': _rot180, 'rot270': _rot90,
          'flip_lr': _flip_lr, 'flip_ud': _flip_ud, 'transpose': _transpose, 'anti': _anti}
D4_NAMES = list(D4_FWD)

@dataclass(frozen=True)
class Augmentation:
    d4: str = 'identity'
    color: Tuple[int, ...] = tuple(range(10))
    example_order: Optional[Tuple[int, ...]] = None
    def inv_color(self):
        inv = [0] * len(self.color)
        for c, nc in enumerate(self.color):
            inv[nc] = c
        return tuple(inv)

def _recolor(arr, mapping):
    return np.asarray(mapping, dtype=arr.dtype)[arr]

def apply_grid(aug, grid):
    a = np.asarray(grid, dtype=np.int64)
    return _recolor(D4_FWD[aug.d4](a), aug.color).tolist()

def invert_grid(aug, grid):
    a = np.asarray(grid, dtype=np.int64)
    return D4_INV[aug.d4](_recolor(a, aug.inv_color())).tolist()

def apply_task(aug, task):
    def pair(p):
        q = {'input': apply_grid(aug, p['input'])}
        if 'output' in p:
            q['output'] = apply_grid(aug, p['output'])
        return q
    train = [pair(p) for p in task['train']]
    if aug.example_order is not None:
        train = [train[i] for i in aug.example_order]
    return {'train': train, 'test': [pair(p) for p in task['test']]}

# --- data.py / select.py / eval.py ---
def load_consolidated(challenges_path, solutions_path=None, n=None):
    ch = json.loads(Path(challenges_path).read_text(encoding='utf-8'))
    sol = json.loads(Path(solutions_path).read_text(encoding='utf-8')) if solutions_path else {}
    if n:
        ids = list(ch)[:n]
        ch = {k: ch[k] for k in ids}
        sol = {k: sol[k] for k in ids if k in sol}
    return ch, sol

def write_submission(submission, challenges, path):
    out = {}
    for tid, task in challenges.items():
        preds = submission.get(tid, [])
        entry = []
        for i in range(len(task['test'])):
            p = preds[i] if i < len(preds) else {}
            a1 = p.get('attempt_1') or [[0]]
            entry.append({'attempt_1': a1, 'attempt_2': p.get('attempt_2') or a1})
        out[tid] = entry
    Path(path).write_text(json.dumps(out))
    return out

def select_top2(candidates):
    agg = {}
    for g, lp in candidates:
        if not g:
            continue
        k = tuple(tuple(int(v) for v in r) for r in g)
        if k not in agg:
            agg[k] = [0, 0.0, g]
        agg[k][0] += 1
        agg[k][1] += lp
    if not agg:
        return [[0]], [[0]]
    ranked = sorted(agg.values(), key=lambda v: (v[0], v[1] / v[0]), reverse=True)
    return ranked[0][2], (ranked[1][2] if len(ranked) > 1 else ranked[0][2])

def normalize(grid):
    return [[int(v) for v in row] for row in grid]

def score_submission(submission, solutions):
    correct = total = 0
    for tid, gts in solutions.items():
        preds = submission.get(tid, [])
        for i, gt in enumerate(gts):
            total += 1
            gt_n = normalize(gt)
            if i < len(preds) and (preds[i] or {}):
                p = preds[i]
                if any(k in p and normalize(p[k]) == gt_n for k in ('attempt_1', 'attempt_2')):
                    correct += 1
    return (correct / total if total else 0.0), correct, total
print('pipeline ready')
"""),

md(r"""
## 2. No-TTT AIRV baseline

For each test input, generate one prediction under each of the 8 dihedral (D4) augmentations, invert
each back to the original frame, and vote to two attempts. The pipeline owns the llama-server
lifecycle and restarts it on the rare crash.
"""),

code(r"""
HEALTH = 'http://127.0.0.1:8080/health'
COMPL  = 'http://127.0.0.1:8080/completion'
SERVER_ARGS = [SERVER, '-m', MODEL_GGUF, '-ngl', '99', '-c', '16384', '-np', '1',
               '--host', '127.0.0.1', '--port', '8080', '--no-webui',
               '--no-cache-prompt', '--no-cache-idle-slots']
AUGS = [Augmentation(d4=n) for n in D4_NAMES]

def healthy():
    try:
        return json.loads(urllib.request.urlopen(HEALTH, timeout=3).read()).get('status') == 'ok'
    except Exception:
        return False

def ensure_server():
    if healthy():
        return True
    flags = 0x08000000 if os.name == 'nt' else 0
    _slog = open('llama-server.log', 'ab')  # server logs to a file, not the notebook (Kaggle UI lag)
    subprocess.Popen(SERVER_ARGS, stdout=_slog, stderr=_slog, creationflags=flags) if os.name == 'nt' else subprocess.Popen(SERVER_ARGS, stdout=_slog, stderr=_slog)
    for _ in range(120):
        time.sleep(2)
        if healthy():
            return True
    return False

def query(prompt_ids, n_predict=1024, retries=3):
    body = json.dumps({'prompt': prompt_ids, 'n_predict': n_predict,
                       'temperature': 0.0, 'cache_prompt': False}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(COMPL, data=body, headers={'Content-Type': 'application/json'})
            return json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception:
            if a < retries - 1:
                ensure_server()
    return None

def parse(text):
    rows = [[int(c) for c in s.strip()] for s in text.split('\n')
            if s.strip() and all(c in '0123456789' for c in s.strip())]
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None

def airv(task):
    attempts = []
    for ti in range(len(task['test'])):
        cands = []
        for aug in AUGS:
            at = apply_task(aug, task)
            r = query(build_prompt_ids(at['train'], at['test'][ti]['input']))
            g = parse(r.get('content', '')) if r else None
            if g:
                cands.append((invert_grid(aug, g), 0.0))
        a1, a2 = select_top2(cands)
        attempts.append({'attempt_1': a1, 'attempt_2': a2})
    return attempts
"""),

code(r"""
assert ensure_server(), 'llama-server failed to start (check the attached prep build / accelerator)'
tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
submission = {}
t0 = time.time()
for i, tid in enumerate(tasks):
    submission[tid] = airv(tasks[tid])
    write_submission(submission, tasks, OUT)
    print('[%d/%d] %s  t=%.0fs' % (i + 1, len(tasks), tid, time.time() - t0), flush=True)
print('wrote', OUT)
if sols:
    acc, c, n = score_submission(submission, sols)
    print('no-TTT AIRV score: %d/%d = %.4f' % (c, n, acc))
"""),

md(r"""
## 3. Per-task TTT (next)

v2 layers TTT on top: per task, fit a rank-16 LoRA in PyTorch, convert it to a GGUF adapter,
hot-swap it on llama-server, and run the same AIRV at adapter scale 1.25.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.x"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "arc_submission.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
json.load(open(out, encoding="utf-8"))
print("wrote", os.path.abspath(out), "cells:", len(cells))
