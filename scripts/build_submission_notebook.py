"""Build paper/arc_submission.ipynb, the runnable pipeline notebook we grow iteratively.
v1: config + setup + no-TTT AIRV -> submission.json (+ score if solutions present). Reuses the
proven src/arc modules so the notebook is not new risk. Detects Kaggle vs rig/local for paths."""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# ARC-AGI-2 on a single 16 GB GPU, runnable pipeline

AIRV (Augment-Infer-Reverse-Vote) plus per-task test-time training, served through a llama.cpp LoRA
bridge. Best result on the public eval: a blended 16.28 percent, up from a 12.79 percent no-TTT
baseline. This notebook runs the pipeline end to end and writes `submission.json`.

It grows in stages: v1 is the no-TTT AIRV baseline below; TTT and the scale/vote levers layer on
top in later cells. ARC Prize is an offline code competition, so on Kaggle the model, the llama.cpp
binary, and this repo's `src/arc` must be attached as datasets (no download at runtime); the config
cell detects Kaggle vs a local/rig run and sets paths accordingly.
"""),

md(r"""
## Offline submission, datasets to attach

ARC Prize is an offline code competition (no internet at submission). Attach these so the notebook
runs with internet off:

1. Base model (HF transformers) for TTT training and adapter conversion: the Kaggle model
   `sorokin/qwen3_4b_grids15_sft139`.
2. The Q8_0 GGUF (our conversion of that model) for serving: upload as a dataset.
3. A Linux llama.cpp build (`llama-server` plus `convert_lora_to_gguf.py`): upload as a dataset.
   The custom-vocab patches are baked into the GGUF at conversion time, so the server itself is a
   stock build.
4. This repository (`src/arc`): attach as a dataset or a utility script.

No `pip install` runs at submission. The notebook forces `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`
so nothing tries to reach the network. The no-TTT path (v1) needs only items 2-4; TTT (v2) adds
item 1 plus PyTorch and PEFT (pre-load PEFT as a wheel if the Kaggle image lacks it).
"""),

md("## 0. Configuration"),

code(r"""
import os, sys, json, time, subprocess, urllib.request

# Offline code competition: no internet at submission, so never let transformers/HF phone home,
# and never pip-install at runtime (pre-load any wheels as a dataset and install with --no-index).
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ON_KAGGLE = os.path.isdir('/kaggle')

if ON_KAGGLE:
    # TODO set these to the datasets you attach to the notebook
    REPO       = '/kaggle/input/arc-prize-2026-code'          # this repo (for src/arc)
    MODEL_GGUF = '/kaggle/input/arc-qwen4b-gguf/arc-qwen4b-q8_0.gguf'
    SERVER     = '/kaggle/input/llamacpp-linux/llama-server'  # Linux build, +x
    CH         = '/kaggle/input/arc-prize-2026/arc-agi_test_challenges.json'
    SOL        = None                                          # hidden at submission time
    OUT        = '/kaggle/working/submission.json'
else:
    REPO       = r'D:\arc'
    MODEL_GGUF = r'D:\arc\models\arc-qwen4b-q8_0.gguf'
    SERVER     = r'D:\arc\llamacpp\llama-server.exe'
    CH         = r'D:\arc\data\kaggle\arc-agi_evaluation_challenges.json'
    SOL        = r'D:\arc\data\kaggle\arc-agi_evaluation_solutions.json'
    OUT        = r'D:\arc\experiments\submission_nb.json'

sys.path.insert(0, REPO)

# best configuration found on the public eval
SCALE, STEPS, RANK, DROPOUT = 1.25, 20, 16, 0.1
N_LIMIT = 2 if not ON_KAGGLE else 0      # small smoke run locally, all tasks on Kaggle
print('on_kaggle=%s  n_limit=%s' % (ON_KAGGLE, N_LIMIT))
"""),

md("## 1. Setup, model, engine, code"),

code(r"""
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission

for label, path in [('model', MODEL_GGUF), ('server', SERVER), ('challenges', CH)]:
    print('%-11s %s  %s' % (label, 'ok' if os.path.exists(path) else 'MISSING', path))
"""),

md(r"""
## 2. No-TTT AIRV baseline

For each test input we generate one prediction under each of the eight dihedral (D4) augmentations,
invert each prediction back to the original frame, and vote to two attempts. The pipeline owns the
llama-server lifecycle and restarts it on the rare crash, the same code that produced the 12.79
percent baseline.
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
    subprocess.Popen(SERVER_ARGS, creationflags=flags) if os.name == 'nt' else subprocess.Popen(SERVER_ARGS)
    for _ in range(60):
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
ensure_server()
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

The next cells will layer TTT on top: per task, fit a rank-16 LoRA in PyTorch, convert it to a GGUF
adapter, hot-swap it on llama-server, and run the same AIRV at adapter scale 1.25. That is the step
that lifts the trainable tasks from 12.8 to 17.6 percent. Added in v2.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.x"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "..", "paper", "arc_submission.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
json.load(open(out, encoding="utf-8"))
print("wrote", os.path.abspath(out), "cells:", len(cells))
