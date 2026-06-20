"""Build the Paper Track public notebook paper/arc_pipeline.ipynb from a cell list. Keeps the
notebook JSON well-formed (no hand-escaping). Code cells show the real pipeline logic; the notebook
documents and reproduces the approach (full execution needs the rig llama.cpp + GGUF model)."""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# ARC-AGI-2 on a Single Consumer GPU

### AIRV + per-task Test-Time Training via a llama.cpp LoRA bridge

The strongest ARC-AGI-2 results come from multi-GPU clusters. This notebook documents a fully
offline pipeline that runs the 2025 winning recipe, a small grid language model with
Augment-Infer-Reverse-Vote (AIRV) selection and per-task test-time training (TTT), on a single
16 GB consumer GPU (RTX 5070 Ti, Blackwell, native Windows), and reports where it walls out.

Headline numbers on the 120-task public evaluation set:

| system | accuracy |
|---|---|
| no-TTT AIRV baseline | 12.79% |
| regularized TTT on the 84 trainable tasks | 16.8% (vs 12.8% no-TTT) |
| blended full-120 (TTT where trainable) | ~16% |

The contribution is an honest accounting of what each stage costs on one card, and the engineering
needed to make per-task TTT serve at all on this hardware.
"""),

md(r"""
## 1. The constraint

A single RTX 5070 Ti, 16 GB, Blackwell (sm_120), native Windows. No cluster, no Linux toolchain.
We build on the released NVARC `qwen3_4b_grids15_sft139` checkpoint (a Qwen3-4B trained for ARC
grids) rather than retraining. The goal is not to top the leaderboard but to map precisely where a
single card stops being enough.
"""),

md(r"""
## 2. Model and prompt format

The model carries a 16-token vocabulary: digits 0-9 are the ten grid colors, plus a newline, two
role markers, and three specials. A grid is serialized one token per cell, and a color value
equals its token id. The role and special tokens cannot be produced by tokenizing text, so prompts
are assembled at the token-id level as ChatML few-shot sequences.
"""),

code(r"""
NL, USER, ASSISTANT, PAD, IM_START, IM_END = 10, 11, 12, 13, 14, 15

def grid_to_ids(grid):
    ids = []
    for i, row in enumerate(grid):
        if i:
            ids.append(NL)
        ids.extend(int(v) for v in row)   # color value == token id (0..9)
    return ids

def _turn(role_id, content_ids):
    return [IM_START, role_id, NL] + content_ids + [IM_END]

def build_prompt_ids(train_pairs, test_input):
    seq = []
    for p in train_pairs:
        seq += _turn(USER, grid_to_ids(p["input"])) + [NL]
        seq += _turn(ASSISTANT, grid_to_ids(p["output"])) + [NL]
    seq += _turn(USER, grid_to_ids(test_input)) + [NL]
    seq += [IM_START, ASSISTANT, NL]
    return seq
"""),

md(r"""
## 3. The engine is the bottleneck

The single most consequential finding: on the standard Hugging Face eager path, prefill is healthy
(~3,400 tok/s) but decoding runs at only ~13 tok/s, a tenth of what the card's bandwidth allows,
because of per-token framework overhead with no CUDA graphs. The usual fix (compiling decode) is
unavailable: `torch.compile` needs Triton, which has no native-Windows backend.

The fix is to change engines. We convert the model to GGUF and serve it with llama.cpp (two small
converter patches handle the custom vocabulary: map the unknown tokenizer hash to the qwen2
pre-tokenizer, and add two throwaway merges so the WordLevel tokenizer passes the loader). The
prebuilt CUDA 12.4 binary runs on Blackwell through PTX JIT. Measured decode: 170 tok/s at Q8_0, a
13x speedup, near-lossless, 3.6 GB footprint. On one card, the inference engine dominated every
other knob.
"""),

md(r"""
## 4. AIRV: augment, infer, reverse, vote

For each test input we generate a prediction under each of the eight dihedral (D4) augmentations,
optionally with color permutations, invert each prediction back to the original frame, and vote.
The recipe relies on label-preserving symmetries: geometry and recoloring commute, so the inverse
undoes both.
"""),

code(r"""
import numpy as np

D4_FWD = {"identity": lambda a: a, "rot90": lambda a: np.rot90(a,1), "rot180": lambda a: np.rot90(a,2),
          "rot270": lambda a: np.rot90(a,3), "flip_lr": np.fliplr, "flip_ud": np.flipud,
          "transpose": lambda a: a.T, "anti": lambda a: np.fliplr(np.fliplr(a).T)}

def select_top2(candidates):
    # rank distinct grids by vote count, then mean log-prob; return the two best
    agg = {}
    for g, lp in candidates:
        if not g:
            continue
        k = tuple(map(tuple, g))
        agg.setdefault(k, [0, 0.0, g])
        agg[k][0] += 1
        agg[k][1] += lp
    if not agg:
        return [[0]], [[0]]
    ranked = sorted(agg.values(), key=lambda v: (v[0], v[1] / v[0]), reverse=True)
    return ranked[0][2], (ranked[1][2] if len(ranked) > 1 else ranked[0][2])
"""),

md(r"""
## 5. Test-time training and the serving bridge

TTT fits a fresh rank-16 LoRA adapter per task with a leave-one-out objective and a random
augmentation each step. Capping the few-shot context to ~1,600 tokens keeps a step at ~2s. Training
is the easy part; the hard part is generation. TTT must train in PyTorch (where decode is the slow
13 tok/s, intermittently quadratic after a step), while the fast engine does not train.

The bridge: train in PyTorch, save each adapter, convert it to a GGUF adapter, and serve it on
llama-server, which hot-swaps adapters per request via its `/lora-adapters` endpoint (~115 tok/s
with an adapter applied). Run as three phases (train all, convert all, then serve and generate) so
the trainer and the engine never share VRAM. Large-grid tasks are too VRAM-bound to train and are
excluded.
"""),

code(r"""
# Per-task training loop (PyTorch side); each step: a random augmentation + leave-one-out target,
# loss on the output tokens only. A wall-clock break and a largest-pair filter keep big-grid tasks
# from exhausting 16 GB.
def test_time_train(peft_model, init, task, rng, steps=20, lr=1e-4, time_budget=150, max_ctx=1600):
    reset_adapter(peft_model, init)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    t0 = time.time()
    for _ in range(steps):
        ids, labels = leave_one_out_example(task, rng, max_ctx)   # masked to output tokens
        loss = peft_model(input_ids=ids, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        if time.time() - t0 > time_budget:
            break
    # then: convert_lora_to_gguf.py adapter --base <model> ; llama-server --lora ... ;
    # POST /lora-adapters [{"id": j, "scale": s}] to apply this task's adapter at generation time.
"""),

md(r"""
## 6. Results

On the 120-task public evaluation set:

- **no-TTT AIRV baseline: 12.79%** (22/172 test outputs), one stream, ~78 min.
- **Regularized TTT (20 steps, rank 16, dropout 0.1) on the 84 trainable tasks: 16.8%** vs 12.8%
  no-TTT on the same tasks (+5 outputs; paired 6:1, McNemar p~0.13, consistent across a 15-task
  sweep and the 84-task run).
- **Adapter scale matters**: accuracy rises monotonically with the LoRA application scale (12.0% at
  0.5 up to 17.6% at 1.25 on the 84 tasks).
- **More votes help**: adding 8 color-permuted augmentations lifts the no-TTT baseline from 12.8 to
  14.4% on the 84 tasks.
- **Forcing TTT on the 36 big-grid tasks does not help** (10.6% vs 12.8% no-TTT): the VRAM
  exclusion is a genuine ceiling, so the blended system uses no-TTT there.

The naive configuration (40 steps, rank 32, no dropout) overfits to training loss 0.000 and ties
the baseline; regularization is what turns TTT into a lift on this compute-poor setup.
"""),

md(r"""
## 7. Reproducibility and availability

The full system is open source: the GGUF conversion recipe for the custom-vocabulary model, the
AIRV pipeline (`run_airv_llamacpp.py`), the per-task TTT bridge (`ttt_phase1/1b/2.py`), the sweep
and scale harnesses, and a development log recording every measurement and dead end.

This notebook documents the approach and reports measured numbers. Full end-to-end execution needs
the rig environment (a CUDA llama.cpp build serving the Q8_0 GGUF model, plus PyTorch + PEFT for
TTT). Porting to the Kaggle runtime (Linux) is a follow-up: build llama.cpp for Linux, upload the
GGUF model and adapters as a Kaggle dataset, and run the three phases within the notebook time
limit. The accuracy-per-GPU-hour framing and the map of walls (the decode ceiling, the VRAM limit
on large-grid TTT, the overfitting of naive adapters) are the transferable contribution.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.x"}},
      "nbformat": 4, "nbformat_minor": 5}

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "paper"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "..", "paper", "arc_pipeline.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
# validate round-trip
json.load(open(out, encoding="utf-8"))
print("wrote", os.path.abspath(out), "cells:", len(cells))
