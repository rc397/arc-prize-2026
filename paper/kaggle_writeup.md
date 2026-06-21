# Kaggle Paper Track submission, paste-ready fields

Each section below maps to one field in the writeup form. Numbers match DEVLOG.md.

---

## Title (63 / 80)

Test-Time Training on a Budget: ARC-AGI-2 on a Single Consumer GPU

## Subtitle, one sentence (128 / 140)

How far the cluster-scale ARC-AGI-2 recipe (a 4B grid model, AIRV, and per-task test-time training) gets on one 16 GB consumer GPU.

## Card / thumbnail image

560 x 280 (2:1). Suggested content: the title plus the headline jump (no-TTT 12.79 percent to a
blended 16.28 percent), or a simple pipeline strip: model -> llama.cpp bridge -> AIRV + TTT -> vote.

## Attachments / project links

- Code (GitHub): https://github.com/rc397/arc-prize-2026
- Public notebook: notebooks/arc_pipeline.ipynb (in the repo; documents and reproduces the pipeline)

---

## Project Description (paste into the description field)

### Summary

The strongest published ARC-AGI-2 results come from multi-GPU clusters. We ask the opposite
question: how far does the 2025 winning recipe, a small grid language model with
Augment-Infer-Reverse-Vote (AIRV) selection and per-task test-time training (TTT), go on a single
16 GB consumer GPU (RTX 5070 Ti, Blackwell, native Windows), and where exactly does it wall out?
Building on the released NVARC Qwen3-4B grid model, we report a reproducible offline system: a
12.79 percent no-TTT baseline on the 120-task public evaluation set, a regularized, scale-tuned TTT
that reaches 17.6 percent on the 84 trainable tasks (vs 12.8 percent no-TTT), and a best blended
16.28 percent across the full 120. The contribution is an honest accounting of what each stage
costs on one card and the engineering needed to make per-task TTT serve at all.

### The constraint and the model

A single RTX 5070 Ti, 16 GB, native Windows, no cluster and no Linux toolchain. We build on the
NVARC `qwen3_4b_grids15_sft139` checkpoint. It carries a 16-token vocabulary: digits 0-9 are the
ten grid colors, plus a newline, two role markers, and three specials. A grid is serialized one
token per cell and a color value equals its token id, so all prompts are assembled at the token-id
level as ChatML few-shot sequences.

### The engine is the bottleneck

Our most consequential finding concerns throughput, not model quality. On the standard Hugging Face
eager path, prefill is healthy (about 3,400 tokens per second) but decoding runs at only about 13
tokens per second, a tenth of what the card's bandwidth allows, because of per-token framework
overhead with no CUDA graphs. The usual fix, compiling the decode step, is unavailable: torch
compile needs Triton, which has no native-Windows backend. The fix is to change engines. We convert
the model to GGUF and serve it with llama.cpp (two small converter patches handle the custom
vocabulary). Measured decode rises to 170 tokens per second at Q8_0, a 13x speedup, near-lossless,
3.6 GB. On one card, the inference engine dominated every other knob we touched.

### AIRV and the test-time training bridge

The no-TTT pipeline generates one prediction per test input under each of the eight dihedral
augmentations, inverts each back to the original frame, and votes to two attempts. It scores 12.79
percent on the public eval. TTT then fits a fresh rank-16 LoRA adapter per task with a leave-one-out
objective. Training is the easy part once the few-shot context is capped to about 1,600 tokens; the
hard part is generation, because TTT must train in PyTorch (slow 13 tokens per second, intermittently
quadratic after a step) while the fast engine does not train. We bridge them: train in PyTorch, save
each adapter, convert it to a GGUF adapter, and serve it on llama.cpp, which hot-swaps adapters per
request through its lora-adapters endpoint (about 115 tokens per second with an adapter applied). The
three phases (train all, convert all, then serve and generate) keep the trainer and the engine from
sharing VRAM. Large-grid tasks are too VRAM-bound to train and are excluded.

### Results

- No-TTT AIRV baseline: 12.79 percent (22 of 172 test outputs), one stream, about 78 minutes.
- Regularized TTT (20 steps, rank 16, dropout 0.1) on the 84 trainable tasks: 16.8 percent at unit
  adapter scale vs 12.8 percent no-TTT on the same tasks. The lift is consistent across a 15-task
  sweep and the 84-task run (paired 6 to 1, exact McNemar p about 0.13).
- The naive configuration (40 steps, rank 32, no dropout) overfits to training loss 0.000 and ties
  the baseline. Regularization is what turns TTT into a lift on this compute-poor setup.
- Adapter scale matters: accuracy climbs with the LoRA application scale to a peak of 17.6 percent
  at scale 1.25 on the 84 tasks (12.0 at 0.5), then declines (16.0 at 1.4, 16.8 at 1.5).
- More votes help the baseline but not TTT: adding eight color-permuted augmentations lifts no-TTT
  from 12.8 to 14.4 percent, but TTT stays at 17.6 percent either way, because the adapter trains
  with random recoloring and is already color-robust.
- Forcing TTT onto the 36 big-grid tasks does not help (10.6 vs 12.8 percent no-TTT): the VRAM
  exclusion is a genuine ceiling, so the blend keeps no-TTT there.
- Best blended full-120: 16.28 percent (TTT at scale 1.25 where trainable, no-TTT elsewhere), up
  from the 12.79 percent baseline.

### Why this is worth reporting

Two lessons generalize beyond our particular card. First, on a single GPU the inference engine is
the dominant design choice; quantization and sampling mattered far less than the 13x we recovered by
moving decode off the eager path. Second, the real obstacle for TTT on one card is the serving
bridge: training needs a gradient framework, fast generation needs an optimized engine, and per-task
TTT produces a fresh adapter the engine must serve. We show that bridge is buildable on one card,
and that once it exists the open question is not plumbing but regularization, because the small
per-task training set overfits easily. The right framing for compute-poor ARC work is accuracy per
GPU-hour, and the map of walls reported here (the decode ceiling, the VRAM limit on large-grid TTT,
the overfitting of naive adapters) tells a practitioner with similar hardware which problems to
solve first.
