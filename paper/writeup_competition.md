# Test-Time Training on a Budget: ARC-AGI-2 on a Single Consumer GPU

## Abstract

The strongest published results on ARC-AGI-2 come from multi-GPU clusters. We ask how far the
2025 winning recipe, a small grid language model with per-task test-time training (TTT) and
Augment-Infer-Reverse-Vote (AIRV) selection, goes on a single 16 GB consumer GPU, and where it
walls out. Building on the released NVARC Qwen3-4B grid model, we report a reproducible offline
system with a 12.79 percent no-TTT baseline on the 120-task public evaluation set. We find that
the binding constraint is generation throughput, not model quality or memory, and that the single
largest lever is the inference engine, which buys a 13x decode speedup. We then show how to run
per-task TTT on this hardware by serving each task's adapter on the fast engine, and report that
naive TTT ties the no-TTT baseline while a regularized configuration recovers a small lift. The
contribution is an honest accounting of what the recipe costs and where the effort should go when
you cannot rent a cluster.

## 1. Introduction

ARC-AGI-2 is a hard abstraction benchmark, and the offline competition setting rules out frontier
API models. The 2025 ARC Prize was won by NVARC at 24.0 percent using a 4B grid model with
per-task TTT, and the broadly reproduced recipe across the top teams is the same: fine-tune a
small model on augmented grid data, train a fresh low-rank adapter on each test task, then sample
many augmented candidates and vote. The recipe is effective but expensive, and the public reports
were produced on clusters of H100 or GH200 GPUs.

Most researchers do not have that hardware. We characterize the recipe under a constraint we think
is widely shared: a single RTX 5070 Ti with 16 GB, on a Blackwell architecture, running native
Windows. The aim is not to top the leaderboard, which is unrealistic here, but to build a working
reproducible system, measure honest numbers, and map where one card stops being enough.

## 2. System

### 2.1 Model and prompting

We use the NVARC `qwen3_4b_grids15_sft139` model, a Qwen3-4B with a 16-token vocabulary: the ten
grid colors as digits, a newline, two role markers, and three specials. A grid is serialized one
token per cell, and a color value equals its token id. Because the role and special tokens cannot
be produced from text, we assemble all prompts at the token-id level as ChatML sequences.

### 2.2 The engine is the bottleneck

Our most consequential finding concerns throughput. On the standard Hugging Face eager path,
prefill is healthy at about 3,400 tokens per second, but decoding runs at only about 13 tokens per
second, roughly a tenth of what the card's memory bandwidth allows, because of per-token framework
overhead with no CUDA graphs. The usual remedy, compiling the decode step, is unavailable here:
`torch.compile` needs Triton, and Triton has no native-Windows backend.

The fix is to change engines. We convert the model to GGUF and serve it with llama.cpp. The custom
vocabulary needed two small converter patches, both documented in our repository: the tokenizer
hash is unknown to the loader but its pre-tokenizer is exactly the qwen2 regex, so we map it, and
the WordLevel tokenizer has no byte-pair merges while the loader requires a non-empty list, so we
add two throwaway merges among role tokens that never fire on grid text. The prebuilt CUDA 12.4
binary runs on the Blackwell card through PTX just-in-time compilation. Measured decode rises to
102 tokens per second in bf16, 170 in Q8_0, and 244 in Q4_K_M, against the 13 of the eager path.
We adopt Q8_0: a 13x speedup, near-lossless, and a 3.6 GB footprint. We verified faithfulness by
reproducing 14 of 15 training tasks exactly under greedy decoding. One stability note: the server
enables a prompt cache by default that saves about 680 MiB per unique prompt; since every ARC
prompt is unique, this yields no reuse and destabilizes sustained generation, so we disable it and
run a single supervised stream.

### 2.3 AIRV and TTT

Our no-TTT pipeline generates one prediction per test input under each of the eight dihedral
augmentations, inverts each back to the original frame, and votes to two attempts, scored with the
exact competition metric. For TTT we fit a per-task LoRA with a leave-one-out target and a fresh
random augmentation each step. Capping the few-shot context to about 1,600 tokens drops the cost
of a training step from about 13 seconds to about 2, so training is no longer the expensive part.

The harder problem is serving the trained adapter. TTT must run in PyTorch, where decoding is the
slow 13 tokens per second and intermittently becomes quadratic after a training step. The fast
engine does not train. We bridge them: train in PyTorch, save each adapter, convert it to a GGUF
adapter, and serve it on llama.cpp, which hot-swaps adapters per request through its
`/lora-adapters` endpoint. With an adapter applied, decode holds about 115 tokens per second with
no stalls. We run this as three phases (train all, convert all from a clean process, then serve
and generate all) so PyTorch and llama.cpp never share VRAM. Large-grid tasks remain too
VRAM-bound to train and are excluded.

## 3. Results

### 3.1 No-TTT baseline

The AIRV pipeline without TTT scores **12.79 percent** (22 of 172 test outputs) on the 120-task
public evaluation set, a clean run completing in about 78 minutes at a single stream. For
reference the 2025 winner reached 24.0 percent with TTT.

### 3.2 The engine speedup

The 13x decode improvement from changing engines is the single most important result for this
hardware class. It turned an overnight, frequently-failing process into a sub-90-minute
evaluation. On one consumer GPU, the inference engine dominated every other knob we touched.

### 3.3 Test-time training

With generation fixed, we compared TTT against no-TTT on a fixed set of 15 small-context tasks,
varying the adapter configuration:

| configuration | TTT | no-TTT |
|---|---|---|
| 40 steps, rank 32, no dropout | 3/23 | 3/23 |
| 20 steps, rank 16, dropout 0.1 | 4/23 | 3/23 |
| 8 steps, rank 8, dropout 0.05 | 3/23 | 3/23 |

The standard configuration overfits, its training loss reaches 0.000, and ties the no-TTT
baseline. A regularized configuration with fewer steps, lower rank, and dropout recovers a small
lift, and an over-aggressive reduction undertrains and ties again. The lift is one task on a
15-task sample, so it is modest and needs larger-scale confirmation, but the trend is coherent
with the overfit diagnosis rather than random. The honest summary: naive per-task TTT does not
beat a strong no-TTT AIRV baseline on this setup, while a regularized TTT can.

## 4. Discussion

Two lessons generalize beyond our particular card. First, on a single GPU the inference engine is
the dominant design choice; quantization and sampling mattered far less than the 13x we recovered
by moving decode off the eager path. Second, the real obstacle for TTT is the serving bridge:
training needs a gradient framework, fast generation needs an optimized engine, and per-task TTT
produces a fresh adapter the engine must serve. We show this bridge is buildable on one card by
converting each adapter to GGUF and hot-swapping it, and that once it exists the open question is
not plumbing but regularization, because the small per-task training set overfits easily.

We argue the right framing for compute-poor ARC work is accuracy per GPU-hour, and the map of
walls reported here, the decode ceiling, the VRAM limit on large-grid TTT, and the overfitting of
naive adapters, is a first step toward it. The findings tell a practitioner with similar hardware
exactly which problems to solve first.

## 5. Availability

The full system is open source: the GGUF conversion recipe for the custom-vocabulary model, the
AIRV pipeline, the per-task TTT bridge and sweep harness, and a development log recording every
measurement and dead end behind these numbers.
