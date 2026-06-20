# Test-Time Training on a Budget: An Honest Account of ARC-AGI-2 on a Single Consumer GPU

## Abstract

The strongest published results on ARC-AGI-2 come from teams running multi-GPU clusters. We ask
the opposite question: how far does the 2025 winning recipe, a small grid language model with
per-task test-time training (TTT) and Augment-Infer-Reverse-Vote (AIRV) selection, go on a single
16 GB consumer GPU, and where exactly does it wall out? Building on the released NVARC Qwen3-4B
grid model, we report a reproducible offline system and a measured no-TTT baseline of 12.79
percent on the 120-task public evaluation set. We find that on this hardware the binding
constraint is not model quality or memory but generation throughput: naive decoding runs at about
13 tokens per second, and the single largest lever we found, changing the inference engine,
recovers a 13x speedup. We also report a negative result. Per-task TTT trains cheaply once we cap
its context, but its generation half remains too slow and too unstable to produce a clean
evaluation pass on our setup, so we could not measure a TTT lift. We argue that this kind of
honest accounting, the cost of each stage and the order in which the walls appear, is itself a
useful contribution for the many researchers who cannot rent a cluster.

## 1. Introduction

ARC-AGI-2 is a deliberately hard test of abstraction and reasoning, and the offline competition
setting rules out frontier API models. The 2025 ARC Prize was won by NVARC at 24.0 percent using
a 4B grid language model with per-task test-time training, and the broadly reproduced recipe
across the top teams is the same: fine-tune a small model on heavily augmented grid data, train a
fresh low-rank adapter on each test task, then sample many augmented candidates and vote. That
recipe is effective, but it is also expensive, and the public reports were produced on clusters
of H100 or GH200 GPUs.

Most people interested in ARC do not have that hardware. We set out to characterize the recipe
under a hard constraint that we think is widely shared: a single consumer GPU, here an RTX 5070
Ti with 16 GB of memory, on a Blackwell architecture, running native Windows. Our goal was not to
beat the leaderboard, which on this hardware is not realistic, but to build a working,
reproducible system, measure an honest baseline, and map precisely where a single card stops
being enough. The constraint is the contribution: we report what each stage of the recipe costs,
which costs dominate, and which engineering choices matter most when compute is scarce.

## 2. Background

The winning recipe has three stages, and each is compute-hungry in a different way.

Fine-tuning produces a base model that emits grids token by token. We do not retrain it; we build
on the released NVARC checkpoint directly.

Test-time training fits a small LoRA adapter to a single task at inference time, using the task's
own training pairs as supervision, typically with a leave-one-out objective and a fresh random
augmentation at each step. This is where the published systems spend most of their budget: a fresh
adapter per task means a short training run for every one of the test problems.

AIRV selection generates several candidate solutions per task by transforming the task under
label-preserving symmetries (the eight dihedral operations on a grid, optionally with color
permutations), generating a prediction for each, inverting the transform, and voting. The cost
here is multiplicative: every augmentation is a full generation, so the selection stage scales the
decoding bill by the number of augmentations.

On a cluster these costs are absorbed by parallelism. On one card they appear in series, and as we
show next, the decoding cost is the one that hurts.

## 3. System and methods

### 3.1 Model and prompt format

We use the NVARC `qwen3_4b_grids15_sft139` model, a Qwen3-4B trained for ARC grids. It carries a
tiny custom vocabulary of 16 tokens: the digits 0 through 9 as the ten grid colors, a newline, two
role markers (user and assistant), and three special tokens. A grid is therefore serialized with
one token per cell, and a color value equals its token id. Because the role and special tokens
cannot be produced by tokenizing ordinary text, we assemble all prompts at the token-id level as
ChatML sequences rather than as strings. This detail matters for correctness and is the kind of
thing that is easy to get subtly wrong.

### 3.2 The engine is the bottleneck

Our first and most consequential finding concerns inference throughput. Running the model through
the standard Hugging Face eager path, prefill is healthy at about 3,400 tokens per second, but
decoding runs at only about 13 tokens per second, roughly a tenth of what the card's memory
bandwidth should allow. The cause is per-token framework overhead with no CUDA graphs, the classic
slow-generate problem. The standard remedy, compiling the decode step, is unavailable to us:
`torch.compile` depends on Triton, and Triton has no native Windows backend, so the compile path
fails outright on this platform.

The fix that worked was to change engines. We convert the model to GGUF and serve it with
llama.cpp. The custom vocabulary required two small patches to the converter, both documented in
our repository. First, the model's tokenizer is recognized by its hash, and ours was unknown; its
pre-tokenizer regex is byte-for-byte the qwen2 regex, so we map the hash to the qwen2
pre-tokenizer. Second, the tokenizer is WordLevel and therefore has no byte-pair merges, but the
llama.cpp loader requires a non-empty merge list, so we write two throwaway merges among the role
tokens. These never affect digit or newline tokenization, and since we drive the model at the
token-id level they are never exercised. The prebuilt CUDA 12.4 binary runs on the Blackwell card
through PTX just-in-time compilation, so no from-source build was needed.

The payoff is large. Measured decode throughput rises to 102 tokens per second in bf16, 170 in
Q8_0, and 244 in Q4_K_M, against the 13 of the eager path. We adopt Q8_0 as the operating point: a
13x decode speedup, near-lossless quantization, and a 3.6 GB footprint that leaves room for the
key-value cache. We verified that the converted model is faithful by reproducing 14 of 15 training
tasks exactly under greedy decoding, which confirms that the conversion and the token-id pipeline
agree with the original.

One stability caveat is worth recording for anyone repeating this. The llama.cpp server enables a
prompt cache by default, which saves roughly 680 MiB of state per unique prompt. Because every ARC
prompt is unique, this cache yields no reuse and instead grows until the server becomes unstable
under sustained generation. Disabling it (the `--no-cache-prompt` and `--no-cache-idle-slots`
flags), running a single stream, and supervising the server so it restarts on the rare crash made
the pipeline robust enough to run unattended.

### 3.3 AIRV pipeline

Our no-TTT pipeline generates, for each test input, one prediction under each of the eight
dihedral augmentations, inverts each prediction back to the original frame, and votes to two
attempts. Scoring uses the exact competition metric: two attempts per test output, a point if
either matches the ground truth exactly, averaged over all test outputs.

### 3.4 Capped test-time training

Our TTT implementation fits a rank-32 LoRA per task with a leave-one-out target and a fresh random
augmentation each step. The naive version is slow for a non-obvious reason: a leave-one-out prompt
that packs in several large training grids can run past 3,800 tokens, and at that length a single
training step costs about 13 seconds under eager training. Capping the few-shot context to roughly
1,600 tokens, keeping at least one example, brings the step cost down to about 2 seconds with no
change to the rest of the recipe. After this fix, training is no longer the expensive part.

## 4. Results

### 4.1 No-TTT baseline

Our AIRV pipeline without TTT scores **12.79 percent** (22 of 172 test outputs) on the 120-task
public evaluation set. The run was clean: every task produced a full set of eight candidates, and
the whole evaluation completed in about 78 minutes at a single stream. For reference, the 2025
winner reached 24.0 percent with TTT, and the TTT literature reports lifts of roughly two times or
more, so a 13 percent no-TTT floor is a reasonable platform.

### 4.2 The engine speedup

The 13x decode improvement from the engine change is, in our experience, the single most important
result for this hardware class. It is what turned an overnight, frequently-failing process into a
sub-90-minute evaluation. On a single consumer GPU, the choice of inference engine dominated every
other tuning knob we touched.

### 4.3 Test-time training: a negative result

We were unable to obtain a clean TTT score on this rig, and we report that honestly rather than
omit it. TTT training itself works and is cheap after the context cap. The wall is generation. The
fast llama.cpp engine does not train, so TTT must run in PyTorch, where decoding sits back at about
13 tokens per second. Worse, after a TTT step the key-value cache is intermittently not used,
which makes decoding quadratic in the prompt length, so a single large-output task can pin the GPU
for tens of minutes. A full TTT plus AIRV pass over 120 tasks projects to roughly 7 to 10 hours
under these conditions, and the instability makes even that estimate optimistic.

To decide whether such a run was worth committing, we ran a controlled single-shot probe comparing
TTT against no-TTT on the same tasks. Over the five tasks it completed before the quadratic-decode
regression stalled it, the two were tied at one correct out of seven test outputs each, so we
observed no measurable lift. Five tasks is far too small to conclude anything about TTT in general,
and we draw no such conclusion. What we can say is specific and useful: on this hardware, as
wired, we could not make TTT pay off, because its generation half is both too slow and too
unreliable to evaluate.

## 5. Discussion

Two lessons generalize beyond our particular card.

First, on a single GPU the inference engine is the dominant design choice. Model selection,
quantization, and sampling parameters all mattered less than the 13x we recovered by moving decode
off the eager path. Anyone reproducing this recipe under tight compute should treat engine
throughput as the first-order concern, not an afterthought.

Second, the real unsolved problem for TTT on one card is what we call the serving bridge. Training
must happen in a framework that computes gradients; fast generation must happen in an optimized
engine; and per-task TTT produces a fresh adapter that the engine must then serve. Crossing that
bridge for every task, while sharing one 16 GB card between the trainer and the engine, is the
crux. On a Linux toolchain this bridge largely dissolves: `torch.compile` makes training and
generation fast, and vLLM serves per-task LoRA adapters dynamically. On our native-Windows
Blackwell setup, neither is available, and that absence, not the model and not the memory, is what
parks TTT.

We think the right framing for compute-poor ARC work is accuracy per GPU-hour rather than accuracy
alone, and we view the map of walls reported here as a first step toward that framing. The honest
negative result is part of the contribution: it tells a practitioner with similar hardware exactly
which problem to solve first.

## 6. Conclusion

We built a working, reproducible, fully offline ARC-AGI-2 system on a single 16 GB consumer GPU,
measured a 12.79 percent no-TTT baseline, and identified generation throughput as the binding
constraint, recovering a 13x decode speedup by changing the inference engine. We were candid about
where the recipe stops: per-task test-time training is cheap to train but, on this hardware, too
slow and unstable to generate from, so we could not measure its lift. The next lever is the
training-to-serving bridge, either by serving each per-task adapter in the fast engine or by moving
to a toolchain where compilation and dynamic adapter serving are available. We hope the cost
accounting here is useful to others working under the same constraint.

## 7. Availability

The full system is open source, including the GGUF conversion recipe for the custom-vocabulary
model, the AIRV pipeline, the capped TTT implementation, and a development log that records every
measurement and dead end behind the numbers above.

---

*Notes for trimming to the Paper Track word limit: sections 2 and 3.2 carry the most detail and can
be compressed first; the abstract, section 4, and section 5 should be preserved as the core
contribution.*
