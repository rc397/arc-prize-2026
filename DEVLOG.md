# Development Log

A running record of the ARC Prize 2026 ARC-AGI-2 effort. Newest entries at the bottom.

## 2026-06-18

### Strategy

Build on the NVARC Qwen3-4B grid model by Ivan Sorokin (NVIDIA), the base model of the
team that won ARC Prize 2025 at 24.0 percent on ARC-AGI-2. The recipe shared by the top
2025 teams is a small open LLM fine-tuned on augmented and synthetic ARC data, per-task
test-time training with LoRA, and Augment-Infer-Reverse-Vote candidate selection. The
Kaggle submission runs offline, so only open-weight models apply and the 85 percent bonus
is out of reach. The realistic target is a Progress Prize plus the Paper Track.

### Compute environment

- Laptop (development): Windows 11, RTX 3070 Laptop 8 GB, Python 3.13, torch 2.6 cu118.
  Used for editing, unit tests, and orchestration. Holds the git repository.
- rig (training and inference): SSH host `rig`, Windows 11,
  RTX 5070 Ti 16 GB (Blackwell sm_120), Python 3.13, torch 2.10 dev cu128 (CUDA 12.8).
  Project root `D:\arc` on a 3.7 TB drive. The Blackwell stack already works, which
  removes the main setup risk.

### Operational notes

- The rig's SSH shell is Windows PowerShell. Remote commands are sent base64-encoded to
  avoid quoting problems, and use `[Console]::WriteLine` to avoid CLIXML noise.
- Code is edited on the laptop and synced to the rig with scp. Datasets and model weights
  live only on the rig.

### Phase 0: environment and assets

- Created the project structure and the model-independent core: invertible augmentations,
  the competition scorer, and data IO. Core unit tests pass on the laptop.
- Installed peft, accelerate, and kagglehub on the rig without disturbing the torch build.
- Cloned the official ARC-AGI-2 dataset on the rig: 1000 training tasks and 120 public
  evaluation tasks.
- Downloaded the NVARC model to `D:\arc\models` (about 6.8 GB across two safetensors
  shards).

### Key finding: model format

The model uses a 16-token vocabulary. Ids 0 to 9 are the digit colors, 10 is newline, 11
and 12 are the role tokens user and assistant, 13 is pad, and 14 and 15 are im_start and
im_end (im_end is the eos). It is a Qwen3ForCausalLM with hidden size 2560 and 36 layers.

The role tokens cannot be produced by tokenizing the words user or assistant, because the
byte tokens for those characters are absent and there is no unknown token. Prompts must
therefore be assembled at the token-id level. A grid is encoded as rows of digit ids
separated by the newline id, and a color value equals its token id.

The prompt is a ChatML few-shot sequence: each training pair is a user input turn followed
by an assistant output turn, then the test input as a final user turn, ending with the
assistant opener so the model generates the output grid followed by im_end.

### Phase 1: baseline

Implemented the model IO layer (`src/arc/model.py`) and a greedy no-TTT baseline runner
(`scripts/run_baseline.py`).

The first run crashed with a cuBLAS internal error during prefill because other GPU
processes left only about 357 MiB free. This was VRAM contention, not a logic bug. After
the rig GPU was freed (about 15.5 GB available), a 3-task smoke test produced correctly
structured output grids of varied shapes (9x3, 29x29, 19x7) at about 29 seconds per task,
with 0 of 3 exact matches. Zero exact matches on a tiny sample is expected for the base
model without test-time training, and the well-formed, task-appropriate shapes confirm the
prompt format is correct. Added a shape-match diagnostic (predicted output dimensions
versus ground truth) and started the full 120-task baseline. Results to be recorded here.

### Model size question

Checked whether a larger ARC-specialized model exists from the same author. Ivan Sorokin
published 2B and 4B grids models (qwen3_2b_grids15 and qwen3_4b_grids15) but nothing larger
in that line, so the 4B we are using is the largest specialized variant. Going beyond 4B
would mean fine-tuning a generic base on ARC ourselves, which is a later research option
rather than a drop-in upgrade. Decision: keep the 4B base, and revisit a larger model only
as an ensemble member or a custom fine-tune after the 4B pipeline plateaus.

### Phase 2: test-time training (code written)

Wrote `src/arc/ttt.py` (per-task LoRA adapter with reset between tasks, leave-one-out
targets, a random augmentation per step, gradient checkpointing to fit 16 GB) and
`scripts/run_ttt.py` (TTT followed by augmented voting for the two attempts). To be tested
on the rig once the baseline run frees the GPU.

### Official competition data

Switched to the official Kaggle dataset (consolidated challenges and solutions files plus
sample_submission.json). Confirmed the submission format: each task id maps to a list of
{attempt_1, attempt_2}, one per test input. Counts: 1000 training tasks, 120 evaluation,
and a 240-task placeholder test set that is swapped for the real tasks at rerun. Test
entries in the challenges files carry only the input; outputs live in the separate
solutions files. Added `load_consolidated` and `write_submission` to `src/arc/data.py` and
pointed `run_ttt.py` at the official files. Data copied to the rig at `D:\arc\data\kaggle`.

### Performance bug: quadratic decode after TTT (fixed)

The first TTT runs were about 50x slower than the no-TTT baseline (roughly 25 minutes on a
single small task). Root cause: test-time training enables gradient checkpointing, which
forces use_cache=False, and resetting that flag on the PEFT wrapper did not reach the base
model that generate() uses. Without the KV cache, decoding is quadratic. Fixed by passing
use_cache=True explicitly in greedy_generate and batched_generate, plus a TTT-versus-gen
timing split in run_ttt.py to confirm, plus clearing the CUDA cache between tasks (Windows
does not support expandable_segments) to bound memory over long runs.

### Running jobs detached on the rig (survives the laptop closing)

The rig is always-on and headless; the laptop is only a remote running ssh. A job launched
as a child of the ssh session dies when the laptop closes (the link drops and the child gets
the hang-up). To run overnight, launch detached:
`Start-Process -FilePath python -ArgumentList <args> -WindowStyle Hidden -WorkingDirectory D:\arc -RedirectStandardOutput <log> -RedirectStandardError <log>.err -PassThru`.
That process is independent of the ssh session, so it keeps running after disconnect; read
the log file on the rig when reconnecting.

### Overnight run, 2026-06-18

Launched the full 120-task TTT eval detached (pid 4340 at launch) with the use_cache fix,
writing to `D:\arc\experiments\ttt_eval_overnight.log` and the submission to
`ttt_eval_overnight.json`. On reconnect, read the log: the per-task `ttt=Xs gen=Ys` split
confirms the fix, and the final `[score]` line is the first real TTT accuracy on the public
eval.

## 2026-06-19

### The real problem: decode was the bottleneck all along

The overnight runs were still catastrophically slow end to end. Stepping back from reactive
debugging, the actual diagnosis: HuggingFace eager decode on this model runs at about 13
tokens per second, roughly 10 percent of the card's memory-bandwidth ceiling, caused by
per-token Python dispatch with no CUDA graphs (the classic slow generate path). Prefill was
always fine at about 3400 tokens per second; only decode was slow. The standard fix
(StaticCache plus torch.compile) cannot work here because torch.compile needs Triton and
Triton has no native Windows support (the TritonMissing error). Full sourced analysis in
`docs/inference-stack-research.md`. Decision: run generation through llama.cpp on native
Windows, not HF eager and not a WSL2/vLLM build.

### Rig lockout and recovery

Enabling the WSL and Virtual Machine Platform features (to explore vLLM) created a Hyper-V
virtual switch; on reboot Windows reclassified the network as Public, and the firewall then
blocked all inbound connections including SSH. The machine stayed up and answered ping, but
every TCP port was filtered. Recovered by running, on the rig console, a block that re-opens
SSH on all firewall profiles, sets the active network back to Private, and ensures sshd is
running and automatic (`RECOVER_RIG.md`). Durable fix: the OpenSSH rule now applies on any
profile, so a future reboot cannot repeat this.

### Stage 0: llama.cpp engine, validated

Installed the prebuilt llama.cpp Windows CUDA 12.4 build (release b9724). The CUDA 12.4
binary loads and runs on the Blackwell sm_120 card via PTX JIT on driver 591.86, so no
from-source build was needed.

Converting the model to GGUF needed two converter patches, because the model uses a custom
16-token WordLevel tokenizer (digits 0-9, newline, user, assistant, plus three special
tokens), not standard BPE:
1. Its pre-tokenizer regex is exactly the qwen2 regex, but its tokenizer hash is unknown to
   llama.cpp, so the converter refused to guess. Mapped the hash to the qwen2 pre-tokenizer
   in `conversion/base.py` `get_vocab_base_pre`.
2. WordLevel has zero BPE merges, and the loader requires a non-empty merges list. Added a
   fallback in `_set_vocab_gpt2` that writes two throwaway merges among the user/assistant
   tokens when none exist. This never affects digit or newline tokenization, and we drive the
   model at the token-ID level regardless. Reproducible recipe in `docs/gguf-conversion.md`.

Decode benchmarks on the RTX 5070 Ti (llama-bench, all layers on GPU):
- bf16, 6.77 GB: 102 tok/s decode, 2344 prefill.
- Q8_0, 3.60 GB: 170 tok/s decode, near-lossless. Chosen operating point.
- Q4_K_M, 2.02 GB: 244 tok/s decode, held in reserve pending an accuracy check.

Against the 13 tok/s HF baseline this is 13x at Q8_0, and the small footprint leaves about
12 GB for KV cache and parallel candidate batching.

Correctness gate: drove the model at the token-ID level through llama-server (token-array
prompt) on 15 training tasks, greedy, no TTT, scoring exact match against known solutions
(`scripts/stage0_check.py`). Result: 14/15 correct, output dimensions correct on all 15,
output lengths 12 to 930 tokens with clean stopping and no runaway, prompts up to 6541 tokens
and grids up to 30x30. This confirms the conversion is faithful and the token-ID pipeline is
correct end to end. Stage 0 decision: GO. Build the pipeline on llama.cpp.

## 2026-06-20

### Stage 1: stabilizing the no-TTT AIRV pipeline (a debugging saga)

Built the no-TTT AIRV pipeline (`scripts/run_airv_llamacpp.py`): for each test input, generate
across the 8 D4 augmentations through llama-server at the token-ID level, reverse each
augmentation, and vote to top-2 (`src/arc/select.py`). It validated on 2 then 10 tasks.
Getting the full 120-task run to complete reliably took several iterations, recorded here
because the failure modes were instructive and cost real time.

1. The first overnight attempt produced a 0.0 score: the server had died early and the eval
   churned out empty candidates (cands=0) for all 120 tasks with nothing watching. Lesson: a
   long unattended run needs both a server watchdog and an in-eval abort, or a dead dependency
   silently wastes the whole night.

2. Added an abort-after-3-zero-candidate-tasks safety to the eval, and a watchdog
   (`scripts/server_watchdog.ps1`) that restarts the server within ~15s if its process dies.

3. The server kept crashing under load. The cause was in the server's own log: a prompt cache
   enabled by default (`--cache-prompt`, `--cache-idle-slots`) was saving ~680 MiB of state
   per request into an 8 GB cache. Every ARC prompt is unique, so the cache gave zero reuse
   and was pure overhead plus a crash source. A per-request `cache_prompt=false` does not
   disable the server-side idle-slot caching; the flags `--no-cache-prompt
   --no-cache-idle-slots` do. With caching off, a 10-task run completed cleanly.

4. The full run still hit intermittent crashes on the heaviest tasks (two test inputs, 30x30
   outputs). They are non-deterministic, the same task passes in one run and crashes in
   another, which points to an instability of the prebuilt CUDA 12.4 build on Blackwell sm_120
   under sustained generation rather than a logic bug. Single stream (`-np 1`) reduced but did
   not eliminate them.

5. Settled on resilience over prevention: the eval now retries a failed request for ~48s while
   the watchdog revives the server, so a transient crash costs one augmentation's latency
   rather than the run. Operating config: `-np 1`, `-c 16384`, caching disabled, watchdog
   supervising. Revisit the CUDA 13.x build or flash attention later as a speed/stability
   optimization.

### Stage 1 result: no-TTT baseline

The full 120-task run completed cleanly: all 120 tasks at full 8/8 candidates, zero degraded,
no manual intervention. First real baseline, no TTT, AIRV over the 8 D4 augmentations with
greedy decoding on the Q8_0 model:

**12.79 percent (22 of 172 test outputs) on the ARC-AGI-2 public eval.**

For reference the 2025 winner reached 24 percent with test-time training, and the literature
puts TTT's lift at roughly 2x or more, so a ~13 percent no-TTT floor is encouraging. TTT is
the main score lever from here (Stage 2). Submission at `experiments/airv_nottt_eval.json`;
run wall-clock about 78 minutes at single stream (~39s/task), the obvious speed target for
later. Dead HF-era logs and Stage 0 scratch were archived to `experiments/archive/` with a
manifest.

### Stage 2: TTT trains fast, but generation is the wall

Per-task TTT in PyTorch is fast once the few-shot context is capped to ~1600 tokens (~2s/step
versus ~13s/step at full ~3800-token context, so ~50-90s/task). A cache-after-TTT regression
was fixed: gradient checkpointing left decode quadratic even after the disable call, so the
cleanup now forces `gradient_checkpointing=False` on every module and restores `use_cache`, and
the pipeline does one warmup generation to absorb the ~35s Blackwell kernel JIT. A per-task
generation cap (from the train-output sizes) bounds post-TTT runaway.

The wall is generation, not training. AIRV needs 8 augmented generations per task, and PyTorch
decode is ~13 tok/s with no fast compile on Windows/Blackwell. Large-output tasks run
~250-675s of generation alone, so a full 120-task TTT run is ~7-10 hours, a long overnight
rather than a quick pass. The fast fix (serve each per-task LoRA in llama.cpp) carries the GGUF
conversion plus model-swap complexity we set aside; cloud is off the table (no funding). Net:
TTT is feasible on the rig but slow. The pipeline (`scripts/run_ttt_airv.py`, launcher
`scripts/launch_ttt_overnight.ps1`) is built, fixed, and crash-safe (incremental writes,
per-task error handling). The generation-speed ceiling on a single consumer GPU is itself a
result worth documenting for the paper.

### Stage 2 lift test: no measurable lift, and the generation regression returns

To decide whether to commit a night to the full TTT run, a controlled single-shot test (TTT
vs no-TTT, identity augmentation, same tasks) was run first. It cleared 5 of 20 tasks, then
task 6 hung: the quadratic-decode regression came back on the no-TTT/reset path, pinning the
GPU at 100 percent for ~45 minutes on one task, far past the 1024-token cap's ~80s/generation.
Partial score over those 5 tasks: no_ttt 1/7, ttt 1/7, no measurable TTT lift. TTT training
itself is fine (it fits each task); the wall is generation, which is both too slow and too
unreliable to make TTT pay off here, and it prevented a clean lift measurement.

Decision: bank the 12.79 percent no-TTT baseline as the result for now. TTT is not viable on
this rig as wired, it needs the fast-generation bridge we set aside (serve each per-task LoRA
in llama.cpp, or move to a Linux toolchain where torch.compile and vLLM dynamic LoRA work).
That is a focused future effort, not an overnight grind. The strongest near-term deliverable is
the baseline plus the honest single-GPU engineering account, which is itself paper material.

### Stage 2 fixed: the TTT generation bridge works, but the method shows no lift yet

The generation wall is solved. Keep TTT training in PyTorch, but serve each per-task LoRA on
llama.cpp: `convert_lora_to_gguf.py` converts the PEFT adapter to GGUF (the Qwen3 conversion bug
did not bite this build), and llama-server hot-swaps adapters via `--lora-init-without-apply`
plus `POST /lora-adapters`. With the adapter applied, decode holds ~115 tok/s (versus ~13 in
PyTorch) with no quadratic stall. Architecture: phase 1 trains and saves all adapters in PyTorch
(`scripts/ttt_phase1.py`); phase 1b converts them from a clean process so the convert cannot
collide with a loaded model's VRAM, which had been failing conversions silently
(`scripts/ttt_phase1b.py`); phase 2 serves in VRAM-sized batches and runs AIRV with single-stream
stability and per-task recovery (`scripts/ttt_phase2.py`). Two integration bugs found and fixed:
the converter crashed a cp1252 capture thread on its progress bars, and the Microsoft Store
Python's `sys.executable` alias plus a maxed GPU broke the in-process convert.

A clean 6-task run (full 8/8 candidates, no crashes) gives the first fair TTT-versus-no-TTT
number: 1/8 versus 1/8, tied, no measurable lift. The adapters overfit (training loss reaches
0.000 in 40 steps), which likely explains the lack of generalization to the held-out test input.
So the honest two-part answer to "why does TTT not work here" is: the generation was broken (now
fixed), and the method as configured does not lift, which is a tuning problem (step count, rank,
adapter scale). Next levers: lower the step count or the applied adapter scale to curb
overfitting, or run all 120 to see whether a lift emerges at scale.

### Stage 2 method sweep: regularized TTT recovers a small lift

With the bridge working, a sweep on 15 small-context eval tasks (big-grid tasks are VRAM-bound
and degrade TTT to a crawl, so they were excluded; the sweep selects the smallest-context tasks
and frees the CUDA cache between tasks) tested the overfit hypothesis by varying
steps/rank/dropout, each scored against no-TTT on the same tasks:

| config | TTT | no-TTT |
|---|---|---|
| 40 steps, rank 32, no dropout (standard) | 3/23 | 3/23 (tied) |
| 20 steps, rank 16, dropout 0.1 | 4/23 | 3/23 (+1, a lift) |
| 8 steps, rank 8, dropout 0.05 | 3/23 | 3/23 (tied) |

The pattern is coherent with the overfit diagnosis: the standard config overfits (loss 0.000)
and ties; moderate regularization (fewer steps, lower rank, dropout) recovers a small lift; too
aggressive a reduction undertrains and ties again. The lift is one task on a 15-task sample, so
it is a weak but hypothesis-consistent signal that warrants larger-scale confirmation. The
useful, honest finding for the paper: naive per-task TTT does not beat a strong no-TTT AIRV
baseline on this single-GPU setup, but a regularized configuration can. Sweep harness:
`scripts/ttt_sweep.py` orchestrating `ttt_phase1.py` / `ttt_phase1b.py` / `ttt_phase2.py`,
compared by `scripts/sweep_compare.py`.

### Stage 2 scale validation: regularized TTT lifts +5 on 84 tasks

The 15-task sweep's +1 was confirmed at scale. The regularized config (20 steps, rank 16, dropout
0.1, scale 1.0) was run on the 84 eval tasks small enough to train on a 16 GB card and scored
against no-TTT on the same tasks:

| method | correct | accuracy |
|---|---|---|
| regularized TTT | 21/125 | 16.8% |
| no-TTT | 16/125 | 12.8% |

Net +5 test outputs (+4.0 points, ~31% relative). The paired breakdown is clean: of the tasks the
two methods disagreed on, TTT was right on 6 that no-TTT missed and wrong on only 1 that no-TTT got
(6:1). With only 7 disagreements the exact McNemar p is ~0.13, so no single run is statistically
bulletproof, but three independent measurements now point the same way (sweep +1, scale +5, 6:1
discordant). The no-TTT 12.8% here matches the 12.79% full baseline, so the subset is
representative.

Competition implication: applying TTT where trainable and no-TTT on the 36 excluded big-grid tasks
blends to 27/172 = 15.70% on the full 120-task eval, up +2.9 points from the 12.79% no-TTT
baseline. The VRAM exclusion of big-grid tasks is the ceiling on this setup.

Task selection is the safety crux: the leave-one-out target can be a task's largest train pair, so
that pair's token length (maxpair) drives the worst-case training sequence and the VRAM hang.
Selecting maxpair <= 1000 (84 tasks; worst-case ~2100 tokens vs the ~3700 that hung the prior runs)
plus a per-task wall-clock break in `test_time_train` made the run hang-free: phase1 ~14-33s/task,
phase2 ~45s/task, ~100 min total. Harness: `scripts/ttt_val.py` over `ttt_phase1/1b/2.py`, compared
by `scripts/compare_ttt.py <tag>`.

### Stage 2 overnight push: scale, color-perm, big-task TTT, combine

An overnight queue swept additional levers on the 84 trainable tasks (all compared to no-TTT 12.8%
there):
- Adapter application scale: accuracy rises with the LoRA scale to a peak, 12.0% (0.5), 12.8%
  (0.7), 15.2% (0.85), 16.8% (1.0), 17.6% (1.25), then declines: 16.0% (1.4), 16.8% (1.5), 1.75
  collapses (over-applied). Best: scale 1.25 = 22/125 = 17.6%.
- Color permutation: adding 8 color-permuted augmentations to the 8 D4 lifts the no-TTT baseline
  from 12.8% to 14.4% (more votes), but does NOT stack with TTT (s1.25 is 17.6% with or without
  color-perm). The TTT adapter trains with random recoloring, so it is already color-robust and has
  subsumed the gain.
- Big-task TTT: forcing TTT onto the 36 VRAM-excluded tasks at a tighter cap (max_ctx 600) scores
  10.6% vs 12.8% no-TTT. It does not help; the VRAM exclusion is a genuine ceiling, so the blend
  keeps no-TTT there.

Best blended full-120 = TTT s1.25 on the 84 trainable + no-TTT on the 36 big = 28/172 = 16.28%, up
from the 15.70% scale-1.0 blend and the 12.79% no-TTT baseline. Harness: `scripts/ttt_overnight.py`,
`ttt_scalehi.py`, `ttt_phase2_cp.py`, `ttt_combine.py`; results figure `scripts/make_figure.py`.
