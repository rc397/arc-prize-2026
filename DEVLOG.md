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
  (0.7), 15.2% (0.85), 16.8% (1.0), 17.6% (1.25), then settles onto a plateau just below the peak:
  16.0% (1.4), 16.8% (1.5), 16.8% (1.75). A real optimum at 1.25, but over-scaling degrades gently
  rather than collapsing. Best: scale 1.25 = 22/125 = 17.6%.
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

### Paper Track report figures: six measured-data figures

Extended `scripts/make_visuals.py` with six report figures (PNG, 200 dpi, the shared Times/blue
style), each backing one claim, every number cross-checked to the runs and to `DEVLOG.md`:
- fig_roofline: decode throughput against the 896 GB/s memory roof. Confirmed the GGUF weight sizes
  first (bf16 6.77, Q8_0 3.60, Q4_K_M 2.02 GB); eager and llama.cpp bf16 run the same weights at the
  same 132 tok/s roof, so their ~8x gap is engine overhead, not the card. Measured fill: eager 10%,
  bf16 77%, Q8_0 68%, Q4_K_M 55%.
- fig_vram_wall: a 3D scatter of all 120 tasks (test height x width x largest train pair), trainable
  green / excluded red, with the 16 GB training budget as a plane at maxpair 1000. The split is
  clean because maxpair is the actual cutoff metric (trainable top out at 996 tokens, excluded start
  at 1010), not the test grid size.
- fig_scale_flip: a per-output heatmap of which of the 84-subset's 125 test outputs each adapter
  scale solves; the column counts reproduce the scale curve exactly (16/15/16/19/21/22/20/21/21),
  confirming the 1.25 peak and the plateau are real at the task level, not an aggregate artifact.
- fig_cost_frontier: accuracy vs GPU-minutes (log) on the one card, reconstructed from the run logs
  (no-TTT 78.5 min at 12.79%, TTT-blended 95.7 min at 16.28%, the eight scales stacked at one cost
  since the adapters train once). +3.5 points costs +17 minutes; an eager ghost marks where the same
  no-TTT run would land at 13 tok/s (~17 hours), the engine's leverage.
- fig_airv_votes and fig_phases are written and staged but need a GPU run (an instrumented no-TTT
  AIRV pass for per-transform vote agreement, and a live nvidia-smi trace of one task's
  train/convert/serve bridge); both are queued behind the running verifier.

Data dumps live in `paper/data/` (task_dims.csv, scale_flip.csv, timings.csv) and the figure code
reads them, so every number is auditable. Scripts: `dump_task_dims.py`, `dump_scale_flip.py`,
`parse_timings.py`, plus `dump_airv_votes.py`, `dump_vram_trace.py`, `b6_train_one.py` staged for the
two GPU figures.

## 2026-06-21 / 06-22

### v1 eligibility unblocked
The Kaggle prep notebook now stages the full llama.cpp `bin/` (llama-server plus all shared
libraries, including libggml-cuda.so) alongside the Q8_0 GGUF, so the offline server actually
starts on the submission side. Committed prep output is ~7.9 GB. v1 (no-TTT, ~12.79%) is down to
attaching the prep output + the ARC Prize 2026 competition and running the submission notebook.

### Licensing and provenance
Relicensed the repo to MIT-0 (was effectively all-rights-reserved; the rules require CC0 or
MIT-0). Added THIRD_PARTY_LICENSES.md and NOTICE covering Qwen3-4B (Apache-2.0) -> NVARC SFT, the
official ARC-AGI-2 tasks (Apache-2.0), and RE-ARC (MIT). Citations wired into the paper.

### Continued base fine-tune (gated)
bitsandbytes does not load on Blackwell+Windows, so the planned 4-bit QLoRA became a bf16 LoRA
(per-task TTT already proves bf16 fits 16 GB). build_ft_corpus.py mints a corpus from the official
ARC-2 training pool + RE-ARC generations using the exact inference prompt format (~137k examples).
finetune_base.py runs a bf16 LoRA (rank 32) and merges to qwen3-4b-arc2ft for GGUF conversion.
Pilot (20k examples, 1 epoch) running on the rig at ~3.45s/step (~19h); val 0.039 tracking train
0.024 at step 2000, no overfit yet. Keep only if it clears the gate: no-TTT(FT) > 12.79% and
FT+TTT > 16.28%.

### Paper as a LaTeX PDF
No published word cap for the Paper Track (rubric-judged on six criteria with a required
structure). make_visuals.py now emits every figure as vector SVG + PDF. Authored paper/writeup.tex
and built a 13-page PDF via MiKTeX: reader's guide + table of contents + PDF bookmarks, typeset
math (roofline r_max = B/M, AIRV vote, TTT loss), all 10 figures, and six appendices
(reproducibility incl. the two GGUF converter patches, extended per-scale results + McNemar,
engineering log of dead ends, prompt/vocab format, provenance + BibTeX).

### Submission notebook
Redirected llama-server stdout/stderr to llama-server.log instead of the notebook output: the
per-request server logs were flooding the Kaggle cell output and lagging the browser during the
model run. Recommended workflow is Save & Run All (Commit), not interactive watching.

## 2026-06-23

### v1 on the ARC-AGI-2 main-track leaderboard: 6.53
The v1 no-TTT AIRV submission was scored: 6.53 on the hidden 240-task test set (P100, offline,
submission.json produced, "Succeeded"). The Paper-Track team now has its required main-track entry
(eligibility met). Public-to-hidden gap: 12.79% on the 120 public eval vs 6.53 on the hidden set.
Likely mostly the hidden set being harder (it resists the public eval's patterns), but a ~2x drop is
large enough to also suspect per-task failures in the scored rerun (the prompt cache was active in
the run log; big grids are VRAM-heavy on P100). This run predated the run.log capture, so the next
submission will show empty-vs-wrong per task. Report both numbers in the paper; the gap is an honest
single-GPU finding. Score-climb path unchanged: v2 TTT, then the self-trained base if it clears the gate.

## 2026-06-23 (later)

### Keep/drop gate: self-trained base DROPPED (honest negative result)
Converted qwen3-4b-arc2ft to Q8_0 (the rig's convert_hf_to_gguf is already vocab-patched) and ran the
no-TTT AIRV eval on the 120 public set: 11.63% (20/172) vs the stock base's 12.79% (22/172). The FT is
slightly worse, so it fails the gate (needed > 12.79%); FT+TTT was not run since the first condition
already failed. Decision: DROP the fine-tune, ship the stock NVARC base. The base is already
well-adapted to the grid format; continued FT on official ARC-2 + RE-ARC (ARC-1-derived) did not
transfer, consistent with the barely-moving val loss (0.039 -> 0.036). This is the negative result we
agreed to report either way; it goes in the paper as an honest single-GPU finding. v2 is unaffected (it
uses the stock base). The dual-path framing held: a clean drop, no sunk-cost.

## 2026-06-24

### v2 (TTT + colour-perm) on the main-track leaderboard: 11.94
The v2 submission (160 tasks with per-task TTT at scale 1.25, 80 no-TTT, 8 D4 + 4 colour-perm, 2.45h)
scored 11.94 on the hidden 240-task set, up from v1's 6.53 no-TTT. That is +5.41 absolute, ~83%
relative from TTT, a LARGER lift than the public eval saw (12.79 -> 16.28, +27%). Per-task adaptation
pays off more on the harder held-out distribution; this is the headline result of the entry.

### v2.1: spend the 12h headroom on accuracy
v2 used only 2.45h of 12h. Scaled the v2 notebook knobs up (still per-task try/except + budget-guarded):
MAX_TTT_TASKS 160->240, MAXPAIR 1000->1500 (L4 24GB trains bigger tasks the rig's 16GB excluded),
N_COLORPERM 4->8 (16 votes), TTT context 1600->2000 tokens (more train pairs per step), TRAIN_BUDGET_H
4->7. Kept TTT_STEPS=20 (more steps overfit per the sweep). Projected ~4-6h, well inside 12h.

### v2.1 run completed: 3.90h, 232 of 240 tasks trained
The scaled-up v2.1 ran clean on L4 x4. Phase A+B trained 232 of 240 tasks (maxpair <= 1500; only 8
big-grid tasks excluded, up from v2's 160 trained), 2.13h. Phase C served 8 no-TTT + 232 TTT across
the 4 GPUs, 1.77h. Total 3.90h wall-clock (vs v2's 2.45h), every adapter ok=True, no disk or OOM
fault, comfortably inside the 12h budget. Hidden score pending submission. Run-time series so far:
v1 ~3.2h est (P100 x1, the hidden run was not instrumented), v2 2.45h, v2.1 3.90h (both L4 x4), all
under the 12h cap. Added two version-progression figures, fig_version_score and fig_version_time,
built by scripts/make_version_progress.py from a single VERSIONS table (append a row per submission).

### Self-training (STaR) scaffolded; oracle self-solve saturates, pivot to a near-miss curriculum
User directed self-training on the rig. Built scripts/self_train.py: harvest (AIRV over the ARC-2
TRAINING tasks, keep what the model self-solves) -> curriculum corpus (reuses build_ft_corpus
official_pools + mint) -> bf16 LoRA distil (reuses finetune_base) -> convert + quantize + no-TTT AIRV
gate vs stock 12.79%, then iterate. Honest catch recorded up front: on the training set a verified-correct
solution equals the ground-truth target, so plain self-training shares the failure mode of the dropped
continued-FT (11.63%). The smoke (12 tasks, port 8090) validated the pipeline end to end but self-solved
12/12 under oracle any-match: the lenient criterion saturates, so that curriculum is ~the FT's data.
Pivot: the harvest now records voted top-2 correctness AND oracle any-match, and the distil targets the
NEAR-MISS set (oracle-solved but not voted), the tasks at the base's edge, the only genuinely new signal
vs the FT. A 200-task harvest (skip-train, port 8090) is running to measure the oracle/voted/near-miss
split. Train + gate are blocked on a stale llama-server (pid 27900) holding port 8080 + ~3.6 GB that the
distil (VRAM) and the gate eval (port) need. Dual-path: ship if it clears the gate, else honest negative
result #2 (does self-distillation move ARC-AGI-2 on one GPU, or has the SFT'd base saturated it).

### Self-training result: the base is saturated, near-miss curriculum empty, DROPPED
The 200-task harvest settled it. The stock base self-solves 194/200 training tasks by oracle any-match
(97%) and 191/200 by the robust AIRV vote (95.5%); the near-miss set (reached but not robustly solved,
the only exploitable self-training signal) is 3/200 (1.5%). The minted curriculum (15,203 examples from
the 194 solved) is the same ground-truth data the dropped continued-FT used. So self-training has no
purchase here: the training tasks split into already-solved (nothing to learn) and unsolved (no correct
trace), almost nothing between. This answers the open question (the SFT'd base IS saturated on its own
training distribution), a clean second negative on weight refinement that also explains the FT failure.
Recorded in paper section 5.3 (writeup.tex/.pdf + paper.html). Pivot to the inference-time lever:
scripts/refine_loop.py (verifier-guided selection via honest leave-one-out on the un-adapted base, so it
avoids the saturated check that sank the earlier E1 vote-reweighting), validating on the public eval now.

### Refinement loop v1 (verifier-guided selection): null on the eval slice, same bimodal wall
scripts/refine_loop.py: generate under each D4 aug, score each aug by honest leave-one-out train fidelity
(un-adapted base, so not the saturated E1 check), select the test answer from the top-fidelity augs, fall
back to the plain vote when none reproduce the task. On a 20-task eval slice it scored 2/32 = 6.25%,
IDENTICAL to no-TTT AIRV on the same tasks (2/32), because most eval tasks had fidelity 0 (the base
reproduces no held-out train pair for puzzles it does not understand), so the selector had nothing to act
on and fell back to the vote. UNIFYING FINDING: this is the same wall self-training hit. The base is
bimodal on ARC tasks (robust solve or total miss, almost no partial-understanding middle: the training
harvest was 95.5% robust vote / 1.5% near-miss; eval fidelity is ~0 on the tasks it misses), so every
refinement lever (self-training on near-misses, verifier-guided selection) has nothing to grip. The one
lever that works, per-task TTT, MANUFACTURES capability rather than refining partial understanding. The
lever that would break the wall is program search (partial credit on a program is a real gradient even
where the LLM is lost). Paper write-up of the refinement family held pending the call to test the active
re-prompt variant or bank it.

### Active refinement variant: also null, refinement family closed
The active form (verification-gated temperature re-sampling, 6 extra samples on the tasks the base is
unsure of; our 16-token model cannot take natural-language feedback, so the canonical NL re-prompt is
not expressible) scored 2/26 = 7.69% on a 15-task slice, IDENTICAL to no-TTT AIRV on the same tasks.
Both refinement forms (passive selection, active search) are null. Refinement family closed: the
bimodal base gives no partial-credit gradient to act on, and the 16-token interface forbids the
natural-language critique the canonical loop needs. Consolidated into the paper (sections 5.1 + 5.3).

## 2026-06-24 (later): program-search line scouted

### Scouting report + verification audit
A research scouting pass (docs/scouting_report.md) ranked the post-bimodal directions; the user audited
every load-bearing claim against primary sources. Headline catch: the regimes did not match. Pang (26%)
and Berman (29.4%) on ARC-2 were UNCONSTRAINED frontier-NL runs (Grok-4, $8-30/task); the in-box SOTA is
NVARC's 24.03% by TTT, and NO published program-synthesis result beats TTT in the offline / 4B / grid
regime. So program search has no in-regime evidence over TTT; its case rests solely on its unique
partial-credit property. Folded the audit + five figure corrections (GridCoder params, PeARL 30->77,
ARChitects threshold T=9% and "runner-up not winner", Akyurek 53.0/47.1/61.9 not "+35%", arc-dsl 400 not
400+) into the artifact. Re-weighting: lead with the in-regime secondary lever (honest LOO-TTT); program
search is the parallel higher-risk bet.

### Program-search Stage 0: arc-dsl grader built and validated
Cloned arc-dsl (MIT, Hodel: 160 dsl.py primitives, 400 ARC-1 reference solvers) + the ARC-1 training set
(400 tasks) on the rig. scripts/grade.py scores an arc-dsl program as fraction-of-train-pairs-reproduced
(CPU, no GPU), the partial-credit grader the program-search line needs. Validation: 399/400 reference
solvers reproduce their whole task (99.8%, mean fractional grade 0.999), confirming the grader + DSL
execution end to end. Throughput 252 program-exec/s single-core; below the report's optimistic gate but
the executor is not the bottleneck (neural-guided search proposes few programs/task), so it passes the
real bar. Next gate for program search is the proposer (the 16-token model cannot emit DSL: needs a
trained DSL proposer, Stage 1.5). Stage 0 done.

### v2.1 on the leaderboard: 13.61 (rank 468 / 4000)
The scaled-up v2.1 (232/240 tasks TTT, 16 augmentations, 2000-token context, 3.90h on L4 x4) scored
13.61 on the hidden 240-task main-track set, up from v2's 11.94 (+1.67 absolute, +14% relative), rank
468 / 4000 (top ~12%). The scale-up paid off: more TTT coverage (160 -> 232 tasks) plus richer context
lifted the score. Trajectory: v1 6.53 (no-TTT) -> v2 11.94 -> v2.1 13.61, climbing toward the in-box
SOTA (NVARC 24.03% on 4x L4). Confirms the audit's read that in-regime TTT-family headroom remains,
reinforcing the lead on honest LOO-TTT (Stage 1). Version-progression figures updated with 13.61.

### Stage 1 LOO probe: weak positive (held-out accuracy predicts test correctness, thinly)
scripts/ttt_loo.py ran overnight on the rig (40 trainable tasks, greedy single-seed: one adapter on n-1
pairs predicting the held-out pair, one adapter on n predicting the test). Verdict: leave-one-out IS
directionally predictive. LOO-hit tasks (9): 22% test-correct; LOO-miss (31): 3%; contingency
{(T,T):2,(T,F):7,(F,T):1,(F,F):30}. The 0.19 gap clears the script's 0.15 threshold, but it rests on only
3 test-correct tasks in a greedy proxy (no AIRV/scale, base rate 7.5% << the 16.28% pipeline). So a real
but thin positive: it confirms CORRELATION, not yet that acting on it lifts the score. Green light to
build v4 = LOO-based adapter SELECTION (train a few seeds/task, keep only the best-held-out seed's GGUF so
disk stays 1/task) and measure it on the rig against 16.28% before spending Kaggle quota. ("launch failed
exit 255" was just the SSH link dropping when the laptop slept; the detached probe finished 40/40 clean.)

### v3 ran: multi-scale voting, 7.06h (submitted, score pending)
v3 (multi-scale: the per-task adapter applied at scales 1.0/1.25/1.5, AIRV votes over the union) ran full
on L4 x4: Phase A+B 232/232 adapters (2.13h), Phase C multi-scale serve completed, 7.06h total. Serve was
~4.9h, ~2.8x v2.1's 1.77h, confirming the 3-scale vote engaged (not a silent no-op). Valid submission.json
(8 no-TTT, 232 TTT), no fallbacks. The earlier smoke fell back to no-TTT only because the accelerator was
set to None/CPU ("Torch not compiled with CUDA"); with L4 x4 selected it trained fine. Base
mirzamilanfarabi (the always-used Kaggle base, correct). Submitted, hidden score pending (~7h marking).
Honest expectation: modest lift or flat over 13.61. Run-time series: v1 ~3.2h est, v2 2.45h, v2.1 3.90h,
v3 7.06h (the multi-scale serve is the cost).

### v4 go/no-go: leave-one-out adapter SELECTION test (running on the rig, free)
scripts/ttt_loo_select.py: per trainable task, train a baseline adapter (all n pairs) and K=3 leave-one-out
adapters (each holding out a different pair); keep only the LOO adapters that reproduce their held-out
pair; pool their AIRV (4 D4) candidates and vote; compare to the baseline by PAIRED FLIPS (recovered
miss->hit vs lost hit->miss, the sensitive read for a thin signal). Tests whether the probe's weak LOO
correlation translates into an actual selection lift. 30 tasks, ~7h on the single 16GB GPU, zero Kaggle
quota. If selection recovers more than it loses, v4 is worth building as a full submission; else drop LOO
and v3 (multi-scale) stays the lead.

## 2026-06-25  Paper prose pass (clean style)
Rewrote the paper body into clean prose, removing the semicolon/colon/comma-splice dash-substitutes the user flagged as a shoddy em-dash replacement. ~50 clause-splices fixed across the abstract, sections 1-6, and appendices A-E, in both paper/writeup.tex and paper/paper.html (kept in sync, sentence for sentence). Numbers, citations, and figures unchanged. Fixed the stale HTML TOC label (Two -> Three lessons). PDF recompiled, 16 pages, no LaTeX errors. Commits db1c9ae (abstract template) and bb23e32 (full pass). Figure captions left in their conventional dense form. Memory working-preferences updated with the no-substitute rule.
Pending content folds (per plan 1d): v3 hidden score (Kaggle still marking) and the leave-one-out selection null (rig job 4208 at 20/30, A0==A1 on every task, zero flips, ~4h left).

## 2026-06-25 (cont)  Diverse second-model lever: Tier-0, LOO-selection, TRM coverage test
Deep research + experiments on whether a second model adds DISJOINT coverage (the only thing it can, given the bimodal wall). User time-boxed the TRM attempt ("we can never know if we don't try; pivot if it fails").
- Tier 0 (free, existing dumps): stock-4B no-TTT solves 14/120, FT-4B-arc2ft no-TTT 13/120, overlap 11. FT adds +2 disjoint (1818057f, 5961cc34); stock has 3 the FT misses; oracle union 16. A second transductive model (even our own FT variant) adds a SMALL disjoint set (+2), weaker overall. Marginal but nonzero. (Union 16 is an oracle bound; a real 2-attempt submission gets less.)
- LOO-selection (ttt_loo_select.py), stopped at 21/30 to free GPU: baseline single-adapter hits 2 tasks; keep+vote leave-one-out adapters that reproduce their held-out pair hits 3 = recovered 1 (36a08778), lost 0, net +1. WEAK POSITIVE, not the 0-flip null it showed through 20/30. CORRECT the paper: LOO-selection is a weak positive, not a null.
- TRM coverage test (the time-boxed "try"): released ARC-Prize-verified arc_v2_public checkpoint (6.2% pass@2). Env works on the rig (TRM imports vs system torch 2.10, adam-atan2 stubbed, DISABLE_COMPILE). Rebuilt the exact arc2concept-aug-1000 dataset (deterministic; num_puzzle_identifiers=1191730 MATCHES the checkpoint puzzle_emb -> alignment confirmed). eval_only.py loads clean (missing 0 unexpected 0, ALIGN MATCH). BUT the eval is impractical on native Windows: TRM's recursive ACT forward is meant to run torch.compile'd, which needs triton (no native-Windows backend -- the SAME wall as our torch.compile finding), so eager mode runs >6 min/batch -> ~a day-plus for the full 464-batch / 118k-augmented-test eval on one consumer GPU. Stopped. Precise TRM coverage number uncollected on the rig; would need Linux/Kaggle.
- Synthesis: every line of evidence (Tier-0 +2, LOO +1, NVARC shipped 4B-alone and found 4B+TRM only "slight", BARC: transductive models overlap and the coverage-adding diversity is INDUCTION) says a second transductive model is low-value, and TRM is impractical to even evaluate here. Pivot to program search (the inductive arm). Bonus paper point: TRM hits the same native-Windows compile/triton wall we document for our own pipeline.
Artifacts: scripts/trm_eval_only.py, scripts/trm_coverage.py; TRM env on rig at D:\arc\trm (.venv, ckpt/arc_v2_public, data/arc2concept-aug-1000).

## v3.1 two-scale submission (2026-06-25)

Forked the v3 multi-scale notebook down to two scales, 1.0 and 1.25, for margin under the marking wall. The three-scale v3 ran about 7h in dev and was still marking at 11h. Checked the rules. The Kaggle rerun gets a single 12-hour wall-clock window and fails with no score once it runs over, so v3 was genuinely close to timing out. v3.1 should land near 5.4h dev and about 9h marking. Built notebooks/arc_submission_v3_1.ipynb, ready to fire the moment v3 fails or a quota slot frees.

One item for later. The ARC-AGI-2 page says most competitive offline submissions call out to third-party compute like Modal or Lambda, which sits oddly next to the no-internet rule and may be how the 50-something scores happen. Worth investigating before the next scaling push.

## Compute-loophole research and the offline ceiling (2026-06-26)

Ran the third-party-compute question to ground. ARC-AGI-2 has two tracks. The prize-eligible Kaggle competition is offline, internet disabled, no API systems, four L4s, 12h. The internet-and-$10k allowance (Modal, Lambda) belongs only to the Verified and Community leaderboards, which are not eligible for the Kaggle prize, so we cannot use external compute on our track.

Pulled the live 2026 Kaggle public leaderboard from the API. Leader nvbanana at 50.56, the chasing pack at 32 to 35, all offline under our exact rules. So roughly 50 is achievable offline and the gap is method, not compute access. The 54 percent Gemini and the 70-plus frontier numbers are the internet-allowed track, a separate thing. Folded the dated leaderboard snapshot and the two-track framing into Section 5.4. Saved to memory as arc-agi-2-two-track-ceiling.

Also confirmed v3 came back as a Kaggle system error with no score, so v2.1 at 13.61 stays the floor. v3.1 is built and parked on the weekly Kaggle GPU quota.

## Program-search disjoint coverage on ARC-2 (2026-06-28)

Ran the DSL proposer over the 101 ARC-2 eval tasks the 4B-TTT misses, the only ones that can be disjoint. Result, proposer solved 0, disjoint coverage 0, with 2 of the biggest-grid tasks skipped on OOM. The 1.5B arc-dsl proposer adds nothing the 4B misses on ARC-2. Clean no-go. The arc-dsl primitives are ARC-1-designed and the proposer trained on ARC-1, so it does not express ARC-2's new concepts, and the 4B-missed tasks are disproportionately the biggest grids where generation is heaviest. ARC-1 held-out was about 1 in 9, ARC-2 disjoint is 0. Program search in this form is out, the 96 GB method work redirects to the refinement loop.

Harness fixes needed along the way, the rig python binary is python3.13.exe so the python.exe process checks misread live jobs as dead, an OOM cap with fewer return sequences and expandable_segments, and a process-isolated executor (dsl_exec.py) that can actually kill infinite-loop generated programs since a thread cannot be killed.

## Robust rig validation harness (2026-06-28)

Both heavy rig runs this session stalled a few tasks in and died, the program-search probe and the more-candidates AIRV test that hung at task 3 of 40. Cause, the HF-eager harnesses call model.generate with no timeout, so one bad task on the 16 GB card hangs the whole run, the per-task try/except catches exceptions but not hangs, and there is no resume so a restart loses everything.

Fixed by hardening scripts/run_airv_llamacpp.py into the single rig validation harness. It runs through llama-server (the 13x, far lower VRAM) where every request has a 300s cap plus ensure_server recovery, so no task can hang the run. Added resume-from-partial (filtering the [[0]] padding write_submission emits for un-done tasks), per-task checkpoint, per-task try/except, a heartbeat eval_status.json so monitoring is a file read not a process guess, and a K/TEMP sampling knob for the more-candidates lever (K=1 = greedy baseline). Standard launch is Start-Process with the real python3.13 interpreter detached, monitored via eval_status.json, never a laptop-side ssh monitor.

Verified end to end. Smoke, 3 tasks via llama.cpp in 64s including server start, with heartbeat, per-task checkpoint, and SCORE. Resume, killed at 3 of 8, relaunch loaded exactly 3 real tasks and re-ran the rest rather than restarting. The K=4 more-candidates verdict over the 120 eval now runs on it, compared against the recorded K=1 baseline airv_nottt_eval.json at 12.79.

## v3.1 regression and the more-candidates win (2026-06-28)

v3.1 scored 12.78 on the hidden set, a regression from v2.1's 13.61. Cause, v3.1 added scale 1.0 to the vote and 8 colour-perm augmentations on top of the proven single-scale-1.25 8-D4 config. AIRV is a popularity vote, so the weaker scale-1.0 outputs and noisier colour-perm views diluted the consensus. Matches the paper finding that colour-perm stacking did not help. v2.1 stays the best.

The rig more-candidates verdict, the first run on the hardened llama.cpp harness that finished cleanly where the HF-eager runs stalled, is a win. K=4, sampling each of the 8 D4 augs 4 times at temp 0.8 for 32 candidates, scored 13.95 no-TTT (24/172) against the K=1 greedy baseline 12.79 (22/172), +1.16 and +2 tasks. So more candidates helps when they are more samples of the good D4 transforms and hurts when they are weaker different transforms. The lever is candidate quality, not raw count.

Next, confirm more-candidates stacks with TTT (needs the TTT-on-llama.cpp validation), then v4 = v2.1's proven config plus more-candidates sampling, dropping multi-scale and colour-perm.

## Paper fold-in and workspace cleanup (2026-06-28)

Folded the session's results into the paper. The candidate-quality ablation (sampling each D4 view 4 times lifts no-TTT 12.79 to 13.95, while multi-scale and colour-perm dilute the vote) is in Section 4.4, the program-search 0 of 101 disjoint-coverage negative in Section 5.3, the robust llama.cpp validation harness in the engineering log, and the v3 error, the v3.1 regression to 12.78, and v4 in Section 4.1 plus the regenerated version-progress figures. PDF recompiled to 18 pages, HTML mirrored, no em-dashes or section symbols.

Archived the superseded experiment scripts (the v3 and v3.1 builders, airv_morecand, grade, ttt_loo_select, and the closed dsl and trm harnesses) and the v3 and v3.1 notebooks into archive/, leaving scripts/ and notebooks/ holding only the active pipeline.

## v4 more-candidates regressed (2026-06-28)

v4 (the more-candidates lever, sampling each D4 view 4 times) scored 13.19 on the hidden set, below v2.1's 13.61 and slower. The +1.16 lift the lever gave the no-TTT base on the local eval did not transfer to the TTT submission, because the per-task adapter already concentrates the model on its best answer and the extra temperature-0.8 samples only add noise. Same pattern as colour permutation, helps no-TTT, does not stack with TTT.

Third straight attempt to beat v2.1 that failed (v3 errored, v3.1 12.78, v4 13.19). v2.1's eight greedy D4 views at a single scale with TTT is a robust local optimum the candidate and voting side does not improve. The candidate-side levers on the 4B are exhausted, keep v2.1 as the entry. The only real lever left is a bigger non-bimodal base on the 96 GB. Corrected the paper (Sections 4.1 and 4.4) and the version figure.

## White-box filter is neutral (2026-07-01)

Built and gated the white-box plausibility filter (colour palette plus size rule, the LongT5 27% recipe) on the rig before any submission. On our pipeline it is neutral, 0 of 95 no-TTT tasks changed, the filtered score identical to the baseline (11.05 on both, 19 of 172). Our eight greedy D4 views already produce plausible candidates, so the rare implausible one never wins the vote and filtering changes no top-2. The LongT5 filter cleaned a noisier 200M model, ours is already clean, so the recipe does not transfer in isolation.

v5 (v2.1 plus filter) would match v2.1 at 13.61, so it is not submitted, keep v2.1 as the entry. The gate did its job, catching the neutral lever on the rig before burning a slot, the discipline v4 lacked. Fourth AIRV-side lever that fails (multi-scale, colour-perm, more-candidates regress, filter neutral). The 4B-AIRV pipeline is at its 13.61 ceiling, moving the number needs a different system not a tweak. The paper is done and honest, the entry is v2.1.

## White-box filter, corrected and completed (2026-07-01)

The neutral read above was from a partial run and was incomplete. Finishing the full 120-task no-TTT filter run and scoring it properly corrected the story before it settled, so here is the verified account.

The filter is not benignly neutral. On 117 tasks it keeps at least one candidate and changes no top-2, byte-identical to the baseline. On 3 tasks (38007db0, a32d8b75, e87109e9) its constant-size rule empties the whole candidate list. That rule infers a fixed output size from train pairs that only coincidentally share one, while the true rule is variable, so it rejects the correct variable-size output. 38007db0 was a baseline solve on both its test outputs, so the raw filter discards 2 correct answers.

The v5 notebook survives this because its selection is select_top2(f or cs), falling back to the unfiltered candidates when the filter empties. So v5 recovers those 3 tasks and ties the baseline, 12.79 no-TTT and 13.61 with TTT, a genuine neutral. The rig harness first read 11.63 because it lacked that fallback and padded the emptied tasks, a harness-versus-notebook mismatch and not a property of v5. Fixed run_airv_llamacpp.py to carry the same fallback (filtered or cands), re-ran, and the harness now lands on 22 of 172 = 12.79, matching the baseline exactly.

Net, the white-box filter earns nothing on our already-clean candidates and its rules actively reject correct outputs, saved from a real regression only by the empty-list fallback. A dead lever with latent harm, a sharper honest-negative than plain neutral and worth folding into the paper's ablation. v5 not submitted, v2.1 stays the entry at 13.61. Rig-python note, python and python3.13 resolve to a non-functional WindowsApps stub over SSH, so resolve the real interpreter with Get-AppxPackage PythonSoftwareFoundation.Python.3.13 before running inline.

## v5 verification pass, notebook filter aligned with the validated code (2026-07-02)

A full double-check of the v5 artifacts found one real divergence. The notebook's inline plausible() was not the function the rig validated. src/arc/filter.py treats the size rule as one of same-as-input or constant, same taking precedence, while the notebook copy applied both checks unconditionally. On the 32 eval tasks whose train pairs all share one size and also satisfy in-equals-out, the two checks contradict each other whenever the test input is a different size, so the notebook filter rejected every candidate including correct ones, 42 disagreements against filter.py on real grids and 19 ground truths rejected. The empty-list fallback masked all of it, under the conflict nothing passes, the list empties, and select_top2(f or cs) reverts to the unfiltered set, so the attempts and the score were unchanged. Still, the shipped artifact must match the validated code, so the builder's inline filter now mirrors filter.py exactly.

Verified after the fix by extracting plausible() from the built notebook and comparing against filter.py on all 516 grids constructible from the eval set (ground truths plus baseline attempts), zero disagreements, and the only rejected ground truths are the 5 outputs of the 3 known const-rule-flaw tasks where the fallback provably restores baseline behaviour. All notebook code cells parse, the builder and the committed notebook are byte-identical on rebuild, and the v4-to-v5 diff is exactly the intended change set, greedy reversion plus filter plus naming. Also refreshed the stale v2-era markdown and comments in the notebook (it described colour-perm layering that is off in this config). v5 remains unsubmitted, v2.1 stays the entry.

## v5 regressed on the hidden set (2026-07-04)

Ryan submitted v5 for a hidden-set read and it scored 12.78, down 0.83 from v2.1's 13.61 and coincidentally equal to v3.1's number. The save-version run was clean, 232 of 232 adapters trained, zero errors or fallbacks, 3.03h on the placeholder set, so this is not an execution failure. The pipeline is v2.1's exact stack plus the filter, decoding is greedy and the TTT seed fixed, so the filter is the only designed difference, and the loss is almost exactly two tasks.

The mechanism fits the tail risk flagged before submission. The filter rejects a correct output when the train pairs coincidentally share a size pattern the true rule does not follow, and the empty-list fallback only rescues a task when every candidate is rejected. On the public 120 every such rejection emptied the list, which is why the gate read neutral. On the hidden 240 it evidently kept a plausible wrong candidate on a couple of tasks, the fallback never fired, and those solves were clipped. Nondeterministic bf16 TTT training could contribute some of the gap and cannot be excluded from outside, but the flagged channel predicts exactly this failure shape and magnitude.

The lesson now has a price. A lever that measures neutral on the public eval has zero upside and only tail downside on the hidden set, so public-neutral means do not submit, and this slot bought that rule for 0.83. Fifth attempt to beat v2.1 fails (v3 errored, v3.1 12.78, v4 13.19, v5 12.78). v2.1 stays the selected entry at 13.61, the version figures now carry the v5 point, and the 4B-AIRV ceiling stands confirmed from a fifth direction.

## Paper revision on external review (2026-07-04)

A reader review of the paper came back with five recommendations and four of them landed as real improvements, applied to the tex and mirrored to the HTML, PDF recompiled to 19 pages. The abstract now opens with the constraint and the binding-constraint finding rather than burying the 13x in the fifth sentence. The public-to-hidden transfer gap is promoted from a limitations aside to a headline result, a new contributions bullet, a fourth generalizing lesson in 5.1, and a reframed 5.3, now backed by the full pattern, levels halve for the lean recipe and deltas reverse, every public-validated candidate lever regressed when submitted. Section 4.1 is restructured around an explicit two-regime signpost (single 5070 Ti public eval versus four-L4 hidden leaderboard) with a new leaderboard-progression table carrying v1 through v5. Section 5.4's scaling case is restated as a falsifiable hypothesis, the prediction is a non-empty near-miss band on a more capable base against our measured 1.5 percent, with the LongT5-based 200M report (arXiv:2603.06590) cited as the warning against reading it as a bare parameter-count claim, and the promissory closing sentence is gone. The Figure 4 caption now states outright that the VRAM surface is a calibrated model and only the points and cutoff are measured.

The review also folded the v5 chapter in as a side effect, the filter is the fifth ablation in 4.4 with the public-neutral hidden-harm mechanism, which resolves the inconsistency where the regenerated version figure showed a v5 point the prose never mentioned. The reviewer's typo note did not reproduce, a mechanical misspelling and doubled-word sweep plus a full re-read found none, so it likely referred to an earlier build.

## The routed dual-arm plan and the zero-shot induction probe (2026-07-15)

New direction after the 4B-AIRV ceiling, a consensus-routed dual-arm solver. The grid model's AIRV vote agreement, already measured to predict correctness, routes each task. High-consensus tasks bank the grid answer byte-identical to v2.1, low-consensus tasks route to an induction arm, a code LLM writing plain-Python transforms executed against the train pairs, with a verified program taking attempt_1 and the AIRV answer falling to attempt_2. Do-no-harm by construction, the arm only touches tasks the base already misses, so it cannot regress the 13.61 floor the way the five candidate-side levers did. Built scripts/pyexec.py (process-isolated executor forked from the archived arc-dsl one, plain namespace, kill-on-timeout, numpy-tolerant) and scripts/coder_probe.py (coder GGUF on a second llama-server, fenced-block extraction, resume, heartbeat). Added the whole system to the paper as Section 5.5 with a live status slot. Plan and gate ladder recorded, zero disjoint kills the direction, one to four iterates one bounded week, five or more builds the v6 submission.

First probe, zero-shot Qwen2.5-Coder-14B Q4_K_M on the rig (benched 79 tok/s), K=16 at temperature 0.8, over exactly the 101 eval tasks the 4B misses. Result, 0 of 101 disjoint solves, and across roughly 1600 programs only one task had a program reproduce even one train pair, an essentially empty near-miss band. Verified this is real and not a harness artifact, the executor and grid comparison correctly score a control program at 1 of 4, and the coder returns 16 distinct valid programs per task. Pulled the actual programs, they are fluent and sensible, not the arc-dsl expression failure. On a variable-output-size task the model wrote clean extract-and-sort code that crashed on the fixed-shape assumption, and on a narrower-output task it wrote correct code for the wrong rule and kept the input width. So the bottleneck moved from expression (the 1.5B DSL) to inference (this 14B), and a general coder does not close it zero-shot on one card. A sharper negative than the arc-dsl zero, it brackets induction from both ends of the model-size spectrum. Recorded in paper 5.5. Ryan's call, keep pushing on the rig rather than close it, so the next probe primes the coder with a few verified worked-example solvers (in the system prompt, cache-friendly) to attack rule inference directly, run over the same 101.

Few-shot result (2026-07-15), also 0 of 101. Three verified exemplars in the system prompt (recolor, tile, variable-size crop) lifted the partial signal only from 1 task with any train pass to 2, and one task reached 2 of 4 pairs, still nothing near a full solve, and the run was faster (1.6h) because the priming made the model write terser programs. So neither sampling nor worked examples close the inference gap at 14B on one card, a decisive two-form negative on top of the arc-dsl one. The one move left that changes the mechanism rather than repeating it is a reasoning-class model of the same size (DeepSeek-R1-Distill-Qwen-14B class, fits 16GB at Q4) that reasons through the transformation before emitting code, aimed at the inference bottleneck directly. Same harness, longer generation budget for the think trace. Recorded both probes in paper 5.5.
