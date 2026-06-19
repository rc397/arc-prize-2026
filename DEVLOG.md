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
