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
