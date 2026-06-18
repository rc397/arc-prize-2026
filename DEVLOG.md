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
(`scripts/run_baseline.py`). Validating the format on a small subset of the evaluation
set, then running the full 120-task baseline. Numbers to be recorded here once available.
