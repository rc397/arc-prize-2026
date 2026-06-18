# ARC Prize 2026: ARC-AGI-2

An offline solver for the Kaggle ARC Prize 2026 ARC-AGI-2 competition, with a companion
Paper Track writeup. The approach builds on the NVARC Qwen3-4B grid model (the base model
of the team that won ARC Prize 2025) using per-task test-time training and an
Augment-Infer-Reverse-Vote candidate selection scheme.

Target: a top-8 Progress Prize and a strong paper. The 85 percent bonus is not reachable
with an offline open-weight model, so it is out of scope.

## Compute

- Laptop (development): Windows 11, RTX 3070 Laptop 8 GB, Python 3.13, torch 2.6 cu118.
  Holds the git repository and runs unit tests.
- rig (training and inference): SSH host `rig`, Windows 11, RTX 5070 Ti 16 GB
  (Blackwell sm_120), Python 3.13, torch 2.10 dev cu128. Project root `D:\arc` on a
  3.7 TB drive.

Code is edited on the laptop and synced to the rig with scp. Datasets and model weights
live only on the rig. See `DEVLOG.md` for the full development record.

## Layout

- `src/arc/`: core pipeline. `data` (IO and serialization), `augment` (invertible D4,
  color, and reorder augmentations), `eval` (competition scorer), `model` (model loading
  and ID-level ChatML prompt assembly). Later phases add test-time training, candidate
  generation, selection, and verification.
- `scripts/`: runnable entry points, including the baseline runner.
- `configs/`: hyperparameters.
- `tests/`: model-independent unit tests.
- `notebooks/`: the offline Kaggle submission notebook.
- `paper/`: Paper Track materials.

## Status

Phase 0 (environment and assets) is complete. Phase 1 (data IO, scorer, and a no-TTT
baseline) is in progress. Remaining phases: test-time training with AIRV selection, a
novel contribution for the paper, Kaggle packaging, and the writeup.
