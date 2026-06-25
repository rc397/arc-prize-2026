# The 16GB Ceiling, ARC-AGI-2 on One Consumer GPU

An offline solver and paper for the ARC Prize 2026 ARC-AGI-2 competition. The system builds on the NVARC
Qwen3-4B grid model, the base of the team that won ARC Prize 2025, and runs per-task test-time training
with augment-infer-reverse-vote selection. The paper studies how far that recipe goes on a single 16 GB
consumer GPU, then lays out the path that scales it onto the full 96 GB the contest provides.

The Paper Track writeup is the deliverable, `paper/the-16gb-ceiling.pdf`.

## Results

On the hidden ARC-AGI-2 main-track leaderboard the v2.1 submission scores 13.61%, which sat around the top
12% of about 4000 entries. On the 120-task public evaluation set the system scores 12.79% without
test-time training and 16.28% with a regularized, scale-tuned per-task adapter blend. Every measurement
and dead end is recorded in `DEVLOG.md`.

## Compute

Development happens on a Windows laptop with an RTX 3070 8 GB, which holds the git repository and runs the
unit tests. Training and inference run on the rig, a Windows box with an RTX 5070 Ti 16 GB on the Blackwell
architecture and a torch 2.10 cu128 stack, reached over SSH. Datasets and model weights live only on the
rig. The Kaggle submissions run on the contest box, four L4 GPUs offline.

## Layout

`paper/` holds the one paper, `the-16gb-ceiling.{tex,pdf,html}`, plus its figures. `src/arc/` is the core
pipeline, covering IO and serialization, invertible augmentations, the competition scorer, model loading
and ID-level ChatML prompt assembly, top-2 selection, and per-task test-time training. `scripts/` holds the
runnable entry points and the figure builders. `notebooks/` holds the current Kaggle submission notebook.
`archive/` keeps superseded experiments and drafts out of the way. `tests/` holds the model-independent
unit tests.

## License

All code, the conversion recipe, and the development log are released under MIT-0. Third-party licenses and
attribution are in `THIRD_PARTY_LICENSES.md` and `NOTICE`.
