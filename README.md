# The 16GB Ceiling, ARC-AGI-2 on One Consumer GPU

An offline solver and paper for the ARC Prize 2026 ARC-AGI-2 competition and its Paper Track. The system
builds on the NVARC Qwen3-4B grid model, the base of the team that won ARC Prize 2025, and runs the
ARChitects lineage of per-task test-time training, depth-first decoding and augmentation-vote selection.
The paper measures how far that recipe goes on a single 16 GB consumer GPU, prices the gap to the contest's
four-L4 node, and spends most of its length on what it takes to trust a hidden-set score.

Two deliverables live in `paper/`. One is the 1,500-word Kaggle writeup, `writeup.md`, and the other
is the full paper, `the-16gb-ceiling.{tex,pdf}`, with every figure regenerable from `scripts/`.

## Results

| configuration | hidden ARC-AGI-2 score |
|---|---|
| best single-card build, v6.1 on an RTX 5070 Ti 16 GB | 18.19 |
| the reproduced pipeline on the contest's four L4s, one immutable kernel version executed twelve times | mean 29.90, sd 1.34, range 28.47 to 32.64 |
| a fresh build of the same configuration, submitted as a control on the build script | 31.39 |

The twelve-run series is the paper's central measurement. The same artifact spans 4.17 points, ten task
credits, so one submission cannot separate configurations less than about four points apart, and every
build this project called a regression, plus both public notebooks run as controls, falls inside 1.5
standard deviations of the series mean. Forty levers were measured on top of the reproduction and none
moved the hidden score. The paper's ledger section records what closed each one. Three training-side gates
(rule count, rule kind, training duration) are null at their own optimum, and a corpus of 128
execution-verified ARC-AGI-2 generators was built for under three dollars along the way and is released in `corpus/`.

Every measurement and dead end is recorded in the paper, with working notes in `docs/`.

## Compute

Development happens on a Windows laptop with an RTX 3070 8 GB, which holds the git repository and runs the
unit tests. Training and inference run on the rig, a Windows box with an RTX 5070 Ti 16 GB on the Blackwell
architecture and a torch 2.10 cu128 stack, reached over SSH. Datasets, model weights and the raw
generation logs live only on the rig. The Kaggle submissions run on the contest box, four L4 GPUs offline.

## Layout

`paper/` holds the writeup, the paper `the-16gb-ceiling.{tex,pdf}`, the Kaggle card and cover, and the
figures, each regenerable from `scripts/`. The PDF is the record. `src/arc/` is the
single-card pipeline, covering IO and serialization, invertible augmentations, the competition scorer, model
loading and ID-level ChatML prompt assembly, top-2 selection, and per-task test-time training. `scripts/` holds the
runnable entry points, the Kaggle notebook builder (`build_v7_from_architects.py`), the rig experiment
drivers, and the figure builders. `notebooks/` holds every submitted Kaggle notebook by version.
`arc_submission_v7_2_replicate.ipynb` is the exact artifact behind the twelve-run series (kernel version
339772104) and `arc_submission_v7_2_original.ipynb` the build that scored 30.97 (337769272). `corpus/`
holds the released ARC-AGI-2 generators and their verification results. `docs/` holds the working notes, `archive/` keeps superseded experiments and drafts out of the way, and `tests/`
holds the model-independent unit tests.

## License

All code, the conversion recipe, and the development log are released under MIT-0. Third-party licenses and
attribution are in `THIRD_PARTY_LICENSES.md` and `NOTICE`.
