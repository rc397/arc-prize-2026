# The 16GB Ceiling, ARC-AGI-2 on One Consumer GPU

An offline solver and paper for the ARC Prize 2026 ARC-AGI-2 competition and its Paper Track. The system
builds on the NVARC Qwen3-4B grid model, the base of the team that won ARC Prize 2025, and runs the
ARChitects lineage of per-task test-time training, depth-first decoding and augmentation-vote selection.
The paper measures how far that recipe goes on a single 16 GB consumer GPU, prices the gap to the contest's
four-L4 node, and spends most of its length on what it takes to trust a hidden-set score.

## Start here

| if you want | open |
|---|---|
| the paper | [`paper/the-16gb-ceiling.pdf`](paper/the-16gb-ceiling.pdf) |
| the short version, the 1,500-word Kaggle writeup | [`paper/writeup.md`](paper/writeup.md) |
| the exact notebook behind the twelve-run series | [`notebooks/arc_submission_v7_2_replicate.ipynb`](notebooks/arc_submission_v7_2_replicate.ipynb) |
| the 128 verified ARC-AGI-2 generators | [`corpus/`](corpus/) |
| how the system works, in one document | [`docs/CONTEXT.md`](docs/CONTEXT.md) |
| the contamination audit | [`docs/contamination-audit.md`](docs/contamination-audit.md) |

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

## Layout

| folder | what's in it |
|---|---|
| `paper/` | the paper (`the-16gb-ceiling.tex` and `.pdf`), the writeup, the Kaggle card and covers, every figure, and the data behind the figures in `paper/data/` |
| `src/arc/` | the single-card pipeline, covering IO and serialization, invertible augmentations, the competition scorer, model loading and prompt assembly, top-2 selection, and per-task test-time training |
| `scripts/` | runnable entry points, the Kaggle notebook builder (`build_v7_from_architects.py`), the rig experiment drivers, and the figure builders |
| `notebooks/` | every submitted Kaggle notebook by version, with the experiment matrix in `notebooks/EXPERIMENTS.md` |
| `corpus/` | the released generators and their verification results |
| `docs/` | the context document, the contamination audit, the pool briefing and the GGUF conversion recipe |
| `tests/` | model-independent unit tests |
| `configs/` | the default run configuration |
| `archive/` | superseded scripts, notebooks, paper drafts and the dated lab notes, described in `archive/README.md` |

`notebooks/arc_submission_v7_2_replicate.ipynb` is the exact artifact behind the twelve-run series (kernel
version 339772104), `arc_submission_v7_2_original.ipynb` is the build that scored 30.97 (337769272), and
`arc_submission_v7_2.ipynb` is the fresh rebuild that scored 31.39. `DEVLOG.md` is the development log as it
stood when it was frozen, and the paper supersedes it.

## Running it

The unit tests need only Python, numpy and pytest.

```bash
python -m pytest tests
```

Everything else needs data and weights that aren't in the repository. The ARC-AGI-2 tasks come from the
[ARC Prize](https://github.com/arcprize/ARC-AGI-2) repository and go in `data/`. The base model is NVARC's
`qwen3_4b_grids15_sft139`, which the Kaggle notebooks load as a Kaggle model source. Development ran on a
Windows laptop, training and inference on a Windows box with an RTX 5070 Ti 16 GB (Blackwell, torch 2.10
with CUDA 12.8), and the submissions on the contest's offline four-L4 machine.

## License

All code, the conversion recipe, and the development log are released under MIT-0. Third-party licenses and
attribution are in `THIRD_PARTY_LICENSES.md` and `NOTICE`.
