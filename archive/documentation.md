# ARC Prize 2026 (ARC-AGI-2) - Technical Documentation

A complete reference for the solver: architecture, components, how to run it, the compute
setup, and the issues fixed so far. For the chronological narrative see `DEVLOG.md`; for the
week-by-week plan see `ROADMAP.md`.

## 1. Overview

Goal: an offline ARC-AGI-2 solver for the Kaggle ARC Prize 2026 competition, plus a Paper
Track writeup. Approach: build on the NVARC Qwen3-4B grid model (the released base model of
the 2025 winner) with per-task test-time training (TTT) and Augment-Infer-Reverse-Vote (AIRV)
candidate selection. Realistic target: a top-8 Progress Prize and a strong paper. The 85
percent bonus is unreachable offline, so it is out of scope.

## 2. Compute environment

- Laptop (development): Windows 11, RTX 3070 Laptop 8 GB, Python 3.13, torch 2.6 cu118. Holds
  the git repo, edits code, runs unit tests. It is only a remote control for the rig.
- rig (training and inference): SSH host `rig`, always-on and
  headless, Windows 11, RTX 5070 Ti 16 GB (Blackwell sm_120), Python 3.13, torch 2.10.dev
  cu128 (CUDA 12.8). Project root `D:\arc` on a 3.7 TB drive.

Driving the rig (its SSH shell is Windows PowerShell, not bash):
- Send remote scripts base64-encoded:
  `$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script)); ssh rig powershell -EncodedCommand $enc`
- Use `[Console]::WriteLine` (not `Write-Host`) to avoid CLIXML noise. Set
  `$env:PYTHONIOENCODING="utf-8"` before python.
- Sync code with `scp` (forward-slash remote paths, e.g. `rig:D:/arc/src/`). Data and weights
  live only on the rig.

Running jobs that survive the laptop closing (the rig stays on): launch DETACHED, so the job
is not a child of the SSH session and does not die when the link drops:
```
Start-Process -FilePath python -ArgumentList <args> -WindowStyle Hidden \
  -WorkingDirectory D:\arc -RedirectStandardOutput <log> -RedirectStandardError <log>.err -PassThru
```
Read the log file on the rig when reconnecting.

## 3. Data

Official Kaggle consolidated format, on the rig at `D:\arc\data\kaggle\`:
- `arc-agi_training_challenges.json` / `_solutions.json`: 1000 tasks.
- `arc-agi_evaluation_challenges.json` / `_solutions.json`: 120 tasks (local validation).
- `arc-agi_test_challenges.json`: 240-task placeholder, swapped for the real tasks at rerun.
- `sample_submission.json`: the required output shape.

Schema: `challenges[task_id] = {"train": [{input, output}...], "test": [{input}...]}` (test
entries carry only the input). `solutions[task_id] = [grid, ...]` (one per test input). A grid
is a list of lists of ints 0-9, up to 30x30.

Submission: `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one entry per test
input; every task_id and both attempts must be present. `data.write_submission` enforces this.

## 4. The model

NVARC Qwen3-4B grids model (`kaggle: sorokin/qwen3_4b_grids15_sft139`), on the rig under
`D:\arc\models\kagglehub\...\transformers\bfloat16\1`. It is a `Qwen3ForCausalLM` (hidden 2560,
36 layers) with a 16-token vocabulary:
- ids 0-9: the grid colors (a color value equals its token id).
- id 10: newline (row separator).
- ids 11/12: the role tokens `user` / `assistant`.
- id 13: `<|endoftext|>` (pad); ids 14/15: `<|im_start|>` / `<|im_end|>` (eos).

Critical detail: the role tokens cannot be produced by tokenizing the words "user"/"assistant"
(the byte tokens are absent and there is no unk token), so prompts are assembled at the
token-id level, not via strings. The prompt is ChatML few-shot: each train pair is a
user(input) then assistant(output) turn, then the test input as a final user turn, ending with
the assistant opener so the model generates the output grid followed by im_end.

## 5. Pipeline architecture (`src/arc/`)

- `data.py`: task IO (`load_consolidated`, `load_json`), grid/text `GridSerializer`, and
  `write_submission`.
- `augment.py`: invertible augmentations. 8 D4 geometric ops, color permutation, train-example
  reorder. `apply_grid`/`invert_grid` map a grid into and back out of the augmented frame;
  geometry and recoloring commute so the inverse undoes both. Unit-tested for exact inversion.
- `eval.py`: the competition scorer (`score_submission`): for each test output, 1 if either
  attempt matches exactly, averaged over all test outputs.
- `model.py`: model loading and ID-level ChatML. `grid_to_ids`/`ids_to_grid`,
  `build_prompt_ids`, `greedy_generate`, and `batched_generate` (generates many augmentation
  prompts in one left-padded, chunked call; returns each candidate with its sequence
  log-probability).
- `ttt.py`: per-task test-time training. `make_adapter` wraps the base in a LoRA adapter and
  snapshots its init; `test_time_train` resets the adapter, then trains ~150 steps with a fresh
  random augmentation and a leave-one-out target each step (predict one train pair's output
  from the others), loss on the output tokens only, gradient checkpointing on.
- `select.py`: `select_top2` aggregates candidates by vote count then mean log-probability and
  returns the two best distinct grids (confidence-weighted AIRV).

End-to-end per task (`scripts/run_ttt.py`): fit a fresh LoRA adapter, then for each test input
generate one candidate per D4 view in a batched call, invert each back to the original frame,
and select the top two distinct grids.

## 6. Running the code

Baseline (no TTT) on the rig:
```
python scripts/run_baseline.py --model-dir <model> --data-dir <per-task dir> --n 10
```
TTT pipeline (official consolidated files):
```
python scripts/run_ttt.py --model-dir <model> \
  --challenges <eval_challenges.json> --solutions <eval_solutions.json> [--n N] \
  [--steps 150] [--n-aug 8] [--lr 1e-4] [--rank 32] [--do-sample] [--temperature 1.0] [--out path]
```
Key flags: `--n` limits tasks; `--steps` TTT steps; `--n-aug` augmentations/candidates per
task; `--do-sample`/`--temperature` enable sampled (more diverse) candidates; `--rank` LoRA
rank. Output prints `[i tid] loss=.. ttt=Xs gen=Ys correct1=..` per task and a final
`[score] exact c/t` line, and writes a submission json.

Tests (laptop, no GPU): `python tests/test_core.py` (augmentation invertibility, serialization
round-trip, scorer).

## 7. Performance notes

- Decoding is autoregressive and memory-bandwidth bound: at batch 1 each token re-reads all
  ~6.8 GB of weights, so a 30x30 output (~900 tokens) is the dominant cost. `batched_generate`
  amortizes that read across candidates (~5-7x). Planned: FP8 E4M3 weight-only quant (~2x more,
  same on Blackwell as MXFP8 for speed), then a WSL2 + vLLM spike for a larger inference win.
- The Windows allocator does not support `expandable_segments`, so PyTorch reserves the peak
  memory of the most demanding task and holds it; `run_ttt.py` calls `torch.cuda.empty_cache()`
  between tasks to bound this over long runs.
- The Kaggle submission budget (240 tasks in 12 hours on L4x4) is independent of the rig; the
  rig only accelerates development.

## 8. Issues fixed

- Quadratic decode after TTT (~50x slowdown): TTT enables gradient checkpointing, which forces
  `use_cache=False`, and resetting that on the PEFT wrapper did not reach the base model that
  `generate` uses, so decoding ran without a KV cache. Fixed by passing `use_cache=True`
  explicitly in the generate calls.
- First baseline CUBLAS error: VRAM contention from desktop apps, not a code or model issue.
- Jobs dying when the laptop closed: they were children of the SSH session; fixed by launching
  detached with `Start-Process` (section 2).

## 9. Status

- Pipeline complete and committed: data IO, augmentation, scorer, model IO, TTT, batched
  generation, confidence-weighted selection, on the official data format.
- Running: full 120-task TTT eval (default settings) detached on the rig, plus a queued sweep
  of variations (more candidates, more TTT steps, higher rank).
- Next: confirm the first TTT accuracy number, wire in FP8 E4M3, then multi-GPU candidate
  generation and the WSL2 + vLLM spike. See `ROADMAP.md`.
