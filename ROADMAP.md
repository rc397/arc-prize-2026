# Roadmap

From today (2026-06-18) to the final submission, about 19 weeks. Entries are week-of.
Key deadlines: entry and team-merge October 26, ARC-AGI-2 final submission November 2,
Paper Track submission November 9.

## Status and active plan (updated 2026-06-21)

We are far ahead of the original June phase schedule below. Done: the pipeline, the measured result
(16.28% blended at adapter scale 1.25, vs the 12.79% no-TTT baseline), both paper drafts, the public
notebook, the 560x280 card, the Kaggle prep and submission notebooks, and the repo cleanup. What
remains is getting the offline notebook onto the ARC-AGI-2 main track (required so the Paper-Track
team is eligible, since the rubric's Accuracy criterion reads the main-track leaderboard) and
submitting the paper. With Nov 2 (main) and Nov 9 (paper) deadlines, the buffer is ~4.5 months, so
the goal now is correctness, not speed. The next moves branch on results:

### A. Prep notebook (build the offline assets: Linux llama.cpp + Q8_0 GGUF)
- A1. nvcc present on the CPU image: prep runs on no-accelerator (zero GPU quota). Go to B.
- A2. nvcc absent: run prep once on any GPU (~0.25h quota, GPU idle, present only for nvcc), or attach
  a public prebuilt llama.cpp CUDA dataset and skip the build cell. Go to B.
- A3. Conversion patches fail (llama.cpp drift from release b9724): upload the rig's validated
  `arc-qwen4b-q8_0.gguf` as a dataset and skip the convert cells; still build and stage `llama-server`.
- A4. Build errors (cmake or CUDA): fix the flags, or fall back to a prebuilt binary dataset.

### B. v1 submission notebook (no-TTT AIRV, offline) to the ARC-AGI-2 main track
- B1. Runs offline, valid `submission.json`, under 12h: SUBMIT. Locks ~12.79% and satisfies the rule
  that the Paper-Track team must have a main-track submission. This is the minimum viable entry.
- B2. `llama-server` will not start on the Kaggle GPU: rebuild for the exact arch (P100 sm_60,
  T4 sm_75, L4 sm_89) and confirm the CUDA runtime libs are staged in the assets.
- B3. Projected over 12h: cut to 4 augmentations, use the P100 (fastest decode), or both.
- B4. Submission rejected on format: match the exact ARC submission schema in `write_submission`.

### C. v2 submission (layer TTT) for a better main-track score
- C1. TTT trains and serves in-notebook under 12h: ~16.28% blended, resubmit (main track allows
  one submission per day).
- C2. TTT too slow or OOM on the Kaggle GPU: shrink it (fewer steps or rank, the maxpair filter to
  the smaller tasks), or cap TTT to the smallest tasks; if it cannot fit the budget, ship v1.
- C3. The GGUF LoRA bridge fails on Kaggle: debug `convert_lora_to_gguf`; worst case ship v1.

### D. Paper Track submission (the prize deliverable)
- D1. Once v1 (or v2) is on the board: submit the paper, the writeup fields from
  `paper/kaggle_writeup.md`, the 560x280 card, the notebook, and the public GitHub repo link.
- D2. If time allows, strengthen for the rubric. We are weakest on Accuracy and Progress (raise via
  v2) and strongest on Theory, Novelty, and Completeness (the single-consumer-GPU characterization).

### Priority order
1. Prep assets working (A). 2. v1 on the main-track board (B1), for eligibility. 3. Paper submitted
(D1). Those three make a complete, eligible entry. 4. v2 for a better Accuracy score (C). 5. Extras
only if quota and time allow: L4 x4 parallelization, more levers, a larger TTT confirmation.

---

## Original June schedule (kept for reference; we are well ahead of it)

## Phase 1: working pipeline (now)
- Jun 18: record the no-TTT baseline (exact and shape match). Validate the TTT pipeline on a
  small slice. First TTT versus baseline comparison.
- Jun 25: TTT sweep (steps, augmentations, learning rate, rank). First full 120-task TTT run
  on the public eval. Goal: reproduce the low-20s.

## Phase 2: strengthen the system
- Jul 2: better candidate selection (product-of-experts scoring, voting) plus a verifier that
  rejects candidates failing the train pairs. Measure the lift.
- Jul 9: faster inference (batching) to shorten eval runs. Choose the novel contribution and
  prototype the options.
- Jul 16 to Jul 29: build and ablate the novel contribution (for example a neuro-symbolic
  verifier or a refinement loop). Keep only what beats the baseline.

## Phase 3: raise the ceiling
- Jul 30 to Aug 12: optional base improvement (continue fine-tuning on RE-ARC, BARC, Nemotron)
  or a two-model ensemble. Run the approved 8B QLoRA comparison. Keep what helps.
- Aug 13 to Aug 26: first end-to-end Kaggle submission (offline notebook, submission.json, dry
  run, then a real leaderboard score). Do this early, not at the end.

## Phase 4: compete and write
- Aug 27 to Sep 16: iterate on the leaderboard, more sweeps, refine the novel edge.
- Sep 17 to Sep 30: lock the best configuration and fit it to the 12-hour, 240-task Kaggle
  budget. Keep a buffer.
- Oct 1 to Oct 14: draft the Paper Track writeup (method, theory, ablations, results, figures).
- Oct 15 to Oct 21: polish the submission and the paper. Final ablations.

## Phase 5: deadlines
- Oct 22 to Oct 28: accept the rules before October 26. Final leaderboard submissions.
- Oct 29 to Nov 2: final ARC-AGI-2 submission on November 2, with buffer.
- By Nov 9: submit the Paper Track writeup.

## Working notes
- The rig runs as a job queue: one task per GPU, continuously. Each finished run triggers the
  next, so the card stays busy.
- Only the final submission is scored, so we keep a shippable submission from mid-August on and
  improve it, rather than waiting for a perfect one.
- A second 5070 Ti would roughly halve experiment wall-clock (task-parallel), but it is a
  nice-to-have, not a requirement.
