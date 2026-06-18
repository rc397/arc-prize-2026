# Roadmap

From today (2026-06-18) to the final submission, about 19 weeks. Entries are week-of.
Key deadlines: entry and team-merge October 26, ARC-AGI-2 final submission November 2,
Paper Track submission November 9.

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
