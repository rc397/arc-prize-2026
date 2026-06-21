# Roadmap

From today (2026-06-18) to the final submission, about 19 weeks. Entries are week-of.
Key deadlines: entry and team-merge October 26, ARC-AGI-2 final submission November 2,
Paper Track submission November 9.

## Status and active plan (updated 2026-06-21, researched)

We are far ahead of the original June schedule below. Done: the pipeline, the measured result
(16.28% blended at adapter scale 1.25 vs the 12.79% no-TTT baseline), both paper drafts, the public
notebook, the 560x280 card, the Kaggle prep and submission notebooks, and the repo cleanup. What
remains is getting the offline notebook onto the ARC-AGI-2 main track (required so the Paper-Track
team is eligible, since the rubric's Accuracy criterion reads the main-track score) and submitting
the paper. With Nov 2 (main) and Nov 9 (paper) deadlines, the buffer is ~4.5 months.

Submission constraints (from the ARC-AGI-2 competition pages): the scored run is OFFLINE on the 2026
"upgraded accelerators", reported as four NVIDIA L4 GPUs (24 GB each); P100 is the older default and
a dev-dropdown option, so confirm on the Code Requirements tab. It must solve 240 tasks (120
semi-private + 120 private) in under 12 hours, with about a $50 compute budget. The `submission.json`
schema `{task_id: [{attempt_1, attempt_2}, ...]}` is already what `write_submission` produces, and
solutions must be open-sourced (our repo is public, CC-BY 4.0). Implication: v1 (no-TTT) fits one L4
(~7h for 240 tasks); v2 (TTT) on 240 tasks needs sharding across the four L4s or a TTT subset.

### A. Prep notebook (build the offline assets)
- A1. nvcc on the CPU image: prep runs on no-accelerator (zero GPU quota). A2. nvcc absent: one
  ~0.25h GPU run, or a prebuilt llama.cpp CUDA dataset. A3. Conversion patch fails (llama.cpp drift
  from b9724): upload the rig's validated `arc-qwen4b-q8_0.gguf` and skip the convert, still stage
  `llama-server`. A4. Build error: fix flags or use a prebuilt.

### B. v1 no-TTT submission, the eligibility floor
- B1. Runs offline, valid `submission.json`, 240 tasks under 12h on one L4: SUBMIT. Locks ~12.79% and
  satisfies the team-must-have-a-main-submission rule.
- B2. `llama-server` will not start on L4: rebuild for sm_89 and stage the CUDA runtime libs.
- B3. Over 12h: drop to 4 augmentations. B4. Format rejected: match the schema (already matched).

### C. v2 TTT submission, the score lift
- The 240-task TTT will not fit one L4 in 12h. 2a (best): shard the 240 across the four L4s, ~4x
  throughput (new multi-GPU dispatcher around `ttt_phase1/1b/2`). 2b (fallback): TTT only on the
  smaller tasks (maxpair filter) + no-TTT on the rest, single L4. Branch: still too slow -> 2b -> v1.

### D. Paper Track submission (the prize deliverable)
- Once v1 (or v2) is on the board: submit the paper (the `paper/kaggle_writeup.md` fields, the card,
  the notebook, the public repo link). We are weak on Accuracy/Progress, strong on
  Theory/Novelty/Completeness.

### E. Novel score improvement (the 2-month-buffer research; lifts the score and the paper rubric)
2025 grounding: top Kaggle ARC-AGI-2 was 24.03% (NVARC, whose model we use); the official analysis
names the refinement loop with verification as the central advance and the top 2026 direction, and
we do not have it (our AIRV is a one-shot vote). Staged, cheapest first:
- E1. Verifier-guided selection (16 -> ~20%): score each candidate against the train pairs
  (leave-one-out consistency) instead of plain voting. No training, the biggest lever we lack.
- E2. Refinement loop (~20 -> ~24%, matches 2025 SOTA): generate -> verify -> re-prompt with the
  discrepancy -> regenerate, keep the best-verified. The defining 2025 technique.
- E3. Scale TTT toward the NVARC/MindsAI recipe (augmentation ensembles, tokenizer dropout).
- E4. Stretch (~26 -> 38+, real research): evolutionary program synthesis with self-training on
  traces (SOAR-style) or a TRM-style recursive module.
Paper framing: "refinement loops, the 2025 frontier, on a single consumer GPU." Realistic target
~24% from E1-E3; 38%+ needs E4 and is a genuine stretch (only frontier models reached it).

### Priority order
1. Prep working. 2. v1 on the board (eligibility). 3. Paper submitted. Those three are the complete,
eligible floor. 4. E1 -> E2 -> E3, resubmitting as it climbs toward ~24%. 5. E4 as the stretch
toward 30%+. 6. C (multi-GPU sharding) only once TTT plus the loop no longer fit one L4 in 12h.

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
