# Pending results and where each one lands

Standing checklist so nothing measured goes unreported. Every row names the artifact to update, not
just "the paper", because the paper, the HTML mirror, the 1500-word writeup and the experiment
matrix drift apart otherwise. Delete a row only when its result is folded in everywhere listed.

## Landed and folded in

| result | value | folded into |
|---|---|---|
| LB33 unmodified control | 28.47 | tex phantom section, HTML, EXPERIMENTS.md, writeup, fig_phantom |
| v7.5a AIF prune | 27.92 | tex escape-route passage, versions table, writeup, EXPERIMENTS.md |
| v7.2 replicate | 30.56 | tex variance passage, EXPERIMENTS.md, fig_version_score, writeup |
| no-dead-AIF | 30.00 | tex variance passage widened to a one-point bar, EXPERIMENTS.md, writeup |
| contamination audit 2x2 | z = 8.0 | tex section, HTML, fig_contamination, writeup, memory |
| invariant filter (A2) | 0 recovered, 11/19 truths killed | score-levers section 2, writeup escape-route passage, memory |
| best-first replay (T1) | median 2.9%, 16 tasks | tex appendix passage, HTML, score-levers, memory |
| pre-fitting prior art | adjacent-negative, not unexplored | docs/testinput-ttt-priorart.md, tex Related Work (TTT name collision, the ARChitects' explicit exclusion of input grids), tex sec:prefit positioning, 6 new bib entries |
| ragged DFS batches | 0/120 eval, 4/24 RE-ARC | tex engineering log, decode_pool split-by-length fix, scripts/rig_batchlen.py |

| replicate 2, sub 55261090 | **32.64**, rank 43 | tex variance passage rewritten around the immutable-kernel pair, abstract, intro, versions table, EXPERIMENTS.md, writeup, fig_version_score band, memory |
| dual-expert on fresh pools | **oracle gap 0 of 18** | tex escape-route ledger, score-levers F3 |
| subset-disagreement TTT | **+0 coverage, but a routing signal** | 20 outputs compared, union of subsets equals the full adapter exactly. The usable half: mean pairwise pool Jaccard is 0.373 where the full adapter found the truth and 0.015 where it did not, a 0.358 separation on n=7 vs n=4. Same-config replicate Jaccard 0.543. Belongs in a new tex passage on underdetermination AND in the budget-policy section, since it is a candidate for the progress signal training loss fails to provide |

## Outstanding

| result | expected | where it must land |
|---|---|---|
| cap 1200, kernel `rc7530/v7-2-cap1200` v1 | submit when the UTC day rolls, score ~9h later | tex appendix cap passage, which currently says the cap is untested; EXPERIMENTS.md "Where the cap question actually stands"; memory. Report against the corrected band, not against a single number |
| perceptual pre-fitting, 3 arms | re-run after subsetttt | tex sec:prefit, replacing the in-flight paragraph with the outcome and scoring the pre-registered null; score-levers section 2 if it is a kill; writeup only if positive |
| subset-disagreement TTT | ~10:00 local, 34/48 cells done | new tex passage on demonstration underdetermination, this is the headline if the number is interesting; writeup only if it displaces something |
| composition learnability, `rig_composelearn.py` | 14 pairs, 4 done at 21:40 on 10 Aug | decides whether the 78,792-example compositional corpus is worth training on. Early and not yet a result: single 4/4, composed 1/4 |
| batch-02 queue, `rig_queue_batch02.py` (pid 12900) | maskloss then dfsfloor then dimprefix, chained behind composelearn | tex sec:ledger rewritten ONCE with the five offline closures and these three, rather than three times; EXPERIMENTS.md; idea-batch-01.md Part 6 |
| five offline closures, 10 Aug | done, no GPU | folded into idea-triage.md Part 2 and idea-batch-01.md Part 6. **Still owed to the paper**: candidate-filter replay, depth-decay dominated by the flat floor, palette canon no-op proof, column-major already in D4, augmentation damage numbers |

## Reading any new hidden score

The family is $\{30.00, 30.56, 30.56, 30.97, 32.64\}$, mean 30.95, sd 1.01. A single submission
resolves nothing below about two points. Before calling any new score a result, state where it sits
in that band, and prefer "not distinguishable" to a directional story. The 30.00 was called a
regression on the old sd of 0.24 and was not one.

## Rules for folding in

Correct in place rather than appending. If a new number falsifies an earlier claim, edit the earlier
claim and say it was corrected, which is what the phantom and cap sections already do.

Update the writeup only when the result changes what a judge should conclude. It is capped at 1500
words and currently sits at 1499, so every addition must displace something.

Recompute derived statistics rather than patching them. The variance passage quotes a range, a
standard deviation and a sigma count for four regressions; all of them move when a family member is
added.

Regenerate figures from their scripts rather than editing images. fig_version_score, fig_phantom and
the two heroes all read their numbers from source and must be re-run.

Section~\ref{sec:prefit} carries a pre-registered prediction of a null. Score it explicitly when the
numbers land, whichever way it goes, rather than quietly deleting it.

## Rig operating notes

The rig's python processes are named `python3.13`, not `python`. `Get-Process python` returns
nothing while a run is perfectly healthy, and reading that as "the queue died" once came within one
command of killing a live job. Use `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`,
which also gives the command line and the parent, so the queue and its child are distinguishable.

The queue streams each phase's output to `experiments/novel_queue.log` line by line. It used to
buffer with `capture_output`, which hid progress for hours and lost the entire log when the queue
was killed mid-phase.
