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
| replicate runs 5 and 6, subs 55365698 and 55400700 | **30.56 and 30.56** | EXPERIMENTS.md table updated. **Still owed**: tex variance passage and abstract, which quote a five-member family; the same-kernel series is now n=6, mean 30.70, sd 1.36, range 4.17, CI on sd [0.85, 3.34] |
| composition learnability, `rig_composelearn.py` | **DONE, 28 records.** single 14/14 truth-in-pool, composed **5/14**; paired composed worse on 9, tied on 5, better on 0; mean final loss 0.0019 against 0.0176 | This is the pre-registered "clearly between" outcome: harder than a single rule and still learnable, which is what a corpus meant to teach composition should look like. It clears the gate on the 78,792-example corpus. Lands in a new tex passage and gates the fine-tune |
| prompt budget, `sim_token_budget.py` | at Kaggle's 8192, **0 of 240 training views and 6 of 344 eval views** lose a demonstration, 6 of 1046 total | kills the whole token-saving family by removing its motivation. Also a methodology note worth the paper: the rig's 6144 loses 18, so every rig probe runs a slightly harder problem than the submission and is biased toward null |
| **unchanged-cell loss masking, `rig_maskloss.py`** | **DONE.** base 6/14 in pool, masked 3/14, random control 6/14. Error on CHANGED cells 0.286 to 0.062; error on COPIED cells 0.024 to 0.095 | The mechanism worked and the idea still lost, with a control that rules out the "less supervision" confound. This deserves its own tex passage: it is the only lever that moved a quantity by 4.6x, and the reason it failed is the 88% copy fraction, which the paper already establishes. idea-batch-01.md Part 7, idea-triage.md Part 2 |
| **DFS floor sweep, `rig_dfsfloor.py`** | **DONE.** floor 0.20 = 6 of 14 outputs, 56 candidates; 0.35 = 5, 48; 0.45 = 5, 43. Worse on 1 task, better on 0, tied on 7. All 24 decodes exhausted the 600s wall clock | closes the floor. Also scores the two offline replays against each other: the POOL replay was right, the tree replay's 13 covered outputs were too small a sample. Belongs in the tex passage on offline replay as a method, since it is a case where two replays of the same system disagreed and the run adjudicated. **Known gap**: batches-completed was not recorded, so we cannot say whether the tighter floor got further and found nothing, or simply needed less |
| batch-02 queue, `rig_queue_batch02.py` (pid 12440) | maskloss DONE, dfsfloor DONE, dimprefix running (pid 15864, 10 of 12) | tex sec:ledger rewritten ONCE with the five offline closures and these three, rather than three times; EXPERIMENTS.md; idea-batch-01.md Part 6 |
| batch-03 queue, `rig_queue_batch03.py` (pid 13040) | **has not started**, still waiting on batch-02. The compositional fine-tune has not run | pilot fine-tune then `rig_ftgate.py` |
| replicate run 10, sub **55424200** | submitted 2026-08-11 05:17 UTC, PENDING, score in ~9h | seventh draw of the same-kernel series. Recompute mean, sd and the CI on sd when it lands; the paper's variance passage quotes them |
| five offline closures, 10 Aug | done, no GPU | folded into idea-triage.md Part 2 and idea-batch-01.md Part 6. **Still owed to the paper**: candidate-filter replay, depth-decay dominated by the flat floor, palette canon no-op proof, column-major already in D4, augmentation damage numbers |

## Reading any new hidden score

Use the same-kernel series, because it is the only one with nothing else varying:
$\{30.56, 32.64, 28.47, 31.39, 30.56, 30.56\}$ from `rc7530/v7-2-replicate` v1, six runs of one
immutable artifact. Mean 30.70, sd 1.36, range **4.17**, 95% CI on the standard deviation
[0.85, 3.34].

A single submission resolves nothing below about three points. Before calling any new score a
result, state where it sits in that band, and prefer "not distinguishable" to a directional story.
The 30.00 was called a regression on an old sd of 0.24 and was not one. The wider family that mixes
one-factor variants in ($\{30.00, 30.97, ...\}$) is still useful for the ledger but is the wrong
reference for judging a new run, since its spread confounds the change with the day.

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

## Submitting from the API

`api.competition_submit_code` needs `kernel_version` and it is **not optional**. Omitting it, or
passing the 0 that `kernels_list` reports as `current_version_number`, returns a bare
`403 Forbidden` with no message. That reads exactly like a permissions failure and it is not one:
`kernel_version=1` submits fine on the same token in the same session. Two attempts were abandoned
and a submission slot nearly lost to that misreading.

The lesson generalises. The Kaggle SDK raises `requests.HTTPError`, whose `str()` shows only the
status line; the server's explanation is in `e.response.text`. Print the body before concluding
anything about permissions. `scripts/rig_submit.py` now does.

## Rig operating notes

The rig's python processes are named `python3.13`, not `python`. `Get-Process python` returns
nothing while a run is perfectly healthy, and reading that as "the queue died" once came within one
command of killing a live job. Use `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`,
which also gives the command line and the parent, so the queue and its child are distinguishable.

The queue streams each phase's output to `experiments/novel_queue.log` line by line. It used to
buffer with `capture_output`, which hid progress for hours and lost the entire log when the queue
was killed mid-phase.
