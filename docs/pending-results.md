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
| batch-03 pilot fine-tune | **DONE, 26.0 hours.** val loss 0.1087 / 0.1047 / 0.0988 / **0.0981** at steps 2k/4k/6k/8k, train 0.0937 at the end. Monotone, train and val together, so it fit the corpus without overfitting | the checkpoint is at `D:\arc\models\comp_pilot\qwen3-4b-arc2ft`. Note the val loss is next-token loss on serialized grids and 88% of a grid is a copy, so it is a weak proxy for whether a rule was learned. The gate decides |
| `rig_ftgate.py` | **DONE, 40 records.** composed: better on 1 task, worse on 0, tied on 7 of 8, candidates +13%. ARC-AGI-2: better on 0, worse on 1, tied on 11 of 12, candidates **-29%** | **The pilot is a null in both directions**, and the first verdict this script printed was wrong: it read the totals (3 to 4, and 5 to 4) and announced both that the fine-tune had learned the rules and that it had cost grid ability, when each rested on ONE task. Script corrected to report paired sign counts. Read with the training curve, which moved val loss only 9.8%: the intervention was too small to test the hypothesis, not evidence the hypothesis is false. The only quantity with size to it is the 29% drop in ARC candidates, and it points toward distribution shift |

### The eleven-day figure was an allocator bug, and it is now about one day

`rig_ftbench.py` plus a traced trainer, 13 Aug. **The full corpus is ~25 hours for one epoch, not
eleven days.** Sequence lengths in this corpus are all different, so every step asked CUDA's caching
allocator for a block size it had never seen, nothing was ever reused, reserved memory climbed to the
card's limit and every step then paid for synchronising frees. Rounding each sequence up to a
multiple of 256, with the padding masked out of both the loss and the attention, leaves at most eight
distinct shapes and the allocator reuses them forever.

| | steps 0-20 | 20-40 | 40-60 | s per 1k tokens | one epoch, 78,792 examples |
|---|---|---|---|---|---|
| unpadded | 4.36 s | 4.15 s | - | 2.9 | 3.8 days |
| **pad to 256** | **1.13 s** | **1.14 s** | **1.17 s** | **0.75** | **1.05 days** |
| pilot as realised | | | | | 10.7 days |

Two hypotheses were wrong on the way and both cost a measurement to kill. **sdpa attention is
slower here than eager**, 966 against 1133 tokens/s, so the "eager is the 13 s/step path" note in
`src/arc/ttt.py` does not apply to this workload. And the card was **not** contended: the decisive
re-test ran with the GPU at 293 MiB and 2% utilisation and was still slow, so the desktop
applications holding contexts were a red herring.

Also measured: the **val pass costs 185.6 s** for its 200 examples, once per `eval_every`. Cheap in
the pilot at four calls, but it is 200 forward passes for one number.

**Verified at full pilot scale.** The same 8,000 examples the pilot took **26.0 hours** to train
finished in **2.42 hours** wall clock, including corpus load, model load and the merge-and-save.
That is 1.06 s/step against the pilot's 11.6, a **10.9x** speedup, and step 299 was still at
0.99-1.21 s so it never degraded. One epoch of the full 78,792-example corpus is therefore
**23.2 hours**, not eleven days.

The checkpoint that run produced is a throwaway: `--eval-every` was set to 100000 to keep the val
pass out of the timing, so `best` stayed at infinity and it saved the final adapter rather than the
best one. A real full run wants `--eval-every` back at something sane, and the val pass is now cheap
enough to afford.

### Two things the pilot already tells us

**It is 27 hours, not the 1 to 3 I estimated.** 12.0 s/step at rank 32, batch 1, 2048 tokens, with the
card at 97.6% of its 16 GB. So the FULL corpus, 78,792 examples, is 9.85x the pilot and lands near
**eleven days** on this hardware. The full fine-tune is not affordable on one 5070 Ti at this
configuration, and that is a finding, not a scheduling note: it is the concrete price of the 16 GB
constraint on the one lever that changes the model. Scaling it needs rented compute or a much
shorter `max_len`, and the token-budget measurement says most sequences are far below 2048 anyway,
so length-bucketed batching is the cheap fix to try first.

**There is no step-0 baseline, so 0.1087 is uninterpretable on its own.** `finetune_base.py` only
evaluates at `eval_every` intervals, so we never measured the stock base's val loss on this corpus.
Train and val are near identical (0.1084 vs 0.1087), which says it is not overfitting, but not how
much it learned. The gate is the decision-relevant comparison and it does not depend on this, so
this is a gap to fill afterwards rather than a reason to restart.
| replicate run 10, sub **55424200** | **28.89.** Series is now n=7: mean 30.44, sd 1.42, range 4.17, CI on sd [0.91, 3.12], CI on mean [29.13, 31.75] | EXPERIMENTS.md done. **Still owed to the tex**: the abstract, intro and variance passage all quote 32.64 as the result. It is the maximum of seven draws from a distribution centred on 30.44, so it should be reported as an order statistic with the band, not as the score |
| five offline closures, 10 Aug | done, no GPU | folded into idea-triage.md Part 2 and idea-batch-01.md Part 6. **Still owed to the paper**: candidate-filter replay, depth-decay dominated by the flat floor, palette canon no-op proof, column-major already in D4, augmentation damage numbers |

## The full compositional fine-tune: it learned, and it cost more than it learned

67,671 examples, one epoch, 20.9 hours, best val loss **0.0719** against the pilot's 0.0981. Unlike
the pilot this moved things, in both directions.

| | coverage | tasks better/worse/tied | sign p | candidates |
|---|---|---|---|---|
| composed rules, no TTT | 3/8 -> **6/8** | 3 / 0 / 5 | 0.25 | 13 -> **39 (+200%)** |
| ARC-AGI-2, no TTT | 5/20 -> **3/20** | 0 / 2 / 10 | 0.50 | 36 -> **19 (-47%)** |
| ARC-AGI-2, under per-task TTT | 6/17 -> **4/17** | 0 / 2 / 8 | 0.50 | 54 -> 54 |

**The corpus teaches what it contains and damages what it does not.** Composed-rule coverage doubles
and candidate generation on composed tasks triples, which settles the pilot's ambiguity: that null
was underpowering, not absence of an effect. On the real target, coverage falls and candidate
generation halves, and **test-time training does not rebuild it** - a fresh rank-256 adapter per task
still lands at 4 of 17 against the stock base's 6.

Read the strengths honestly. Every individual sign test is weak (p = 0.25, 0.50, 0.50). What carries
the result is that three independent measurements point the same way and that the candidate counts,
which have far more samples than the task counts, move by +200% and -47%. **The checkpoint is dead
for submission** and the direction is not: this is textbook catastrophic forgetting on a narrow
distribution, and the standard remedy, mixing the composed corpus with the base's own distribution
rather than training on composed data alone, was never tried. That is the obvious next experiment
and it costs the same 23 hours.

## Unchanged-cell loss masking, topped up to 12 tasks

base 8 of 20 outputs, 74 candidates, copy error 0.019, change error 0.222. Masked arm worse on 3
tasks, tied on 9, better on 0. **The random control is tied on all 12**, up from all 8, which is the
half that matters: the confound stays excluded on a larger sample. The four new tasks were all ties,
so the sign test is unchanged at 3 against 0.

## Does the variance generalise beyond one artifact? Partly answered

**ANSWERED.** Three runs of `scriptVersionId=337769272`, the v7.2 build that had scored 30.97 once:
it went **30.56** then **28.47**.

| script version | runs | mean | sd | range |
|---|---|---|---|---|
| 339772104, the replicate | 7 | 30.44 | **1.42** | 4.17 |
| 337769272, the v7.2 build | 3 | 30.00 | **1.34** | 2.50 |
| pooled | 10 | 30.31 | **1.34** | 4.17 |

Variance ratio **F = 1.12**. Two separately built artifacts of the same configuration reproduce the
run-to-run spread to two significant figures, so the standard deviation is a property of the
pipeline and not of the one kernel we hammered. 95% CI on the pooled sd $[0.92, 2.44]$. Folded into
the tex as Table~\ref{tab:twoartifacts}.

It also kills an intermediate reading we held for a day. After two draws that artifact showed a gap
of 0.41 and looked quieter; two draws from sd 1.42 differ by more than that 84% of the time, and the
third came in at 28.47. **A pair of runs cannot measure a spread**, which is the same lesson the
three-run estimate taught, arriving a second time by a different route.

28.47 is 1.37 sd below the pooled mean, from code identical to the run that scored 30.97. It is an
ordinary draw and specifically not a regression.

One oddity recorded rather than interpreted: across all ten draws, 30.56 appears 4 times and 28.47
twice, with everything else once. That is more repetition than a smooth distribution suggests, and
ten draws is far too few to claim it.

## Reading any new hidden score

Use the same-kernel series, because it is the only one with nothing else varying:
$\{30.56, 32.64, 28.47, 31.39, 30.56, 30.56, 28.89\}$ from `rc7530/v7-2-replicate` v1, seven runs of
one immutable artifact. **Mean 30.44, sd 1.42, range 4.17**, 95% CI on the sd [0.91, 3.12].

A single submission resolves nothing below about three points, and a properly powered comparison
needs about **3.4 points of separation with three runs per arm**. Before calling any new score a
result, state where it sits in that band, and prefer "not distinguishable" to a directional story.
The 30.00 was called a regression on an old sd of 0.24 and was not one. The wider family that mixes
one-factor variants in ($\{30.00, 30.97, ...\}$) is still useful for the ledger but is the wrong
reference for judging a new run, since its spread confounds the change with the day.

**And stop quoting 32.64 as the result.** It is the maximum of seven draws from a distribution
centred on 30.44. The honest headline is 30.4 plus or minus 1.4.

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

## What the score variance is actually made of

Pulled the per-submission `total_bytes` for all 28 submissions, 15 Aug. It is the only per-run
quantity the Kaggle API exposes and it turns out to be an instrument.

Across the ten same-configuration draws, **submission size and score correlate at r = +0.64**, and
the relationship holds independently inside each artifact (+0.61 on seven runs, +0.80 on three). The
size spread is 25.5 kB on a 778 kB file; at ~2.8 kB per output that is about **nine outputs' worth of
content**, and nine outputs is 3.75 points against an observed range of 4.17.

An empty pool is written out as the `[[0]]` fallback, a 1x1 grid, which is tiny and certainly wrong.
So a slow session produces more empty pools, a smaller file and a lower score together. **The
variance is substantially throughput**, and the whole observed range is accounted for by fewer than a
dozen tasks producing no answer at all.

This refines rather than contradicts the tail-dropping refutation: that simulation was about the
GLOBAL twelve-hour wall, which does finish with 14% slack. The deadlines that bite are the per-task
1800s and the 540s inside a DFS call.

It also partly displaces the coin-flip account. sd 1.34 is consistent with ~41 marginal outputs,
which matches the measured 12.5% same-seed pool disagreement almost exactly, and that effect is real.
But size correlation says a large share is the coarser thing: not which answer a task gave but
whether it gave one. We cannot apportion them, because the API exposes no per-submission log.

**The actionable consequence points opposite to the obvious one.** A CHEAPER search should raise the
mean and cut the variance together by converting empty pools into answers on slow sessions. The rig
floor sweep found no gain from tightening at a fixed 600s budget, but that is not a contest worker at
1800s with a variable machine underneath it. Testing it properly is three runs per arm, six days.

## Mixing the corpus: encouraging, and confounded

`rig_queue_batch06.py`, 15 Aug. 8,000 steps on 4,000 composed interleaved with 4,000 single-rule,
2.3 hours training plus a 2.2-hour ft-arm-only gate. Compared at **equal compute** against the pilot,
which spent its 8,000 steps on composed data alone.

| training | composed | ARC-AGI-2 | ARC candidates |
|---|---|---|---|
| pilot, pure composed, 8k steps | +1 task, +13% cands | **-1 task** | **-29%** |
| full, pure composed, 67.7k steps | +3 tasks, +200% cands | **-2 tasks** | **-47%** |
| **mixed 50/50, 8k steps** | +1 task, +8% cands | **0 tasks** | **0%**, 36 to 36 |

The damage is gone and the (weak) composed gain survives, which is what the anti-forgetting
hypothesis predicts.

**The confound, which is not small.** The mixed run contains 4,000 composed examples where the pilot
had 8,000, so "mixing prevented the damage" is not separated from "half the composed exposure did
half the damage". Two further signs point at the second reading: 7 of 12 ARC tasks produced a
byte-identical candidate count, and the total is exactly 36 to 36, which is what a model that barely
moved would look like rather than one that moved and was protected.

**The decisive run isolates it**: 8,000 composed, matching the pilot exactly, plus 8,000 single-rule,
so composed exposure is held fixed and only the protective half is added. 16,000 steps is about 4.5
hours plus a 2.2-hour gate. If ARC is preserved there, mixing is confirmed.

Do not compare the val losses across these runs. The pilot's 0.0981 is on the composed val set and
the mixed run's 0.0601 is on a mixed val set that is half single-rule, so the difference is mostly
the easier half being averaged in.

## The compositional direction, closed by a complete 2x2

The confound is resolved: at MATCHED composed exposure, adding protective data prevents the damage.
It was the mixing, not the reduced exposure.

| run | composed | protective | ARC-AGI-2 | ARC cands | composed | composed cands |
|---|---|---|---|---|---|---|
| pilot | 8k | 0 | 4/20 (**-1**) | **-29%** | 4/8 (+1) | +13% |
| full | 67.7k | 0 | 3/20 (**-2**) | **-47%** | 6/8 (+3) | **+200%** |
| mixed | 4k | 4k | 5/20 (0) | 0% | 4/8 (+1) | +8% |
| **mixed** | **8k** | **8k** | **5/20 (0)** | **-6%** | 4/8 (+1) | -15% |

Base is 5/20 on ARC and 3/8 on composed in every row.

**And the direction is closed anyway, for a reason the confound was hiding.** No configuration at any
scale has ever put ARC coverage above the base's 5/20. Best case is neutral. The large composed gain
needed the unmixed full corpus, which is precisely the run that damaged ARC, and protecting ARC
also removes the gain: composed candidates went from +200% unmixed to **-15%** mixed at 8k exposure.

The one untested cell is full-scale mixed, 67.7k composed plus 67.7k protective, about 135,000 steps
and **41 hours**. Its best plausible outcome is "ARC unchanged, composed better", and composed is our
own synthetic benchmark rather than the target. That is not a score, so the cell is not worth 41
hours unless something else motivates it.

## no-dead-AIF, two draws

30.00 and 29.03, mean 29.52 against the banked control's 30.31 (n=10, sd 1.34). Difference -0.79,
se 1.04, **t = -0.77, not distinguishable**, and both draws sit below the control mean.

A third draw narrows the interval to about +/-0.88 and the resolution at n=3 is 2.5 points, so it
cannot make a -0.8 effect significant either way. Stopping at n=2 IS optional stopping and should be
reported as such: **n=2, not distinguishable, point estimate slightly negative.** The slot is better
spent on v7.3, which has a live mechanism.

## Operational: not every past build is resubmittable

Discovered 16 Aug trying to re-test v7.3. `competition_submit_code` returns
**400 FAILED_PRECONDITION, "Did not find provided Notebook Output File"** for
`rc7530/arc-agi-2-v7-3-and-onwards-26-07-26` at BOTH version 1 and version 2, so neither v7.3
(338070403, 28.61) nor v7.4 (338271550, 29.44) can be re-run from the existing kernel. Kaggle no
longer holds an output for it. A 400 creates nothing, so the failed attempts cost no slot.

Still resubmittable, all verified by landing on the intended `scriptVersionId`:
`v7-2-replicate` v1, `arc-agi-2-after-import-bug-25-07-26` v1,
`v7-2-without-the-discarded-aif-work` v2.

**Consequence for the re-test plan.** Of the five builds rejected on a single hidden score, only
no-dead-AIF is directly re-testable. v7.3, v7.4 and cap1200 would each need rebuilding with
`build_v7_from_architects.py` and pushing as a new kernel. Pushing costs no interactive quota, so
that is affordable, but it means the greedy-fallback test is a build job rather than a submission.

Given that, today's slot went to **no-dead-AIF draw 3 (sub 55543143)**, which completes the arm at
the pre-registered n=3 instead of stopping at two. That was the honest thing to do with a slot that
had no better use, and it removes the optional-stopping caveat from the record.

## Pushing a new submission kernel: three ways it fails silently

Building v7.6 (greedy fallback isolated) found three, all of which let `kernels_push` report success:

1. **`title` must slug to `id`.** "ARC-AGI-2 v7.6 greedy fallback only" created
   `arc-agi-2-v7-6-greedy-fallback-only`, not the id we asked for, so we were editing a kernel we had
   not named.
2. **`model_sources` is validated but not enforced.** `mirzamilanfarabi/qwen3-4b-grids15-sft139/...`
   comes back in `invalidModelSources` and the push still succeeds, leaving a kernel with no model
   whose run dies at the model-path assert. The working string, from `v7-2-replicate`'s own metadata,
   is **`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`**, capital T. The memory file
   asserted the opposite and has been corrected.
3. **`machine_shape` must be `NvidiaL4`.** Omitted, the kernel gets generic `Gpu` and Kaggle may
   allocate a **Tesla P100 (sm_60)**, where torch 2.8+cu128 fails with *"no kernel image is available
   for execution on the device"* since it supports sm_70 and up. The scoring rerun uses L4s and would
   have been fine; it is the COMMIT run that dies, and a kernel whose commit fails has no output, so
   `competition_submit_code` then returns 400 FAILED_PRECONDITION.

That third one also explains why v7.3 and v7.4 are unreachable: no retained output, same precondition.

`push_v76.py` now asserts on `invalidModelSources` and pins `machine_shape`, and the three failure
modes are recorded here because each cost a push cycle and none of them raises.

**Resolved.** With `machine_shape: NvidiaL4` pinned, the v7.6 commit run completed in 24 minutes
and `rc7530/arc-agi-2-v7-6-greedy-fallback-only` is submittable. It is queued for the next slot as
draw 1 of 3 against the banked n=10 control.
