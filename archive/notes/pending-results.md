# Pending results and where each one goes

This is a standing checklist so nothing we measure goes unreported. Every row names the file to update,
not just "the paper", because otherwise the paper, the HTML mirror, the 1,500-word writeup and the
experiment matrix drift apart. A row only comes off once its result is folded in everywhere it lists.

## Landed and folded in

| result | value | folded into |
|---|---|---|
| LB33 unmodified control | 28.47 | the phantom section of the tex, the HTML, EXPERIMENTS.md, the writeup, fig_phantom |
| v7.5a AIF prune | 27.92 | the escape-route passage in the tex, the versions table, the writeup, EXPERIMENTS.md |
| v7.2 replicate | 30.56 | the variance passage in the tex, EXPERIMENTS.md, fig_version_score, the writeup |
| no-dead-AIF | 30.00 | the variance passage, widened to a one-point bar, EXPERIMENTS.md, the writeup |
| contamination audit, 2x2 | z = 8.0 | its own tex section, the HTML, fig_contamination, the writeup, memory |
| invariant filter (A2) | recovered 0, killed 11 of 19 truths | section 2 of score-levers, the escape-route passage of the writeup, memory |
| best-first replay (T1) | median 2.9%, 16 tasks | an appendix passage in the tex, the HTML, score-levers, memory |
| prior art on pre-fitting | next to a negative result, not unexplored | docs/testinput-ttt-priorart.md, Related Work in the tex (the clash over the name TTT, and the ARChitects' explicit exclusion of input grids), how sec:prefit is positioned, 6 new bib entries |
| ragged DFS batches | 0 of 120 on eval, 4 of 24 on RE-ARC | the engineering log in the tex, the fix in decode_pool that splits batches by length, scripts/rig_batchlen.py |
| replicate 2, sub 55261090 | **32.64**, rank 43 | the variance passage rewritten around the pair of runs on the unchanged kernel, the abstract, the introduction, the versions table, EXPERIMENTS.md, the writeup, the band in fig_version_score, memory |
| two experts on fresh pools | **oracle gap of 0 out of 18** | the escape-route ledger in the tex, F3 in score-levers |
| TTT on subsets, reading disagreement | **no gain in coverage, but a routing signal** | 20 outputs compared, and the union of the subsets equals the full adapter exactly. The useful half is that the mean pairwise Jaccard between pools is 0.373 where the full adapter found the truth and 0.015 where it didn't, a gap of 0.358 on 7 tasks against 4. Replicates of the same configuration give a Jaccard of 0.543. It belongs in a new tex passage on underdetermination and also in the section on budget policy, since it could be the progress signal that training loss fails to give |

## Outstanding

| result | expected | where it has to go |
|---|---|---|
| cap 1200, kernel `rc7530/v7-2-cap1200` v1 | submit when the UTC day rolls over, score about 9 hours later | the cap passage in the tex appendix, which still says the cap is untested, the part of EXPERIMENTS.md called "Where the cap question actually stands", and memory. Report it against the corrected band, not against a single number |
| perceptual pre-fitting, 3 arms | run again after subsetttt | sec:prefit in the tex, replacing the paragraph about work in progress with the outcome and scoring the pre-registered null, section 2 of score-levers if it's a kill, and the writeup only if it's positive |
| TTT on subsets, reading disagreement | about 10:00 local time, 34 of 48 cells done | a new tex passage on demonstrations that don't pin down the rule. This is the headline if the number is interesting. The writeup only if it pushes something out |
| replicate runs 5 and 6, subs 55365698 and 55400700 | **30.56 and 30.56** | EXPERIMENTS.md table updated. **Still owed** to the variance passage and abstract in the tex, which quote a family of five. The series on the same kernel is now 6 runs, mean 30.70, sd 1.36, range 4.17, with a confidence interval on the sd of 0.85 to 3.34 |
| whether composition can be learned, `rig_composelearn.py` | **Done, 28 records.** With single rules the truth was in the pool on 14 of 14, and with composed rules on **5 of 14**. Paired, composed was worse on 9, tied on 5 and better on 0. Mean final loss 0.0019 against 0.0176 | This is the pre-registered outcome of "clearly in between", harder than a single rule and still learnable, which is what a corpus meant to teach composition should look like. It clears the gate on the corpus of 78,792 examples. It goes in a new tex passage and gates the fine-tune |
| prompt budget, `sim_token_budget.py` | at Kaggle's 8,192 tokens, **0 of 240 training views and 6 of 344 eval views** lose a demonstration, 6 of 1,046 in all | This kills the whole family of token-saving ideas by taking away their reason. It's also a point about method worth putting in the paper. The rig's limit of 6,144 loses 18, so every probe on the rig solves a slightly harder problem than the submission does and leans toward a null |
| **masking the loss on unchanged cells, `rig_maskloss.py`** | **Done.** Base had 6 of 14 in the pool, masked 3 of 14, and a random control 6 of 14. Error on changed cells went from 0.286 to 0.062 and error on copied cells from 0.024 to 0.095 | The mechanism worked and the idea still lost, with a control that rules out the explanation of less supervision. It deserves its own tex passage. It's the only lever that moved a quantity by 4.6 times, and it failed because 88% of cells are copied, which the paper already shows. idea-batch-01.md Part 7, idea-triage.md Part 2 |
| **sweep of the DFS floor, `rig_dfsfloor.py`** | **Done.** A floor of 0.20 gave 6 of 14 outputs and 56 candidates, 0.35 gave 5 and 48, and 0.45 gave 5 and 43. Worse on 1 task, better on 0, tied on 7. All 24 decodes ran out the 600 second wall clock | This closes the floor. It also settles which of the two offline replays was right. The replay of the pool was right, and the 13 outputs the tree replay covered were too small a sample. It belongs in the tex passage on offline replay as a method, as a case where two replays of the same system disagreed and the real run decided. **A known gap** is that the number of batches completed wasn't recorded, so we can't tell whether the tighter floor got further and found nothing or just needed less |
| batch-02 queue, `rig_queue_batch02.py` (pid 12440) | maskloss done, dfsfloor done, dimprefix running (pid 15864, 10 of 12) | rewrite sec:ledger in the tex once, with the five offline closures and these three, rather than three times. EXPERIMENTS.md. idea-batch-01.md Part 6 |
| batch-03 pilot fine-tune | **Done, 26.0 hours.** Val loss 0.1087, 0.1047, 0.0988 and **0.0981** at steps 2k, 4k, 6k and 8k, with train at 0.0937 at the end. It fell steadily and train and val moved together, so it fit the corpus without overfitting | The checkpoint is at `D:\arc\models\comp_pilot\qwen3-4b-arc2ft`. The val loss is next-token loss on serialized grids, and 88% of a grid is a copy, so it says little about whether a rule was learned. The gate decides |
| `rig_ftgate.py` | **Done, 40 records.** On composed tasks it was better on 1, worse on 0 and tied on 7 of 8, with 13% more candidates. On ARC-AGI-2 it was better on 0, worse on 1 and tied on 11 of 12, with **29% fewer** candidates | **The pilot is a null both ways**, and the first verdict this script printed was wrong. It read the totals (3 to 4, and 5 to 4) and announced both that the fine-tune had learned the rules and that it had cost grid ability, when each rested on one task. The script now reports paired sign counts. Read alongside the training curve, which moved val loss by only 9.8%, the intervention was too small to test the idea, which isn't evidence the idea is wrong. The only number with any size to it is the 29% drop in ARC candidates, and that points to a shift in distribution |

### The eleven-day figure was a bug in the allocator, and it's really about one day

Measured on 13 August with `rig_ftbench.py` and a traced trainer. **The full corpus takes about 25 hours
for one epoch, not eleven days.** Every sequence in this corpus has a different length, so every step
asked CUDA's caching allocator for a block size it had never seen. Nothing got reused, reserved memory
climbed to the card's limit, and from then on every step paid for synchronizing frees. Rounding each
sequence up to a multiple of 256, with the padding masked out of both the loss and the attention, leaves
at most eight distinct shapes, and the allocator reuses them forever.

| | steps 0 to 20 | 20 to 40 | 40 to 60 | seconds per 1k tokens | one epoch, 78,792 examples |
|---|---|---|---|---|---|
| no padding | 4.36 s | 4.15 s | - | 2.9 | 3.8 days |
| **padded to 256** | **1.13 s** | **1.14 s** | **1.17 s** | **0.75** | **1.05 days** |
| the pilot as it actually ran | | | | | 10.7 days |

Two guesses were wrong along the way, and each one took a measurement to kill. **sdpa attention is
slower here than eager**, at 966 tokens a second against 1,133, so the note in `src/arc/ttt.py` that
eager is the 13 second per step path doesn't apply to this workload. And the card **wasn't** being
shared. The deciding re-test ran with the GPU at 293 MiB and 2% use and was still slow, so the desktop
apps holding contexts were a red herring.

We also measured that **the val pass costs 185.6 seconds** for its 200 examples, once per `eval_every`.
That's cheap in the pilot at four calls, but it's 200 forward passes for one number.

**Checked at the full pilot's scale.** The same 8,000 examples the pilot took **26.0 hours** to train
finished in **2.42 hours** of wall clock, counting loading the corpus, loading the model, and merging and
saving. That's 1.06 seconds a step against the pilot's 11.6, a **10.9 times** speedup, and step 299 was
still at 0.99 to 1.21 seconds, so it never slowed down. One epoch of the full corpus of 78,792 examples
is therefore **23.2 hours**, not eleven days.

The checkpoint from that run is a throwaway. `--eval-every` was set to 100000 to keep the val pass out of
the timing, so `best` stayed at infinity and it saved the final adapter rather than the best one. A real
full run should put `--eval-every` back at something sensible, and the val pass is now cheap enough to
afford.

### Two things the pilot already told us

These two points were written before the padding fix above and are kept as they were.

**It took 27 hours, not the 1 to 3 I'd guessed.** That's 12.0 seconds a step at rank 32, batch size 1 and
2,048 tokens, with the card at 97.6% of its 16 GB. The full corpus of 78,792 examples is 9.85 times the
pilot, which comes to about **eleven days** on this hardware. So the full fine-tune can't be afforded on
one 5070 Ti in this configuration, and that's a finding, not a scheduling note. It's the concrete price
of the 16 GB limit on the one lever that changes the model. Scaling it up needs rented compute or a much
shorter `max_len`, and the token budget measurement says most sequences are far under 2,048 anyway, so
batching by length is the cheap fix to try first.

**There's no baseline at step 0, so 0.1087 means nothing on its own.** `finetune_base.py` only evaluates
every `eval_every` steps, so we never measured the stock base's val loss on this corpus. Train and val
are almost the same (0.1084 and 0.1087), which says it isn't overfitting but not how much it learned.
The gate is the comparison that matters for the decision and it doesn't depend on this, so this is a gap
to fill later rather than a reason to start over.

| result | value | where it has to go |
|---|---|---|
| replicate run 10, sub **55424200** | **28.89.** The series is now 7 runs, mean 30.44, sd 1.42, range 4.17, confidence interval on the sd 0.91 to 3.12, and on the mean 29.13 to 31.75 | EXPERIMENTS.md is done. **Still owed to the tex.** The abstract, introduction and variance passage all quote 32.64 as the result. It's the highest of seven draws from a distribution centred on 30.44, so it should be reported as the top of a range with the band, not as the score |
| five offline closures, 10 Aug | done, no GPU | folded into idea-triage.md Part 2 and idea-batch-01.md Part 6. **Still owed to the paper**, the replay of the candidate filter, decay with depth being beaten by the flat floor, the proof that canonical palettes do nothing, column-major order already being in D4, and the numbers on damage from augmentation |

## The full compositional fine-tune learned something and cost more than it learned

It ran on 67,671 examples for one epoch, taking 20.9 hours, with a best val loss of **0.0719** against
the pilot's 0.0981. Unlike the pilot, this one moved things, in both directions.

| | coverage | tasks better, worse, tied | sign test p | candidates |
|---|---|---|---|---|
| composed rules, no TTT | 3 of 8 to **6 of 8** | 3, 0, 5 | 0.25 | 13 to **39 (+200%)** |
| ARC-AGI-2, no TTT | 5 of 20 to **3 of 20** | 0, 2, 10 | 0.50 | 36 to **19 (-47%)** |
| ARC-AGI-2, with per-task TTT | 6 of 17 to **4 of 17** | 0, 2, 8 | 0.50 | 54 to 54 |

**The corpus teaches what's in it and damages what isn't.** Coverage of composed rules doubles and the
candidates generated on composed tasks triple, which settles the pilot's ambiguity. That null came from
too little power, not from there being no effect. On the real target, coverage falls and the number of
candidates halves, and **test-time training doesn't bring it back**. A fresh rank-256 adapter per task
still ends up at 4 of 17 against the stock base's 6.

The strength of this evidence should be read honestly. Each sign test on its own is weak (p = 0.25, 0.50
and 0.50). What carries the result is that three independent measurements point the same way, and that
the candidate counts, which rest on far more samples than the task counts, move by +200% and -47%.
**The checkpoint is dead for submission**, but the direction isn't. This is textbook catastrophic
forgetting on a narrow distribution, and the standard fix, mixing the composed corpus with the base's
own kind of data instead of training on composed data alone, was never tried. That's the obvious next
experiment, and it costs the same 23 hours.

## Masking the loss on unchanged cells, topped up to 12 tasks

The base got 8 of 20 outputs and 74 candidates, with an error of 0.019 on copied cells and 0.222 on
changed ones. The masked arm was worse on 3 tasks, tied on 9 and better on 0. **The random control is
tied on all 12**, up from all 8, and that's the half that matters, because the confound stays ruled out
on a bigger sample. The four new tasks were all ties, so the sign test is still 3 against 0.

## Does the variance hold beyond one artifact? Partly answered

**Answered.** We ran `scriptVersionId=337769272`, the v7.2 build that had scored 30.97 once, three
times. It went **30.56** and then **28.47**.

| script version | runs | mean | sd | range |
|---|---|---|---|---|
| 339772104, the replicate | 7 | 30.44 | **1.42** | 4.17 |
| 337769272, the v7.2 build | 3 | 30.00 | **1.34** | 2.50 |
| pooled | 10 | 30.31 | **1.34** | 4.17 |

The variance ratio is **F = 1.12**. Two separately built artifacts of the same configuration reproduce
the spread between runs to two significant figures, so the standard deviation belongs to the pipeline
and not to the one kernel we ran over and over. The 95% confidence interval on the pooled sd is 0.92 to
2.44. This went into the tex as the table tab:twoartifacts.

It also kills a reading we held for a day in between. After two draws that artifact showed a gap of
0.41 and looked quieter. But two draws from an sd of 1.42 differ by more than that 84% of the time, and
the third came in at 28.47. **Two runs can't measure a spread**, which is the same lesson the estimate
from three runs taught, arriving a second time by another route.

28.47 is 1.37 sd below the pooled mean, from code identical to the run that scored 30.97. It's an
ordinary draw and specifically not a regression.

One oddity is worth recording without reading anything into it. Across all ten draws, 30.56 shows up 4
times and 28.47 twice, with everything else once. That's more repetition than a smooth distribution
would suggest, and ten draws is far too few to claim anything.

## How to read any new hidden score

Use the series on the same kernel, because it's the only one where nothing else varies. That's 30.56,
32.64, 28.47, 31.39, 30.56, 30.56 and 28.89 from `rc7530/v7-2-replicate` v1, seven runs of one unchanging
artifact. **Mean 30.44, sd 1.42, range 4.17**, with a 95% confidence interval on the sd of 0.91 to 3.12.

A single submission settles nothing below about three points, and a properly powered comparison needs
about **3.4 points of separation with three runs per arm**. Before calling any new score a result, say
where it sits in that band, and prefer "can't be told apart" to a story about direction. The 30.00 was
called a regression against an old sd of 0.24, and it wasn't one. The wider family that mixes in
variants differing by one factor (30.00, 30.97 and so on) is still useful for the ledger, but it's the
wrong reference for judging a new run, because its spread mixes up the change with the day.

**And stop quoting 32.64 as the result.** It's the highest of seven draws from a distribution centred on
30.44. The honest headline is 30.4 give or take 1.4.

## Rules for folding results in

Correct things where they are instead of adding to the end. If a new number disproves an earlier claim,
edit the earlier claim and say it was corrected, the way the phantom and cap sections already do.

Only update the writeup when a result changes what a judge should conclude. It's capped at 1,500 words
and sits at 1,499, so anything added has to push something out.

Recompute derived statistics instead of patching them. The variance passage quotes a range, a standard
deviation and a count of sigmas for four regressions, and all of them move when a run is added to the
family.

Regenerate figures from their scripts instead of editing the images. fig_version_score, fig_phantom and
the two hero images all read their numbers from source and have to be run again.

The section sec:prefit carries a pre-registered prediction of a null. Score it openly when the numbers
come in, whichever way it goes, instead of quietly deleting it.

## Submitting through the API

`api.competition_submit_code` needs `kernel_version`, and it's **not optional**. Leaving it out, or
passing the 0 that `kernels_list` reports as `current_version_number`, gets a bare `403 Forbidden` with
no message. That looks exactly like a permissions failure, and it isn't one. `kernel_version=1` submits
fine with the same token in the same session. Two attempts were given up and a submission slot nearly
lost because of that misreading.

The lesson is general. The Kaggle SDK raises `requests.HTTPError`, and its `str()` only shows the status
line. The server's explanation is in `e.response.text`. Print the body before concluding anything about
permissions. `scripts/rig_submit.py` now does.

## Notes on running the rig

The rig's Python processes are called `python3.13`, not `python`. `Get-Process python` returns nothing
while a run is perfectly healthy, and once, reading that as the queue having died came within one
command of killing a live job. Use `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`, which
also gives the command line and the parent, so the queue and its child can be told apart.

The queue now streams each phase's output to `experiments/novel_queue.log` line by line. It used to
buffer with `capture_output`, which hid progress for hours and lost the whole log when the queue was
killed in the middle of a phase.

## What the score variance is actually made of

On 15 August we pulled `total_bytes` for all 28 submissions. It's the only per-run quantity the Kaggle
API gives out, and it turns out to be a useful instrument.

Across the ten draws of the same configuration, **submission size and score correlate at r = +0.64**,
and the relationship holds separately inside each artifact (+0.61 over seven runs and +0.80 over three).
Sizes spread by 25.5 kB on a file of 778 kB. At about 2.8 kB per output, that's about **nine outputs'
worth of content**, and nine outputs is 3.75 points, against an observed range of 4.17.

An empty pool gets written out as the fallback `[[0]]`, a 1 by 1 grid, which is tiny and certainly
wrong. So a slow session gives more empty pools, a smaller file and a lower score all together. **The
variance is largely throughput**, and the whole observed range can be explained by fewer than a dozen
tasks producing no answer at all.

This refines the earlier refutation of tail-dropping rather than contradicting it. That simulation was
about the global twelve-hour limit, which really does finish with 14% to spare. The deadlines that bite
are the 1,800 seconds per task and the 540 seconds inside a DFS call.

It also partly replaces the coin-flip account. An sd of 1.34 fits about 41 outputs on the edge, which
matches the measured 12.5% disagreement between pools with the same seed almost exactly, and that effect
is real. But the correlation with size says much of it is cruder than that. It's not which answer a task
gave but whether it gave one at all. We can't split the two, because the API gives no log per
submission.

**What to do about it runs opposite to the obvious idea.** A cheaper search should raise the mean and
cut the variance together, by turning empty pools into answers on slow sessions. The floor sweep on the
rig found nothing from tightening at a fixed budget of 600 seconds, but that isn't a contest worker with
1,800 seconds and a varying machine underneath. Testing it properly takes three runs per arm, which is
six days.

## Mixing the corpus is encouraging, but confounded

On 15 August, `rig_queue_batch06.py` ran 8,000 steps on 4,000 composed examples interleaved with 4,000
single-rule ones, taking 2.3 hours of training plus a 2.2 hour gate on the fine-tuned arm only. It's
compared at **equal compute** with the pilot, which spent all its 8,000 steps on composed data.

| training | composed | ARC-AGI-2 | ARC candidates |
|---|---|---|---|
| pilot, composed only, 8k steps | +1 task, +13% candidates | **-1 task** | **-29%** |
| full, composed only, 67.7k steps | +3 tasks, +200% candidates | **-2 tasks** | **-47%** |
| **mixed half and half, 8k steps** | +1 task, +8% candidates | **0 tasks** | **0%**, 36 to 36 |

The damage is gone and the weak gain on composed tasks survives, which is what the idea of preventing
forgetting predicts.

**The confound isn't small.** The mixed run has 4,000 composed examples where the pilot had 8,000, so
"mixing prevented the damage" can't be told apart from "half the composed data did half the damage".
Two more signs point to the second reading. 7 of 12 ARC tasks produced exactly the same number of
candidates, and the total is exactly 36 to 36, which looks like a model that barely moved rather than one
that moved and was protected.

**The deciding run separates them.** Use 8,000 composed examples, exactly matching the pilot, plus 8,000
single-rule ones, so composed exposure stays the same and only the protective half is added. 16,000 steps
is about 4.5 hours plus a 2.2 hour gate. If ARC holds up there, mixing is confirmed.

Don't compare val losses across these runs. The pilot's 0.0981 is on the composed val set, and the mixed
run's 0.0601 is on a mixed val set that's half single-rule, so most of the difference is the easier half
being averaged in.

## The compositional direction, closed by a complete 2x2

The confound is settled. With composed exposure matched, adding protective data prevents the damage, so
it was the mixing and not the smaller exposure.

| run | composed | protective | ARC-AGI-2 | ARC candidates | composed | composed candidates |
|---|---|---|---|---|---|---|
| pilot | 8k | 0 | 4 of 20 (**-1**) | **-29%** | 4 of 8 (+1) | +13% |
| full | 67.7k | 0 | 3 of 20 (**-2**) | **-47%** | 6 of 8 (+3) | **+200%** |
| mixed | 4k | 4k | 5 of 20 (0) | 0% | 4 of 8 (+1) | +8% |
| **mixed** | **8k** | **8k** | **5 of 20 (0)** | **-6%** | 4 of 8 (+1) | -15% |

The base is 5 of 20 on ARC and 3 of 8 on composed tasks in every row.

**And the direction is closed anyway, for a reason the confound was hiding.** No configuration at any
scale has ever pushed ARC coverage above the base's 5 of 20. The best case is no change. The big gain on
composed tasks needed the full unmixed corpus, which is exactly the run that damaged ARC, and protecting
ARC also takes away the gain. Composed candidates went from +200% unmixed to **-15%** mixed at 8k
exposure.

The one cell left untested is mixed at full scale, 67.7k composed plus 67.7k protective, about 135,000
steps and **41 hours**. The best it could plausibly do is leave ARC unchanged and improve composed tasks,
and composed tasks are our own synthetic benchmark, not the target. That's not a score, so the cell isn't
worth 41 hours unless something else gives a reason to run it.

## no-dead-AIF, two draws

The draws were 30.00 and 29.03, a mean of 29.52 against the banked control's 30.31 (10 runs, sd 1.34).
The difference is -0.79 with a standard error of 1.04, so **t = -0.77, and the two can't be told apart**.
Both draws sit below the control mean.

A third draw narrows the interval to about plus or minus 0.88, and three runs resolve 2.5 points, so it
can't make an effect of -0.8 significant either way. Stopping at two draws is optional stopping and
should be reported that way, as **two draws, can't be told apart, point estimate slightly negative**. The
slot is better spent on v7.3, which has a live mechanism.

## Not every past build can be submitted again

We found this on 16 August trying to re-test v7.3. `competition_submit_code` returns **400
FAILED_PRECONDITION, "Did not find provided Notebook Output File"** for
`rc7530/arc-agi-2-v7-3-and-onwards-26-07-26` at both version 1 and version 2, so neither v7.3 (338070403,
28.61) nor v7.4 (338271550, 29.44) can be run again from the existing kernel. Kaggle no longer keeps an
output for it. A 400 creates nothing, so the failed attempts didn't cost a slot.

These can still be resubmitted, each checked by landing on the intended `scriptVersionId`.
`v7-2-replicate` v1, `arc-agi-2-after-import-bug-25-07-26` v1 and
`v7-2-without-the-discarded-aif-work` v2.

**What it means for the plan to re-test.** Of the five builds rejected on a single hidden score, only
no-dead-AIF can be re-tested directly. v7.3, v7.4 and cap1200 would each have to be rebuilt with
`build_v7_from_architects.py` and pushed as a new kernel. Pushing doesn't use interactive quota, so it's
affordable, but it makes the test of the greedy fallback a build job rather than just a submission.

So today's slot went to **the third draw of no-dead-AIF (sub 55543143)**, which completes the arm at the
pre-registered three instead of stopping at two. That was the honest use of a slot with nothing better
to do, and it takes the optional-stopping caveat off the record.

## Three ways pushing a new submission kernel fails without saying so

Building v7.6, with the greedy fallback on its own, turned up three, and in each case `kernels_push`
reported success.

1. **`title` has to slug to `id`.** "ARC-AGI-2 v7.6 greedy fallback only" created
   `arc-agi-2-v7-6-greedy-fallback-only`, not the id we asked for, so we were editing a kernel we hadn't
   named.
2. **`model_sources` is checked but not enforced.** `mirzamilanfarabi/qwen3-4b-grids15-sft139/...` comes
   back in `invalidModelSources` and the push still succeeds, leaving a kernel with no model, whose run
   dies at the assert on the model path. The string that works, taken from `v7-2-replicate`'s own
   metadata, is **`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`**, with a capital T. An
   earlier note said the opposite and has been corrected.
3. **`machine_shape` has to be `NvidiaL4`.** Without it the kernel gets a generic `Gpu`, and Kaggle may
   hand out a **Tesla P100 (sm_60)**, where torch 2.8 with cu128 fails with *"no kernel image is
   available for execution on the device"*, since it supports sm_70 and up. The scoring rerun uses L4s
   and would have been fine. It's the commit run that dies, and a kernel whose commit fails has no
   output, so `competition_submit_code` then returns 400 FAILED_PRECONDITION.

The third one also explains why v7.3 and v7.4 can't be reached. They have no kept output, so they hit
the same precondition.

`push_v76.py` now asserts on `invalidModelSources` and pins `machine_shape`. The three failures are
written down here because each one cost a push cycle and none of them raises an error.

**Resolved.** With `machine_shape: NvidiaL4` pinned, the v7.6 commit run finished in 24 minutes, and
`rc7530/arc-agi-2-v7-6-greedy-fallback-only` can be submitted. It's queued for the next slot as the first
of three draws against the banked control of ten.

## no-dead-AIF is complete, and it's a null

The third draw came back at **30.56**, so the pre-registered three are 30.00, 29.03 and 30.56, a mean of
**29.86** against the banked control's 30.31 (10 runs, sd 1.34). The difference is **-0.44** with a
standard error of 0.88, so **t = -0.50**. At this sample size the resolution is 2.46 points, so the
verdict is that **they can't be told apart**, with a point estimate slightly under the control.

This is the first of the five builds rejected on a single score to be re-tested properly, and it neither
clears nor condemns the build. What it does settle is the point about method. The original 30.00 was
called a regression, and three draws say the build sits inside the noise. The arm was finished instead
of stopped at the two draws that already looked flat, so there's no optional-stopping caveat on it.

Nothing else can be re-tested directly. v7.3, v7.4 and cap1200 all fail the output precondition, which
is why v7.6 was rebuilt from scratch.

## Where each arm stands against the banked control

The control is `339772104` plus `337769272`, **10 runs, mean 30.31, sd 1.34**. Never spend another slot
on it.

| arm | draws | mean | t | verdict |
|---|---|---|---|---|
| no-dead-AIF | 3 (30.00, 29.03, 30.56) | 29.86 | -0.50 | **closed, can't be told apart** |
| v7.6 greedy fallback | 1 of 3, sub 55572442 pending | | | in progress |
| v7.3, v7.4 and cap1200 | can't be reached, no kept output | | | need rebuilding like v7.6 |

Three draws resolve 2.46 points. Nothing this project has produced is that big, so the realistic outcome
for each arm is "can't be told apart", and the value lies in retiring the verdicts based on one score
rather than in finding a winner.

## v7.6 scored 30.56, and the number it landed on is the result

The greedy fallback **fired**. The commit log shows `[Rank 2] greedy fallback filled ...` twice and
`GFB=1` in the config banner, so this is different code doing something different. It scored **30.56**.

That value has now come up **7 times from five different builds** (336595817, 337769272, 339772104,
340004463 and 342748321). Across 21 submissions since v7, with 13 distinct values, a third land on
exactly 30.56, and 28.47 has come up 3 times from 3 different builds. Under a smooth normal with an sd
of 1.34, the most common bin, 0.4167 wide, holds 12.4% of the mass, so the chance of 7 or more of 21
landing in one bin is **about 1%**. **The score distribution isn't smooth. It has spikes.**

That changes our picture of the mechanism twice.

**The model of 41 coin flips is wrong as stated.** It matched the sd and the 12.5% disagreement between
pools with the same seed, but independent outputs on the edge predict a spread-out histogram, and we see
values the scores keep returning to. Something makes many different perturbations come back to the same
total.

**And it corrects how I read the correlation between size and score.** I argued that since output size
correlates with score at r = +0.64, and empty pools become `[[0]]`, filling those pools should buy score.
v7.6 filled them and bought nothing. The likelier reading is that both depend on a third thing. A run
that gets further through the search produces more content and also solves more tasks. Adding content
artificially doesn't move the score, because a greedy guess on a task the full search couldn't solve is
almost never right.

So v7.6 is an honest answer to the idea about empty pools rather than a null. The mechanism worked, and
the outputs it filled in were for tasks that couldn't be solved, not tasks that just hadn't been tried.
Draws 2 and 3 will say whether 30.56 was a coincidence, but there's much less reason to expect a gain
now.

## The compositional 2x2 is complete, and mixing doesn't save it at scale

The 41-hour cell finished after 135,342 steps and 38.8 hours, with a best val loss of 0.0414.

| composed | protective | ARC tasks | ARC candidates | composed tasks | composed candidates |
|---|---|---|---|---|---|
| 8k | 0 | -1 | -29% | +1 | +13% |
| 67.7k | 0 | **-2** | **-47%** | +3 | +200% |
| 4k | 4k | 0 | 0% | +1 | +8% |
| 8k | 8k | 0 | -6% | +1 | -15% |
| **67.7k** | **67.7k** | **-3** | **+3%** | **+3** | **+223%** |

The base is 5 of 20 on ARC and 3 of 8 on composed tasks in every row.

**At full scale, protection doesn't save ARC coverage.** The mixed full run is the worst of the five on
ARC, worse on 3 of 12 tasks, while learning composition better than any of them, at +223% candidates.
And the way it fails has changed. ARC candidate generation is kept (+3%, against -47% without
protection), so the model still produces plenty of candidates, and fewer of them are right. Mixing
protected the fluency and not the rule.

So the direction closes properly. Every configuration of this corpus lands at or below the base on ARC,
and the two that learn composition best are the two that hurt ARC most. **Composed ARC-1 rules aren't a
substitute for ARC-2 rules**, which is the premise the generation plan rests on, and now the best
argument for it.

**What this means for Phase 0.** As designed, it varies the number of rules inside the composed corpus,
and at 8,000 examples both arms will probably sit at no change in ARC coverage, the same lack of power
the pilot had. It's still worth the nine hours, because **candidate counts are the sensitive readout**,
not coverage. The composed corpus moved ARC candidates by -29% at 8k and -47% at 67.7k, so an effect of
diversity should show up there first. Read Phase 0 on candidates and treat flat coverage as telling us
nothing rather than as a null.

## 4 September. The state at the close, with everything folded in

We pulled the whole submission record (45 rows, with script versions checked) and reconciled every
document against it. The numbers below override everything above.

| result | value | folded into |
|---|---|---|
| replicate series `339772104` | **twelve** draws, 30.56, 32.64, 28.47, 31.39, 30.56, 30.56, 28.89, 28.89, 30.56, 28.47, 28.89 and 28.89. Mean **29.90**, sd **1.34**, range 4.17, confidence interval on the sd 0.95 to 2.27, and on the mean 29.05 to 30.75 | the tex (abstract, results, tables, ledger and conclusion), the writeup, fig_version_score, README, CONTEXT |
| original artifact `337769272` | 30.97, 30.56 and 28.47 (3 runs, mean 30.00, sd 1.34), with F against the replicate of 1.01 | tab:twoartifacts in the tex |
| fresh rebuild `346281718`, sub 55916584 | **31.39**. The build script isn't a confound, and the idea of drift is dead | the tex, the writeup, the versions table, CONTEXT |
| pooled over three artifacts | 16 runs, mean 30.01, sd 1.30, confidence interval on the sd 0.96 to 2.01 | tab:twoartifacts in the tex |
| v7.6, greedy fallback on its own | 30.56, 29.72 and 29.44, mean 29.91, t = 0.02 against the series | a ledger row in the tex, the versions table |
| v7.7, DFS cache crop | 29.72, 27.36 and 29.03, mean 28.70, t = -1.49, p = 0.22 | the frontier section of the tex, a ledger row, the writeup |
| v7.7b, fair view scheduling | **complete** at 29.44, 27.78 and 28.61 (the third draw was 56006884, on 4 Sep). Mean 28.61, t = -2.09, p = 0.09, all three under the control mean. With the crop that's six of six under, a pattern noticed after the fact and reported as one | "direction three" in the frontier section of the tex, a ledger row, the versions table, the writeup, fig_version_score, CONTEXT (folded in 2 Oct) |
| no-dead-AIF | 30.00, 29.03 and 30.56, mean 29.86 | the versions table |
| power at three runs per arm | **4.1 points** by an exact two-sample t test (Cohen's d of 3.07). The 3.4 quoted before came from a normal approximation and understated it | the tex (five places), the writeup |
| correlation between size and score | recomputed over 15 draws of the two artifacts with several runs, see the tex | the tex, two places |
| generator corpus | 110 verified from the pass without reasoning ($1.57), 17 more from low effort over 25 failures (68%, about 2.5 CNY), and 1 more from a retry control. **128 verified, 127 with samplers** | the gencorpus results in the tex, the writeup, CONTEXT |
| size of the ledger | 33 rows plus 7 arms from the end of the season (greedy fallback, cache crop, decode budget, fair scheduling, rule count, rule kind, duration) makes **40**, with 32 closed offline | the ledger in the tex and every "thirty-three" |

Provenance was settled the same day. `notebooks/arc_submission_v7_2_replicate.ipynb` is the exact
artifact behind the twelve draws (code hash d81f62ce), `arc_submission_v7_2_original.ipynb` is the build
that scored 30.97, `arc_submission_v7_2.ipynb` is the fresh rebuild that scored 31.39, and the v7.6, v7.7
and v7.7b files match the bytes pushed to Kaggle. Two stray local rebuilds of v7.6 and v7.7 were
reverted.

Still owed are a public URL for the PDF before the writeup is filed, the choice of cover image, pushing
the repo, which hadn't been pushed since 5 August, and the decision to make it public.

## 2 October. The board moved, and the paper's claims about the frontier followed it

| result | value | folded into |
|---|---|---|
| public leaderboard, 1 Oct | Tufa Labs 83.06, rabbithole 79.72, nvbanana 75.42, Yi-Chia Chen 55.14, then 38.89 and down. 10th is 36.25 and 25th is 33.06. We're 61st of 2,310 at 32.64 (43rd on 6 Aug) | the abstract, introduction, contributions, scaling section and conclusion of the tex, the HTML snapshot, CONTEXT |
| "reached 30.97" in the scaling section | that's one draw, not the result, and it's now the mean of 29.90 over twelve runs | the tex, the HTML snapshot |

## 4 October. The release and the history

| result | value | folded into |
|---|---|---|
| generator corpus, checked again in full | demonstrations 128 of 128, held-out test outputs 120 of 128 (94%), 127 samplers | corpus/README, gencorpus in the tex, the writeup, README, CONTEXT |
| covers | the triptych and the field rebuilt on the twelve runs and the 1 October board, with 18.19 and 29.90 on the title page | paper/, scripts/make_hero_final.py, scripts/make_cover_hero.py |
| history | every commit rewritten to the GitHub noreply address, locally, with the force push still to do | CONTEXT |
| daily replicate series | the script is written but was never started | CONTEXT |
