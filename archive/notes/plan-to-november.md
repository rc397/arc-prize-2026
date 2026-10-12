# Plan to November, revision 2

Rewritten 2026-08-08 after two results landed in one night and closed most of revision 1. Read the
"what changed" section first, because the strategy inverts.

| date | what | days from 2026-08-08 |
|---|---|---|
| **2 Nov 2026** | Kaggle submissions due | 86 |
| **8 Nov 2026** | papers due | 92 |
| 4 Dec 2026 | results announced | 118 |

## What changed

**1. The variance is far larger than any effect we can produce.** One immutable kernel version,
three executions: 30.56, 32.64, 28.47. Family of six: mean 30.53, sd 1.35, range 4.17 points. Every
build this project called a regression is inside two sigma, and so are both public controls. Third
widening, same direction each time.

**2. Execution-verified synthesis is closed for our hardware.** The gate ran the full public eval
set, 120 tasks, 952 programs from a 14B coder at K=8. Zero solved. One task reproduced any
demonstration pair. Median per-cell lift over identity +0.000, only 5% of tasks above +0.05, though
68% got the output shape right. Evolution amplifies a baseline and there is none, and there is no
gradient to climb either. Revision 1 made this the centrepiece; it is now a measured negative.

**3. A reasoning error in the first draft of this revision, corrected.** I wrote that Accuracy was
now unimprovable because sd 1.35 makes effects unmeasurable. That conflates two different things.
The runs-per-arm needed to detect an effect scales as $1/\delta^2$:

| true effect | runs per arm to detect at 80% power |
|---|---|
| 1 point | 29 |
| 2 points | 7 |
| 3 points | 3 |
| **4 points** | **2** |
| 5 points | 1 |

High variance is an argument against **sub-two-point tinkering**, and no argument at all against a
four-point lever, which is three sigma and shows up in two or three runs. Rank 3 on the board is
+4.58 from where we sit. That is inside the range a real mechanism could deliver and inside the range
we could confirm.

## The strategic consequence, stated plainly

Two things are true at once and the plan has to hold both.

**The measurement is our strongest and most novel contribution** and deserves the slots to make it
unimpeachable. Nobody else has played it; the 2025 winners mention 1-2 points of re-run variability
in passing and never measured it.

**And the variance finding points AT a score lever rather than away from one.** This is the part the
first draft missed. The score moved 4.17 points on machine speed alone, which means the run is
throughput-limited: the search stops where the wall arrives, and candidates it would have found
never exist. So raw speed converts directly into coverage, and coverage is score. That is different
from the reallocation the restart null killed, which moved time between tasks inside the same total.

The bar for spending slots is therefore not "never chase score", it is **only chase a lever with a
mechanism worth more than three points, and gate it on the rig first.**

## W0. Throughput to coverage: the decode batch width

The one lever we have with a mechanism worth more than three points, and it comes straight out of
the variance result.

The deployed decoder splits the sixteen evaluation views into four batches of four, fixed by
hard-coded index offsets, and runs four sequential depth-first searches. Four trees cost the sum of
their step counts; one tree over all sixteen lanes costs the max. Single-token decode is bounded by
reading 8 GB of weights out of HBM rather than by arithmetic, so a wider batch should cost nearly the
same per step.

Measured, over 120 evaluation tasks: the sixteen views take exactly **one** distinct token length for
39.5% of test outputs and exactly **two** for the other 60.5%, and never more. So every task decodes
in at most two batches instead of four, 40% of them in one. The gaps are single-digit token counts
from grid separators. Weighted, that is about a **2.5x reduction in sequential trees**.

Why it should convert to score rather than just to time: twelve of sixteen rig decodes truncate at
their budget, and the hidden-set variance proves the deployed run is wall-limited too. A search that
truncates and is then given 2.5x the effective budget goes deeper and produces candidates that
previously did not exist.

Gates, in order, and each can kill it:

1. **bench** Forward-pass time at width 1, 2, 4, 8, 16. Flat means the width is free. Linear means
   decode is compute-bound here and the idea is dead with no further spend.
2. **A/B** Per task, decode twice under an identical wall budget, deployed 4-wide against
   length-grouped up to 16, comparing candidates and truth-in-pool. A within-task comparison, so
   hidden-set noise does not enter.
3. **correctness** Batch composition changes logits, so wider batching is not automatically
   semantics-preserving. If the wide arm loses truths anywhere, it does not ship regardless of speed.
4. Only then a submission, where a real effect of this size needs two or three runs to confirm.

`scripts/rig_widebatch.py` implements all of it.

## W1. Make the variance result unimpeachable

Currently n=6, and the 95% interval on the standard deviation is `[0.85, 3.32]`. We are claiming
"about 1.35" from data consistent with anything between 0.85 and 3.3. That is the weak point a
reviewer will find, and it is fixable with the resource we have.

**W1a. Extend the replicate series to n≈18-20.** Roughly 12 more slots, spread across the runway
rather than in a block so machine-load seasonality is sampled rather than aliased. At n=20 the
interval on the sd tightens to roughly ±25% instead of a factor of four. This is measurement, not
noise-mining, and the paper must say which it is doing.

**W1b. Report the distribution, not just the sd.** With 20 points we can say whether the spread is
Gaussian, bimodal or heavy-tailed. Bimodal would be a much sharper finding, because it would suggest
a discrete failure (a worker dropping its tail) rather than smooth speed variation.

**W1c. Decompose the mechanism offline, with no submissions at all.** This is the experiment I most
want to run and it needs zero GPU and zero slots. We have per-task wall times and outcomes for a
40-task measurement run. Simulate the real thing: 240 tasks over 4 workers against a 42480-second
wall, tasks drawn from the measured timing distribution, machine speed varied over a plausible
range, and compute the resulting score distribution. Then compare its standard deviation against the
measured 1.35.

  - If the simulation reproduces ~1.35, throughput explains the variance and the paper can say so
    quantitatively rather than by argument.
  - If it produces much less, the residue belongs to decoder nondeterminism, which we have already
    measured directly at 12.5% same-seed output disagreement and 0.543 same-config pool Jaccard.
  - Either way the answer is a decomposition, which is a better result than the sd alone.

Kaggle exposes no runtime on submissions (checked: date, status, scores and nothing else), so the
correlational version of this test is unavailable and the simulation is the way in.

## W2. Re-base every remaining claim onto offline evidence

This is the quiet but important half. Several conclusions currently rest on single hidden-set
comparisons that we have now shown cannot carry them. Each needs re-grounding on a within-task
offline measurement, where run-to-run noise does not apply.

| claim | currently rests on | must rest on |
|---|---|---|
| oracle accuracy precondition | v7.5a's 27.92 (1.9 sd, dead) | invariant mining: 0 recovered, 11/19 truths killed |
| selection is exhausted | partly on version deltas | the theorem, plus 0/18 oracle gap on fresh tasks |
| the phantom notebooks | "below our band" (dead) | displayed-vs-executed, which never needed a gap |
| cap / timing policy | 29.31 (0.9 sd, dead) | the budget table and W1c's simulation |

The general rule for the rest of the project: **an offline within-task comparison is worth more than
a hidden-set delta, because it has no noise floor.** We should have been operating this way for
months.

## W3. Rework the paper around its actual centre

The narrative has shifted from "we tried things and they regressed" to "we tried things and the
leaderboard cannot tell, and here is how much it cannot tell by". That is a better paper and it
needs the spine rewritten, not just the numbers patched. Sections affected: abstract and intro
(done), the results variance passage (done), the phantom section (withdrawn claim, done), the
escape-route ledger (re-based, done), plus the conclusion and the five-lessons section, which still
argue from regressions that no longer exist.

Rubric read after the change:

| criterion | before | after | why |
|---|---|---|---|
| Accuracy | weak-mid | weak-mid | unchanged and unimprovable |
| Universality | strong | **stronger** | the variance result applies to every leaderboard-driven field |
| Progress | mid-strong | **stronger** | stopping the field from over-reading ablations is real progress toward 85% |
| Theory | strong | strong | ceiling theorem intact, now grounded offline |
| Completeness | strong | strong | |
| Novelty | weakest | **mid** | nobody has measured this; it is measurement novelty, not method novelty |

## Submission budget, 86 slots

| purpose | slots | notes |
|---|---|---|
| W0 throughput lever | 6 | only after all three rig gates pass; 3 sigma needs 2-3 runs per arm |
| W1a variance series | 12-14 | the single highest-value use of a slot we have |
| final submission selection | 2 | 1-2 November |
| reserve | 64 | and it will not feel like enough in October |

Score-chasing is allowed but rationed: a lever needs a mechanism worth more than three points and a
passing rig gate before it costs a slot. Anything below that is smaller than our own instrument's
resolution and would be the exact error this paper documents.

## Hard blockers, unchanged and still not research

- [ ] **Repo is private.** Solutions must be open-sourced for prize eligibility. Decide by 1 Oct.
- [ ] Paper must link to a Kaggle code submission. `55261090` at 32.64, the maximum of the twelve-draw series; the fresh-build control `55916584` at 31.39 is the fallback.
- [x] Writeup capped at 1500 words, held near 1,450 so any counter clears it (4 Sep).

## What I dropped from revision 1, and why

- **W1 synthesis arm, 16 slots.** Closed by measurement, 0/120 with a flat gradient.
- **Score-chasing generally.** We cannot resolve what we could build.
- **The routing-policy build.** The subset-disagreement signal (0.373 vs 0.015) stays in the paper as
  an offline lead. Building the policy would produce an improvement we could not measure.

## Where the score could still come from

Ranked by mechanism strength, not by hope.

1. **W0, decode batch width.** Measured 2.5x reduction in sequential trees available, on a run we
   have proven is wall-limited. Three rig gates before a slot is spent. This is the only live one.
2. **Anything else that raises throughput.** The same argument applies to any real speedup: prefill
   reuse across views sharing a prefix, cheaper augmentation scoring, avoiding recomputation between
   the four batches. Nobody has profiled the deployed path end to end. Worth one rig session.
3. **Nothing on the selection side.** Closed by the theorem and by 0/18 on fresh tasks.
4. **Nothing on the synthesis side at our budget.** Closed at 0/120 with a flat gradient.

The honest position: one live lever with a real mechanism, a second worth profiling, and a
measurement contribution that is strong regardless of how those land. That is a better place to be
than the first draft of this revision suggested.
