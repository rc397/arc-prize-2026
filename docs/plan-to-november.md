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

**3. Therefore Accuracy is fixed at 32.64 and cannot be improved measurably.** Not "is hard to
improve" — *cannot be measured*. With sd 1.35 and one submission per day, resolving a 1-point real
effect needs about 30 runs per arm. We have 86 slots total. Anything we could plausibly build is
smaller than our own instrument's resolution.

## The strategic consequence, stated plainly

**We cannot win on Accuracy, so the paper must win on the other five criteria, and the variance
result is the strongest card we hold.** It is also the only one nobody else has played: the 2025
winners mention 1-2 points of re-run variability in passing and never measured it.

That flips the priority order. Revision 1 spent the runway chasing score. Revision 2 spends it
making the measurement contribution unimpeachable, and re-bases every other claim onto evidence that
hidden-set noise cannot touch.

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
| W1a variance series | 12-14 | the single highest-value use of a slot we have |
| final submission selection | 2 | 1-2 November |
| reserve | 70 | and it will not feel like enough in October |

Nothing is allocated to score-chasing. If a genuinely new idea appears with a mechanism worth more
than three points, it comes out of reserve and goes through a rig gate first.

## Hard blockers, unchanged and still not research

- [ ] **Repo is private.** Solutions must be open-sourced for prize eligibility. Decide by 1 Oct.
- [ ] Paper must link to a Kaggle code submission. Currently `55261090` at 32.64.
- [ ] Writeup capped at 1500 words, currently 1498.

## What I dropped from revision 1, and why

- **W1 synthesis arm, 16 slots.** Closed by measurement, 0/120 with a flat gradient.
- **Score-chasing generally.** We cannot resolve what we could build.
- **The routing-policy build.** The subset-disagreement signal (0.373 vs 0.015) stays in the paper as
  an offline lead. Building the policy would produce an improvement we could not measure.

## The decision I need from you

**Do we accept that 32.64 is our final Accuracy and spend the runway on the measurement
contribution?**

My recommendation is yes. The alternative is to keep spending slots on levers whose effects are
smaller than our instrument's resolution, which is precisely the error this paper now exists to
document. But it means accepting a mid-field score in a competition, and that is your call, not
mine.
