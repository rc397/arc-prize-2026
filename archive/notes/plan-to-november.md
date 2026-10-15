# Plan to November, second version

Rewritten on 8 August 2026, after two results landed in one night and closed most of the first
version. Read "What changed" first, because the strategy flips.

| date | what | days from 8 August |
|---|---|---|
| **2 Nov 2026** | Kaggle submissions due | 86 |
| **8 Nov 2026** | papers due | 92 |
| 4 Dec 2026 | results announced | 118 |

## What changed

**1. The noise is far bigger than any effect we can produce.** One unchanged kernel version, run three
times, scored 30.56, 32.64 and 28.47. Across a family of six runs the mean is 30.53, the standard
deviation 1.35 and the range 4.17 points. Every build this project called a regression sits within
two standard deviations, and so do both public notebooks we ran as controls. This is the third time
the estimate has widened, each time in the same direction.

**2. Synthesis checked by running the program is closed on our hardware.** The gate ran the full
public evaluation set, 120 tasks and 952 programs from a 14B coding model with eight tries each. It
solved none. Only one task reproduced even one demonstration pair. The median per-cell improvement over
just copying the input was zero, only 5% of tasks did better than 0.05, though 68% got the output
shape right. Evolution amplifies a baseline, and there isn't one, nor any gradient to climb. The first
version made this the centrepiece. It's now a measured negative.

**3. I made a reasoning mistake in the first draft of this version, and here's the fix.** I wrote that
Accuracy could no longer be improved, because a standard deviation of 1.35 makes effects
unmeasurable. That mixes up two different things. The number of runs per arm you need to detect an
effect grows with one over the square of the effect size.

| true effect | runs per arm to detect it at 80% power |
|---|---|
| 1 point | 29 |
| 2 points | 7 |
| 3 points | 3 |
| **4 points** | **2** |
| 5 points | 1 |

High noise is an argument against tinkering for gains under two points. It's no argument at all
against a four-point lever, which is three standard deviations and shows up in two or three runs.
Third place on the board is 4.58 points above us. That's within reach of a real mechanism and within
what we could confirm.

## What that means for strategy

Two things are true at once, and the plan has to hold both.

**The noise measurement is our strongest and most original contribution**, and it deserves the
submission slots needed to make it airtight. Nobody else has done it. The 2025 winners mention one to
two points of re-run variability in passing and never measured it.

**And the noise result points toward a score lever, not away from one.** This is what the first draft
missed. The score moved 4.17 points on machine speed alone, which means the run is limited by
throughput. The search stops when time runs out, and the candidates it would have found never exist.
So raw speed turns straight into coverage, and coverage is score. That's different from the
reallocation the restart experiment ruled out, which only moved time between tasks inside the same
total.

So the bar for spending a slot isn't "never chase score". It's "only chase a lever whose mechanism is
worth more than three points, and test it on the rig first".

## W0. Turning throughput into coverage with the decoding batch width

This is the one lever we have with a mechanism worth more than three points, and it comes straight out
of the noise result.

The deployed decoder splits the sixteen evaluation views into four batches of four, using hard-coded
index offsets, and runs four depth-first searches one after another. Four trees cost the sum of their
steps, while one tree over all sixteen lanes costs only the longest. Decoding one token at a time is
limited by reading 8 GB of weights from memory, not by arithmetic, so a wider batch should cost about
the same per step.

Measured over 120 evaluation tasks, the sixteen views have exactly one distinct token length for 39.5%
of test outputs and exactly two for the other 60.5%, never more. So every task could decode in at
most two batches instead of four, and 40% of them in one. The length differences are single-digit
token counts from grid separators. Weighted together, that's about a 2.5 times reduction in searches
run one after another.

It should turn into score and not just saved time, because twelve of sixteen decodes on the rig hit
their time limit, and the hidden-set noise proves the real run is time-limited too. A search that gets
cut off and is then given 2.5 times the effective budget goes deeper and finds candidates that didn't
exist before.

Here are the gates, in order, and any one of them can kill it.

1. **Benchmark.** Time a forward pass at widths 1, 2, 4, 8 and 16. Flat means the extra width is free.
   Linear means decoding is compute-bound here, and the idea dies with no more spending.
2. **A/B test.** Decode each task twice under the same time budget, the deployed 4-wide against
   grouping by length up to 16 wide, and compare candidates and how often the truth reaches the pool.
   It's a comparison within each task, so hidden-set noise doesn't come into it.
3. **Correctness.** Changing what's in a batch changes the logits, so wider batching doesn't
   automatically give the same answers. If the wide version loses a true answer anywhere, it doesn't
   ship however fast it is.
4. Only then a submission, where a real effect this size needs two or three runs to confirm.

`scripts/rig_widebatch.py` implements all of it.

## W1. Make the noise result airtight

We have six runs, and the 95% interval on the standard deviation runs from 0.85 to 3.32. We're claiming
"about 1.35" from data that fits anything between 0.85 and 3.3. A reviewer will find that weak spot,
and we can fix it with what we have.

**W1a. Extend the replicate series to about 18 or 20 runs.** That's roughly 12 more slots, spread
across the remaining weeks rather than bunched together, so any day-to-day pattern in machine load
gets sampled instead of hidden. At 20 runs the interval on the standard deviation tightens to about
25% either side instead of a factor of four. This is measurement, not fishing for a lucky draw, and the
paper has to say which it's doing.

**W1b. Report the whole distribution, not just the standard deviation.** With 20 points we can tell
whether the spread is normal, bimodal or heavy-tailed. Bimodal would be a much sharper finding, because
it would point to a discrete failure, like one worker dropping its last tasks, rather than smooth
variation in speed.

**W1c. Break down the mechanism offline, with no submissions.** This is the experiment I most want to
run, and it needs no GPU and no slots. We have per-task times and outcomes from a 40-task measurement
run. Simulate the real thing with 240 tasks over four workers against a 42,480-second limit, drawing
tasks from the measured timing distribution and varying machine speed over a plausible range. Then
compute the score distribution that comes out and compare its standard deviation with the measured
1.35.

- If the simulation gives about 1.35, throughput explains the noise, and the paper can say so with a
  number rather than an argument.
- If it gives much less, the rest comes from the decoder's own nondeterminism, which we've already
  measured directly at 12.5% disagreement between runs with the same seed and 0.543 overlap between
  pools from the same configuration.
- Either way we get a breakdown, which is a better result than the standard deviation alone.

Kaggle doesn't show runtimes for submissions. I checked, and it gives the date, status and scores and
nothing else. So the correlational version of this test isn't possible, and the simulation is the way
in.

## W2. Move every remaining claim onto offline evidence

This is the quiet but important half. Several conclusions still rest on single hidden-set comparisons,
which we've now shown can't carry them. Each needs grounding in an offline comparison within tasks,
where run-to-run noise doesn't apply.

| claim | rests on now | should rest on |
|---|---|---|
| a check has to beat the model's own accuracy | v7.5a's 27.92, at 1.9 standard deviations, which is dead | invariant mining, which recovered nothing and killed 11 of 19 true answers |
| selection is exhausted | partly on differences between versions | the theorem, plus a gap of 0 of 18 on fresh tasks |
| the phantom notebooks | "below our band", which is dead | displayed against executed, which never needed a gap |
| the time cap and timing policy | 29.31, at 0.9 standard deviations, which is dead | the budget table and the W1c simulation |

The rule for the rest of the project is that an offline comparison within tasks is worth more than a
difference on the hidden set, because it has no noise floor. We should have worked this way for
months.

## W3. Rebuild the paper around what it's really about

The story has shifted from "we tried things and they regressed" to "we tried things, the leaderboard
can't tell, and here's by how much it can't tell". That's a better paper, and it needs the backbone
rewritten, not just the numbers patched. The abstract, introduction, the noise passage in the
results, the phantom section and the escape-route ledger are done. The conclusion and the five-lessons
section still argue from regressions that no longer exist.

Here's how the rubric reads after the change.

| criterion | before | after | why |
|---|---|---|---|
| Accuracy | weak to middling | weak to middling | unchanged, and can't be improved |
| Universality | strong | **stronger** | the noise result applies to every field run on leaderboards |
| Progress | middling to strong | **stronger** | stopping a field from over-reading its ablations is real progress toward 85% |
| Theory | strong | strong | the ceiling theorem holds, now grounded offline |
| Completeness | strong | strong | |
| Novelty | weakest | **middling** | nobody has measured this, which is new measurement rather than a new method |

## The submission budget, 86 slots

| purpose | slots | notes |
|---|---|---|
| W0 throughput lever | 6 | only after all three rig gates pass, since three standard deviations needs two or three runs an arm |
| W1a noise series | 12 to 14 | the most valuable use of a slot we have |
| choosing final submissions | 2 | 1 and 2 November |
| reserve | 64 | and it won't feel like enough in October |

Chasing score is allowed but rationed. A lever needs a mechanism worth more than three points, and a
passing rig test, before it costs a slot. Anything smaller sits below what our own instrument can
resolve, and chasing it would be the very mistake this paper documents.

## Hard blockers, which aren't research

- [ ] **The repo is private.** Solutions must be open-sourced to be eligible for prizes. Decide by 1
  October.
- [ ] The paper has to link to a Kaggle code submission. That's `55261090` at 32.64, the best of the
  twelve-run series, with the fresh-build control `55916584` at 31.39 as the fallback.
- [x] The writeup is capped at 1,500 words. It's held near 1,450 so any word counter clears it (4
  September).

## What I dropped from the first version, and why

- **The W1 synthesis arm, 16 slots.** Closed by measurement, 0 of 120 with no gradient to follow.
- **Chasing score in general.** We can't resolve what we could build.
- **Building the routing policy.** The signal from disagreement between subset adapters, 0.373 against
  0.015, stays in the paper as an offline lead. Building the policy would produce an improvement we
  couldn't measure.

## Where score could still come from

Ranked by how strong the mechanism is, not by hope.

1. **W0, the decoding batch width.** A measured 2.5 times reduction in searches run one after another,
   on a run we've proven is time-limited. Three rig gates come before any slot. This is the only live
   one.
2. **Anything else that raises throughput.** The same argument applies to any real speedup, such as
   reusing prefill across views that share a prefix, cheaper scoring of augmentations, or avoiding
   recomputation between the four batches. Nobody has profiled the deployed path end to end. It's
   worth one session on the rig.
3. **Nothing on the selection side.** Closed by the theorem and by 0 of 18 on fresh tasks.
4. **Nothing on the synthesis side at our budget.** Closed at 0 of 120 with no gradient.

Honestly, that's one live lever with a real mechanism, a second worth profiling, and a measurement
contribution that holds up however those land. That's a better position than the first draft of this
version suggested.
