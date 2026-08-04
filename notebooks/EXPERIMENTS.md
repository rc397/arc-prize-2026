# v7 experiment matrix

Every variant runs the same restored pipeline. Only the config cell differs, so a variant is a documented
set of knob values rather than a fork. Build any of them with

    python scripts/build_v7_from_architects.py [v7.1|v7.2|v7.3|v7.4|v7.5a|v7.5b|v7.5c]

`scripts/build_v7_from_architects.py` is the source of truth for what each variant sets. This file is the
history of what each one actually ran and scored, which is not always the same thing.

There is no v7.3a. The variants are v7.1, v7.2, v7.3, v7.4, then the v7.5 bisection a, b and c.

## The family tree

| variant | file | knobs set | cap ACTUALLY run | hidden score |
|---|---|---|---|---|
| v7.1 | arc_submission_v7.ipynb | upstream defaults | 1800 | 30.56 |
| v7.2 | arc_submission_v7_2.ipynb | `ARC_DUMP_POOL=1` | 1800 | **30.97** |
| v7.3 | arc_submission_v7_3.ipynb | `ARC_LOSS_CUT=0.05`, `ARC_GREEDY_FALLBACK=1`, cap 1200 | 1800 | 28.61 |
| v7.4 | arc_submission_v7_4.ipynb | cap 1200, cuts off | 1200 | 29.44, void |
| v7.5a | arc_submission_v7_5a.ipynb | `ARC_AIF=prune`, batch 4, cap 1800 | 1800 | **27.92** |
| v7.5b | arc_submission_v7_5b.ipynb | `ARC_AIF=off`, batch 2, cap 1800 | not run | pending |
| v7.5c | arc_submission_v7_5c.ipynb | `ARC_AIF=full`, batch 2, cap 1800 | not run | pending |
| control | `rc7530/lb33-control-unmodified` | mikelou1's notebook unmodified, sorokin's attachments | 1200 | **28.47** |
| replicate | `rc7530/v7-2-replicate` | v7.2 from c7b591e, only commit size changed | 1800 | **30.56** |
| no-dead-AIF | `rc7530/v7-2-without-the-discarded-aif-work` v2 | the replicate minus the discarded AIF work | 1800 | **30.00** |

The "cap actually run" column exists because it disagreed with the knob for the first four builds. The
builder's replace targeted a `spend_time` form the source does not use, so it never matched, and v7.1,
v7.2 and v7.3 all ran the hardcoded 1800 while their config banners printed whatever we had asked for.
v7.4 was the first build in which the cap was really 1200.

## Corrections to the previous version of this file (2026-08-01)

The 26 July version of this file predates the v7.4 post-mortem and asserted three things that are false.

It listed v7.1 as running "cap 1200". It did not. The knob was not wired until v7.4.

It attributed v7.2's 0.41 gain over v7.1 to a 1200-to-1800 cap change. There was no cap change, both
builds ran 1800. That 0.41 is one task credit, which is exactly the noise floor, so it is unexplained
and should not be attributed to anything.

It named `koushikrudra/failed-in-aimo` as the 33.89 notebook. It is not. We forked that one as a control
and it scored 28.06. The build carrying 33.89 is `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`, and
reading its source on 2026-08-01 shows a literal 1200-second per-task cap inside its own decode loop at
line 851 of the flattened notebook, inherited from nothing. The claim that it forks a source hardcoding
1800, and the consequent withdrawal of the cap hypothesis, were both based on the wrong notebook. The
hypothesis is reinstated. The same correction is applied in the paper and in the v7.4 variant note.

## The control result, 2026-08-01

`mikelou1/arc-agi2-lb33-89-minimal-perfpatch` forked unmodified scores **28.47**, not 33.89. The
notebook was byte-identical, SHA256 `EE49384A...`, with only the ownership metadata changed, and it
mounted sorokin's own wheel tree and model upload exactly as its metadata specifies.

That is the second public notebook to display a high score and land near 28 when actually run. The
first was `koushikrudra/failed-in-aimo` at 28.06. Our own v7.2 scores 30.97. There was never a
deficit to close. We were ahead of both.

Three things die with this result.

The environment hypothesis is refuted rather than merely unsupported. The claim was that mounting
sorokin's original attachments instead of the rc7530 fork was worth 3.33 points. This control mounted
exactly those attachments and scored 28.47, which is below every build we have run on our own mounts.

The cap hypothesis weakens sharply. LB33 runs cap 1200 and scores 28.47. Our v7.2 runs 1800 and
scores 30.97. That is confounded by everything else that differs between the two builds, so it does
not settle the one-factor question, but it removes the reason to expect 1200 to help.

The v7.5 bisection loses its motivation. It was built to explain a 30.56-versus-33.89 gap that does
not exist. v7.5a survives on its own merits as the ceiling theorem's program-execution channel, which
no configuration on either side has ever run. v7.5b and v7.5c were checks on a phantom.

One caveat kept deliberately. A run that hits the twelve-hour wall and drops the tail of its queue
also finishes COMPLETE with a low score, and the scoring rerun's log is not visible from outside. So
28.47 is what that artefact delivers on this account, which is the question the control was built to
answer, and it is not a proof about the code considered in isolation.

## The replicate settles it, 2026-08-03

`rc7530/v7-2-replicate`, the `c7b591e` notebook with only the commit size changed, scored **30.56**
against the original's 30.97. A difference of 0.41, which is exactly one task credit, exactly the
noise floor.

That gives three runs of what is effectively the same method, since v7.1 is v7.2 without the pool
dump and both ran the hardcoded 1800 cap.

| run | date | score |
|---|---|---|
| v7.1 | 2026-07-20 | 30.56 |
| v7.2 | 2026-07-25 | 30.97 |
| v7.2 replicate | 2026-08-03 | 30.56 |

Mean 30.70, range 0.41, standard deviation 0.24. Run-to-run variance on this pipeline is about one
task credit, not several points.

So the bimodality is real and the regressions are real. Measured against the LOWEST baseline run
rather than the best, to be conservative:

| build | score | below 30.56 | in credits | in sd |
|---|---|---|---|---|
| v7.4 | 29.44 | -1.12 | 2.7 | 4.7 |
| v7.3 | 28.61 | -1.95 | 4.7 | 8.2 |
| LB33 control | 28.47 | -2.09 | 5.0 | 8.8 |
| v7.5a | 27.92 | -2.64 | 6.3 | 11.2 |

Every one is far outside the noise. The wall-clock explanation for the bimodality is not supported
either, since a binding twelve-hour wall with variable session speed would show up as spread in these
three baseline runs and it does not.

Read honestly, three runs is a small sample and the standard deviation above is an estimate from it,
not a population value. What it does establish is that the "our changes were fine and the scores are
noise" hypothesis is dead. The changes were harmful. The record stands as written.

This also retires the last defence of the two things we spent the most time on. The 33.89 does not
reproduce, and every modification we made to chase it cost real points.

## v7.5a, and the variance question it forces

v7.5a scored **27.92** against v7.2's 30.97, a loss of 3.05 points or about seven task credits. It ran
on v7.2's environment, the rc7530 unsloth fork recorded in `scripts/kaggle_pushall.py`, so the
environment is not the difference.

Two explanations fit, and they have opposite consequences.

SELECTION HARM. The prune adds `abs(gen_count - expected_count) * 0.5` to every beam score whose
connected-component count disagrees with AIF's prediction. When that prediction is wrong the penalty
demotes correct candidates. AIF returns `expected_count` on every branch including its no-rule
fallback, so a wrong count is not rare. This would say the ceiling theorem's program-execution escape
is real but needs an oracle more accurate than the model's own beliefs, and a noisy oracle is worse
than none.

TIME HARM. The prune launches a `ThreadPoolExecutor` per task before training. On a 240-task rerun
against a twelve-hour wall, work that buys nothing can push the queue tail past the wall, and dropped
tasks score zero. This would say nothing about selection at all.

These are distinguishable, and the last cell added on 2026-08-01 already prints the number that
separates them, the count of tasks the solver actually decoded. If that count is near 240 the loss is
selection. If it is well short, the loss is wall clock. The commit log came back empty from the API,
so this needs reading in the Kaggle UI.

THE PATTERN THAT MATTERS MORE. Every build after v7.2 has scored below it, now five in a row, and the
scores are bimodal: v7.1 and v7.2 at 30.56 and 30.97, then v7.4, v7.3, the LB33 control and v7.5a at
29.44, 28.61, 28.47 and 27.92. Nothing lands between. We have never measured the run-to-run variance
of a single unchanged configuration, and every comparison in this file assumes it is small. The
twelve-hour wall is a plausible mechanism for several points of variance, because whether the queue
tail completes depends on session speed rather than on method.

So the next slot should re-run v7.2 unchanged. If it returns near 31 our changes really are harmful
and the record stands. If it returns near 28 then 30.97 was a favourable draw, five "regressions" were
noise, and much of the last fortnight's reasoning rests on a single lucky run. That question now
gates the value of every other experiment, which makes the replicate the highest-information slot
available rather than a filler.

## The no-dead-AIF run, 2026-08-05, and what it does to the error bar

Deleting the discarded AIF work scored **30.00** against the replicate's 30.56. The change has no
path to the computed answers except timing, and timing is not a null channel because the decode is
deadline-bounded in three places, so speed moves where the search truncates and therefore which
candidates exist. Two readings, not distinguishable from one run: removing CPU load shifted decode
timing enough to lose about one credit's worth of candidates, or same-method noise is wider than the
three pure replicates suggested. Conservatively the family {30.00, 30.56, 30.56, 30.97} spans about
one point, matching the 1-2 point variability NVARC reported in passing and never measured. Under
the wider bound the three unconfounded regressions (v7.3, the LB33 control, v7.5a) remain real at
3.5 to 5.2 sd below the family's lowest run, and v7.4's 29.44, already void, is no longer
distinguishable from a re-run. The paper now uses the wider bound throughout.

## The rebuilt v7.2 is not v7.2

Found on 2026-08-02 while preparing the replicate, and it invalidates the assumption that rebuilding
a variant reproduces it.

The v7.2 that scored 30.97 (`c7b591e`) launched a `ThreadPoolExecutor` and ran `run_aif_analysis` on
every task unconditionally, collected the result, and then overwrote it with a neutered dict. The arm
was disabled by discarding its output, not by skipping its work. Every later build gates the launch on
`ARC_AIF != "off"`, so rebuilding v7.2 today produces a notebook that never pays that cost.

The rebuilt v7.2 is therefore strictly faster than the scored v7.2, on a run whose scores we suspect
are wall-clock sensitive. So a rebuild is not a replicate, and this file's table describes what each
build DID, which is not always what rebuilding its name gives you today.

The replicate pushed as `rc7530/v7-2-replicate` is the `c7b591e` notebook with exactly one line
changed, `ARC_N_EVAL` from 40 to 4, which line `if not rerun_mode and N_EVAL > 0` gates out of the
scored run. Two differing lines in the whole notebook, verified by diff.

This also creates a lever worth its own slot later. Deleting the discarded AIF work from v7.2 frees
per-task wall clock for nothing given up, since the result was already being thrown away.

## Where the cap question actually stands

Never cleanly tested. v7.4 is the only build that really ran 1200 and it scored 29.44, but the same run
fired 24 unbudgeted greedy fills through a gate that treated the string `"0"` as true, so two factors
moved and the result is void rather than adverse. The internal timing model says four tasks running past
1200 seconds consumed 26.3 percent of the budget and returned nothing, which predicts the change is
neutral to positive on a 240-task queue. A clean one-factor run is worth a submission day.

## Commit cost

Kaggle sets `KAGGLE_IS_COMPETITION_RERUN` only when it re-runs a notebook for scoring, so our own commits
read the public evaluation set and can queue a subset. Ours subsets only when `ARC_N_EVAL > 0`, and until
2026-08-01 the config cell set it to 40 and then overrode it to 0, which means all 120. Every commit had
been running the full public evaluation set for no reason, at five to six hours of the weekly GPU
allowance each time.

Fixed. Every variant now commits at `ARC_N_EVAL=4`, and the builder prints the commit cost when it emits
a notebook. Four rather than one because `NPROCS=4`, so a single task leaves ranks 1 to 3 never doing any
work and a broken worker would pass unnoticed. This is almost certainly why LB33 pins exactly four keys.

The scoring rerun is unaffected in every case. Line `if not rerun_mode and N_EVAL > 0` gates the subset on
commit only, so the submitted run always does all 240 hidden tasks. The builder asserts that gate still
exists, because losing it would silently restore the five-hour commit.

The last cell of every emitted notebook now verifies `submission.json` covers every task in the relevant
challenge file and fails loudly if any key is missing. Undecoded tasks carrying default answers are
expected on a small commit and treated as fatal on the rerun. Without this, a malformed submission is
only discovered at submit time, hours later.

## Reading the results honestly

At 120 tasks a one-point difference is about one task and cannot be separated from noise. Even three
points is borderline once you count how many tasks actually flipped. This matrix can resolve large
effects and genuinely cannot resolve small ones, so a narrow win here is not evidence, and a lever worth
pursuing has to move the number enough to survive that. The hidden-set run-to-run noise has never been
measured, and every comparison in the table above silently assumes it is small.

## Settled, do not revisit

Splitting the two attempt slots across the two selectors gains nothing. On the v7.1 run `score_kgmon`
solved 16 subkeys and `score_full_probmul_3` solved 17, and the 16 are a strict subset of the 17. The union
equals probmul alone, so there is no complementary coverage to harvest.

The two search-width arms that used to sit here, floor 0.1 and floor 0.35, were measured dead. The DFS
floor changed the deployed output by nothing, so their notebooks were removed. The finding is in the
DEVLOG and `ARC_DFS_FLOOR` remains a knob for anyone who wants to re-open it.

Selection reweighting is also dead as of 2026-08-01. The exact Appendix F bound on the dumped pool is 18
of 60 against kgmon's attained 15, but a re-weighted monotone selector fit with leave-one-task-out
cross-validation attains 14, a loss of one against a pre-registered bar of plus two. See
`docs/score-levers.txt`.
