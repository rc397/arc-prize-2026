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
| v7.5a | arc_submission_v7_5a.ipynb | `ARC_AIF=prune`, batch 4, cap 1800 | not run | pending |
| v7.5b | arc_submission_v7_5b.ipynb | `ARC_AIF=off`, batch 2, cap 1800 | not run | pending |
| v7.5c | arc_submission_v7_5c.ipynb | `ARC_AIF=full`, batch 2, cap 1800 | not run | pending |
| control | `rc7530/lb33-control-unmodified` | mikelou1's notebook unmodified, sorokin's attachments | 1200 | **28.47** |

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
