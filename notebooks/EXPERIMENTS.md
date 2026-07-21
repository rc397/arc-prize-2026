# v7 experiment matrix

Every variant runs the same restored pipeline. Only the config cell differs, so a variant is a documented
set of knob values rather than a fork. Build any of them with

    python scripts/build_v7_from_architects.py [v7.1|v7.2|v7.3]

The config cell prints its knobs on startup. Line one of every log should read

    CONFIG N_EVAL=... RANK=256 MAXSEQ=8192 TIMEOUT=1800 DFS_FLOOR=...

If that line does not say what you intended, kill the run rather than discovering it hours later.

## Variants

| variant | file | change from v7.1 | question it answers | hidden score |
|---|---|---|---|---|
| v7.1 | arc_submission_v7.ipynb | none, upstream defaults | baseline | **30.56** |
| v7.2 | arc_submission_v7_2.ipynb | `ARC_DFS_FLOOR=0.1` | does wider search buy solves | not run |
| v7.3 | arc_submission_v7_3.ipynb | `ARC_DFS_FLOOR=0.35` | does narrower search lose them | not run |

v7.2 and v7.3 exist as a pair on purpose. One variant gives a point, two either bracket the 0.2 default or
point consistently in one direction, and the search-budget lever then closes in a single batch instead of
three sequential guesses.

## Procedure

Smoke first. Set `ARC_N_EVAL = 6` and run all arms concurrently, roughly fifteen minutes each. The score
is meaningless at six tasks. What matters is that the CONFIG line is right, that nothing crashes, and the
wall-clock per task.

Project before promoting. Take total seconds, divide by six, multiply by 240, divide by four workers. Under
about eleven hours leaves submission headroom against the twelve-hour wall. v7.1 cleared it by roughly an
hour at floor 0.2, so v7.2 is the arm at risk, and a wider floor slows the tasks that previously finished
early, which makes the slowdown worse than proportional.

Then promote survivors to `ARC_N_EVAL = 120`, paired against the v7.1 baseline on the same task set.

## Reading the results honestly

At 120 tasks a one-point difference is about one task and cannot be separated from noise. Even three points
is borderline once you count how many tasks actually flipped. This matrix can resolve large effects and
genuinely cannot resolve small ones, so a narrow win here is not evidence, and a lever worth pursuing has
to move the number enough to survive that.

## Settled, do not revisit

Splitting the two attempt slots across the two selectors gains nothing. On the v7.1 run `score_kgmon`
solved 16 subkeys and `score_full_probmul_3` solved 17, and the 16 are a strict subset of the 17. The union
equals probmul alone, so there is no complementary coverage to harvest.

Selection headroom on the same run was 1.53 task-credits, about 4 points, resting on three missed cases.
That is an anecdote rather than a sample, and tuning a selector against it would repeat the v3 to v5
failure where public-eval gains reversed on the hidden set. The 120-task run exists partly to turn those
three cases into enough of a sample to look for a mechanism.
