# v7 experiment matrix

Every variant runs the same restored pipeline. Only the config cell differs, so a variant is a documented
set of knob values rather than a fork. Build any of them with

    python scripts/build_v7_from_architects.py [v7.1|v7.2]

The config cell prints its knobs on startup. Line one of every log should read

    CONFIG N_EVAL=... RANK=256 MAXSEQ=8192 TIMEOUT=1800 DFS_FLOOR=...

If that line does not say what you intended, kill the run rather than discovering it hours later.

## Variants

| variant | file | change from v7.1 | purpose | score |
|---|---|---|---|---|
| v7.1 | arc_submission_v7.ipynb | none, upstream defaults | the submission | **30.56** hidden |
| v7.2 | arc_submission_v7_2.ipynb | `ARC_DUMP_POOL=1` | dump the candidate pool for selection analysis | measurement run, not submitted |

v7.1 is what you submit. v7.2 is identical plus a gated candidate-pool dump, run once at `ARC_N_EVAL=40`
to write `pool_dump.json`, which lets any selector or verifier be tested offline against the roughly
five-point selection ceiling without further quota. It is a measurement build, never a submission.

The two search-width arms that used to sit here (floor 0.1 and floor 0.35) were measured dead, the DFS
floor changed the deployed output by nothing, so their notebooks were removed. The finding is in the
DEVLOG. `ARC_DFS_FLOOR` remains a knob for anyone who wants to re-open it.

## Procedure

For v7.2, upload it, keep the defaults (`ARC_N_EVAL=40`, `ARC_DUMP_POOL=1` are already set), Save & Run All,
roughly two hours. Confirm the log ends with `POOL DUMP: N outputs -> pool_dump.json`, then the pool is
pulled and analysed offline. No promotion or submission from this build.

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
