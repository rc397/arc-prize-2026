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
| v7.1 | arc_submission_v7.ipynb | none, upstream defaults, cap 1200 | the first submission | 30.56 hidden |
| v7.2 | arc_submission_v7_2.ipynb | `ARC_DUMP_POOL=1`, cap at the 1800 default | pool dump for selection analysis | **30.97** hidden |
| v7.3 | arc_submission_v7_3.ipynb | `ARC_LOSS_CUT=0.05`, `ARC_GREEDY_FALLBACK=1` | the scheduler build | pending |

v7.2's dump wrote `pool_dump.json` for offline selection analysis, and its submission doubled as the cap
read. The 0.41 over v7.1 is the 1200-to-1800 cap change, proven by the sixty-way public-notebook tie at
exactly 30.97.

v7.3 carries the two scheduler rules priced on the v7.2 measurement run, where 71 percent of task-seconds
went to tasks yielding nothing. A task whose adapter finishes training above loss 0.05 skips the half-hour
search and is served by a greedy decode instead, which reclaimed 2330s on the measurement slice while
touching nothing the selector harvests. A task whose search returns empty gets the same greedy fill, which
can only add answers to otherwise-forfeited slots, four per 120 tasks.

v7.3 is also rebased onto the 33.89 deployment condition, and the attachment is not optional. The public
notebook koushikrudra/failed-in-aimo is byte-equivalent to our v7.1 in every module, loader, starter,
decoder and solver all cancel in diff, yet it scores 33.89 where our v7.1 scored 30.56 at the same 1200s
cap. Its kernel metadata shows the one difference. It mounts sorokin's original
pip-install-unsloth-flash-patch wheel tree and sorokin's own model upload, where our builds mount the
rc7530 slim fork whose pins predate the current Kaggle image, and our own run logs carry the warning,
Skipping import of cpp extensions due to incompatible torch version. Identical code submitted twice
prices the environment at 3.33 points, about eight tasks. So when uploading v7.3, attach
`sorokin/pip-install-unsloth-flash-patch` as the utility source and `sorokin/qwen3_4b_grids15_sft139` as
the model, not the rc7530 fork and not the mirzamilanfarabi mirror. The cap is 1200 to match the 33.89
condition exactly, so any delta against 33.89 belongs to the scheduler.

Run v7.3 once at `ARC_N_EVAL=6` as a smoke, about forty minutes. Task 136b0064 trains to loss 0.09, so the
log must show `skipping search for 136b0064` followed by `greedy fallback filled 136b0064_0...`, which
exercises both new paths end to end. Watch the version prints at startup, unsloth should NOT report
2025.9.7 if sorokin's tree mounted correctly, and the cpp-extensions warning should be gone. Then flip to
`ARC_N_EVAL=0` for the full 120, about six hours, and gate on the public read against v7.1's 33.8 credits
on the same denominator. The hidden gate is 33.89, at or above it the scheduler paid, below 30.97 the
rebase itself failed and the attachment is the first suspect.

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
