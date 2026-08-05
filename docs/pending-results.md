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

## Outstanding

| result | expected | where it must land |
|---|---|---|
| replicate 2 (sub 55261090) | tonight | tex variance passage as the 5th family member, recompute the spread and the sd; EXPERIMENTS.md family table; writeup error-bar paragraph; fig_version_score band |
| dual-expert on fresh pools | overnight | tex escape-route ledger (route 2 on the hidden set); score-levers F3; writeup theorem paragraph if it changes the verdict |
| subset-disagreement TTT | overnight | new tex passage on demonstration underdetermination, this is the headline if the number is interesting; writeup only if it displaces something |
| perceptual pre-fitting, 3 arms | overnight | tex only if the control arm separates grid fluency from step count; otherwise score-levers section 2 as a kill |
| pre-fitting prior-art check | running | docs/testinput-ttt-priorart.md, then the positioning of any pre-fitting claim |

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
