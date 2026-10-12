# Invariant-mining candidate filter (lever A2): measured, and dead by its own bar

Measured 2026-08-04 with `scripts/pool_invariants.py` on `paper/data/pool_dump.json` (60 outputs,
39 tasks), demonstrations and solutions from the public evaluation JSONs. Reproduce with
`python scripts/pool_invariants.py [POOL] [CHALLENGES] [SOLUTIONS]`. The dump's `correct` flags are
re-verified against the solutions file at load. Baseline confirmed before filtering: truth in pool
19, kgmon top-2 attains 15, leaked outputs 142ca369_0, 3a25b0d8_0, 3dc255db_0, 4c7dc4dd_1.

Design: 15 invariant types mined per task from its train pairs only, kept only if the property held
on every pair (every pool task has at least 2 pairs; 214 invariants total, median 6 per task, no
task with zero). A candidate violating any kept invariant is killed.

## (a) False kills, the kill condition: 11 of 19 truths die under the full set

| invariant              | tasks mined | wrong kills | false kills |
|------------------------|------------:|------------:|------------:|
| background_preserved   | 27          | 7           | 0           |
| col_parity             | 21          | 41          | 2           |
| colour_count_conserved | 26          | 168         | 5           |
| components_eq_input    | 4           | 8           | 0           |
| components_fixed       | 4           | 16          | 3           |
| immutable_colours      | 22          | 84          | 2           |
| mirror_lr              | 2           | 2           | 2           |
| mirror_ud              | 3           | 6           | 0           |
| palette_additive       | 4           | 0           | 0           |
| palette_fixed          | 6           | 4           | 1           |
| palette_subset         | 35          | 0           | 0           |
| row_parity             | 26          | 71          | 4           |
| shape_eq_input         | 25          | 0           | 0           |
| shape_fixed            | 8           | 29          | 2           |
| shape_scaled           | 1           | 0           | 0           |

False-killed truths: 2d0172a1_1, 4c3d4a41_0, 1ae2feb7_2, 53fb4810_0, 3a25b0d8_0, 45a5af55_0,
38007db0_1, 135a2760_0, 3dc255db_0, 3a25b0d8_1, 38007db0_0. Attained after filtering with the full
set: 6 of 60, a catastrophe. Spot checks confirm these are faithful minings, not bugs: 38007db0 has
both train outputs 19x7 but test truths 29x8 and 25x7; 3a25b0d8 has both train outputs
left-right symmetric but neither test truth is; 4c3d4a41 conserves colour 6 on every train pair but
not on the test pair. With 2 or 3 demonstrations, coincidental regularities are common, and
ARC-AGI-2 test pairs are constructed to break exactly the regularities that are coincidental.

Pruning the 8 offending types leaves background_preserved, components_eq_input, mirror_ud,
palette_additive, palette_subset, shape_eq_input, shape_scaled. False kills after pruning: 0 of 19.

## (b) Payoff on the 4 leaked outputs: zero, with a sharp mechanism

With the pruned set, 0 blockers die on any leaked output and no truth enters the top 2 (0 recovered
of the 3-4 available). The full unpruned set shows why this is structural, not bad luck:

- 3a25b0d8_0 and 3dc255db_0: the full set kills every blocker (2 of 2, 5 of 5), but the same
  aggressive types (components_fixed, row_parity, mirror_lr; colour_count_conserved,
  immutable_colours, col_parity) kill the truth too. Selectivity and safety are the same axis.
- 142ca369_0 and 4c7dc4dd_1: no mined invariant of any type separates blockers from truth; the
  blockers satisfy everything the demonstrations certify.

The safe types are demo-evident properties (palette subset, input shape, background) that TTT has
already conditioned the model on, so its wrong candidates virtually never violate them.

## (c), (d) Overall

kgmon top-2 attained: 15 before, 15 after the pruned filter (no decrease, and no increase). Of the
41 truth-absent outputs, 4 have their entire pool killed by the pruned set and could route to a
fallback if one existed; count only, no such fallback is built.

## Leave-one-task-out

Invariant parameters are mined per task from its own train pairs and never cross tasks; the one
cross-task fitted choice is which invariant TYPES survive pruning. Scoring that choice honestly
(prune on the other 38 tasks' false kills, evaluate the held-out task) gives LOO attained 11
versus baseline 15, with 5 false kills (mirror_lr, shape_fixed and palette_fixed each false-kill
on only one task, so no other task's evidence prunes them). The in-sample zero-false-kill property
does not transfer even within this slice.

## Verdict against the pre-registered bar

Bar was +2 outputs recovered with 0 false kills. Result: 0 recovered, 0 false kills only after
in-sample pruning, LOO net -4 outputs. NOT MET, on both the payoff side and the honest-safety
side. A2 joins A1 in section 2 of score-levers: the selection headroom (bound 18, attained 15) is
real but reachable by neither statistic reweighting nor demonstration-mined hard constraints.

## Hidden-set transfer risk, honestly

The pruned filter's zero-false-kill property was fitted on this slice's 19 truths, and LOO already
shows that choice failing to transfer within the slice (11 versus 15), so on the hidden set the
expected effect of any variant of this filter ranges from zero to several lost credits. The zero
payoff is the transferable part: safe invariants are demo-evident properties the TTT-adapted model
already respects, a mechanism independent of which evaluation set is used. Deploying A2 in any form
risks real losses for a measured upside of zero, so it must not be queued.
