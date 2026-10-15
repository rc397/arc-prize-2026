# The invariant-mining candidate filter (lever A2) is dead by its own bar

Measured on 4 August 2026 with `scripts/pool_invariants.py` on `paper/data/pool_dump.json`, which
holds 60 outputs across 39 tasks, with demonstrations and solutions from the public evaluation files.
To reproduce it, run `python scripts/pool_invariants.py [POOL] [CHALLENGES] [SOLUTIONS]`. The script
re-checks the dump's `correct` flags against the solutions file when it loads.

Before filtering, the baseline matched what we expected. The truth is in the pool for 19 outputs, the
kgmon selector gets 15 of them into its top two, and the four it misses are 142ca369_0, 3a25b0d8_0,
3dc255db_0 and 4c7dc4dd_1.

## The design

For each task the script mines 15 kinds of invariant from its training pairs alone, and keeps one only
if it held on every pair. Every task in the pool has at least two pairs. That gave 214 invariants, a
median of six per task, and no task ended up with none. Any candidate that breaks a kept invariant
gets killed.

## False kills, which was the kill condition

With the full set, 11 of the 19 true answers die.

| invariant              | tasks mined | wrong candidates killed | true answers killed |
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

The true answers it killed were 2d0172a1_1, 4c3d4a41_0, 1ae2feb7_2, 53fb4810_0, 3a25b0d8_0,
45a5af55_0, 38007db0_1, 135a2760_0, 3dc255db_0, 3a25b0d8_1 and 38007db0_0. After filtering with the
full set the selector gets just 6 of 60, which is a disaster.

Spot checks show these are honest minings, not bugs. In 38007db0 both training outputs are 19 by 7,
but the test answers are 29 by 8 and 25 by 7. In 3a25b0d8 both training outputs are mirror-symmetric
left to right, but neither test answer is. In 4c3d4a41 colour 6 is conserved on every training pair
but not on the test pair. With two or three demonstrations, coincidences are common, and ARC-AGI-2
test pairs are built to break exactly the regularities that are coincidental.

Dropping the eight types that kill true answers leaves background_preserved, components_eq_input,
mirror_ud, palette_additive, palette_subset, shape_eq_input and shape_scaled. With those alone, no
true answer dies.

## What it buys on the four missed outputs

Nothing, and the reason is clear-cut. With the pruned set, none of the wrong candidates blocking a
missed output gets killed, and no true answer moves into the top two. The full set shows why this is
built in and not bad luck.

- For 3a25b0d8_0 and 3dc255db_0 the full set kills every blocker, 2 of 2 and 5 of 5. But the same
  aggressive types kill the true answer too. Those are components_fixed, row_parity and mirror_lr in
  the first case, and colour_count_conserved, immutable_colours and col_parity in the second. The
  filter can't be selective without being unsafe, because those are the same knob.
- For 142ca369_0 and 4c7dc4dd_1 no invariant of any type separates the blockers from the true answer.
  The blockers satisfy everything the demonstrations vouch for.

The safe types are properties you can see straight from the demonstrations, like the palette, the
input shape and the background. Test-time training has already taught the model those, so its wrong
candidates almost never break them.

## Overall

The selector gets 15 into its top two before the pruned filter and 15 after, with no loss and no gain.
Of the 41 outputs where the truth never appears, the pruned set kills the entire pool for four. Those
could be routed to a fallback if one existed. I only counted them and built no fallback.

## Leave-one-task-out

The invariants themselves are mined per task from that task's own training pairs and never cross
tasks. The one choice fitted across tasks is which types survive pruning. Scoring that choice
honestly, pruning on the false kills of the other 38 tasks and testing on the one held out, gives 11
in the top two against the baseline's 15, with 5 false kills. mirror_lr, shape_fixed and
palette_fixed each false-kill on only one task, so no other task's evidence prunes them. The clean
zero-false-kill result from pruning on everything doesn't hold up even within this slice.

## Against the bar set in advance

The bar was two outputs recovered with no false kills. The result is none recovered, no false kills
only after pruning on the same data, and a net loss of four outputs under leave-one-task-out. It fails
on both the payoff and an honest test of safety. A2 joins A1 in section 2 of the score-levers note.
The selection headroom, a bound of 18 against 15 attained, is real, but neither reweighting statistics
nor hard constraints mined from demonstrations reaches it.

## Would it transfer to the hidden set

The pruned filter's zero false kills were fitted to this slice's 19 true answers, and
leave-one-task-out already shows that choice failing inside the slice, 11 against 15. On the hidden
set, any version of this filter would land somewhere between no effect and several lost credits. The
zero payoff is the part that does transfer. The safe invariants are properties visible in the
demonstrations that the adapted model already respects, whichever evaluation set you use. Deploying
any form of A2 risks real losses for a measured upside of nothing, so it must not be queued.
