# Idea batch 01, sorted

About 114 proposals came in, which boiled down to roughly 70 distinct mechanisms. I screened them
against [idea-triage.md](idea-triage.md). I also ran two CPU measurements that day to settle whole
clusters of them, and both changed the answer.

Five survive to a rig experiment, and one of those is the clear top pick. Four are high-risk, so I
priced them rather than throwing them out. The remaining 60 or so are closed, with the evidence below.

A word on the ideas that break the rules on purpose. The gates are rules of thumb drawn from our own
measurements, not laws. Gate 3, the 16-token vocabulary, is the deepest constraint in the system, and
nothing we've tried has gone after it. An idea that breaks it is expensive and likely to fail, but it's
also the only place a big surprise could come from. Those ideas are in Part 4 with costs attached,
not in the kill list.

## Part 1. The two measurements that settled whole clusters

**The background is not 80% of a grid, and colour 0 usually isn't the background.**

| across 530 output grids | 10th pct | 25th pct | median | 75th pct | 90th pct |
|---|---|---|---|---|---|
| share of colour 0 | 0.00 | 0.00 | **0.00** | 0.48 | 0.76 |
| share of the most common colour | 0.39 | 0.51 | **0.66** | 0.80 | 0.89 |

Only 26% of output grids are 80% background. More than half contain no colour 0 at all, so any rule
that treats "not zero" as foreground gets the background wrong on most tasks. Down-weighting the
background changes the loss by about a factor of two, not ten.

**But 88% of a typical output is copied straight from its input.**

| among pairs where input and output are the same shape | 10th pct | 25th pct | median | 75th pct | 90th pct |
|---|---|---|---|---|---|
| share of cells identical to the input | 0.62 | 0.76 | **0.88** | 0.94 | 0.96 |

370 of the 530 pairs keep the same shape (69.8%), and 81 of 120 tasks keep it throughout (67.5%).
Among those, 69.5% of pairs are at least 80% unchanged.

So the gradient is dominated by copying, not by background. That moves the whole loss-shaping cluster
onto a different and bigger quantity.

## Part 2. What survives

These are ranked. Every one still has poor odds, since 24 levers are already closed and none moved the
score, but these have a mechanism and measurable room to work.

### 1. Mask the loss on unchanged cells (top pick)

The proposal was to drop the loss entirely on tokens that stay the same between input and output.

It's the only idea in the batch whose room we actually measured, at 88%. It doesn't assume which colour
is the background, which is where the background ideas fall down. It moves gradient away from echoing
the input and toward the transformation.

Here's why it might still come up empty. The model has to learn where to copy as well as what to
change, and masking takes away the supervision that teaches copying. The obvious failure is a model
that transforms correctly and then corrupts the regions it should have left alone. So the experiment
has to check both, not just whether the truth reaches the pool.

It applies cleanly to the 67.5% of tasks that keep their shape. There are three arms. One is the base,
one is masked, and one is a control that masks a random subset of the same size, so we can tell less
supervision apart from supervision on the right cells.

### 2. A binary change mask as a side objective

This one trains the adapter to output a mask of which cells change before it predicts the colours.

It's the same insight as number 1, expressed as an extra objective rather than a masked loss, and it
rides on the same 88%. It's weaker than number 1 because it adds tokens the model was never trained to
produce, which partly runs into Gate 3. It's only worth running if number 1 shows something.

### 3. Sorting the palette into a fixed order first

This remaps colours in a fixed way, sorted by how many pixels of each colour appear in each
demonstration.

It's genuinely untested and acts early in the pipeline. It removes a degree of freedom the adapter
currently has to absorb.

The catch nobody has priced is that the vote depends on colour-permutation diversity across the 16
views. Fixing the palette makes the views more alike, which should help each view's accuracy and shrink
the vote's diversity at the same time. Those pull in opposite directions, and you can't predict the net
effect by argument. That makes it a real experiment rather than a guess.

### 4. A search threshold that shrinks with depth

This lowers the cumulative log-probability threshold as the search tree gets deeper.

It's one of the few decoding ideas that changes what ends up in the pool, rather than the order things
are found in. Best-first ordering came up empty precisely because it only reordered. A threshold that
depends on depth lets in subtrees that a flat 0.2 floor cuts off.

It's cheap to test offline on the search trees we already logged, 30,480 recorded forward passes,
without any new decoding on the GPU.

### 5. A contrastive loss between augmented views

This adds a contrastive objective between augmented views of the same demonstration pair.

It acts early, it's untested, and it targets the model's representation rather than its fit, which is
the one thing the warm-start result didn't already rule out. It's the least likely of the five, and
it's on the list because nothing we have closes it.

## Part 3. Closed, with the evidence

### Optimisation tweaks, about 30 ideas killed by one measurement

These were per-layer learning rates, cyclic schedules, unfreezing layers in sequence, gradient noise on
a plateau, weight averaging, resetting optimiser momentum, a dynamic LoRA alpha, batch size one,
variants of gradient accumulation, a curriculum by grid size, reverse-chronological ordering, learning
rate scaled by grid area, stopping early on loss variance, pruning rank on the fly, ramping weight
decay, freezing the bottom half of the layers, training only K and V, training only the feed-forward
layers, updating layers in reverse, and interleaving zero-learning-rate steps.

The evidence is direct, not inferred. Starting an adapter from a neighbour that shares the same rule
gave a 30% lower final loss on 7 of 8 tasks and changed coverage by nothing, 6 against 6. Training loss
already sits between 0.0006 and 0.09 and doesn't separate solved tasks from missed ones. We measured
that better optimisation just gets to the same place faster. Every idea in this group improves
optimisation.

### Filtering and pruning candidates after the fact, about 12 ideas

These were budgeting by palette frequency, pruning by object size, stopping early on colour ratios,
pruning by sequence length, capping branches, penalising early tokens for lack of diversity, penalising
logits for control tokens, nudging logits with a spatial prior, and limiting the search to colours that
appear in the test input.

Gate 1 rules these out, and so does a direct measurement. Using mined invariants as a filter recovered
no candidates and killed 11 of 19 true answers. Here, being selective and being safe are the same
dial. Any rule sharp enough to prune wrong candidates prunes right ones at about the same rate.

### Already measured, about 8 ideas

| idea | outcome |
|---|---|
| inverting output to input (proposed three times) | 4 against 4 on 7 matched tasks |
| cropping to connected components or tiling sub-grids (twice) | 3 against 3 on 5 matched tasks, and only 5.8% of cells are removable margin |
| chaining RE-ARC generators | done, with 402 clean rules, the corpus built and the learnability probe running |
| reconstructing the input as a side objective | pre-fitting gave 7, 7 and 6, an exact null |
| changing the decoding batch width on the fly | measured, with a memory cliff on our rig and unknown behaviour on L4s |
| allocating steps dynamically or stopping training early | the time-cap test landed inside the noise, and loss doesn't separate outcomes, so the signal it needs doesn't exist |

### Loss shaping keyed on the background, about 8 ideas

These were down-weighting the background, focal loss on rare colours, weighting by distance from
non-zero pixels, exponential decay on the background, masking the most common colour, loss only on
boundaries, normalising loss by the number of colours, and scaling gradients by distance from the
centre.

Measured that same day, the median background share is 66%, not 80%, and the median share of colour 0
is zero. The premise is wrong on both size and identification. Number 1 supersedes them, since it works
on the 88%.

### Augmentations that change the task, about 6 ideas

These were wrap-around translation, random translation with zero padding, checkerboard masking of the
input, single-pixel noise, shuffling blank rows and columns, inserting a dummy background row, putting
two tasks side by side, and recolouring objects at random while keeping the target fixed.

Gate 5 rules these out. ARC rules routinely depend on grid edges, absolute position and which object
is which. A shape that wraps around the border is a different shape. A translated grid breaks rules
tied to the edges. Recolouring with a fixed target teaches that colour doesn't matter, when for many
tasks colour is the whole rule. These feed in false supervision, and training would treat it as true.

## Part 4. High risk, priced rather than dismissed

These break Gate 3 knowingly. Gate 3 is also the deepest constraint we've never attacked, so I ranked
them by how much it costs to find out, not by how likely they are to work.

**Run-length encoding with delimiters, or reusing control tokens as structural markers.** The training
code, the serializer and the search's output parser would all have to agree on a syntax the base has
never seen, and one parse error corrupts a whole grid. That's about two days of engineering before a
single measurement. Run-length coding saves 13.9% of tokens, and that room doesn't justify the cost.
But the general move, teaching the base a new syntax inside its vocabulary during test-time training,
is the only route we have to more expressive output, and this is the cheapest version of it.

**Interleaving coordinates, prefixing digit positions, or forcing a size prefix.** Same family. The best
argument for them is that the decoder currently can't know how tall the output will be until it writes
it, and the depth-dependent threshold in number 4, along with priors on row count, would benefit from
an explicit size token. The best argument against is that every one of these makes the prompt longer,
and prompt length drives the memory cliff and the truncation.

**Writing the output column by column.** The cheapest of the four to try, since it's a serialization
flag with no parser change, and it tests whether the model's learned row-by-row habit carries weight.
It probably does, and this probably fails, but it costs one night on the rig and would tell us how much
the base's serialization habit contributes.

**A two-by-two summary grid before the full output.** This asks the model to commit to a rough plan
first. It's the most interesting of the risky set, because it's the only proposal that tries to give the
model something like intermediate reasoning inside a 16-token vocabulary. It's also the most likely to
just confuse a model trained to write grids directly.

## Part 5. What I'd actually run

1. **Masking the loss on unchanged cells**, with three arms, on the 67.5% of tasks that keep their
   shape. One night on the rig. It has the most measured room of anything here.
2. **The depth-dependent search threshold**, replayed offline against the 16 logged trees, with no GPU
   at all.
3. **Sorting the palette**, if the first two come up empty, because its interaction with vote diversity
   makes it genuinely unpredictable.

Everything else waits on those, and honestly all three will probably come up empty like the 24 before
them. What's different is that the top pick is aimed at a quantity we measured at 88%, rather than one
we assumed.

## Part 6. What happened when the plan ran

I decided to work through the whole list rather than sample it. Anything that needed the card
went in the queue, and anything that didn't ran on the CPU anyway, including the ideas already crossed
out. Five CPU measurements went out that evening. Two of the three survivors above didn't make it, and
what replaced them came from neither list.

### Survivor 2 died and left a better question

I replayed the 16 logged search trees exactly under every threshold schedule that never loosens
(`sim_dfs_budget.py`). Tightening is exact offline, because cumulative NLL only grows along a path, so
if a node survives, its ancestors do too.

| | true answers | wrong answers |
|---|---|---|
| median room left under the 1.609 ceiling | **1.600** | 0.846 |
| median depth when emitted | **199** | 143 |

True answers are near-certain completions that arrive deep. Wrong ones are marginal and arrive
shallow. The proposal penalises depth, so it penalises exactly the answers it meant to protect, and at
the same cost a flat tightening beats it on every row. What the replay turned up instead was the time
budget. 12 of the 16 decodes hit their wall-clock limit, and 29.8% of lane-nodes sit in subtrees that
never emit anything.

So the real lever is the flat floor. The pool replay (`sim_prune_replay.py`, 60 outputs and 432
candidates) disagrees with the tree replay about how far it goes. It's free up to 0.25, and beyond that
19 true answers fall to 16. Neither replay can see what the freed-up time would buy, because those
candidates were never generated. That's now a queued run rather than an argument.

### Survivor 3 died twice

Run before the usual augmentation, sorting the palette is provably a no-op (`sim_palette_canon.py`).
`permute_rnd_all_` draws evenly from all 10! colour permutations, and a fixed remapping followed by a
uniformly random one is still uniformly random. Applied per demonstration as proposed, the remappings
disagree about where a shared colour goes in 92% of tasks, and on half the tasks the palette itself
carries the rule. The only version that changes anything sorts the palette instead of permuting it,
which is really an experiment in removing colour diversity and should be argued as one.

### The rule-breaker in Part 4 was already in the pipeline

Writing column by column is just the transposed view, and the augmentation step already keeps it
(`mod(np.transpose, keep=True)`, arc_loader.py line 250). Every training set already contains the
column-wise reading, so no run was needed.

### The crossed-out filters, priced one at a time

I replayed nine rules mined from the tasks over the pool. Seven kill true answers. The other two, a
palette limited to colours already seen and a shape equal to the test input's on shape-keeping tasks,
kill nothing at all, on any candidate. That's the more useful half. Those constraints aren't worth
enforcing because the model already obeys them. Gate 1 was right, and this is what it looks like rule
by rule instead of by analogy.

### The crossed-out augmentations, priced too

Using `sim_aug_damage.py`, content touches an edge on 79% of grids and 93% of tasks. The cells a rule
changes touch the border on 65% of shape-keeping pairs. The number of uniform rows and columns changes
across 72% of them. So wrap, translation and line shuffling aren't slightly noisy views. They're false
demonstrations on most of the set, and training would treat them as true.

### What's queued on the card

These run in order as one resumable process, chained behind the composition probe.

1. `rig_maskloss.py`, survivor 1, with three arms. I checked the alignment on the CPU first. 58% of
   the answer spans qualify, and 91% of their cells get masked, which matches the 88% the grids
   predicted.
2. `rig_dfsfloor.py`, the floor, with three decodes from one adapter per task at a fixed wall clock.
3. `rig_dimprefix.py`, the size header, breaking Gate 3 on purpose. It round-trips exactly in a 16-token
   vocabulary. Gate 4 already warns it will be weak, because the decoder agrees with itself about shape
   on 82% of pools. So the header has little to reject, and what it really measures is the cost of
   teaching this base new syntax during training.

The tally for the batch is that 2 of the 5 survivors closed offline, 1 was already in the pipeline and
2 are on the card. The best thing to come out of the evening was a decoding constant nobody proposed.

## Part 7. The top pick ran, and it's the most informative negative we have

`rig_maskloss.py` covered 8 tasks and 14 outputs, with three arms at the same number of steps.

| arm | truth in pool | selected | candidates | error on copied cells | error on changed cells |
|---|---|---|---|---|---|
| base | **6** of 14 | 5 | 48 | 0.024 | 0.286 |
| unchanged cells masked | **3** of 14 | 3 | 39 | **0.095** | **0.062** |
| random mask of the same size | **6** of 14 | 6 | 44 | 0.021 | 0.146 |

Task by task, masking was worse on 3, tied on 5 and better on none. The random control tied on all 8.

The idea did exactly what it was built to do. Moving the gradient off the copied cells cut the error on
the cells the rule changes by a factor of 4.6, from 0.286 to 0.062. That's the biggest movement any
lever has produced on any quantity in this project.

And it lost, because it broke the other half. Error on the cells that should have been copied rose
fourfold. Since 88% of a typical output is copied, a model that transforms beautifully but mis-copies
one cell in ten still produces a wrong grid. Coverage fell from 6 to 3.

The control makes this conclusive rather than suggestive. Masking the same number of cells at random
was indistinguishable from the base on every task. So the loss doesn't come from having less
supervision, or from a smaller effective learning rate. It comes specifically from removing all
supervision on copying. A uniform 12% sample of every kind of cell is enough, and no supervision on
copying at all is fatal.

To be honest about how strong this is, the drop in coverage is 3 tasks of 8, which a sign test alone
wouldn't call significant. The split in cell-level error is measured over thousands of cells and isn't
close.

This changes the rest of the batch. The premise behind the whole loss-shaping cluster, that gradient
spent reproducing unchanged cells is wasted, is wrong. It's the half holding everything up. Survivor 2,
the binary change mask, isn't killed by this, since it adds an objective rather than removing
supervision. But its motivation is gone, and it should be argued again before anyone builds it.
