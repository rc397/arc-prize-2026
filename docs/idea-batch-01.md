# Idea batch 01 — triage

~114 proposals, deduplicated to about 70 distinct mechanisms. Screened against
[idea-triage.md](idea-triage.md). Two CPU measurements were run today specifically to settle clusters
of them, and both changed the answer.

**Result: 5 survive to a rig experiment, 1 is the clear top pick, 4 are high-risk and priced rather
than dismissed, the remaining ~60 are closed with evidence.**

A note on the ones deliberately breaking the rules. The gates are heuristics from our own
measurements, not laws. Gate 3 — the 16-token vocabulary — is the deepest constraint in the system
and the one nothing we have tried has attacked. An idea that violates it is expensive and likely to
fail, and it is also the only place a large unexpected win could come from. Those are in Part 4 with
costs attached, not in the kill list.

---

## Part 1: the two measurements that settled clusters

**Background is not 80% of a grid, and colour 0 is usually not the background.**

| over 530 output grids | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| colour-0 share | 0.00 | 0.00 | **0.00** | 0.48 | 0.76 |
| modal-colour share | 0.39 | 0.51 | **0.66** | 0.80 | 0.89 |

Only 26% of output grids reach 80% background. More than half contain **no colour-0 cells at all**,
so any rule keyed on `pixel != 0` mis-identifies the background on most tasks. A background
re-weighting changes the loss by about 2x, not 10x.

**But 88% of a median output is a verbatim copy of its input.**

| among shape-matched pairs | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| cells identical to the input | 0.62 | 0.76 | **0.88** | 0.94 | 0.96 |

370 of 530 pairs are shape-matched (69.8%), and 81 of 120 tasks are shape-matched throughout
(67.5%). Among those, 69.5% of pairs are at least 80% unchanged.

The gradient is dominated by **copying**, not by background. That relocates the whole loss-shaping
cluster onto a different and larger quantity.

---

## Part 2: what survives

Ranked. Every one still has a bad prior — 24 levers are closed and none moved the score — but these
have a mechanism and measurable room.

### 1. Mask the loss on unchanged cells (top pick)

*"Mask out the loss entirely for tokens that remain spatially unchanged between the input and output
grids."*

The only idea in the batch whose structural room we measured at **88%**. It does not assume which
colour is background, which is where the background cluster fails. It redirects gradient from
echoing to transforming.

Why it might still be null: the model has to learn *where* to copy as well as what to change, and
masking removes the supervision that teaches copying. The obvious failure is a model that transforms
correctly and corrupts the untouched regions. **The experiment must therefore check both**, not just
truth-in-pool.

Applies cleanly to the 67.5% of tasks that preserve shape. Three arms: base, masked, and a control
masking a random equal-sized subset so "less supervision" and "supervision on the right cells" are
separable.

### 2. Binary change-mask auxiliary

*"Train the adapter to output a binary mask of modified versus unmodified cells before predicting
exact colour values."*

Same insight as #1, expressed as an auxiliary objective rather than a loss mask. Rides the same 88%.
Weaker than #1 because it adds tokens the model was never pretrained to emit, so it partly collides
with Gate 3. Worth running only if #1 shows something.

### 3. Canonical palette pre-sorting

*"Deterministic colour remapping strictly sorted by total pixel count per demonstration."*

Genuinely untested and upstream. It removes a degree of freedom the adapter currently has to absorb.

The catch nobody has priced: **AIRV's vote depends on colour-permutation diversity across the 16
views.** Canonicalising the palette makes views more alike, which should improve per-view accuracy
and shrink the vote's diversity at the same time. Those pull opposite ways and the net is not
predictable from argument. That makes it a real experiment rather than a guess.

### 4. Depth-decayed DFS threshold

*"Decay the cumulative log probability threshold recursively as tree depth increases."*

One of the few decode ideas that changes **what is in the pool** rather than the order it is found
in. Best-first ordering was null precisely because it only reordered. A depth-dependent threshold
admits subtrees a flat 0.2 floor prunes.

Cheap to test offline on the logged DFS trees we already have — 30,480 recorded forward passes —
without any new GPU decoding.

### 5. Contrastive loss between augmented views

*"Contrastive loss objectives between augmented views of the same demonstration pair."*

Upstream, untested, and it targets representation rather than fit, which is the one thing the
warm-start result did not already rule out. Lowest confidence of the five; listed because it is not
closed by anything we have.

---

## Part 3: closed, with the evidence

### Optimisation dynamics — about 30 ideas, one measurement kills them all

Layer-differentiated learning rates, cyclic schedules, sequential unfreezing, gradient noise on
plateau, stochastic weight averaging, optimizer momentum reset, dynamic LoRA alpha, batch size one,
gradient accumulation variants, curriculum by grid size, reverse-chronological ordering, adaptive LR
by grid area, loss-variance early termination, dynamic rank pruning, weight-decay ramping, freezing
bottom-half layers, K/V-only training, feedforward-only training, layer-reversed updates,
interleaved zero-LR steps.

**The evidence is direct rather than inferential.** Warm-starting an adapter from a same-rule
neighbour produced a **30% lower final loss on 7 of 8 tasks** and changed coverage by **zero** (6 vs
6). Training loss already sits at 0.0006–0.09 and does not separate solved from missed tasks. We have
measured that better optimisation arrives faster at the same place. Every idea in this class improves
optimisation.

### Downstream filtering and pruning — about 12 ideas

Palette-frequency budgeting, object-size pruning, colour-ratio early termination, sequence-length
pruning, branch caps, diversity penalties on early tokens, logit penalties on control tokens,
spatial-prior logit biasing, restricting the search to colours in the test input.

Gate 1, plus a direct measurement: invariant mining as a filter recovered **0** candidates and killed
**11 of 19** truths. Selectivity and safety are the same axis here. Any rule sharp enough to prune
wrong candidates prunes right ones at a similar rate.

### Already measured — about 8 ideas

| idea | outcome |
|---|---|
| output→input inversion (proposed 3 times) | 4 vs 4 on 7 matched tasks |
| connected-component crops / sub-grid tiling (twice) | 3 vs 3 on 5 matched tasks; only 5.8% of cells are removable margin |
| chained RE-ARC DSL generators | done: 402 clean rules, corpus built, learnability probe running |
| auxiliary input reconstruction | pre-fitting: 7 / 7 / 6, exact null |
| dynamic batch width in decode | measured; a memory cliff on our rig, unknown on L4s |
| dynamic step allocation / early TTT termination | the cap test landed inside noise; loss does not separate outcomes, so the gate signal does not exist |

### Background-keyed loss shaping — about 8 ideas

Background down-weighting, focal loss on rare colours, spatial-distance weighting from non-zero
pixels, exponential background decay, masking the modal colour, boundary-only loss, loss normalised
by unique colour count, gradient scaling by distance from centre.

Measured today: median background share **66%**, not 80%, and **colour-0 median share 0.00**. The
premise is wrong in both magnitude and identification. Superseded by #1, which acts on 88%.

### Augmentations that change the task — about 6 ideas

Toroidal wrap-around translation, zero-padded random translation, checkerboard input masking,
single-pixel noise injection, blank row/column permutation, dummy background row injection,
concatenating two tasks side by side, random object recolouring with a static target.

Gate 5. ARC rules routinely reference grid edges, absolute position, and object identity. A shape
that wraps the boundary is a different shape; a translated grid breaks edge-relative rules; recolour
with a static target teaches that colour does not matter when for many tasks it is the whole rule.
These inject **false supervision**, and the collator will train on it as if true.

---

## Part 4: high-risk, priced not dismissed

These break Gate 3 knowingly. Gate 3 is also the deepest thing we have never attacked, so the ranking
is by cost-to-find-out rather than by likelihood.

**Delimiter-based RLE / repurposing control tokens as structural flags.** Requires the collator, the
serializer and the DFS output parser to agree on a syntax the base has never seen, and one parse
error corrupts a whole grid. Cost: about two days of engineering before a single measurement. Room:
run-length saves 13.9% of tokens. **The room does not justify the cost** — but the general move,
teaching the base a new in-vocabulary syntax during TTT, is the only route to more expressive output
we have, and this is the cheapest instance of it.

**Coordinate interleaving / digit position prefixes / forced dimension prefixes.** Same class. The
strongest argument for them is that our decoder currently has no way to know the output height until
it emits it, and #4 above (depth-decayed threshold) plus row-count priors would benefit from an
explicit dimension token. The strongest argument against is that every one of these lengthens the
prompt, and prompt length is what drives the memory cliff and the truncation.

**Column-major output ordering.** Cheapest of the four to try — a serialization flag, no parser
change — and it tests whether the pretrained row-major prior is load-bearing. Probably it is, and
probably it fails, but it costs one rig night and would tell us something about how much the base's
serialization prior contributes.

**Two-by-two summary grid before the full output.** Asks the model to commit to a coarse plan first.
Conceptually the most interesting of the risky set, because it is the only proposal in the batch that
tries to give the model something like intermediate reasoning inside a 16-token vocabulary. Also the
most likely to simply confuse a model pretrained to emit grids directly.

---

## Part 5: what I would actually run

1. **Unchanged-cell loss masking**, three arms, on the 67.5% of tasks that preserve shape. One rig
   night. It has the largest measured room of anything here.
2. **Depth-decayed DFS threshold**, replayed offline against the 16 logged trees. No GPU at all.
3. **Canonical palette pre-sorting**, if 1 and 2 are null, because its interaction with vote
   diversity makes it genuinely unpredictable.

Everything else waits on those, and the honest prior is that all three are null like the 24 before
them. What has changed is that the top pick is aimed at a quantity we measured at 88% rather than at
one we assumed.

---

## Part 6: what happened when the plan above was run

Ryan asked for the whole list to be worked rather than sampled: queue what needs the card, and run
anything that does not need it on CPU anyway, including the ideas already crossed out. Five CPU
measurements went out that evening. Two of the three survivors above did not survive them, and the
thing that replaced them came from neither list.

### Survivor 2 died, and left a better question behind

Replaying the 16 logged DFS trees, exactly, under every non-increasing threshold schedule
(`sim_dfs_budget.py`; tightening is exact offline because cumulative NLL only grows along a path, so
ancestors of a surviving node survive automatically):

| | truths | wrongs |
|---|---|---|
| headroom under the 1.609 ceiling, median | **1.600** | 0.846 |
| depth at emission, median | **199** | 143 |

Truths are near-certain completions that arrive **deep**. Wrongs are marginal ones that arrive
**shallow**. The proposal taxes depth, so it taxes precisely the population it wanted to protect,
and at matched cost a flat tightening beats it on every row of the table. What the replay found
instead is the budget: **12 of 16 decodes hit their wall clock**, and 29.8% of lane-nodes sit in
subtrees that emit nothing at all.

So the real lever is the flat floor. The pool replay (`sim_prune_replay.py`, 60 outputs, 432
candidates) disagrees with the tree replay about how far it goes — free to 0.25, then 19 truths fall
to 16 — and neither can see what the returned budget buys, because those candidates were never
generated. That is now a queued run, not an argument.

### Survivor 3 died twice

`sim_palette_canon.py`. Applied before the deployed augmentation, canonicalisation is provably a
**no-op**: `permute_rnd_all_` draws uniformly from all 10! bijections, and a fixed map composed with
a uniform one is uniform. Applied per demonstration as written, the per-demonstration maps disagree
about where a shared colour goes in **92% of tasks**, on the 50% of tasks where the palette itself
carries the rule. The only version that changes anything is canonicalising *instead of* permuting,
which is an ablation of colour diversity and should be argued as one.

### The rule-breaker in Part 4 was already in the pipeline

**Column-major output ordering** is the transpose view, and `augment` keeps it —
`mod(np.transpose, keep=True)`, arc_loader.py:250. Every training set already contains the
column-major reading. No run needed.

### The crossed-out filters, priced one by one

Nine task-mined rules replayed over the pool. Seven false-kill. The other two — palette contained in
the seen colours, and shape equal to the test input's where the task preserves shape — kill
**nothing at all**, on any candidate, ever. That is the more useful half: those constraints are not
worth enforcing because the model already satisfies them. Gate 1 was right, and this is what it
looks like rule by rule rather than by analogy.

### The crossed-out augmentations, priced too

`sim_aug_damage.py`. Content touches an edge on **79% of grids** and **93% of tasks**; the cells the
rule changes touch the border on **65%** of shape-matched pairs; the count of uniform rows and
columns changes across **72%** of them. Wrap, translate and line-permutation are not slightly noisy
views, they are false demonstrations on most of the set, and the collator trains on them as true.

### What is queued on the card

In order, one process, resumable, chained behind the composition probe:

1. `rig_maskloss.py` — survivor 1, three arms. Alignment verified on CPU first: 58% of assistant
   spans qualify and 91% of their cells mask, which matches the 88% the grids predicted.
2. `rig_dfsfloor.py` — the floor, three decodes off one adapter per task, fixed wall clock.
3. `rig_dimprefix.py` — the dimension header, Gate 3 violated on purpose. Round-trips exactly in a
   16-token vocabulary. Gate 4 already warns it will be weak: the decoder agrees with itself about
   shape on **82%** of pools, so the header has little to reject and what it really prices is the
   cost of teaching this base new syntax during TTT.

Scoreboard for the batch: of 5 survivors, 2 closed offline, 1 was already in the pipeline, 2 are on
the card, and the best thing to come out of the evening was a decode constant nobody proposed.
