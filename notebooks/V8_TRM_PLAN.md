# v8, the TRM ensemble arm

## SHELVED 2026-07-21, refuted by the authors before we built it

The NVARC first-place writeup, by the authors of the exact model we run, reports the TRM ensemble result
directly and it kills this plan. Their numbers.

- TRM standalone is weak. The ARC Prize organisation got 6.9 percent on three nodes of 8xH100, compute we
  cannot use. NVARC reached 7.5 percent on Kaggle and 10 percent post-deadline.
- TRM added to their Qwen3 2B lifted it from 21.53 to 22.50, about one point.
- TRM added to their Qwen3 4B lifted 27.22 to 27.22, exactly zero.

Their mechanism, quoted. "Most of the puzzles solved by TRM were solved by Qwen3, hence TRM added nothing
there. However, about 2 or 3 puzzles solved by TRM were not solved by Qwen3. Unfortunately, these were not
always picked by Qwen3 scoring." That is the selection-ceiling problem we already measured on our own
pipeline, confirmed independently on theirs. The disjoint coverage barely exists and the selector cannot
reliably keep it.

We run the 4B, at 30.56, above their 4B's 27.22. TRM adds a bit only to a weak base, and going to the 2B
to obtain that bit would drop us from 30.56 to the low twenties to chase one point. The bet this document
was built on, disjoint coverage, is real in principle and measured near zero in practice for a strong 4B.
The plan below is kept for the record and for the reasoning, not as work to do.

Source, NVARC solution writeup, ARC Prize 2025, authors ivan/sorokin and CPMP/cpmpml,
https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc

---

## Original plan, retained for the record

Design document, not a runnable artifact yet. Unlike v7, we do not hold the source notebook, so this
records the plan, the integration point, and the honest unknowns, and names the one action that unblocks
a real build.

## The bet, stated plainly

The v7 pipeline is bimodal. A task is solved robustly or missed entirely, with no partial credit, so
refinement levers cannot help. We have now confirmed this the hard way. Search width, the attempt-slot
split, and the selector choice were each measured and each added nothing. The candidate side and the
voting side are saturated.

The only remaining thing that moves a bimodal system is disjoint coverage, a second source that solves
tasks the 4B LLM misses rather than a subset of the ones it already solves. TRM is a Tiny Recursive Model,
a different architecture from an LLM, so there is a real prior that its solves are disjoint. That prior is
the whole reason v8 is worth building. It is also the exact thing that must be measured before a submission,
because a second source that mostly overlaps the first is the failure mode, not the win.

## What v8 reproduces, and attribution

Greg Kamradt, president of the ARC Prize Foundation, publishes a fork of the sorokin pipeline that
ensembles a public TRM checkpoint. Per the standing ruling, v8 is a reproduction of that named public work,
credited explicitly wherever it appears, exactly as v7 credits sorokin and the ARChitects. Our contribution
is the measurement around it, specifically whether the TRM arm adds disjoint coverage on our own 4B's
misses, which is a question no one has answered for this exact stack.

Named public artifacts, to be confirmed on contact rather than trusted from notes.
- The fork notebook, believed to be `batch8_queue_trm2`.
- The TRM checkpoint, believed to be `arc-prize-trm-031-220708`.
- A smaller `qwen3_2b_grids15_sft141` base that the same fork attaches, likely to claw back budget.

## The integration point, which is the elegant part

The v7 selection layer already scores every candidate grid two ways, by how many augmented views produced
it and by its augmentation NLL under the LLM. A TRM output is just a grid. To enter the pool it needs those
same two numbers, and the augmentation NLL is computable for any grid by running the LLM forward on it as
the proposed answer under the eight augmentations, which is the identical machinery that scores an LLM
candidate. So a TRM candidate joins the same kgmon ranking with no new selection logic.

That means v8 changes exactly one thing over v7. It adds a second inference source whose grids are injected
into the candidate pool before selection runs. Everything downstream, selection and submission, is
untouched. This is why the ensemble is a clean addition rather than a rewrite.

## Known versus unknown

Known.
- The pipeline has the scoring hooks a TRM candidate needs, so injection is mechanically straightforward.
- The rig is idle and can host TRM experiments for free, since TRM is small and does not need the L4s.
- Disjoint coverage, if it exists, is worth more than any candidate-side lever because it is additive.

Unknown, and each must be resolved before trusting anything.
- The fork's internals. We have not read it. Every claim about how it wires TRM is inference until we do.
- TRM's coverage on our 4B's specific misses. This is the entire bet and it is unmeasured.
- Budget fit. v7.1 already runs about 11.3 hours of the 11.8 hour wall. A TRM pass per task risks the
  wall, and the 2B base in the fork may exist precisely to buy that time back. Both need measuring.
- The checkpoint's grid encoding and whether it matches our id-level representation.

## Step sequence and quota cost

| step | action | cost |
|---|---|---|
| 1 | Obtain the Kamradt fork notebook and the TRM checkpoint | free, no quota |
| 2 | Read the fork, confirm the TRM-to-pool wiring | free, laptop |
| 3 | Build v8 by transforming the fork the way v7 was built, paths to env, integration kept faithful | free, laptop |
| 4 | Smoke at ARC_N_EVAL=6, prove it runs, measure TRM per-task cost and the new total | small quota |
| 5 | Coverage test on the eval slice, record which tasks TRM solves that the LLM path missed | eval quota |
| 6 | Submit only if step 5 shows disjoint solves AND step 4 shows the budget fits | ~11 h quota |

## The gate, so this does not become another tuned result

v8 earns a submission only if the eval run shows TRM contributing solves the LLM path does not already get.
If TRM's solves are a subset, drop it, the same discipline that killed search width and the selector switch.
A second model that agrees with the first is not an ensemble, it is overhead.

## Immediate next action

Step 1. Pull the fork notebook and the checkpoint. This needs either the Kaggle API with the exact kernel
slug or a browser session, and it is the thing standing between this document and a real builder. Once the
source is in hand, v8 gets built the same afternoon v7 was.
