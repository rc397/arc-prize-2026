# Research brief: improving ARC-AGI-2 on a single consumer GPU

Use this as a prompt for a research agent or as your own checklist. The most important part is the
section on what separates a promising idea from a dead end here, because it encodes the one hard-won
finding that should steer everything.

## Objective
Raise the ARC-AGI-2 solve rate of a 4B open model running offline on one 16 GB consumer GPU, beating
the current best of 16.28% blended on the 120-task public eval (11.94 on the hidden main-track set),
without breaking the constraints below. The deliverable is a measured method, not a hypothesis.

## Hard constraints (a candidate method must fit all of these)
- Compute: one 16 GB consumer GPU (RTX 5070 Ti class) for all measurements; the scored submission runs
  offline on 4x L4 within 12 hours for 240 tasks, about $50.
- Model: open weights only. Base is NVARC qwen3-4b-grids (4B), driven at the token-ID level.
- Interface: a 16-token vocabulary (digits, newline, control tokens). The model emits grids only. It
  cannot read or write natural language, so no NL chain-of-thought, no NL self-critique, no NL feedback
  loop.
- Data: permissive licences only (ARC-AGI-2 Apache-2.0, RE-ARC MIT). Never train on the 120 public eval
  or the hidden sets.
- Stack: inference is llama.cpp (GGUF Q8_0); training is PyTorch LoRA in bf16 (4-bit does not load on
  this Blackwell+Windows stack); the two are never co-resident in VRAM.

## What is already established (do not rediscover or repeat)
Working levers:
- no-TTT AIRV (8 D4 augmentations, reverse, vote): 12.79% public, 6.53 hidden.
- Per-task TTT (rank-16 bf16 LoRA, 20 steps, dropout 0.1, adapter scale 1.25, served via llama.cpp
  hot-swap): 16.28% blended, 11.94 hidden. This is the main working lever.
- The inference engine is the dominant systems lever (13x decode over eager).

Dead ends, with their reasons:
- Continued fine-tune on ARC-2 + RE-ARC: 11.63%, worse than stock. The base is already grid-tuned; more
  public data does not transfer.
- Verifier-guided vote selection: null. With TTT the train-pair check saturates (the adapter memorises
  the pairs); without TTT it is silent on hard tasks.
- Expert-iteration self-training (STaR): null. The base is saturated on its training set (95.5% solved
  robustly, 1.5% near-miss), so a verified-correct trace equals the ground-truth target the fine-tune
  already used.

The unifying finding that must steer all further work:
- The base is bimodal on ARC tasks: it either solves a task robustly or misses it completely, with
  almost no partial-understanding middle. So every method that refines, re-ranks, or selects over the
  model's own understanding has nothing to grip on the tasks it misses. The only lever that helped,
  per-task TTT, works because it manufactures new capability for a task rather than refining existing
  partial understanding.

## The criterion that separates a promising idea from a dead end here
Pursue a direction only if it creates or exploits a partial-credit signal that is valid on tasks the
base cannot already solve. The bimodal base supplies none itself, so the method must introduce one
externally. The cleanest external signal is an artifact that can be checked against the train pairs
deterministically and graded continuously: a program or transformation that reproduces k of n train
pairs is partial credit even when the model is lost. Any idea that only re-ranks, re-weights, or
re-samples the model's grid outputs will hit the bimodal wall and should be deprioritised.

Secondary criteria: it should compound with per-task TTT (not replace it); fit the 16 GB / 12 h /
16-token box; be measurable against 12.79% and 16.28% on the 120 public eval with a pre-registered
keep/drop gate; and not assume natural-language reasoning.

## Directions to research, ranked by fit to that criterion
1. Program / DSL search with execution verification (best fit). Propose candidate transformations
   (from the 4B model, a learned policy, or enumeration over a compact ARC DSL), execute them on the
   train inputs, keep those that reproduce the train outputs, apply the survivor to the test. Partial
   credit is native, it works on tasks the base misses, and verified programs become real self-training
   data, reviving the weight path that saturated. Open question: can a useful search fit 12 h / 16 GB
   with a 4B proposer.
2. TTT as the partial-credit manufacturer, scaled. Leave-one-out adapters (train on n-1 pairs, verify
   honestly on the held-out pair), adapter ensembles, curriculum over augmentation strength. Open
   question: does honest leave-one-out verification re-rank where the saturated checks could not.
3. A grid-native learned verifier or value model (same 16-token interface) trained on held-out
   judgements, used to guide selection or search, avoiding the saturation trap.
4. Diversity ensembling with a second, genuinely different open grid model, since augmentations of one
   model do not stack.
5. Neuro-symbolic object reasoning: parse grids into objects and relations and search over that
   structure.

## Evaluation protocol any proposal must follow
- Measure on the 120 public eval on the single 16 GB GPU; report exact-match on both attempts.
- Compare head-to-head on the same tasks against no-TTT AIRV (12.79%) and the TTT blend (16.28%).
- Pre-register a keep/drop gate (for example, ship only if blended public accuracy beats 16.28% and it
  still fits 12 h on 240 tasks).
- Report negative results with their mechanism; never train on eval; confirm submission timing on
  4x L4 before claiming a leaderboard number.

## Core research questions
1. What is the smallest external partial-credit signal (program execution, honest leave-one-out TTT,
   learned verifier) that re-ranks candidates where the base's own checks are silent?
2. Can program search over a compact ARC-2 DSL, seeded by the 4B model, fit 16 GB / 12 h and beat
   16.28%?
3. Does distilling verified-correct programs (or leave-one-out TTT successes on genuinely hard tasks)
   finally move the weights where flat fine-tuning and STaR did not?
4. Is the bimodal behaviour a property of the 4B scale, the 16-token interface, or ARC-2's
   compositional difficulty, and which can a compute-poor method change?
