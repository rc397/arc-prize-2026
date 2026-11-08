# Research brief for improving ARC-AGI-2 on one consumer GPU

This is a checklist for deciding what to try next. The section that matters
most is the one on telling a promising idea from a dead end, because it carries the one hard-won
finding that should steer everything else.

## The goal

Raise the solve rate of a 4B open model running offline on one 16 GB consumer GPU. The current best is
16.28% blended on the 120 public evaluation tasks, and 11.94 on the hidden main-track set. A new method
has to beat that without breaking any of the constraints below, and the deliverable is a measured
method, not a hypothesis.

## Hard constraints

Any candidate method has to fit all of these.

- **Compute.** Every measurement runs on one 16 GB card, an RTX 5070 Ti or similar. The scored
  submission runs offline on four L4s and must finish 240 tasks within 12 hours, for about $50.
- **Model.** Open weights only. The base is NVARC's qwen3-4b-grids, a 4B model we drive at the level
  of token IDs.
- **Interface.** The vocabulary has 16 tokens, the digits plus a newline and a few control tokens. The
  model only emits grids. It can't read or write natural language, so there's no written
  chain-of-thought, no self-critique and no feedback loop in words.
- **Data.** Permissive licences only, meaning ARC-AGI-2 under Apache 2.0 and RE-ARC under MIT. Never
  train on the 120 public evaluation tasks or the hidden set.
- **Software.** Inference runs in llama.cpp on a Q8_0 GGUF. Training is a bf16 LoRA in PyTorch,
  because 4-bit won't load on this Blackwell and Windows setup. The two never share the card's memory
  at the same time.

## What we already know

These don't need rediscovering.

Here is what works.

- Augment-infer-reverse-vote with no test-time training, using the eight rotations and reflections,
  gets 12.79% on public and 6.53 on hidden.
- Per-task test-time training is the main lever that works. A rank-16 bf16 LoRA trained for 20 steps
  with dropout 0.1, applied at scale 1.25 and hot-swapped into llama.cpp, gets 16.28% blended and 11.94
  on hidden.
- The inference engine is the biggest systems lever, at 13 times the decoding speed of the default
  path.

And here is what doesn't, and why.

- **Continued fine-tuning on ARC-2 and RE-ARC** scored 11.63%, worse than the stock model. The base is
  already tuned on grids, and more public data doesn't transfer.
- **Using a verifier to guide the vote** did nothing. With test-time training the check against the
  training pairs saturates, because the adapter memorises those pairs. Without it the check says
  nothing on hard tasks.
- **Expert-iteration self-training (STaR)** did nothing. The base already solves 95.5% of its training
  set robustly, with only 1.5% near misses, so a verified-correct trace is the same target the
  fine-tune already used.

The one finding that should steer everything is this. The base is bimodal on ARC tasks. It either
solves a task solidly or misses it completely, with almost no partial understanding in between. So any
method that refines, re-ranks or selects over the model's own understanding has nothing to grab onto
on the tasks it misses. Per-task training is the one lever that helped, and it works because it builds
new capability for a task instead of polishing understanding the model doesn't have.

## How to tell a promising idea from a dead end

Only pursue a direction if it creates or uses a partial-credit signal that still means something on
tasks the base can't already solve. The bimodal base gives you no such signal, so it has to come from
outside. The cleanest outside signal is something you can check against the training pairs exactly
and grade smoothly. A program that reproduces k of n training pairs earns partial credit even when the
model is lost. Any idea that only re-ranks, re-weights or re-samples the model's grids will hit the
bimodal wall and should go to the back of the queue.

A good idea should also build on per-task training rather than replace it. It has to fit in 16 GB, 12
hours and 16 tokens. It has to be measurable against 12.79% and 16.28% on the 120 public tasks, with a
keep-or-drop rule written down beforehand. And it can't assume reasoning in natural language.

## Directions worth researching, best fit first

1. **Program or DSL search checked by running it.** Propose candidate transformations, whether from
   the 4B model, a learned policy or enumeration over a compact ARC language. Run each on the
   training inputs, keep the ones that reproduce the training outputs, and apply the survivor to the
   test. Partial credit comes for free, it works on tasks the base misses, and verified programs make
   real self-training data, which revives the weight path that saturated. The open question is
   whether a useful search fits in 12 hours and 16 GB with a 4B model proposing.
2. **Scaling per-task training as the source of partial credit.** That covers leave-one-out adapters
   (train on all but one pair and check honestly on the one held out), ensembles of adapters, and a
   curriculum over how strong the augmentations are. The open question is whether an honest
   leave-one-out check can re-rank where the saturated checks couldn't.
3. **A learned verifier or value model that speaks grids**, using the same 16 tokens, trained on
   held-out judgements and used to guide selection or search without falling into the saturation trap.
4. **Ensembling with a second, genuinely different open grid model**, since augmentations of one model
   don't stack.
5. **Neuro-symbolic reasoning over objects**, parsing grids into objects and relations and searching
   over that structure.

## How any proposal has to be evaluated

- Measure on the 120 public tasks on the single 16 GB card, and report exact match on both attempts.
- Compare head to head on the same tasks against the plain vote (12.79%) and the training blend
  (16.28%).
- Write down the keep-or-drop rule first. For example, ship only if blended public accuracy beats
  16.28% and the method still fits 12 hours on 240 tasks.
- Report negative results along with why they failed. Never train on the evaluation set. Confirm
  timing on four L4s before claiming a leaderboard number.

## The core questions

1. What's the smallest outside source of partial credit that can re-rank candidates where the base's
   own checks say nothing? The options are running programs, honest leave-one-out training and a
   learned verifier.
2. Can program search over a compact ARC-2 language, seeded by the 4B model, fit in 16 GB and 12 hours
   and beat 16.28%?
3. Does distilling verified programs, or leave-one-out successes on genuinely hard tasks, finally move
   the weights where flat fine-tuning and STaR didn't?
4. Does the bimodal behaviour come from the 4B scale, the 16-token interface or ARC-2's compositional
   difficulty, and which of those can a method on a small budget change?
