> **A note added when this was saved, 24 June 2026.** These are early scouting notes. I hadn't
> checked the outside citations and numbers yet when I wrote them, so treat every percentage and
> arXiv ID as a rough guide. Two corrections from what we know about our own
> project override the body below.
>
> 1. **The stock 4B can't be the one proposing DSL programs.** The NVARC base has a 16-token vocabulary,
>    made of ten digits, a newline, two role tokens, a pad and two control tokens. It has no arc-dsl
>    symbols, so it can't be grammar-constrained with GBNF to write DSL programs. The "minimal port" in
>    the body, constraining the existing 4B to arc-dsl, isn't possible on this model. GBNF can only mask
>    tokens the model already has, and this model can't even spell the names of the primitives. Program
>    search therefore needs a proposer that can actually write programs. Either we train a small DSL
>    model at GridCoder scale on pairs of a task and its arc-dsl program, or we switch to a code model,
>    which breaks the idea of staying native to grids. That adds a Stage 1.5 to train the proposer before
>    Stage 2, the search, and it turns the main bet from a port into building a new model. So the
>    leave-one-out TTT hedge matters more, not less.
> 2. **The outside ARC-2 numbers are upper bounds with unlimited compute**, not targets inside our box.
>    Pang (26%), Berman (29.4%, natural-language descriptions at about $8 a task) and SOAR (52%) all run
>    bigger natural-language models well outside our 12 hours, 16 GB, roughly $50 and grids only. Our
>    target hasn't changed. Beat the 16.28% blend on the 120 public evaluation tasks, inside the box.
>
> The order we're actually following is Stage 0 (an arc-dsl grader on CPU) and Stage 1 (honest
> leave-one-out TTT, reusing the existing model) first, since both can be banked. Stage 1.5 (training a
> DSL proposer) then decides whether Stage 2 (search) and Stage 3 (distilling verified programs) go ahead.

## Checking the claims, 24 June 2026

Every claim the plan leans on was checked against the original sources. The headline names and scores
hold up, but there's one problem with framing and five specific errors. Where this section and the body
disagree, this section wins.

**The framing problem, which is the important one. The results being compared come from different
regimes.** The summary says Pang's 26.0% and Berman's 29.4% beat the best neural TTT entry, NVARC's 24%,
which puts results inside the box next to results outside it. NVARC's 24.03% was on the private
ARC-AGI-2 set under the offline contest rules that define our box (12 hours, four L4s, about $0.20 a
task), from a fine-tuned Qwen3-4B, which is where our base comes from, using TTT plus synthetic data.
Pang's 26.0% and Berman's 29.4% were on the unrestricted commercial leaderboard, using Grok-4, a
frontier natural-language model, at $8.42 to $30 a task. So inside our constraints the state of the art
is NVARC's 24%, reached by the same TTT approach we already run, and no published program synthesis
result beats TTT in the offline, 4B, grids-only setting. The top three offline finishers were all TTT
or neural, NVARC at 24.03%, the ARChitects at 16.53% and MindsAI at 12.64%. That means the 26 to 29%
figures are a hopeful upper bound from a frontier natural-language setup, not evidence that a 4B
searching a grid DSL on little compute will clear 16.28%. Whether it can is the open question. This
pushes the plan toward the second lever, honest leave-one-out TTT, which lives in the same regime as
our 16.28%, and keeps program search as the riskier bet, justified only by its unusual ability to give
partial credit. One point does support the report. Berman's fitness function explicitly gives partial
credit, scoring each candidate by the percentage of correct cells, which is the report's central idea
put into practice.

**Errors, with the checked values.**
- GridCoder. Calling it a "452M-parameter VLM" with one decoder layer, embedding size 512 and 4 heads is
  wrong and doesn't hang together, since a one-layer model of width 512 has a few million parameters.
  We couldn't confirm the parameter count. GridCoder is a compact CNN plus Transformer, and it isn't
  guided by execution. It enumerates programs by joint probability under the DSL grammar and then runs
  them afterwards as a filter, so execution plays no part in decoding. Its mechanism and the quote about
  limited ARC examples are confirmed.
- PeARL has 77 primitives, not 30. The figure of 18 out of 400, or 4.5%, is right.
- The ARChitects. The threshold on cumulative sampling probability in their depth-first search is 9%
  (the product-of-experts paper, arXiv 2505.07859). Nothing supports the claim of "17% in 2024 lowered
  to 7% via speculative decoding". The memory figures, 7.3 GB for depth-first search against 14 GB for
  beam search, are right. The ARChitects were runners-up for the 2024 paper award and second on Kaggle
  in 2025, not the 2024 winners.
- Akyurek. 53.0% is TTT applied to the BARC model, 47.1% is on their own model, and 61.9% is when it's
  ensembled with program synthesis. Nothing supports the report's "35% improvement", since 53.0 against
  47.1 is about 13% relative.
- arc-dsl ships 400 worked solvers, one for each ARC-1 training task, not "400+". The count of roughly
  160 primitives is approximate, though we independently counted 160 function definitions in dsl.py on
  the rig.

**Confirmed exactly.** Pang at 77.1 and 26.0. Berman at 79.6 for $8.42, and 29.4 against a "previous
best 25%". NVARC is NVIDIA (Sorokin and Puget), 24.03% private, 12 hours on four L4s. TRM at about 7M
parameters, about 45% and about 8%. SOAR at 52% on ARC-1 (second, MIT licence). CompressARC at 76K
parameters, 20% and 4%, arXiv 2512.06104. BARC at arXiv 2411.02272, with the quote and 42.2%. Akyurek
at arXiv 2411.07279, with leave-one-out. CodeIt at 59 of 400, arXiv 2402.04858. icecuber in 2020 at
about 20%. Greenblatt at 50% on his blog and 42% officially.

**Not checked again, so treat as a rough guide.** Akyurek's exact hyperparameters. SOAR's rank of 256
and 6,000 attempts. BARC's 73.5% after TTT and its 400k dataset. icecuber's 42 functions and 142
variants and about 10^7 pieces per task. LPN's 39M parameters, JAX and the claim that it doubles
out-of-distribution performance (the paper, arXiv 2411.08706, is confirmed, but not those details).

---

# Scouting report on raising the ARC-AGI-2 solve rate of a 4B grid-native model on a 16 GB GPU

## The short version

- **Build program search, guided by the model and checked by running the programs, as the main new
  lever.** Of everything scouted, it's the only method that naturally produces partial credit. A program
  that reproduces k of n training pairs gives a signal even on tasks the base model misses. It fits the
  box without loosening the grids-only rule, and it has real ARC-AGI-2 scores behind it (Eric Pang at
  26.0%, Jeremy Berman at 29.4%). Those scores, though, are only on the unrestricted commercial
  leaderboard with frontier natural-language models (Grok-4, $8 to $30 a task). Inside our offline,
  4B, grids-only box the state of the art is NVARC's 24.03% with TTT, and no program synthesis result
  beats TTT there (see the check above). The case for program search here rests on partial credit, not
  on those numbers from another regime. The plan is to take Greenblatt's structure, generating many
  programs, keeping the ones that pass the training pairs and voting, and move it from natural-language
  Python onto Michael Hodel's arc-dsl (MIT licence), with the 4B reduced to proposing DSL programs under
  a grammar.
- **Run honest leave-one-out TTT as the cheaper and proven second lever.** Akyurek's recipe (MIT
  licence) already builds leave-one-out tasks. Exact match on a held-out pair is a continuous signal for
  re-ranking that the saturated check on training pairs can't give. This is the fallback if program
  search doesn't fit in 12 hours.
- **The wall where tasks are either solved or not touched at all mostly comes from the interface and
  the model's size, not from fate.** A symbolic program can be graded by the fraction of training pairs
  it gets right even when the 4B is lost, so program search removes the wall without needing natural
  language. Distilling verified programs back into the weights is the one self-training route that
  could move the model where plain fine-tuning and STaR couldn't, because the supervision is a new
  symbolic object, not the grid the model has already memorized.

## Main findings

1. **Program search checked by execution works on ARC-AGI-2, but only when a language model guides it.**
   Eric Pang's evolutionary system for synthesizing Python programs, the runner-up paper for ARC Prize
   2025, scored 77.1% on ARC-AGI-1 and 26.0% on ARC-AGI-2 using only tens of programs per task. In his
   words, "My approach scored 77.1% on ARC-AGI-1 and 26.0% on ARC-AGI-2, both beating results of
   frontier models and previous bespoke systems". Jeremy Berman's evolutionary search set a new state of
   the art on ARC-2, "a new high score of 79.6% on ARC v1 at $8.42 per task ... and, more importantly,
   sets a new state-of-the-art (SoTA) of 29.4% on ARC v2 (previous best: 25%)." Both beat the top
   private Kaggle score on ARC-AGI-2, NVARC's 24%, which was pure neural TTT. Brute force over a
   hand-built DSL (icecuber, 20.6% on ARC-1) has no published ARC-AGI-2 score, and it's generally seen
   as beaten by ARC-2's harder compositional and contextual tasks. The lesson is that the neural
   proposer is what makes program search work on ARC-2, and blind enumeration doesn't scale.

2. **Partial credit cleanly separates what works from the dead ends.** Program search scores how many
   training pairs a program reproduces, continuously, on tasks the base misses entirely. That's exactly
   the signal the known dead ends, the verifier-guided vote and STaR, didn't have. Honest leave-one-out
   TTT produces a second signal of this kind, accuracy on the held-out pair. A learned grid verifier,
   ensembling for diversity and parsing object graphs are all weaker on this front and stay shallow.

3. **arc-dsl is a ready, searchable target with a permissive licence.** Michael Hodel's arc-dsl (MIT)
   has roughly 160 primitives (counting the function definitions in dsl.py), and the repo ships worked
   solver programs. It has a typed system of grids, objects and patches in arc_types.py, and it's the
   DSL behind RE-ARC (MIT). A maintained fork, arc-dsl-2, revives it and considers ARC-AGI-2 support. The
   DSL is fully deterministic and runs on CPU, so checking against training pairs costs no GPU time.

4. **The TTT precedents give concrete recipes to copy.** Akyurek et al. (MIT, repo `ekinakyurek/marc`)
   use a per-task LoRA of rank 128 on the MLP, attention and output layers, 2 epochs, batch size 2,
   learning rate 5e-5, leave-one-out task construction plus the eight symmetries, and a hierarchical
   self-consistency vote. Their TTT pipeline reached 53.0% on the ARC-1 public validation set when
   applied to the BARC model, 47.1% on their own model and 61.9% when ensembled with program synthesis.
   The ARChitects, runners-up for the 2024 paper award, use the eight symmetries plus colour permutation
   and example reordering as augmentation, sample with depth-first search cut off at a cumulative
   sampling probability of 9% (arXiv 2505.07859), and pick answers with a product of experts that scores
   candidates by how stable they are across augmentations. Their depth-first search keeps a single path
   in memory, about 7.3 GB of VRAM against about 14 GB for beam search.

5. **Distilling verified programs is the live route to moving the weights.** STaR did nothing because a
   verified correct grid is the same ground-truth target the fine-tune already used. On its own training
   set the base solves 95.5% and gets close on another 1.5%. A verified DSL program is a different kind
   of object. It supervises the proposer's distribution over program tokens, not the grid it already
   produces, and that revives the saturated weight route the way CodeIt's hindsight replay did. CodeIt
   (Butt et al., ICML 2024, arXiv 2402.04858) "solves 59/400 ARC evaluation tasks, achieving
   state-of-the-art performance" by learning from its own relabelled program samples.

## Details

### The main bet, explored in depth. Program and DSL search checked by execution

**The standard shape, from Greenblatt in 2024, and how to move it off natural language.** Ryan
Greenblatt's public ARC-AGI-1 result at Redwood Research is described in his own words. "I recently got
to 50% accuracy on the public test set for ARC-AGI by having GPT-4o generate a huge number of Python
implementations of the transformation rule (around 8,000 per problem)". He kept the ones correct on all
the training examples, took a majority vote over the test outputs of the survivors, and picked the top
candidates. The selection rule is the part worth reusing. Running a program deterministically against
the training pairs is the partial-credit grader. His actual implementation needs natural language,
though. It writes Python reasoned out in natural language, uses few-shot chain-of-thought prompts about
30k tokens long, and revises programs after seeing their outputs. None of that runs on a base with 16
tokens that only knows grids.

The minimal port was to replace GPT-4o writing Python with the 4B writing arc-dsl tokens under a GBNF
grammar. llama.cpp supports grammar-constrained decoding natively, setting to minus infinity every token
that would break the grammar before sampling ("Logits for tokens that would break the grammar rules are
set to -inf"). You'd define a GBNF grammar over the roughly 160 arc-dsl primitives so that the 4B could
only write valid programs. The 4B proposes distributions over program tokens, the executor (CPU,
deterministic) grades, and the training pairs are the specification. There's no natural language
anywhere. It's a clean method that stays native to grids, and the right main bet. [See the first note
at the top. This isn't possible with the stock 16-token model, which needs a trained DSL proposer.]

**An architecture we could build.**
- *Proposer.* The existing NVARC qwen3-4b-grids served by llama.cpp at Q8_0, prompted with the training
  pairs in the existing grid token format and decoding under a GBNF grammar for arc-dsl. Sample N
  candidate programs per task with depth-first sampling, with a cutoff on cumulative probability as the
  ARChitects do, for variety at low VRAM cost. Inference and the executor never share the GPU, and since
  the executor runs only on CPU they can overlap.
- *Executor and grader.* arc-dsl (MIT), with icecuber's DSL in C (42 functions, 142 variants) as a fast
  fallback for the geometric subset. Run every candidate on every training input and score it by the
  fraction of training pairs it reproduces exactly. Keep the ones that get all of them, and among ties
  rank the partial survivors for the self-training buffer.
- *Selector.* A majority vote over the survivors' test outputs, two attempts. If there are fewer than
  two distinct outputs, fall back to AIRV's existing vote.
- *Self-training loop, run offline between rounds.* Collect every verified program, the ones that get
  all the training pairs, into a buffer. Every so often, fine-tune the proposer with LoRA on pairs of
  training examples and verified program tokens. This is the hindsight pattern from CodeIt and SOAR.

**The compute budget. Can it fit 12 hours, 16 GB and four L4s for 240 tasks?** This is the key open
question, and the answer is that the number of programs per task has to stay small, in the tens
to low hundreds, not Greenblatt's 8,000. Pang reached 26.0% on ARC-2 with only tens of programs per
task. icecuber's C++ DSL tries up to about 10^7 candidate pieces per task in tens of seconds on CPU (a
depth-2 run over 419 tasks took about 70 seconds, around 6 tasks a second at depth 2), so the executor
isn't the bottleneck. The bottleneck is decoding from the 4B proposer. The existing engine decodes 13
times faster than eager mode, and AIRV already runs 8 augmentations per task within budget. A program
search pass that samples about 100 to 300 short DSL programs per task, each tens of tokens long and far
shorter than a full grid, is about the same decoding load as AIRV now. A rough budget for 240 tasks on
four L4s in 12 hours is about 180 seconds of GPU decoding per task, around 50 short programs per L4
slot, with execution and selection on CPU. That's comfortably inside the 12 hour and roughly $50
envelope NVARC itself aimed at. NVARC's own limit was "maximum 12 hours on four small GPUs with a budget
of approximately 20 cents per task". As for where the 12 hours goes, roughly 70% is proposer decoding,
20% is execution and checking on CPU, and 10% is LoRA self-training between rounds, done offline and
not on the GPU at the same time.

**Why it gets a grip on the wall.** A program that reproduces 2 of 3 training pairs scores 0.67 even
when the base would write a wrong grid and score 0. That fractional signal ranks candidates and can seed
a curriculum, which is exactly what re-ranking the model's own grids, the known null results, could
never do.

**DreamCoder and library learning, judged for the box.** DreamCoder (Ellis et al., PLDI 2021) grows a
DSL library through wake and sleep phases, refactoring common sub-programs into new primitives by
E-graph matching. On ARC-1 the best result in that style (PeARL, 77 primitives) was only 18 of 400, or
4.5%, beaten by stock GPT-4 and far behind icecuber. The full wake-sleep library learning is too heavy
on compute and too short of data to run in the box, since ARC has too few tasks per concept. GridCoder's
author skipped growing the DSL for this reason ("the limited amount of natural ARC-AGI examples [is] an
obstacle to making this feature particularly useful"). The core of library learning, though, promoting
common verified sub-programs from the self-training buffer into named DSL macros, is a cheap optional
extra that makes the search shallower over rounds. We'd recommend that core, not the whole system.

**Induction against transduction, from BARC (Li et al.).** BARC (best paper at ARC Prize 2024, Wen-Ding
Li, Kevin Ellis et al., arXiv 2411.02272) showed that induction, writing a program, and transduction,
predicting the grid directly, solve different tasks and get close to human level on ARC-1 when
ensembled. "Inductive program synthesis excels at precise computations, and at composing multiple
concepts, while transduction succeeds on fuzzier perceptual concepts." Akyurek found that with their TTT
pipeline, BARC's neural model solved 73.5% of the tasks the program synthesizer solved, up from 42.2%,
so the two complement each other. That's the strategic reason for program search as an addition to the
existing transductive 4B, not a replacement. The program search arm should catch the tasks that need
precise computation and composing several steps, which ARC-2 leans on, exactly where the transductive
base hits the wall. BARC's models (barc0 on HuggingFace, Llama-3.1-8B fine-tuned on a dataset of 400k
tasks generated by GPT-4o-mini) are 8B and write natural-language Python, so they can't be dropped in,
but the split between induction and transduction is the design principle to copy.

**GridCoder is the closest precedent.** Simon Ouellette's GridCoder (ARC Prize 2024 paper "Towards
Efficient Neurally-Guided Program Induction for ARC-AGI", arXiv 2411.17708, repo SimonOuellette35)
trains a compact CNN plus Transformer (the description as 452M parameters, one layer, width 512 doesn't
hang together and isn't confirmed) to output a probability distribution over a DSL grammar of 74 to 98
primitives. It's used as an enumerator that isn't guided by execution, since execution is a filter
afterwards. It enumerates programs in order of joint probability, with a simplifying assumption of
conditional independence that means only a few calls to the model per task. It sorts programs by the
product of their token probabilities, raises the probability threshold until the number of programs is
manageable, runs them in order and returns the first that passes the examples. That's almost exactly
the proposed architecture at a smaller scale, and it shows that enumerating programs conditioned on a
grammar fits modest compute. Its search loop is worth studying directly.

### The second bet, one deliberate thread. TTT as a source of partial credit

**Honest leave-one-out adapters.** The existing per-task TTT (rank-16 LoRA, 20 steps, scale 1.25)
already lifts us to 16.28%. The new idea is to train each adapter on n-1 training pairs and check it on
the pair held out. Exact match on the held-out pair is a signal we can grade on tasks the base misses,
and unlike the saturated check on training pairs, the adapter can't game it by memorizing all the pairs,
because it never saw the held-out one. Akyurek's pipeline already builds leave-one-out tasks ("We start
by creating leave-one-out tasks from the given training examples of the task. These tasks are then
augmented through rule-based transformations to obtain the full TTT dataset"), so the machinery is in
the `marc` repo, a fork built on torchtune. Leave-one-out accuracy can be used to decide whether to
trust a task's TTT adapter at all, as a gate, and to re-rank candidate outputs across adapter seeds. The
open question has an answer. Yes, checking on a held-out pair can re-rank where the saturated check on
training pairs says nothing, because it measures generalizing to an unseen pair rather than fitting the
seen ones.

**Ensembles of adapters and a curriculum on augmentation strength.** Train several per-task adapters
with different augmentation strengths, from just the eight symmetries up to the ARChitects' scheme of
symmetries, colour permutation and example reordering, and with different seeds. Select by leave-one-out
accuracy and vote with AIRV (reverse and vote) over the survivors. This is cheap, since the adapters are
rank 16 in bf16 and can be swapped in llama.cpp on the fly, and it extends the lever we know works. For
the curriculum, start each adapter at low augmentation strength and raise it only if leave-one-out
accuracy is under a threshold, so easy tasks don't get over-regularized.

**Why this is the safe fallback.** Every piece is already proven in our stack (LoRA, hot-swapping and
AIRV). If program search doesn't fit in 12 hours, honest leave-one-out TTT plus adapter ensembles is a
lower-variance way to beat 16.28%.

### A shallow scan, worth coming back to only if the top two stall

**(3) A learned verifier or value model that works on grids.** Train a second head, 4B or smaller, on
the same 16-token interface to predict whether a candidate output is right on the held-out pair, given
the training pairs. That gives partial credit as a calibrated number on hard tasks, and it fits the box
since it's inference only and native to grids. The risk is that the wall comes back. If the base can't
represent a task, a verifier with the same interface probably can't grade it either, which is why this
stays shallow. It's only worth doing if trained on real held-out labels from honest leave-one-out, not
on its own labels, which would bring back the saturation that sank STaR.

**(4) Ensembling for diversity with a second, genuinely different open grid model.** The candidates are
a Tiny Recursive Model trained from scratch (TRM, about 7M parameters, about 45% on ARC-1 and about 8% on
ARC-2, already used inside NVARC's ensemble) and LPN (Latent Program Network, 39M parameters, JAX,
clement-bonnet/lpn, a NeurIPS 2025 spotlight). Both are native to grids and small enough to run
alongside. LPN's headline result is that searching its latent space at test time "doubles its
performance on out-of-distribution tasks." They only give partial credit through the same execution
check, so ensembling helps coverage, solving different tasks, rather than grading. TRM is the cheap
second arm to try, since NVARC already used it. LPN's JAX dependency is friction against our PyTorch and
llama.cpp pipeline.

**(5) Neuro-symbolic reasoning about objects.** Parse grids into objects and relations (object graphs in
the style of ARGA, and the Divide-Align-Conquer line, which solves tasks with about 0.2% of the
candidates icecuber needs) and search over that structure. In principle it's strong on ARC-2's tasks
about objects and interpreting symbols, but building a robust parser is a lot of work, and the search is
a different engine from the DSL line. There's a partial-credit signal (graph edit distance to the
target), but integrating it costs a lot. Put it off unless the DSL turns out too flat for ARC-2's tasks
with contextual rules.

### A side branch on loosening the constraints, kept apart from the main line

Is the wall partly caused by the grids-only interface? Plausibly yes. A model with a channel for partial
credit, a program or a reasoning trace it can write and have graded, has somewhere to put an incomplete
understanding, whereas a grids-only model has to write a whole correct grid or nothing. CompressARC (Liao
and Gu, "ARC-AGI Without Pretraining", arXiv 2512.06104) is suggestive. A model of 76K parameters trained
from scratch on each task to minimize description length, with no pretraining and no outside data,
reaches 20% on ARC-1 (described as 20 to 34%) and 4% on ARC-2 in about 20 minutes a puzzle on one GPU.
That suggests optimizing per task against a continuous objective, description length, pulls out signal
a frozen transductive model can't. SOAR (Pourcel, Colas and Oudeyer, arXiv 2507.14172, MIT licence,
github.com/flowersteam/SOAR), second-place paper at ARC Prize 2025, reaches 52% on the ARC-1 public
evaluation by improving itself on its own program search traces (RS-LoRA at rank 256, 6,000 synthesis
attempts per task, fine-tuning on one H100). But it uses natural-language Python models from 7B to 123B
and reports no ARC-2 number, so it's a pivot that would change the thesis, with a new base, a
natural-language channel and the low-compute story at risk. Berman's own record on ARC-2 came from that
kind of pivot. According to a 2025 interview, his "winning method for the ARC-AGI v2 challenge, achieving
a 29.4% top score, pivoted from Python code generation to evolving natural language descriptions."
**The point that matters for the main line** is that symbolic program search doesn't need natural
language, so it gets the partial-credit channel without giving up the grid specialization or the
low-compute thesis. The constraint therefore doesn't bind the main bet, which is one more reason it's
the right one. Switching to a natural-language model should be treated as a tightly bounded experiment,
not something on the roadmap. It would be slower and heavy on tokens (Greenblatt's prompts run about 30k
tokens), and it would probably blow the 12 hour budget on 240 tasks.

### Answers to the main research questions

1. *What's the smallest outside signal of partial credit that can re-rank where the base says nothing?*
   The fraction of training pairs an executed program reproduces, which is continuous from 0 to 1 and
   works on missed tasks. Second best is accuracy on the held-out pair from honest leave-one-out.
2. *Can DSL search seeded by the 4B fit in 16 GB and 12 hours and beat 16.28%?* Plausibly, if the number
   of programs per task stays in the tens to hundreds (Pang got 26% on ARC-2 with tens), with the CPU
   executor off the critical path and decoding from the 4B as the bottleneck. The budget is worked out
   above. This bet has the most upside and the most risk in building it.
3. *Does distilling verified programs move the weights?* This is the one self-training route with a real
   chance, because the supervision target, program tokens, differs from the grid the model already
   memorized, unlike STaR. CodeIt's hindsight replay (59 of 400 on ARC-1) is the precedent.
4. *Does the solved-or-untouched split come from scale, the interface or how hard ARC-2 is?* Mostly the
   interface and scale. The grids-only channel gives an incomplete understanding nowhere to go, and a 4B
   has limited depth for composing ARC-2's multi-step tasks (ARC-2 was built on purpose around tasks that
   compose several steps and interpret symbols on the fly). A low-compute method can change the effect of
   the interface by giving it a program channel, but not the raw scale. Program search is the change that
   needs no new model. [See the first note at the top. It does need a new proposer model.]

## Recommendations

**Stage 0, one or two days. Build the grader.** Wrap arc-dsl (MIT) so any program gets scored by the
fraction of training pairs it reproduces, on CPU. Measure executor throughput on our hardware and confirm
it never touches VRAM. The gate to go on is an executor that keeps up a few thousand program runs a
second per core. icecuber-class C++ is far faster and arc-dsl in Python is slower, so if Python is too
slow, port the hot primitives to C or use icecuber's geometric core.

**Stage 1, one week. Honest leave-one-out TTT, banking a safe win first.** Extend the existing per-task
TTT to train on n-1 pairs and record exact match on the held-out pair. Use leave-one-out accuracy to gate
and to re-rank adapter seeds, and ensemble the survivors with AIRV. Measure on the 120 public evaluation
tasks, exact match on both attempts, head to head against AIRV without TTT (12.79%) and the TTT blend
(16.28%). **Keep it only if the blended public score is above 16.28% and the run over 240 tasks fits in
12 hours on four L4s.**

**Stage 2, two to four weeks. DSL program search under a grammar.** Write a GBNF grammar over the arc-dsl
primitives, have the 4B propose about 50 to 150 programs per task with depth-first sampling, run them,
keep the survivors and vote for two attempts. Start with the geometric subset of primitives, which is
cheap and productive, before adding the object primitives. Measure the same way. **The gate is a blended
public score above 16.28% and a fit in 12 hours for 240 tasks.** Confirm the timing on four L4s before
claiming any leaderboard number. [See the first note at the top. Stage 1.5, training a DSL proposer, has
to come first, because the stock 4B can't write DSL tokens.]

**Stage 3, alongside and offline. Distilling verified programs.** Buffer every program that gets all the
training pairs right and fine-tune the proposer with LoRA on pairs of training examples and program
tokens. Then run Stage 2 again. Watch for the STaR failure, no gain, against the CodeIt success, a gain
from supervision on new programs. Optionally promote common verified sub-programs to DSL macros, the core
of DreamCoder, to make the search shallower.

**Stage 4, if Stages 2 and 3 stall. An ensemble for coverage.** Add TRM as a second arm native to grids,
as NVARC did, and maybe a verifier trained on leave-one-out labels. Treat parsing object graphs as a
research spike, not something to ship.

**Results that would change the plan.** If program search can't clear about 18% public within the 12
hour budget after Stage 2, freeze it and ship the gain from leave-one-out TTT in Stage 1. If distilling
verified programs shows no movement after one round, the way STaR didn't, stop distilling and keep search
for inference only. If the geometric DSL subset tops out below ARC-2's tasks about objects and context,
move to the object graph spike (5), not to a natural-language model.

## Caveats

- **There's no published ARC-AGI-2 number for SOAR, Greenblatt's original method, icecuber or Hodel's
  arc-dsl ensemble.** The evidence for program search on ARC-2 rests on Pang (26.0%) and Berman (29.4%,
  the current state of the art for custom systems), whose methods are written up in blogs and a
  runner-up paper rather than peer-reviewed venues, so the exact percentages are a rough guide, not
  settled. Pang's and Berman's systems use bigger models that can handle natural language, and Berman's
  record came after switching to natural-language descriptions of programs, so their scores are an upper
  bound on what a 4B proposing DSL programs under a grammar will reach.
- **Brute-force DSL search is documented as beaten on ARC-2.** Any edge has to come from the neural
  proposer guiding the DSL, not from enumerating. A badly conditioned 4B proposer could fall to
  icecuber-class performance, with no signal on ARC-2. ARC-AGI-2 was built on purpose to remove tasks
  brute force can solve.
- **Fitting in 12 hours is the biggest risk for the main bet.** Every compute estimate here is an order
  of magnitude, worked out from second-hand throughput figures (icecuber's DAG sizes, AIRV's decoding
  load, Pang's program counts) and NVARC's stated envelope of 12 hours, four L4s and about $0.20 a task.
  None of it is measured on our stack. Stages 0 and 1 have to confirm the estimates before committing to
  Stage 2.
- **arc-dsl's roughly 160 primitives were built for the ARC-1 training tasks, and nobody has shown they
  cover ARC-2.** ARC-2 was designed on purpose around interpreting symbols and contextual rules, such as
  assigning meaning to a symbol on the fly, which a 2023 DSL for ARC-1 may not express. So how much of
  ARC-2 the DSL covers is itself something to research.
- **Never train on the 120 public evaluation tasks or the hidden sets.** Every self-training buffer has
  to come from RE-ARC (MIT), the ARC-2 training set (Apache-2.0) or synthetic generation. The licences of
  every recommended asset (arc-dsl MIT, RE-ARC MIT, SOAR MIT, LPN open source, GridCoder open source)
  are permissive and confirmed.
- **4-bit doesn't load on our Blackwell and Windows stack.** All LoRA training stays in bf16, and
  inference and training never share the GPU, exactly as in the existing pipeline. None of the
  recommendations change that.
