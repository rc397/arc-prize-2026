# Idea triage

A reference for coming up with ideas on this project and screening them. Read the constraints and the
list of closed ideas first, then the gates. Everything here is measured rather than assumed, and each
rule is here because leaving it out gave a wrong answer at least once.

## Part 0. How the system actually works

When you propose something, point at the stage it changes. I've marked each stage by where it sits
relative to the candidate pool, because that's the strongest predictor we have of whether an idea can
work (see Gate 1).

### The base model

It's `qwen3_4b_grids15_sft139`, NVARC's Qwen3-4B specialised for grids. It has a 16-token vocabulary,
the ten digits plus control ids. It has 36 layers, a hidden size of 2,560, 32 attention heads, 8 KV
heads and a head dimension of 128, in bf16. Prompts are ChatML at the level of token ids. We didn't
train it and we don't modify it. It was already heavily pretrained on synthetic ARC data, which is why
adding more data shaped like ARC-1 was redundant.

### For each task, 240 tasks over four L4s in 12 hours

```
  task (2-6 demonstration pairs + 1-2 test inputs)
        |
   [1] AUGMENT  ---------------------------------------------  UPSTREAM
        |   training: augment(n=16, seed=1) = 2 (transpose) x 4 (rot90)
        |             x 16 random colour permutations = 128 rows,
        |             then the order of examples is shuffled.
        |   evaluation: augment(n=2, seed=2) = 8 x 2 = 16 views.
        |   cut_to_len drops demonstration pairs from the front until the
        |   text fits, 8192 for training text, 8192 - 931 for evaluation input.
        v
   [2] TEST-TIME TRAINING  -----------------------------------  UPSTREAM
        |   a fresh LoRA per task, reset to the saved default weights every
        |   task, at rank 256, alpha 32, rslora (effective scale 2), dropout 0,
        |   on nine modules, the seven projections plus embed_tokens and lm_head.
        |   128 rows, 1 epoch, batch 1, no accumulation = 128 optimizer steps.
        |   AdamW, lr 5e-5, cosine schedule, warmup ratio 0.1, clip 1.0, ~240s.
        |   The collator supervises every assistant span, including the first
        |   demonstration's output (measured at 96 of 96 rows, sim_ttt_shape.py).
        |   35% of tokens carry a loss. So this isn't leave-one-out. It's
        |   next-token prediction over the whole sequence of demonstrations.
        |   No input grid is ever supervised, which is the ARChitects' explicit
        |   choice. The test answer is never seen anywhere.
        v
   [3] DEPTH-FIRST DECODING  ---------------------------------  the POOL is made here
        |   turbo_dfs enumerates every completion whose cumulative
        |   probability stays above 0.2 (max_score = -log 0.2).
        |   It recurses once per token, four views at a time, using
        |   hard-coded index offsets over the sorted view keys.
        |   Candidates are mapped back to the original frame and deduplicated.
        v
   [4] SELECT  -----------------------------------------------  DOWNSTREAM
        |   score_kgmon = multiplicity minus mean augmentation NLL,
        |   rescored under fresh augmentations. The top two become
        |   attempt_1 and attempt_2.
        v
  submission.json
```

### The three deadlines, all in wall-clock seconds

| where | budget |
|---|---|
| the whole run | 12 hours minus a 600-second reserve, so 42,480 seconds |
| each task | 1,800 seconds |
| inside one call to the search | 540 seconds |

Nothing is budgeted in operations. That's why identical code scores differently on different days,
and why anything that changes speed changes which candidates exist.

### Where the score comes from

There are 240 tasks with two attempts each, and a task counts if either attempt is right. A task with
one output is worth about 0.417 points. Our best run scored 32.64.

### Two stacks that aren't the same

| | Kaggle (scored) | our rig (probes) |
|---|---|---|
| hardware | four L4s, 24 GB each | one RTX 5070 Ti, 16 GB |
| operating system | Linux | Windows |
| software | **Unsloth** (`FastLanguageModel`) | plain `transformers` and `peft` |

Unsloth needs Triton, and Triton has no native Windows version, so the rig can't run the deployed
stack. Any measurement of memory or speed on the rig tells you about the rig, not about the scored
runs. This has bitten us once already. A memory cliff in decoding that we measured at 16 GB may not
exist at all on the L4s.

The rig probe is also a strictly weaker pipeline than the submission in three separate ways, and any
null result from the rig has to be read against all three before it's believed.

| | deployed | rig probe |
|---|---|---|
| LoRA targets | 9, including `embed_tokens` and `lm_head` | **7**, with no embedding and no head |
| context | 8,192 | **6,144** |
| optimizer steps | 128 (one epoch over 128 rows) | 128, but **32** in the arm probes with a fixed budget |

The module list is the one that should worry us most. With a 16-token vocabulary, `embed_tokens` and
`lm_head` are each 16 by 2,560, so adapting them is nearly free, and it amounts to re-learning what
each colour token means for this task. The probe can't do that at all.

### What each stage has already absorbed

| stage | tried there | outcome |
|---|---|---|
| [1] augment | completing masked outputs, crops, inverting the rule, run-length serialization | all null or impossible to build |
| [2] training | sweeping rank and depth, pre-fitting, subset adapters, warm starts across tasks, continued fine-tuning | all null on coverage |
| [3] decoding | best-first order, batch width, a per-task time cap, restarting with new seeds | all null |
| [4] selection | reweighting, voting cell by cell, fusing two experts, filtering by invariants | all null, and bounded by proof |

## Part 1. Six facts that constrain every idea

**1. The system is starved of generation, not of selection.** 65% of tasks never produce the right
grid under any of the sixteen augmented views, and they use 71% of the budget doing it. When the truth
does reach the pool, the deployed selector puts it in the top two essentially every time, 18 of 18 on
fresh tasks. Anything that acts after generation is working on the half that already works.

**2. The base model has a 16-token vocabulary.** It writes grid tokens and control ids and nothing
else. There's no chain of thought, no hypotheses in words, no program text, no self-critique, and no
new marker tokens without breaking the pretrained embedding. Most of the current reasoning literature
simply can't be used here. This is the deepest constraint in the system, and nothing we've tried
touches it.

**3. We can't measure improvements under about three points.** The noise between runs of the same
configuration is a standard deviation of about 1.2, from eight runs of one unchanged pipeline. Five of
those are the same unchanged kernel version, scoring 30.56, 32.64, 28.47, 31.39 and 30.56. The cost of
detecting an effect grows with one over its square.

| true effect | runs per arm to detect it | days at one submission a day |
|---|---|---|
| 1 point | 29 | 58 |
| 2 points | 7 | 14 |
| 3 points | 3 | 6 |
| 4 points | 2 | 4 |

An idea worth less than three points can still be measured offline, but it can never be shown on the
leaderboard before the deadline.

**4. Training loss is already near zero, and it isn't the bottleneck.** Solved tasks range from 0.0006
to 0.025 and missed tasks from 0.003 to 0.09, so the two overlap almost completely. The adapter fits
the demonstrations nearly perfectly whether or not the task gets solved. Anything that improves the fit
just gets faster to a place we already reach.

**5. The base is bimodal per task.** It either picks up a rule or locks onto a wrong one, with no
partial credit in between. That's what kills refinement loops, resampling gated on verification, and
every method that assumes partial correctness has a gradient.

**6. Everything is budgeted in seconds of wall clock, not operations.** There are three deadlines, a
12-hour limit on the whole run, an 1,800-second cap per task and a 540-second budget inside the search.
That's why identical code scores differently on different days.

## Part 2. What's already closed

Don't propose these again. A closed lever only reopens on a new measurement, never on a new argument.

### Selection and search, after the pool is made, all null

| lever | what closed it |
|---|---|
| reweighting the selector | refitting under leave-one-task-out scores below the deployed selector |
| voting cell by cell | 0 of 41 missing true answers recovered, because the pool's mistakes are correlated |
| rescoring with a second model, or fusing two experts | an oracle gap of 0 of 18 on fresh tasks |
| any selector built only on the model's beliefs | proved to be bounded by the top-two posterior mass |
| best-first decoding order | a median saving of 2.9% over 16 logged trees |
| restarting with new augmentation seeds | the union of three seeds equals the best single seed exactly |
| mining invariants as a filter | recovered nothing and killed 11 of 19 true answers |
| a per-task cap of 1,200 seconds | 29.31, inside the noise, and the prediction was wrong |
| nine candidate filters mined from tasks | replayed over the 432-candidate pool, 7 of 9 kill true answers and the other 2 kill nothing at all. No candidate the decoder writes ever uses an unseen colour or has the wrong shape on a task that keeps its shape, because the model already obeys those constraints. `sim_prune_replay.py` |
| a search threshold that shrinks with depth | at the same cost it loses to simply raising the flat floor, and its premise is backwards. True answers come out deeper (median 199 against 143) and more confidently (room of 1.600 out of 1.609 against 0.846). Depth isn't what separates them. `sim_dfs_budget.py` |
| **raising the search's probability floor** | run at a fixed 600 seconds of decoding with one adapter per task and three floors. 0.20 gives 6 of 14 outputs and 56 candidates, while 0.35 and 0.45 both give **5**, with 48 and 43. Worse on one task, better on none, tied on seven. The forward passes it freed bought nothing, because every arm still ran out of time. It also settled the disagreement between the two offline replays. The **pool replay was right**, since the truth falls away above 0.25, and the tree replay's 13 covered outputs were too few. `rig_dfsfloor.py` |

### Adaptation

| lever | what closed it |
|---|---|
| adapter rank and depth | an inverted U, and the shipped settings sit on the peak |
| pre-fitting on the test input | 7, 7 and 6 across base, treated and control, so nothing moved |
| subset adapters that leave out one demonstration | no gain in coverage over 20 outputs |
| continued fine-tuning on data shaped like ARC-1 | an honest negative, with the bimodality unmoved |
| refinement loops | there's no partial-credit middle for them to grab |
| **transferring adapters across tasks** | retrieval works (AUC 0.884) and warm starts work (30% lower loss on 7 of 8, graded by how related the tasks are), and coverage is **6 against 6**. The transfer is real and buys nothing |

### The augmentation space

| lever | what closed it |
|---|---|
| completing masked outputs | 8 against 8 on 12 matched tasks |
| cropping to sub-grid bounding boxes | 3 against 3 on 5 matched tasks, and only 5.8% of cells are margin |
| inverting the rule (shape-matched pairs) | 4 against 4 on 7 matched tasks |
| run-length serialization | impossible, since a 16-token vocabulary has no spare token |
| deleting uniform rows and columns losslessly | saves 1.0% of cells and cuts off the answer on 6.7% of tasks |
| the whole family of token-saving ideas (run-length, interleaved coordinates, digit prefixes) | the reason for them is gone. At Kaggle's 8,192, **0 of 240 training views** and 6 of 344 evaluation views lose a demonstration. `cut_to_len` is the only way a saved token can change an answer, and it barely fires. `sim_token_budget.py` |
| **masking the loss on unchanged cells** | the mechanism worked and still lost. Error on the cells the rule changes fell from 0.286 to **0.062**, error on the cells it copies rose from 0.024 to **0.095**, and coverage fell from 6 to 3 of 14. The random control of the same size can't be told apart from the base (6 of 14, tied on all 8 tasks), so the cause isn't less supervision. It's specifically that the supervision teaching the model to copy carries real weight. `rig_maskloss.py` |
| writing the output column by column | it's the transposed view, and `augment` already keeps it with `mod(np.transpose, keep=True)` at arc_loader.py line 250. Every training set already contains the column-wise reading |
| **forcing a size prefix on the output (breaking Gate 3 on purpose)** | a total wipe-out. The base got 4 of 10 outputs and 37 candidates, and the size-prefix arm got **0 of 10** and **2**. Of the 19 completions the prefix arm wrote, 14 declared one shape and then drew another, and 3 had no proper header at all. After 32 training steps the model can't keep to its own stated plan for the length of one grid. That's the price of new syntax inside the vocabulary, and every proposal in that family pays it. `rig_dimprefix.py` |
| sorting the palette into a fixed order | killed twice, independently. Applied before the deployed augmentation it's provably a **no-op**, because `permute_rnd_all_` draws evenly from all 10! colour permutations, and a fixed remapping followed by a uniform one is still uniform. Applied per demonstration as proposed, the remappings disagree about a shared colour in **92% of tasks**. `sim_palette_canon.py` |
| wrap-around shifts, shifts with zero padding, shuffling blank rows and columns | false supervision at scale. Content touches an edge on **79% of grids** and **93% of tasks**, the cells the rule changes touch the border on **65%** of shape-matched pairs, and the number of uniform rows and columns changes across **72%** of them. `sim_aug_damage.py` |

### Program synthesis

| lever | what closed it |
|---|---|
| a 1.5B model proposing DSL programs | 0 of 101, because it couldn't express the transformations |
| a 14B coding model, zero-shot | 0 of 120 on the full set, with a median per-cell gain over copying the input of exactly zero |
| a 14B reasoning model | 0 of 12, with nothing in the near-miss band |
| a symbolic prune on object counts | 27.92 on the hidden set. Offline, a predicted check isn't accurate enough |

So far, 24 levers have been measured against coverage or the hidden score, and none moved it. Five
more were closed offline at no GPU cost, the five new rows above. That's the cheaper way to close
something, and the one to reach for first.

## Part 3. What's still open

**Program synthesis checked by running it, at a budget we can afford.** imbue lifted an open-weights
model from 12.1% to 34.0% with programs scored by running them against the demonstrations, which is
exactly the escape route our ceiling theorem names. Their cost is $2.42 to $8.71 a task, and the
offline track allows about 20 cents. The open question is whether it fits, not whether it works.

**Training data built from compositions.** We have 402 clean two-step rules, built by chaining RE-ARC
verifiers (30 first stages times 28 second stages), and each can be instantiated without limit. A
corpus of 78,792 examples is built, and whether training on it helps is being measured. The bottleneck
in the first stage, where 158,000 of 159,600 pairs never ran because verifiers raise errors on each
other's outputs, is an engineering problem, not a fundamental limit, and fixing it would multiply the
corpus.

**Training the model itself.** Everything above, and everything in Part 2, changes the pipeline around
a fixed model, and none of it moved the score. The composed corpus is the first thing that changes the
model itself. The learnability probe passed its gate at 5 of 14, against 14 of 14 for single rules. A
pilot fine-tune and its gate without test-time training are running.

**A meta-learned starting point.** This isn't the cross-task warm start we closed. That copied a
neighbour's adapter, while this trains a starting point through the adaptation step itself, in the
style of Reptile. It needs a spread of tasks, which the 402 composed rules now give us and which we
didn't have before. It waits behind the fine-tune, since it inherits any damage the corpus does.

**Anything that attacks the 16-token vocabulary.** This is still our deepest constraint, but it's now
priced rather than untouched. The size-prefix run says new syntax inside the vocabulary costs the whole
pool during training. That doesn't close the direction, but it sets the bar. Any idea here has to
explain how it avoids the 14 of 19 self-contradictions the header ran into.

## Part 4. The seven gates, cheapest first

Run these before spending GPU time. Any one of them can close an idea outright.

**Gate 0. Is it in Part 2?** Working out a closed lever again is the most expensive habit there is.

**Gate 1. Does it act before or after the candidate pool?** This is the strongest predictor we have.
Ideas that act after the pool are bounded by the theorem, and so far they've gone nowhere. Both levers
that produced a genuine positive acted before it. This gate alone would have redirected most of a
season.

**Gate 2. Is the mechanism worth more than three points?** See fact 3.

**Gate 3. Can a 16-token vocabulary express it?** This closed run-length serialization on its own.

**Gate 4. Is there room for it to work?** Usually that's one CPU script that runs in under a minute.
Some examples that closed ideas are that 5.8% of cells are removable margin, run-length saves 13.9%,
deleting uniform rows cuts off 6.7% of answers, and the task queue finishes with 14% to spare, so
dropping the tail can't explain our noise. If the quantity an idea acts on is small, the idea is small.

**Gate 5. Could it feed in false supervision?** Training supervises the output grids, so any synthetic
pair whose output isn't what the rule really produces teaches the adapter a wrong rule. Cropped pairs
are only valid where the rule is local, and inverted pairs only where it's one-to-one.

**Gate 6. Prior art.** We've hit two traps. A widely indexed blog credits Sun-style self-supervised
training at test time to BARC, where it doesn't appear. And "test-time training" means two different
procedures in two literatures, so an idea can look unexplored purely because the field stopped using
the other name.

## Part 5. How experiments go wrong here

Every rule below was written after it cost us something.

**Use three arms, not two.** A treated arm against a base can't separate "this objective helps" from
"any extra optimisation helps". Every experiment that gave a readable answer had a third arm doing the
same amount of the same kind of work on the wrong data.

**Keep the step budget the same across arms.** If the arm with the extra objective gets extra steps, a
win can't be attributed. Make it spend half its steps on the extra objective, so a win has to beat the
training it displaced.

**Only score matched pairs.** Arms can have different tasks eligible. One summary compared an arm's
total over 4 tasks with the base's total over 10 and announced a false closure.

**Refuse to give a verdict on an empty run.** Two summaries printed confident closures on runs that
produced no candidates at all. If nothing was decoded, that's a failure of the harness, not a result.

**Check the right quantity.** One verdict judged convergence on the loss at step 16 and reported no
advantage, while the final loss was 30% lower on 7 of 8 tasks.

**Look at the data.** This is the most valuable rule here. Two exhaustive scans over 159,600 pairs
missed what rendering two tasks caught on one screen. One was a "two-step" task whose second step did
nothing on the test instance, and the other was a shape relationship that changed between
demonstrations. Filters that work on aggregates can't see that something is incoherent. Before
training on a corpus, render some of it.

**Be wary of checks that pass when something holds on at least one instance.** All five composition
filters worked that way, and that's how the no-op got through. Tighten them to every instance. The
count fell from 1,440 to 402.

**Write down the prediction first.** It's free, and it's the only thing that separates a measurement
from a hunt for confirmation. Score it afterwards either way.

**Watch which way your numbers move as you look harder.** The composition count went from 851 to 1,440
to 402. The noise estimate went from 0.24 to 1.01 to 1.18 to 1.28. When every refinement moves a number
the same way, the current value is probably still biased in that direction.

**One width per process.** A benchmark that ran five batch widths in one process measured memory
fragmentation, not width. This platform has no `expandable_segments`.

**Check the build itself, not one keyword.** A check that only searched the log for "undefined" passed
a LaTeX table with 22 broken rows. `scripts/build_paper.py` now fails on any line starting with `!`,
alignment errors, undefined references, duplicate labels and runaway arguments.

**Don't write Python patch scripts through bash heredocs.** The heredoc quietly eats one backslash
from `\\n` inside string literals. It broke a LaTeX table and two scripts in one day. Write them with a
file-writing tool instead.

## Part 6. What closing an idea costs

| closed by | cost | examples |
|---|---|---|
| a proof | nothing | selectors built only on the model's beliefs are bounded |
| arithmetic | nothing | run-length coding needs a token the vocabulary doesn't have |
| one CPU script | about a minute | crop margin, the run-length saving, cut-off answers, simulating the queue |
| **rendering the data** | about five minutes | the second step that did nothing, the inconsistent shape relationship |
| one night on the rig | about eight hours | pre-fitting, subset adapters, adapter transfer, three augmentations |
| one submission | a day | the time cap, and it landed inside the noise anyway |

## Part 7. Ideas from the throughput finding (16 August)

The noise turned out to come from empty pools being written as `[[0]]`, and that produced a batch of
ideas the earlier lists couldn't have contained. Two closed themselves on a read of the code.

**Closed for free. A search that times out doesn't waste its work.** `turbo_dfs` leaves its
`while time.time() - start_time < 540` loop and returns whatever endings it has found, and finished
batches are already saved into `dir_outputs` before the outer timeout breaks. There's no half-built
tree thrown away, so there's nothing to reclaim.

**Closed for free. v7.3's loss cut barely does anything, so v7.3 is close to a clean test of its greedy
fallback.** `ARC_LOSS_CUT=0.05` skips the search entirely for a task whose final training loss is above
the cut, which sounded dangerous given that loss doesn't separate solved tasks from missed ones.
Measured across 12 tasks of training in the deployed style, none exceed it, and the highest is 0.026.
Our slice leans toward reachable tasks, and the wider record goes up to 0.09, so it must fire somewhere
across all 240. But it's an effect at the edges, not the explanation for 28.61, and v7.3 can be
submitted as built.

### Still open, cheapest first

**1. Replacing the empty-pool fallback of `[[0]]` with the test input. Dead, settled in 30 seconds.**
The argument was that 88% of a typical output is copied straight from its input, so the input should
be a real lottery ticket where `[[0]]` is none. Measured, the test output equals the test input on 0 of
172 evaluation outputs, and "every training output is identical and equals the answer" is also 0.
Copying the input is worth exactly what `[[0]]` is worth. The 88% is about how much of a grid gets
copied, not about whether the whole grid does, and I read it as the second thing.

The only related quantity that survives is that on 21.5% of outputs the training outputs all share one
shape and it matches the answer's shape. Knowing the shape isn't knowing the answer, so that doesn't
rescue a constant fallback either. Any useful fallback has to be a prediction from the model, which is
what makes the greedy fallback below the only live member of this family.

**2. Submit v7.3 three times.** Its `ARC_GREEDY_FALLBACK` finds keys with no output and decodes them
greedily inside a 180-second budget, which is exactly the mechanism the throughput finding pointed to.
It's already built, and against the stored control of ten runs, three draws resolve 2.5 points.

**3. A second attempt where the pool has exactly one candidate.** It's dead in the form that copies the
input, for the reason above. But the waste underneath is real and wider than empty pools, because
`fill_submission` writes `v[:len(target_dict)]`, so an output with one candidate leaves the second
attempt at `[[0]]`. A second greedy decode under a different augmentation would be a real second
ticket. That costs GPU time rather than nothing, which puts it behind the free items.

**4. Ordering tasks by predicted cost.** Decode cheap tasks first, so more of them clear the
1,800-second cap before a slow session runs out. It's pure scheduling with no change in behaviour, and
it goes after the mechanism directly rather than the symptom.

**5. Fine-tuning anchored with a KL penalty.** This is the principled version of mixing in other data.
Penalise drift from the frozen base on inputs from the ARC distribution, instead of diluting the
training stream. If anything lets full exposure to the composed rules sit alongside a preserved ARC
prior, it's this rather than a ratio of data.
