# Idea triage

A reference for generating and screening ideas on this project. Read the constraints and the closed
list first, then the gates. Everything here is measured, not assumed, and each rule exists because
its absence produced a wrong answer at least once.

---

## Part 0: how the system actually works

Point at a stage when proposing something. Each stage is annotated with where it sits relative to the
candidate pool, because that is the strongest predictor of whether an idea can work (Gate 1).

### The base

`qwen3_4b_grids15_sft139` — NVARC's Qwen3-4B, specialised to grids. **16-token vocabulary**: the ten
digits plus control ids. 36 layers, hidden 2560, 32 attention heads, 8 KV heads, head dim 128, bf16.
Prompts are ID-level ChatML. We did not train it and we do not modify it. It was already heavily
pretrained on synthetic ARC data, which is why adding more ARC-1-shaped data was redundant.

### Per task, 240 tasks over 4 L4s in 12 hours

```
  task (2-6 demonstration pairs + 1-2 test inputs)
        |
   [1] AUGMENT  ---------------------------------------------  UPSTREAM
        |   TRAIN augment(n=16, seed=1) = 2 (transpose) x 4 (rot90)
        |              x 16 random colour permutations = 128 ROWS,
        |              then example-order shuffle.
        |   EVAL  augment(n=2, seed=2) = 8 x 2 = 16 views.
        |   cut_to_len drops demo pairs from the FRONT until the text
        |   fits, 8192 for training text, 8192 - 931 for eval input.
        v
   [2] TEST-TIME TRAINING  -----------------------------------  UPSTREAM
        |   fresh LoRA per task, reset to the saved default weights
        |   every task: rank 256, alpha 32, rslora (effective scale 2),
        |   dropout 0, NINE modules -- the seven projections PLUS
        |   embed_tokens and lm_head.
        |   128 rows, 1 epoch, batch 1, accum 1 = 128 optimizer steps.
        |   AdamW, lr 5e-5, cosine, warmup_ratio 0.1, clip 1.0, ~240s.
        |   The collator supervises EVERY assistant span, including the
        |   first demonstration's output (measured: 96 of 96 rows,
        |   sim_ttt_shape.py). 35% of tokens carry a loss. So this is
        |   NOT leave-one-out; it is next-token prediction over the whole
        |   demonstration sequence. No input grid is ever supervised,
        |   which IS the ARChitects' explicit choice. The test answer is
        |   never seen anywhere.
        v
   [3] DFS DECODE  -------------------------------------------  the POOL is produced here
        |   turbo_dfs: enumerate EVERY completion whose cumulative
        |   probability stays above 0.2 (max_score = -log 0.2).
        |   Recurses once per token. Batched 4 views at a time, by
        |   hard-coded index offsets over the sorted view keys.
        |   Candidates inverted back to the original frame, deduplicated.
        v
   [4] SELECT  -----------------------------------------------  DOWNSTREAM
        |   score_kgmon = multiplicity - mean augmentation NLL,
        |   rescored under fresh augmentations. Top 2 become attempt_1
        |   and attempt_2.
        v
  submission.json
```

### The three deadlines, all in wall-clock seconds

| where | budget |
|---|---|
| global run | 12 h minus a 600 s reserve (42480 s) |
| per task | 1800 s |
| inside one DFS call | 540 s |

Nothing is budgeted in operations, which is why identical code scores differently on different days
and why anything that changes speed changes which candidates exist.

### Where the score comes from

240 tasks, two attempts each, either attempt correct scores the task. Roughly 0.417 points per
single-output task. Our best run: 32.64.

### Two stacks, and they are not the same

| | Kaggle (scored) | our rig (probes) |
|---|---|---|
| hardware | 4x L4, 24 GB each | 1x RTX 5070 Ti, 16 GB |
| OS | Linux | Windows |
| stack | **Unsloth** (`FastLanguageModel`) | plain `transformers` + `peft` |

Unsloth needs Triton, which has no native-Windows backend, which is why the rig cannot run the
deployed stack. Any memory or throughput measurement on the rig is therefore a statement about the
rig, not about the scored runs. This bit us once already: a decode memory cliff we measured at 16 GB
may not exist at all on the L4s.

**And the rig probe is a strictly weaker pipeline than the submission, in three separate ways.** Any
rig null has to be read against all three before it is believed:

| | deployed | rig probe |
|---|---|---|
| LoRA targets | 9, including `embed_tokens` and `lm_head` | **7**, no embedding, no head |
| context | 8192 | **6144** |
| optimizer steps | 128 (one epoch over 128 rows) | 128, but **32** in the fixed-budget arm probes |

The module list is the one that should worry us most. On a 16-token vocabulary `embed_tokens` and
`lm_head` are 16x2560 each, so adapting them is nearly free and is literally re-learning what each
colour token MEANS for this task. The probe cannot do that at all.

### What each stage has already absorbed

| stage | tried there | outcome |
|---|---|---|
| [1] augment | masked-output completion, crops, rule inversion, run-length serialization | all null or unimplementable |
| [2] TTT | rank/depth sweep, pre-fitting, subset adapters, cross-task warm start, continued fine-tune | all null on coverage |
| [3] decode | best-first ordering, batch width, per-task cap, seed restarts | all null |
| [4] select | reweighting, cell-wise voting, dual-expert fusion, invariant filtering | all null, and bounded by proof |

---

## Part 1: the six facts that constrain every idea

**1. The system is generation-starved, not selection-starved.** 65% of tasks never produce the
correct grid under any of the sixteen augmented views, and they consume 71% of the budget doing it.
When the truth *does* reach the candidate pool the deployed selector places it top-two essentially
always (18 of 18 on fresh tasks). Anything downstream of generation is operating on the half that
works.

**2. The base model has a 16-token vocabulary.** It emits grid tokens and control ids and nothing
else. No chain-of-thought, no natural-language hypotheses, no program text, no self-critique, no new
marker tokens without breaking the pretrained embedding. Most of the contemporary reasoning
literature is structurally unavailable. This is the deepest constraint in the system and nothing we
have tried touches it.

**3. We cannot measure improvements under about three points.** Same-configuration noise is
sd ≈ 1.2, from eight runs of one unchanged pipeline, five of them the *same immutable kernel version*
scoring 30.56 / 32.64 / 28.47 / 31.39 / 30.56. Detection cost scales as 1/δ²:

| true effect | runs per arm to detect | days at 1 submission/day |
|---|---|---|
| 1 point | 29 | 58 |
| 2 points | 7 | 14 |
| 3 points | 3 | 6 |
| 4 points | 2 | 4 |

An idea worth less than three points can still be measured *offline* but can never be demonstrated on
the leaderboard before the deadline.

**4. Training loss is already near zero and is not the bottleneck.** Solved tasks span 0.0006 to
0.025, missed tasks 0.003 to 0.09 — overlapping almost entirely. The adapter fits the demonstrations
nearly perfectly whether or not the task is solved. Anything that improves the *fit* arrives faster
at a place we already reach.

**5. The base is bimodal per task.** It internalises a rule or concentrates on a wrong one, with no
partial-credit middle. This is what kills refinement loops, verification-gated resampling, and every
method assuming a gradient of partial correctness.

**6. Everything is budgeted in wall-clock seconds, not operations.** Three deadlines: a 12-hour
global wall, an 1800s per-task cap, a 540s budget inside the search. This is why identical code
scores differently on different days.

---

## Part 2: what has already been closed

Do not re-propose these. A closed lever reopens only on a new **measurement**, never a new argument.

### Selection and search (downstream of the pool — all null)
| lever | what closed it |
|---|---|
| selector reweighting | leave-one-task-out refit scores *below* the deployed selector |
| cell-wise voting | 0 of 41 missing truths recovered; pool errors are correlated |
| second-model rescoring / dual-expert | oracle gap **0 of 18** on fresh tasks |
| any belief-only selector | proof: bounded by top-two posterior mass |
| best-first decode order | median 2.9% saving over 16 logged trees |
| augmentation-seed restarts | union of three seeds = best single seed, exactly |
| invariant mining as a filter | 0 recovered, kills 11 of 19 truths |
| per-task cap at 1200s | 29.31, inside noise; the prediction was wrong |
| task-mined candidate filters, 9 rules | replayed over the 432-candidate pool: 7 of 9 false-kill, and the other 2 kill **nothing at all**. No candidate the decoder emits ever uses an unseen colour or has the wrong shape where the task preserves shape — the model already satisfies those constraints. `sim_prune_replay.py` |
| depth-decayed DFS threshold | dominated at matched cost by simply raising the flat floor, and backwards in premise: truths emit **deeper** (median 199 vs 143) and **cheaper** (headroom 1.600 of 1.609 vs 0.846). Depth is not the axis that separates them. `sim_dfs_budget.py` |
| **raising the DFS probability floor** | run at a fixed 600s decode, one adapter per task, three floors. 0.20 gives 6 of 14 outputs and 56 candidates; 0.35 and 0.45 both give **5** and 48/43. Worse on one task, better on none, tied on seven. The freed forward passes bought nothing: every arm still exhausted its wall clock. It also scored the two offline replays against each other, and the **pool replay was right** (truth falls above 0.25) while the tree replay's 13 covered outputs were too few. `rig_dfsfloor.py` |

### Adaptation
| lever | what closed it |
|---|---|
| adapter rank / depth | inverted U; the shipped config sits on the peak |
| pre-fitting on the test input | 7 / 7 / 6 across base, treated, control — nothing moved |
| leave-one-demo subset adapters | +0 coverage over 20 outputs |
| continued fine-tune on ARC-1-shaped data | honest negative; bimodality unmoved |
| refinement loops | no partial-credit middle to grip |
| **cross-task adapter transfer** | retrieval *works* (AUC 0.884) and warm-start *works* (30% lower loss, 7 of 8, graded by relatedness) — and coverage is **6 vs 6**. The transfer is real and buys nothing. |

### Augmentation space
| lever | what closed it |
|---|---|
| masked-output completion | 8 vs 8 on 12 matched tasks |
| sub-grid bounding-box crops | 3 vs 3 on 5 matched tasks; only 5.8% of cells are margin |
| rule inversion (shape-matched) | 4 vs 4 on 7 matched tasks |
| run-length serialization | unimplementable: no spare token in a 16-token vocabulary |
| lossless uniform row/col deletion | saves 1.0% of cells, truncates the answer on 6.7% of tasks |
| the whole token-saving family (RLE, coordinate interleaving, digit prefixes) | motivation removed: at Kaggle's 8192, **0 of 240 training views** and 6 of 344 eval views lose a demonstration. `cut_to_len` is the only channel by which a saved token can change an answer, and it is barely firing. `sim_token_budget.py` |
| **unchanged-cell loss masking** | the mechanism WORKED and it lost anyway. Error on the cells the rule changes fell from 0.286 to **0.062**; error on the cells it copies rose from 0.024 to **0.095**; coverage fell 6 to 3 of 14. The equal-size random control is indistinguishable from base (6 of 14, tied on all 8 tasks), so this is not "less supervision", it is specifically that the supervision teaching the model to COPY is load-bearing. `rig_maskloss.py` |
| column-major output ordering | it is the transpose view, and `augment` already keeps it: `mod(np.transpose, keep=True)` at arc_loader.py:250. Every training set already contains the column-major reading |
| **forced output-dimension prefix (Gate 3, run on purpose)** | total wipe-out. base 4 of 10 outputs and 37 candidates; dim **0 of 10** and **2**. Of 19 completions the dim arm produced, 14 declared a shape and then drew a different one, and 3 had no well formed header at all. After 32 TTT steps the model cannot hold its own stated intent for the length of one grid. This is the price of new in-vocabulary syntax and every proposal in that family pays it. `rig_dimprefix.py` |
| canonical palette pre-sorting | two independent kills. Applied before the deployed augmentation it is provably a **no-op**, because `permute_rnd_all_` draws uniformly from all 10! bijections and a fixed map composed with a uniform one is uniform. Applied per demonstration as proposed, the maps disagree on a shared colour in **92% of tasks**. `sim_palette_canon.py` |
| toroidal wrap, zero-padded translation, blank row/col permutation | false supervision at scale: content touches an edge on **79% of grids** and **93% of tasks**, the cells the rule changes touch the border on **65%** of shape-matched pairs, and the count of uniform rows/cols changes across **72%** of them. `sim_aug_damage.py` |

### Program synthesis
| lever | what closed it |
|---|---|
| 1.5B DSL proposer | 0 of 101; could not express the transforms |
| 14B coder, zero-shot | 0 of 120 on the full set, median per-cell lift over identity +0.000 |
| 14B reasoner | 0 of 12, empty near-miss band |
| symbolic object-count prune | 27.92 hidden; offline, a *predicted* oracle is not accurate enough |

**Running total: 24 levers measured against coverage or the hidden score, none moved it. Five more
closed offline at zero GPU cost** (the five new rows above), which is the cheaper way to close
something and the one to reach for first.

---

## Part 3: what is still open

**Execution-verified program synthesis at a budget we can afford.** Imbue lifted an open-weights
model from 12.1% to 34.0% using programs scored by execution against the demonstrations — the exact
channel our ceiling theorem names as its escape. Their cost is $2.42–$8.71 per task; the offline
track allows about $0.20. The open question is whether it fits, not whether it works.

**Compositional training data.** 402 clean two-step rules exist, built by composing RE-ARC verifiers
(30 first stages × 28 second stages), each instantiable without limit. A corpus of 78,792 examples is
built. Whether training on it helps is being measured. The first-stage bottleneck — 158,000 of
159,600 pairs never ran because verifiers raise on each other's outputs — is engineering, not a
fundamental limit, and fixing it would multiply the corpus.

**Training the model itself.** Everything above and everything in Part 2 changes the PIPELINE around
a fixed model. None of it moved the score. The compositional corpus is the first thing that changes
the model: 402 two-step rules from composed RE-ARC verifiers, 78,792 examples, each instantiable
without limit. The learnability probe cleared its gate at 5 of 14 against 14 of 14 for single rules.
A pilot fine-tune and its no-TTT gate are running.

**Meta-learned initialization.** Not the same as the cross-task warm start we closed. That transferred
a neighbour's adapter; this trains an initialization THROUGH the adaptation step, Reptile-style. It
needs a distribution of tasks, which is exactly what the 402 composed rules now provide and what we
did not have before. Gated behind the fine-tune, since it inherits any damage the corpus does.

**Anything attacking the 16-token vocabulary.** Still the deepest constraint we have, and now priced
rather than untouched: the dimension-prefix run says new in-vocabulary syntax costs the entire pool
during TTT. That does not close the direction, but it sets the bar. Anything here has to explain how
it avoids the 14-of-19 self-inconsistency the header hit.

---

## Part 4: the seven gates, in cost order

Run these before spending GPU time. Each can close an idea outright.

**Gate 0 — is it in Part 2?** Re-deriving a closed lever is the most expensive habit available.

**Gate 1 — upstream or downstream of the candidate pool?** The strongest predictor we have.
Downstream is bounded by the theorem and empirically 0-for-0. Both levers that produced a genuine
positive were upstream. *This gate alone would have redirected most of a season.*

**Gate 2 — is the mechanism worth more than three points?** See fact 3.

**Gate 3 — can a 16-token vocabulary express it?** This closed run-length serialization by itself.

**Gate 4 — is there structural room?** Usually one CPU script, usually under a minute. Examples that
closed ideas: 5.8% of cells are removable margin; run-length saves 13.9%; uniform deletion truncates
6.7% of answers; the task queue finishes with 14% slack so tail-dropping cannot explain our variance.
**If the quantity the idea acts on is small, the idea is small.**

**Gate 5 — does it risk injecting false supervision?** The collator supervises output grids, so any
synthetic pair whose output is not what the rule actually produces trains the adapter on a wrong
rule. Cropped pairs are valid only where the rule is local; inverted pairs only where it is
injective.

**Gate 6 — prior art.** Two traps we have hit: a widely indexed blog attributes Sun-style
self-supervised TTT to BARC, where it does not appear; and "test-time training" means two different
procedures in two literatures, so an idea can look unexplored purely because the field stopped using
the other name.

---

## Part 5: how experiments go wrong here

Every rule below was written after it cost us something.

**Three arms, not two.** A treated arm and a base cannot separate "this objective helps" from "any
extra optimisation helps". Every experiment that produced a readable answer had a third arm doing the
same *amount* of the same *kind* of work on the wrong data.

**Fix the step budget across arms.** If the auxiliary arm gets extra steps, a win is unattributable.
Make it spend half its steps on the auxiliary so a win must beat the training it displaced.

**Score matched pairs only.** Arms have different eligibility. A summariser compared one arm's total
over 4 tasks against base's over 10 and announced a false closure.

**Refuse a verdict on an empty run.** Two summarisers printed confident closures on runs that
generated zero candidates. If nothing decoded, that is a harness failure, not a result.

**Check the right quantity.** One verdict judged convergence on the loss at 16 steps and reported
"no advantage" while the *final* loss was 30% lower on 7 of 8 tasks.

**LOOK AT THE DATA.** The single most valuable rule here. Two exhaustive scans over 159,600 pairs
did not catch what rendering two tasks caught in one screen: a "two-step" task whose second stage was
a no-op on the test instance, and a shape relationship that changed between demonstrations. Aggregate
filters cannot see incoherence. **Before training on a corpus, render some of it.**

**Beware "at least one instance" predicates.** All five composition filters were "holds on at least
one instance", which is how the no-op survived. Tighten to *every* instance. The count fell from
1,440 to 402.

**Pre-register the prediction.** Free, and the only thing separating a measurement from a search for
confirmation. Score it either way afterwards.

**Watch which direction your numbers move under scrutiny.** The composition count went 851 → 1,440 →
402. The noise estimate went 0.24 → 1.01 → 1.18 → 1.28. When every refinement moves a number the same
way, the current value is probably still biased in that direction.

**One width, one process.** A benchmark that ran five batch widths in one process measured allocator
fragmentation, not width. This platform has no `expandable_segments`.

**Verify the build, not a keyword.** A check that grepped only for "undefined" passed a LaTeX table
with 22 malformed rows. `scripts/build_paper.py` now fails on any `!` line, alignment errors,
undefined references, duplicate labels and runaway arguments.

**Do not write Python patch scripts through bash heredocs.** The heredoc eats one backslash from
`\\n` inside string literals, silently. It broke a LaTeX table and two scripts in one day. Author them
with a file-writing tool instead.

---

## Part 6: what closure costs

| closed by | cost | examples |
|---|---|---|
| proof | none | belief-only selectors are bounded |
| arithmetic | none | run-length needs a token the vocabulary lacks |
| one CPU script | ~1 min | crop margin, RLE saving, answer truncation, queue simulation |
| **rendering the data** | ~5 min | the no-op second stage; the inconsistent shape relation |
| one rig night | ~8 h | pre-fitting, subset adapters, adapter transfer, three augmentations |
| one submission | 1 day | the cap — and it landed inside noise anyway |

---

## Part 8: ideas from the throughput finding (16 Aug)

The variance turned out to be made of empty pools becoming `[[0]]`, and that generated a batch of
ideas the earlier lists could not have contained. Two closed themselves on a code read.

**Closed free: a timed-out DFS does not waste its work.** `turbo_dfs` exits its
`while time.time() - start_time < 540` loop and returns whatever suffixes it has, and completed
batches are already pickled into `dir_outputs` before the outer timeout breaks. There is no discarded
partial tree, so there is nothing to reclaim here.

**Closed free: v7.3's loss cut is nearly inert, so v7.3 is close to a clean test of its greedy
fallback.** `ARC_LOSS_CUT=0.05` skips the search entirely for a task whose final training loss
exceeds the cut, which sounded actively dangerous given loss does not separate solved from missed.
Measured across 12 tasks of deployed-style TTT: **0 exceed it**, max 0.026. Our slice is biased
toward reachable tasks and the wider record reaches 0.09, so it must fire somewhere on the full 240,
but it is a tail effect and not the explanation for 28.61. v7.3 can be submitted as built.

### Open, ranked by cost

**1. ~~Replace the `[[0]]` empty-pool fallback with the test input.~~ DEAD, gated in 30 seconds.**
The argument was that 88% of a median output is a verbatim copy of its input, so the input should be
a real lottery ticket where `[[0]]` is none. Measured: the test output equals the test input on
**0 of 172** evaluation outputs, and "all train outputs identical and equal to the answer" is also 0.
Copying the input is worth exactly what `[[0]]` is worth. The 88% figure is about how much of a grid
is copied, not about whether the whole grid is, and I read it as the second thing.

The only related quantity that survives: on 21.5% of outputs the train outputs share a single shape
and it matches the answer's shape. Knowing the shape is not knowing the answer, so that does not
rescue a constant fallback either. **Any useful fallback has to be a model prediction**, which is
what makes the greedy fallback below the only live member of this family.

**2. Submit v7.3 three times.** Its `ARC_GREEDY_FALLBACK` finds keys with no output and greedy-decodes
them inside a 180-second budget, which is exactly the mechanism the throughput finding identified.
Already built, and against the banked n=10 control three draws resolve 2.5 points.

**3. attempt_2 where the pool has exactly one candidate.** Dead in its input-copy form for the reason
above, but the underlying waste is real and wider than empty pools: `fill_submission` writes
`v[:len(target_dict)]`, so an output with ONE candidate leaves attempt_2 at `[[0]]`. A second greedy
decode under a different augmentation would be a real second ticket. That costs GPU rather than
nothing, which moves it behind the free items.

**4. Task ordering by predicted cost.** Decode cheap tasks first so more of them clear the per-task
1800s cap before a slow session runs out. Pure scheduling, no behavioural change, and it attacks the
mechanism directly rather than the symptom.

**5. KL-anchored fine-tuning.** The principled version of corpus mixing: penalise divergence from the
frozen base on ARC-distribution inputs instead of diluting the training stream. If anything lets full
composed exposure coexist with a preserved ARC prior, it is this rather than a data ratio.
