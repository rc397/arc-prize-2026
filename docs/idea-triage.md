# Idea triage

A reference for generating and screening ideas on this project. Read the constraints and the closed
list first, then the gates. Everything here is measured, not assumed, and each rule exists because
its absence produced a wrong answer at least once.

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

### Program synthesis
| lever | what closed it |
|---|---|
| 1.5B DSL proposer | 0 of 101; could not express the transforms |
| 14B coder, zero-shot | 0 of 120 on the full set, median per-cell lift over identity +0.000 |
| 14B reasoner | 0 of 12, empty near-miss band |
| symbolic object-count prune | 27.92 hidden; offline, a *predicted* oracle is not accurate enough |

**Running total: 24 levers measured, none moved the hidden score.**

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

**Anything attacking the 16-token vocabulary.** Untouched, and the deepest constraint we have.

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
