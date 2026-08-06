# Plan to November

Written 2026-08-06. The runway is 88 days to code, 94 to paper. This exists so the allocation is
decided once, while slack is cheap, rather than improvised in October when it is not.

| date | what | days from 2026-08-06 |
|---|---|---|
| **2 Nov 2026** | Kaggle submissions due | 88 |
| **8 Nov 2026** | papers due | 94 |
| 4 Dec 2026 | results announced | 120 |

## Hard blockers, none of which are research

These sink the entry regardless of score. Check them monthly, not in the last week.

- [ ] **Repo is private.** Solutions must be open-sourced under a permissive license to be prize
      eligible. Repo is MIT-0 but `rc397/arc-prize-2026` is private. Flip it, or host the PDF and
      code separately, before judging. Decide by 1 Oct, do not leave to November.
- [ ] **The paper must link to a Kaggle code submission.** Currently `55261090` at 32.64. If a later
      build wins, the link changes and so does the writeup's submission ID line.
- [ ] **Paper Track needs the main-track submission to exist and be valid.** It does.
- [ ] **Writeup is capped at 1500 words and sits at exactly 1500.** Every addition displaces
      something. Do not discover this in November.

## Where we stand

| | |
|---|---|
| best hidden score | **32.64**, rank 43, tied with 21 others |
| same-method family | {30.00, 30.56, 30.56, 30.97, 32.64}, mean 30.95, sd 1.01 |
| best single-card config | 18.19 |
| the paper's price on 16GB | about 12 points |
| paper | 59 pages, 0 undefined refs, HTML mirrored |
| leader | nvbanana at 67.50 |

## Closed by measurement. Do not re-run these.

Re-deriving a closed lever is the most expensive habit available to us, so the list is explicit.

| lever | verdict | evidence |
|---|---|---|
| selector reweighting | dead | LOO scores below the deployed selector |
| cell-wise voting | dead | 0 of 41 missing truths reconstructed |
| best-first decode order | dead | median 2.9% over 16 logged trees |
| augmentation-seed restarts | dead | union of 3 seeds equals best single seed exactly |
| invariant mining as a filter | dead | 0 recovered, kills 11 of 19 truths |
| dual-expert fusion | dead | oracle gap 0 of 18 on fresh tasks |
| subset-adapter unioning | dead | +0 coverage over 20 outputs |
| adapter rank / depth | at optimum | inverted U, current config sits on the peak |
| bigger base model | wrong bet | 200M LongT5 scores 27, Qwen3 plateaus 0.6B to 8B |

The shape of all of it: **selection and search are exhausted, generation is not.** 65% of tasks never
produce the truth under any view and consume 71% of the budget. That is the only number that matters.

## Workstreams, in priority order

### W1. Execution-verified program synthesis (the centrepiece)

The single highest-value thing left, because it hits four of the six rubric criteria at once.

Appendix F derives that selection over the adapted model's own likelihoods is bounded, and names
execution against the demonstrations as the escape. Imbue then measured it: candidate Python
programs scored 90% by execution against the task's own demonstration pairs lift an open-weights
model from 12.1% to 34.0%. Our own v7.5a failed the same route with an *approximate* oracle and cost
three sigma, which is the theorem's precondition confirmed from the other side.

The open question, which nobody has answered publicly, is whether this fits the offline budget.
Imbue runs at \$2.42 to \$8.71 per task. The offline track allows roughly \$0.20. **A local 4B doing
execution-verified evolution inside 12 hours on four L4s is exactly our paper's efficiency framing
applied to the one route that is not closed.**

Feasibility, rough: 240 tasks over 4 workers in 12h is about 720 GPU-seconds per task. A 4B emitting
a 500-token program at realistic decode speed is on the order of 10s, so roughly 60 program
candidates per task before any batching. Imbue uses at most 16 iterations. It is tight and it is not
obviously impossible, which is the definition of worth trying.

**Prior-negative caution, stated up front.** Our own notes record an induction arm that scored 0 on
the miss set across three model classes. That is a real result and it must not be waved away. But it
was run on the miss set alone, the hardest slice, and without the evolutionary loop or the
execution-fitness signal that is now externally validated. Our own standing lesson is not to
generalise a one-arm negative into a whole-system ceiling. So: rebuild it in the validated shape,
gate it on the rig, and only then spend submissions.

Steps:
1. Build the evolver offline against public eval. Fitness = exact execution on demonstrations, with
   per-cell partial credit. No LLM judge terms initially; they need a second model we cannot afford.
2. Rig-gate: measure solve rate on a stratified slice, and specifically on tasks the current pipeline
   misses, since that is the only place it can add anything.
3. Kill condition: if it recovers **zero** tasks from the current miss set on a 40-task rig slice, it
   is dead and the paper reports it as a measured negative against imbue's positive, which is still
   a publishable result about budget.
4. If it recovers anything, route it as a second arm behind the vote-agreement signal and submit.

### W2. Tighten the variance claim

Our most novel contribution is the error bar, and it currently rests on five runs. The 95% CI on the
standard deviation from n=5 is roughly **[0.60, 2.90]**; at n=15 it is about **[0.74, 1.59]**.

Resubmitting the same kernel to keep a lucky maximum is noise-mining and pointless, because the
displayed board is about half the test data and the final standing uses the other half. Resubmitting
to estimate the variance is the experiment. Same action, different purpose, and the paper must say
which one it is doing.

Spend about 12 slots on this, spread across the runway rather than in a block, so machine-load
seasonality is sampled rather than aliased.

### W3. Finish what is in flight

- Pre-fitting, 3 arms, running now. Pre-registered prediction is a null. Score it either way.
- Cap 1200, submission `55283687`, pending. Read against the band, not as a single number.
- Subset-disagreement as a **routing signal**: Jaccard 0.373 where the truth was found against 0.015
  where it was not, on n=7 and n=4. That is the progress signal training loss fails to provide. Worth
  a rig run at larger n before it can be more than a lead.

## Submission budget, 88 slots

| purpose | slots | notes |
|---|---|---|
| variance series (W2) | 12 | spread out, never in a block |
| W1 synthesis arm | 16 | only after the rig gate passes |
| in-flight and follow-ups | 10 | cap, pre-fitting, routing policy |
| reserve | 50 | it feels like a lot in August and will not in October |

Rough calendar:

- **August (25 left):** finish W3. Build and rig-gate W1. Start the variance series at roughly two
  per week.
- **September (30):** W1 submissions if the gate passed. Variance series continues. First full paper
  pass against the rubric.
- **October (31):** consolidate. Paper finalisation. Hard blockers cleared by 1 Oct. Reserve absorbs
  whatever went wrong.
- **1-2 November:** final submission selected and verified.
- **3-8 November:** paper final, PDF hosted, links checked.

## Decision rules

These exist so a bad result does not turn into a week of explaining it, which has happened here
before and cost the most expensive week of the project.

1. **Run the control first.** If an effect's premise is an artifact whose provenance we cannot audit,
   or is the basis for a programme rather than one comparison, the control is the first experiment,
   not the seventh.
2. **No hidden-set difference under about two points is a result.** State where a score sits in the
   band. Prefer "not distinguishable" to a directional story.
3. **Rig-gate before a submission slot**, in the actual TTT condition, for anything with a mechanism
   we can test offline.
4. **One factor per submission.** The builder must assert its edits landed and diff everything else.
5. **A closed lever stays closed** unless a new measurement, not a new argument, reopens it.
6. **Report negatives at the same weight as positives.** They are most of what we have and they are
   why the paper is worth reading.

## Rubric self-assessment, and where the remaining effort should go

Six equally weighted criteria, scored 0-5.

| criterion | current | where it moves |
|---|---|---|
| Accuracy | weak-mid, 32.64 is mid-field | W1 is the only realistic lift |
| Universality | strong | contamination instrument, selection theorem, replicate control all generalise |
| Progress | mid-strong | the closed-lever map plus the escape-route bracketing |
| Theory | strong | the ceiling theorem and its measured precondition |
| Completeness | strong | 59 pages, controls, negatives, provenance |
| **Novelty** | **weakest** | the method is a reproduction; novelty currently lives in measurement |

**Novelty and Accuracy are the two weak scores and W1 is the only workstream that lifts both.** That
is the argument for it being the centrepiece rather than one of several ideas.
