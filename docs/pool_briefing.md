# Pool briefing, two open problems from an ARC-AGI-2 pipeline

One-page orientation for a mathematician joining cold. The full formal treatment is Appendix F of the
attached paper. Everything marked measured is a number we computed directly, nothing here is folklore.

## The system in three lines

- Solver for ARC-AGI-2 puzzles. Each task gives 2 to 6 demonstration pairs of an unknown grid
  transformation and one or more test inputs. A 4B language model is fine-tuned per task on the
  demonstrations, then a depth-first search over 16 augmented views decodes candidate output grids.
- A selector picks 2 attempts per test output from the candidate pool. Score is one credit per task,
  split evenly over its test outputs. Hidden leaderboard has 240 tasks, so one credit is 0.4167 points.
- Competition constraint is a 12 hour wall on 4 GPUs for the 240 tasks, tasks processed sequentially
  per GPU.

## The dataset (pool_dump.json)

- One measurement run over 40 public evaluation tasks. 60 test outputs across 39 tasks produced
  candidates, one task decoded nothing.
- Schema. Keys are `taskid_testindex`. Each value is a list of distinct candidate grids, each with
  `grid` (2D int array), `count` (how many of the 16 augmented decodes produced it), `beam` (list of
  search-path negative log-probabilities, one per producing decode), `aug` (per-decode lists of
  per-view negative log-likelihoods under fresh augmentations), `correct` (bool, public solutions).
- Totals, measured. 847 decoded subkeys, 432 distinct candidates, 152 correct decodes. The true grid
  is present in the pool on 19 of 60 outputs. The deployed selector recovers 15, so 9.833 of 39
  credits against a present-in-pool ceiling of 12.333.

## Established facts (all measured on this dump)

- The two production selectors are monotone functions of the pool statistics. kgmon scores
  count minus mean augmentation NLL, probmul scores summed (3 minus beam) plus mean summed
  (3 minus aug). On this pool they select identically on all 60 outputs.
- On every one of the 4 present-but-missed outputs, the true grid is strictly Pareto-dominated inside
  its own pool on (count, mean aug NLL, best beam NLL). Dominators are wrong grids. Counts of
  dominators per output are 1, 1, 3, 2. Hence no selector monotone in these statistics can recover
  any of the four. A 16-rule sweep over that family confirmed it empirically, best rule gained one
  output, which is also the null expectation of a 16-rule search.
- The truth, when missed, is near the chosen wrong answer. On one output the correct grid sits 2 cells
  from the selected grid while ranking 6th of 13. Per-cell majority fusion of 13 same-shape candidates
  reproduces the same 2 wrong cells the winner has. Errors are common-mode across candidates, not
  independent noise.
- An external-looking signal failed instructively. Injecting candidate c as an extra demonstration and
  measuring the mean NLL change of held-out real demonstrations (Delta, Appendix F equation 7) ranked
  the truth in the bottom half on 3 of the 4 missed outputs. Measured Delta tracks candidate
  description length, simple grids perturb the conditional less, so minus Delta acts as a smoothness
  prior, again a functional of the model's beliefs. Gate statistics, truth beat 4 synthetic
  distractors on 133 of 172 outputs (chance 20 percent) but median winning margin 0.0043 nats against
  median losing margin 0.0031.
- Noise floor. Two hidden submissions of one identical build differed by 0.41 points, one credit.
  Paired-comparison resolution needs at least 6 unanimous task flips for p below 0.05 (Appendix F.2).

## Problem 1, selection. When can any signal beat the pool?

- The pool statistics are all estimates of functionals of the sampling distribution q the model
  induces. When the model internalizes a wrong rule, q concentrates off the truth, and dominance
  follows. The proposition is trivial, the premise is measured.
- Open, the converse. Characterize the class of side signals that escape. The Delta signal used
  genuinely external data (the held-out demonstrations) but measured it through the same model and
  collapsed to a prior. Is there a clean formalization separating signals that factor through the
  model's likelihood from signals with an independent verification channel, and a proof that the
  former cannot beat argmax q in expectation?
- Open, debiasing. Delta confounds correctness with description length. Does a normalization exist,
  for instance subtracting the perturbation the same candidate causes when paired with a scrambled
  test input, that removes the length term while keeping any rule-consistency term? Testable on this
  dump plus a local GPU within hours.
- Open, pool-only lower bound. Non-monotone rules were never excluded by the proposition. Is there an
  identifiability argument that the pool alone (exchangeable given q) cannot distinguish the truth
  from a mode of q, giving a Bayes bound on any pool-only rule with 2 attempts?

## Problem 2, allocation. Choosing the per-task cap

- Model. Task i has natural completion time tau_i, unknown until run, and step success
  p_i(t) = s_i for t at least tau_i else 0. Policy is a uniform cap C. Sequential processing against
  wall T gives k(C), the number of tasks reached, and S(C), the score (Appendix F equation 8).
  S(C) is non-monotone in C, the maximizer can be interior.
- Data. task_timing.csv has 40 rows of (task, final training loss, wall seconds, outcome in
  solved, in-pool, miss, empty). tau ranged 174 to 1889 s. 71 percent of task-seconds went to tasks
  yielding nothing. Training loss weakly predicts yield, every task finishing above loss 0.05
  produced nothing the selector kept.
- Open, optimal static C. Given the empirical tau distribution and (s, tau) dependence, derive
  E[S(C)] and C*. A bootstrap answer with a predicted gain of 2 or more credits at N = 240 changes
  our next submission directly. Live experiment already running at C = 1200 against a history of
  C = 1800.
- Open, adaptive policies. Training loss is observed before the expensive search. Two-stage caps,
  value of that observation, and whether the setting is isomorphic to a known one, Pandora's box,
  bandits with knapsacks, prophet inequalities, with a competitive ratio we can import.

## Ground rules that keep results real

- Any proposed rule must be stated before it touches new data. We can generate a fresh pool on a
  disjoint 40 tasks for out-of-sample validation on request.
- Effects below one credit are inside the measured noise floor. A claim needs either the 6-flip bound
  on a paired comparison or a mechanism strong enough to pre-register.
