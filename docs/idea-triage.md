# Idea triage: how to price an idea before spending a night on it

Twenty-four levers have been measured on this pipeline and none moved the hidden score. Five of the
closures needed no GPU at all, and several of those were only attempted after the expensive version
had already been planned. With a large batch of ideas incoming, the screen below runs first, in
order, and each step can close an idea outright.

The point is not to reject things. It is that a null costs a night of rig time and a *cheap* null
costs a minute, so anything answerable by arithmetic should be answered by arithmetic.

## Gate 0: is it already in the ledger?

`paper/the-16gb-ceiling.tex` section "The complete ledger" lists every closed lever with the evidence
that closed it. Re-deriving a closed lever is the most expensive habit available to us. A closed
lever reopens only on a new *measurement*, never on a new argument.

## Gate 1: upstream or downstream of the candidate pool?

The single strongest predictor we have.

- **Downstream** (reranking, filtering, voting, selection, attempt allocation, scheduling among
  existing candidates): bounded by the ceiling theorem, and empirically 0 for 0 across every lever we
  tried. On fresh tasks the deployed selector already places the truth top-two 18 times in 18. An
  idea here needs a channel that can be *checked against observable data*, not another statistic of
  the model's own beliefs. Default expectation: null.
- **Upstream** (generation, augmentation, adaptation, initialisation, the base model itself): where
  65% of tasks fail and 71% of the budget goes. Both of the two positive sub-results we obtained live
  here. Default expectation: worth measuring.

## Gate 2: is the mechanism worth more than three points?

Same-configuration noise is sd $\approx$ 1.2 with a 95% interval of roughly [0.8, 2.4]. Detection
cost scales as $1/\delta^2$: a 1-point effect needs ~29 runs per arm, 2 points ~7, 3 points ~3,
4 points ~2. At one submission per day, **anything under three points cannot be confirmed on the
leaderboard before the deadline.** It can still be worth measuring offline, but it can never be
shipped as a demonstrated gain, and the writeup must not claim one.

## Gate 3: can the 16-token vocabulary express it?

The base emits grid tokens and control ids and nothing else. No chain-of-thought, no natural-language
hypotheses, no program text, no new marker tokens without breaking the pretrained embedding. This
gate alone closed run-length serialization: a run marker would have to overload a digit and collide
with a colour.

## Gate 4: is there structural room? (usually one CPU script)

Before assuming a transformation helps, measure how much of it there is to do. Examples that closed
ideas in under a minute:

- sub-grid cropping: **5.8%** of cells are removable margin, median task 1.000, so there is almost
  nothing to crop
- run-length compression: background is 22.3% of cells and RLE saves **13.9%** of tokens
- uniform row/column deletion: saves 1.0% and truncates the true answer on **6.7%** of tasks
- prompt lengths against a memory threshold: median 2198 tokens, p75 3018

If the quantity the idea acts on is small, the idea is small, and no amount of GPU changes that.

## Gate 5: does it risk injecting false supervision?

The collator supervises output grids. Any synthetic pair whose output is not what the task's rule
actually produces trains the adapter on a wrong rule. Cropped pairs are valid only where the rule is
local; inverted pairs only where the rule is injective. This does not disqualify an idea, but it
means the arm needs a control and the writeup needs the caveat.

## Gate 6: prior art

Cheap, and it has changed the framing of every idea it touched. Two specific traps we have hit: a
widely indexed blog attributes Sun-style self-supervised TTT to BARC, where it does not appear; and
"test-time training" means two different procedures in two different literatures, so an idea can look
unexplored purely because the field stopped using the other name.

## If it clears all seven: how to build the experiment

**Three arms, not two.** A treated arm and a base cannot separate "this objective helps" from "any
extra optimisation helps". Every experiment here that produced a readable answer had a third arm
doing the same *amount* of the same *kind* of work on the wrong data.

**Fix the step budget across arms.** If the auxiliary arm gets extra steps, a win is unattributable.
Make it spend half its steps on the auxiliary so a win has to beat the real training it displaced.

**Score matched pairs only.** Arms have different eligibility. Comparing an arm's total over the 4
tasks it could run against base's total over 10 is not a comparison, and a summariser of mine did
exactly that and announced a false closure.

**Refuse a verdict on an empty run.** If no arm decoded anything, that is a budget or harness
failure. Two summarisers here printed confident closures on runs that generated nothing.

**Pre-register the prediction.** Write down the expected outcome before the run and score it after,
either way. It is free, and it is the only thing that distinguishes a measurement from a search for
confirmation.

## The record this screen is drawn from

| closed by | cost | examples |
|---|---|---|
| proof | none | belief-only selectors are bounded by top-two posterior mass |
| arithmetic | none | run-length markers need a token the vocabulary lacks |
| one CPU script | ~1 min | crop margin, RLE saving, answer truncation, queue simulation |
| one rig night | ~8 h | pre-fitting, subset adapters, adapter transfer, three augmentations |
| one submission | 1 day | the cap, and it landed inside noise |
