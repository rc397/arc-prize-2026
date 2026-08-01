# Scoping: a contamination audit of the public ARC checkpoint zoo

Status: COMPLETE, 2026-08-01. All five gates passed. Result in section 8.

## 1. What we actually observed

While testing the ceiling theorem's second escape route (score the candidate pool under a
separately trained model), `scripts/rig_crossscore.py` run with `ARC_SCORER_MODEL` pointed at
`sorokin/qwen3_2b_grids15_sft141` produced:

- median per-token NLL of the true ARC-AGI-2 public evaluation answer: **0.0002 nats**
- the 2B prefers the true grid over every decoded alternative on **54 of 60** outputs, including
  **40 of 41** outputs where our 4B never produced the truth at all

0.0002 nats/token over a grid of several hundred cells is not understanding. Over a 16-token
vocabulary, uniform is 2.77 nats and a confident-but-reasoning model still carries real uncertainty
at ambiguous cells. A number that small means recall.

The same script against `sorokin/qwen3_4b_grids15_sft139`, our base, shows nothing of the kind.

## 2. Why this is not yet a finding

Two failure modes would produce the same number, and neither has been ruled out.

**2a. Degenerate encoding.** The NVARC tokenizer is a WordLevel vocabulary of 16 tokens with no
merges, and it has already bitten us once: `tok.encode("user")` returns the empty list, and both
ChatML turn markers encode to the same pair. If `encoder_for(2B tokenizer)` silently drops or
collapses characters, the "answer" being scored could be a short degenerate token run that any model
predicts trivially. **Gate: decode the scored token ids back to a grid and assert equality with the
truth, for every task, before anything else.**

**2b. Prefix leak.** Checked and clean. `answer_nll` builds `ids = pre + ans` and slices
`logits[len(pre)-1 : -1]`, which has length `len(ans)` and aligns position `len(pre)-1` with the
prediction of `ans[0]`. No off-by-one, no visibility of the answer. Recorded here so it is not
re-litigated.

**2c. It may not be misconduct.** Many ARC training recipes deliberately include the public
evaluation set, because the private set is what gets scored. That is standard practice, not
cheating. So "contamination" is the wrong frame for the paper. The defensible claim is narrower and
more useful:

> The ARC public evaluation set is not a valid measurement instrument for these checkpoints, and it
> is widely used as one.

That is a methodological warning with a measurement behind it, not an accusation, and it does not
depend on establishing intent.

## 3. The instrument: capability-ordering inversion

A likelihood contrast alone cannot separate "memorized it" from "is very good at it". The decisive
version does not need training-data access, does not need the private test set, and does not need
the model card to be honest:

1. Establish a capability ordering between two or more models on a set **provably minted after they
   were published**.
2. Measure the same models on the suspect set.
3. Contamination shows up as an **interaction that inverts the ordering**. A 2B beating a 4B from
   the same author, same recipe family, same data pipeline, on the suspect set while losing to it on
   fresh data is not capability.

The fresh set is the crux, and we already have the generator: `D:\arc\re-arc` ships `generators.py`,
`verifiers.py` and `main.py`, so we mint instances under our own seed. Their concepts are public;
their specific realizations are not, and cannot be in any published checkpoint.

The general form of the method, which is what makes it worth a paper section:

> Given any set of models whose capability ordering is established on freshly generated data,
> contamination of a benchmark is the deviation of that benchmark's ordering from the fresh one.

Benchmark-agnostic, checkpoint-agnostic, needs no cooperation from the model's author.

## 4. Design and gates

The measurement is a 2x2 over {2B sft141, 4B sft139} x {ARC-AGI-2 public eval, fresh RE-ARC},
greedy decode, no TTT. No TTT is the right condition: a scorer or filter is used off the shelf, and
the raw base model is what people are actually reading.

Difficulty differs between the two column sets, which is fine. The design does not compare across
columns. It compares the **model gap within each column**, and asks whether the sign flips.

Gates, in order. Each must pass before the next runs.

| # | Gate | Fails if |
|---|---|---|
| G1 | Token round-trip: decode the scored ids back to a grid, assert equality with the truth on all tasks, both tokenizers | Any mismatch. The 0.0002 is then an encoder artefact and the whole line dies here |
| G2 | Reproduce the NLL result under the round-tripped encoder | Median NLL rises to a normal range |
| G3 | Greedy decode of the 2B on public eval. A model at 0.0002 nats must reproduce the answers | Accuracy is low. Then the NLL and the decode disagree and something is wrong with one of them |
| G4 | Fresh RE-ARC minted under our seed; verify with RE-ARC's own `verifiers.py` that each instance is a correct instance of its concept | Verifier rejects. Do not measure on unverified generated data |
| G5 | The 2x2, and the interaction term | Ordering does not invert. Then it is capability, and we report that instead |

G3 deserves emphasis. It is the honest test that could kill this cheaply: a teacher-forced NLL of
0.0002 makes a hard prediction about free-running decode, and if the decode does not match, the
likelihood number is measuring something other than what we think.

## 5. Cost

No training. Decode only, 2B and 4B, roughly 120 + 120 tasks per model per column.

- Mint and verify fresh RE-ARC: minutes, CPU
- Four decode sweeps: order of 1-2 hours total on the rig
- Fits alongside the score work rather than competing with it

The field-wide version, enumerating public ARC checkpoints on Kaggle and HF and running the same
instrument on each, is the paper-scale extension. Cost scales linearly in checkpoints and is
dominated by download bandwidth, not compute.

## 6. Rubric fit

Against the six criteria ([[paper-track-rules]]):

- **Novelty**: no published audit of ARC checkpoints for evaluation-set validity. The
  ordering-inversion instrument is new as far as we can tell.
- **Universality**: the method is benchmark-agnostic and checkpoint-agnostic, and needs neither
  training-data access nor a private split.
- **Progress**: a field that measures itself wrong cannot reach 85%. This also offers an explanation
  for something the field observes and does not explain, namely the systematic gap between numbers
  displayed beside public notebooks and numbers those notebooks score. Our own 33.89-versus-28.06
  episode is one instance of that genre.
- **Theory**: the signature can be stated precisely rather than as a heuristic.

## 7. Kill conditions

State these now so they are not negotiated later.

- G1 fails: encoder artefact, report nothing.
- G3 fails: likelihood and decode disagree, report the discrepancy as a caution about NLL-based
  contamination tests and stop.
- G5 shows no inversion: the 2B is simply better on this set than we assumed. Report it as a
  negative result in the paper, which is what we did with restarts and with targeted synthetic data.

If it survives all five, we name the checkpoint and state the measurement and the method, and we do
not speculate about intent, exactly as the paper already handles the 33.89.

## 8. Result

All five gates passed. Data in `paper/data/contamination_g{2,3,5}.json`.

**G1**, 172 outputs, both tokenizers, 75,052 scored ids, zero round-trip mismatches. The likelihoods
are computed on the truth.

**G2**, truth-likelihood over 167 outputs of the public evaluation set:

| model | median NLL/token | fraction under 0.01 nats |
|---|---|---|
| 4B sft139 | 0.04039 | 13% |
| 2B sft141 | **0.00014** | **91%** |

**G4**, 120 fresh RE-ARC instances minted under seed 20260801 at difficulty [0.35, 1.0], each
verified by RE-ARC's own verifier, 8 candidates rejected.

**G3 and G5**, the 2x2. Greedy decode, no test-time training, no search, 60 outputs per cell:

| model | ARC-AGI-2 public eval | fresh RE-ARC |
|---|---|---|
| 4B sft139 | 10.0% (6/60) | 78.3% (47/60) |
| 2B sft141 | **80.0%** (48/60) | 65.0% (39/60) |

Within-column model gap: **+70.0 points on eval** (z = 7.7), **-13.3 points on fresh** (z = 1.6).
Interaction **+83.3 points, z = 8.0**.

The fresh column is not at ceiling or floor, so it establishes an ordering, though note honestly
that the fresh-column gap alone is not significant at n = 60. The claim does not need it to be: a
2B that beats a 4B from the same author by eight-fold on the suspect set while **not** beating it on
data minted after both were published cannot be explained by capability, and a null on the fresh
column would suffice for that.

Conclusion, stated as narrowly as the evidence supports:

> The ARC-AGI-2 public evaluation set is not a valid measurement instrument for
> `sorokin/qwen3_2b_grids15_sft141`. Any score, filter, validation or candidate selection performed
> with that checkpoint against public eval is reading recall rather than reasoning.

Nothing here speaks to intent. Training on the public evaluation set is standard practice in this
field because the private set is what gets scored. The finding is about what the artefact can be
used to measure, not about the conduct of whoever built it.

Two consequences for our own work. First, it retroactively validates the call to discard the
cross-scoring experiment that used this checkpoint as a second belief state. Second, and more
useful: the 2B reaches 65% on fresh data against the 4B's 78.3%, so it is a **genuinely capable
model** whose contamination is specific to the public evaluation set. Using it as a second scorer on
the HIDDEN set is therefore legitimate and better motivated than before. That is lever F3 in
docs/score-levers.txt.
