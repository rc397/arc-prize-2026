# A contamination audit of a public ARC checkpoint

Run on 1 August 2026. The data is in `paper/data/contamination_g{2,3,5}.json`, and
`scripts/rig_contamination_audit.py` reproduces it.

## What prompted it

We were testing whether a second, separately trained model could rescore our candidate pools. Pointed
at `sorokin/qwen3_2b_grids15_sft141`, the rescoring script reported a median likelihood of
0.0002 nats per token for the true answers on the ARC-AGI-2 public evaluation set. It also preferred
the true grid over every decoded alternative on 54 of 60 outputs, including 40 of the 41 where our own
4B never produced the truth.

That is too good. Over a 16-token vocabulary a uniform guess costs 2.77 nats a token, and a model that
reasons through a task still carries real doubt at the ambiguous cells. A number that small looks like
recall. The same script on our base model, `sorokin/qwen3_4b_grids15_sft139`, showed nothing like it.

## Why one number was not enough

Two things could fake that result.

The first is a broken encoding. This model family uses a 16-token word-level vocabulary with no merges,
and it had already caught us once, when the word "user" encoded to nothing. If the encoder silently
dropped or merged cells, the "answer" being scored could be a short run of tokens any model predicts
easily. So the first check decodes every scored answer back to a grid and compares it with the truth.

The second is the answer leaking into the prompt. We checked the slicing in the scoring code by hand,
and it is clean.

It also may not be misconduct at all. Many ARC training recipes include the public evaluation set on
purpose, because the private set is what gets scored. So the claim we set out to test is narrower than
"contamination". It is that the public evaluation set is not a valid instrument for measuring this
checkpoint.

## The instrument

A likelihood alone cannot tell memorising from being very good. Capability ordering can.

1. Establish how two models rank against each other on data generated after both were published.
2. Measure the same two models on the suspect benchmark.
3. Look for the ordering to flip. A 2B that beats a 4B from the same author on the suspect set while
   losing to it on fresh data is not showing capability.

This needs no access to training data, no private split and no cooperation from whoever built the
model. For fresh data we minted RE-ARC instances under our own seed. Their concepts are public, but the
specific instances cannot be in any published checkpoint.

## Design

The measurement is a two-by-two of the 2B and the 4B against the public evaluation set and fresh
RE-ARC, with greedy decoding and no test-time training. That is the condition people actually use these
models in when they reach for one as a scorer or a filter. The two columns differ in difficulty, and
that does not matter, because the comparison is the gap between the models within each column.

We fixed five checks in advance, each one gating the next.

| check | what it tests | what would stop the audit |
|---|---|---|
| G1 | decode every scored answer back to a grid, both tokenizers | any mismatch |
| G2 | reproduce the likelihood result with the verified encoding | the likelihood rises to a normal range |
| G3 | greedy-decode the 2B on public eval | low accuracy, meaning likelihood and decoding disagree |
| G4 | verify every fresh RE-ARC instance with RE-ARC's own verifier | any rejected instance kept |
| G5 | the two-by-two and its interaction | no flip, in which case we report capability instead |

G3 was the cheap way for the whole idea to die. A likelihood of 0.0002 nats makes a hard prediction
about free-running decoding, and if decoding had not matched, the likelihood would have been measuring
something else.

## Result

All five checks passed.

G1 compared 172 outputs and 75,052 scored tokens under both tokenizers, with no mismatches.

G2 measured truth-likelihood over 167 outputs of the public evaluation set.

| model | median NLL per token | share under 0.01 nats |
|---|---|---|
| 4B sft139 | 0.04039 | 13% |
| 2B sft141 | 0.00014 | 91% |

G4 minted 120 fresh RE-ARC instances under seed 20260801 at difficulty 0.35 to 1.0. RE-ARC's verifier
rejected 8 candidates, which were dropped.

G3 and G5 used greedy decoding, no test-time training and no search, with 60 outputs per cell.

| model | ARC-AGI-2 public eval | fresh RE-ARC |
|---|---|---|
| 4B sft139 | 10.0% (6 of 60) | 78.3% (47 of 60) |
| 2B sft141 | 80.0% (48 of 60) | 65.0% (39 of 60) |

On the public evaluation set the 2B leads by 70 points (z = 7.7). On fresh data it trails by 13 points
(z = 1.6). The interaction is 83 points at z = 8.0. The fresh-data gap alone is not significant at 60
outputs, and the claim does not need it to be. A 2B that beats its 4B sibling eight-fold on the suspect
set and does not beat it on data minted after both were published is not explained by capability.

## What it means

The ARC-AGI-2 public evaluation set is not a valid instrument for `sorokin/qwen3_2b_grids15_sft141`.
Any score, filter or candidate selection made with that checkpoint against the public set is reading
recall rather than reasoning.

This says nothing about intent. Training on the public evaluation set is normal in this field, because
the private set is what gets scored. The finding is about what the checkpoint can be used to measure.

It also cuts the other way. The 2B reaches 65% on fresh data, so it is a capable model whose
contamination is specific to the public set. Using it as a second scorer on the hidden set remains
legitimate.
