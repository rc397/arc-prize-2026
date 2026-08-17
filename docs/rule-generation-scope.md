# Scoping an LLM rule-generation pipeline

The question is whether to spend money getting from ~800 grid-transformation rules to thousands, in
order to train a base that beats the one we inherited. Written 17 Aug, 76 days to the Kaggle deadline.

## Why this is the direction

The leaderboard says small gains are worth a lot: at our 32.64 peak, **+0.83 reaches top 10 and
+1.80 reaches top 5**, with 25 entries above and 21 tied. And the attractor says the pipeline cannot
produce them: five distinct builds return to exactly 30.56, so what we can change is not what sets
the score. The base model is.

Compute is no longer the blocker. A 78,792-example epoch is ~1 day on the card since the padding fix.
NVARC's base was full-parameter SFT over of order **10^6** LLM-generated puzzles. What we lack is not
examples, it is **distinct rules**: RE-ARC ships 400, our composition work added 402, and every
configuration of that corpus overfit the 402 it had.

## The cost nobody should pay yet

| target | yield | generations | tokens | cheap model | frontier model |
|---|---|---|---|---|---|
| 500 rules | 20% | 2,500 | 4.5M in + 2.2M out | **$4** | $47 |
| 2,000 rules | 20% | 10,000 | 18M in + 9M out | **$16** | $189 |
| 2,000 rules | 5% | 40,000 | 72M in + 36M out | **$65** | $756 |
| 5,000 rules | 5% | 100,000 | 180M in + 90M out | **$162** | $1,890 |

Assumes ~1,800 input tokens (instructions plus three few-shot RE-ARC verifiers) and ~900 output for a
40-80 line program. Price points are per-million rates and should be re-checked rather than trusted.

**The API bill is not the risk.** Even the pessimistic cell is under $200 on a cheap model. Two other
things are the risk, and both are free to test first.

**Local generation makes it nearly zero anyway.** The rig runs a 14B coder already
(`scripts/coder_probe.py`). At ~1.5s per program, 10,000 generations is about four hours of card
time and no money. The filters are execution-based and objective, so low precision costs only
throughput, and volume is exactly what a local model is good for. That 14B scored 0 of 120 at
*solving* ARC tasks, but writing a novel generator is a generative task rather than a search, so its
failure there does not predict failure here.

## Phase 0, before spending anything: is rule COUNT even the lever?

We already own the corpus that answers this. Train two models on **equal example counts** and
different rule diversity, say 8,000 examples drawn from 100 rules against 8,000 drawn from all 402,
and gate both. Both runs are ~2.5 hours plus a 2.2-hour ft-arm-only gate.

If diversity moves nothing at 4x, it will not move anything at 8x, and the whole direction closes for
free. If it does move something, the slope tells us how many rules are worth buying.

**This is the gate. Nothing else should start before it.**

## Phase 1, about $2 or one rig night: what is the yield?

Yield is the only unknown in the cost table above and it swings the bill by 10x. Generate 100
programs, run the existing filter chain, and count survivors. Filters already written:

- executes without raising on the RE-ARC input pools (`pyexec.py`, process-isolated, kill-on-timeout)
- output is a valid grid, 1-30 per side, colours 0-9
- not the identity, and not a no-op on any instance
- the input remains visible in the output, so it is a transformation and not a constant
- shape relationship consistent across every instance

Those are the same predicates that took the composition survey from 1,440 to 402, and the reason to
reuse them is that they were tightened after **rendering two tasks and looking at them**, which is
what caught a second stage that was a no-op on the test instance.

## Phase 2 and 3, if Phase 0 justifies them

Generate at scale, mint with `build_ft_corpus.py`, train ~1 day per 78k examples, gate with
`rig_ftgate.py` against the banked base-arm records. Two weeks end to end, which fits the calendar.

## What would make me stop

Phase 0 showing flat coverage across rule counts. That would say the base's ARC ability is not
limited by the diversity of synthetic rules it has seen, and no amount of generation fixes it.
