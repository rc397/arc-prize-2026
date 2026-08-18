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

## Revision after Ryan's point: batching, and a much better target

**Batching saves less than it looks, and costs the thing we are buying.** Ten programs per call
amortises the prompt from 1,800 tokens each to 180, which takes 10,000 programs from $16 to $11 on a
cheap model, or $189 to $140 on a frontier one. That is a rounding error against a two-week project.
And ten programs generated in one response are correlated with each other, while **diversity is
precisely the commodity we are trying to buy**. Batching therefore trades the variable that matters
for a cost that does not. Batch for throughput and round trips if you like, but vary the prompt per
call, seed each with different few-shot verifiers and an explicit avoid-list, and do not batch to
save tokens we do not need to save.

**The stronger idea, and it changes the target.** A model that solves ARC-AGI-2 at ~90% can do
something much more valuable than invent rules from nothing: it can **write a generator for each of
the 1,000 public ARC-AGI-2 TRAINING tasks**. That is a RE-ARC for ARC-2, which does not exist.

Why this is the right target:

- **It fills the exact gap.** RE-ARC implements ARC-1 rules; our composition work stacks pairs of
  them. Neither is ARC-2-shaped. The training set is, by construction.
- **The gate is objective and free.** A generated program is correct if and only if it reproduces
  that task's own demonstration pairs under execution. No heuristics, no judgement, no rendering
  tasks by hand to catch a no-op. `pyexec.py` already does process-isolated execution with
  kill-on-timeout.
- **It is legitimate.** The ARC-AGI-2 training split is published for training. This is not the
  public-eval contamination trap we diagnosed in another team's checkpoint, and the distinction
  should be stated explicitly in the paper.
- **It is cheap.** 1,000 tasks at 3 attempts each is 3,000 generations. The prompt carries the demo
  pairs, so about 3,000 tokens in and 900 out: **$6 on a cheap model, $68 on a frontier one.**

The hard half is not the transform, it is the **input sampler**. RE-ARC verifiers ship with
generators that produce unlimited fresh inputs. A transform alone gives us only the task's own three
or four pairs, which is no more data than we started with. Either the model writes a sampler too and
we verify it produces inputs the transform handles, or we mint by applying verified transforms to
inputs borrowed from other tasks and lean on the non-degeneracy filters. The first is better and
harder; the second is a fallback that at least multiplies instances.

**Phase 0 still gates all of it.** If training on 402 rules beats training on 100 rules by nothing,
then a thousand ARC-2 rules will not help either, and none of this should be built.

## Measured, 18 Aug: the cost model above was wrong in both directions

The pilot ran 30 ARC-2 training tasks through deepseek-v4-flash. Three assumptions in the table
above did not survive it.

**The prompt is a third the size assumed.** Median prompt is 536 tokens, not the ~1,800 guessed from
"instructions plus three few-shot verifiers". The task's own demonstration pairs are the whole
prompt and they are small.

**V4 Flash is a REASONING model, and that nearly killed the direction.** On a real task it spent
8,000 output tokens on `reasoning_content` and had still not emitted one character of code
(`finish_reason: length`, `content` empty). Reasoning tokens bill at the output rate. The first
pilot returned 0 of 4 with every program empty, which looked like model failure and was actually a
cap landing mid-thought. Sending `thinking: {"type": "disabled"}` returns a complete program in
about 600-1,000 tokens. The difference between those two modes is roughly twenty times the bill.

**Yield is 10 percent, not the 20 percent guessed.** 3 verified of 30, one attempt, thinking off.
The failure shape matters more than the rate: only 1 of 27 failures matched even one demonstration
pair. The model either infers the rule or produces something unrelated, so there is no partial-credit
tail to mine and no repair heuristic worth writing.

| measured, thinking off | value |
|---|---|
| verified yield, 1 attempt | 10.0% |
| output tokens per call | 991 median |
| input cached | 31% (the 313-token prefix rounds to 256 in 64-token blocks) |
| cost per call | $0.00118 off-peak |
| **cost per VERIFIED program** | **$0.0118 off-peak** |
| full 1000 tasks x3, off-peak | $3.53, or about $3.19 with early stop |

**Caching is a rounding error here and the earlier emphasis on it was misplaced.** Output is 92
percent of the bill, so the whole prefix-cache mechanism saves about 9 percent. Early stop, already
implemented, removes four times as much by not making the call at all.

**Concurrency is a cost lever, not a speed one.** Tokens cost the same however many sockets are
open. What concurrency buys is landing inside the 15-hour off-peak block (10:00-01:00 UTC): serial,
the full run is about 19 hours and spills into peak, paying double on the overflow. Six workers
bring it to about three hours, entirely inside the discount.

**Still open.** Whether thinking mode buys enough yield to justify roughly twenty times the cost per
call is being measured on the same tasks. And Phase 0 still gates all of it.
