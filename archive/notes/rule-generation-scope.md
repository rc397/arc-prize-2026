# Scoping an LLM pipeline that writes transformation rules

The question is whether to spend money growing our set of grid-transformation rules from about 800
to several thousand, so we can train a base model that beats the one we inherited. Written on 17
August, with 76 days left before the Kaggle deadline.

## Why this direction

The leaderboard says small gains are worth a lot. From our 32.64 peak, another 0.83 points reaches the
top 10 and 1.80 reaches the top 5. There are 25 entries above us and 21 tied with us. And the score
keeps landing on the same value, which says the pipeline can't deliver those gains. Five different
builds came back at exactly 30.56, so whatever we change in the pipeline isn't what sets the score.
The base model is.

Compute isn't the problem anymore. Since the padding fix, a 78,792-example epoch takes about a day on
the card. NVARC trained their base with a full-parameter fine-tune over something like a million
LLM-generated puzzles. What we're short of isn't examples, it's distinct rules. RE-ARC ships 400, our
composition work added 402, and every variant of that corpus overfit the 402 it had.

## What it would cost, which nobody should pay yet

| target | yield | generations | tokens | cheap model | frontier model |
|---|---|---|---|---|---|
| 500 rules | 20% | 2,500 | 4.5M in, 2.2M out | **$4** | $47 |
| 2,000 rules | 20% | 10,000 | 18M in, 9M out | **$16** | $189 |
| 2,000 rules | 5% | 40,000 | 72M in, 36M out | **$65** | $756 |
| 5,000 rules | 5% | 100,000 | 180M in, 90M out | **$162** | $1,890 |

This assumes about 1,800 input tokens, the instructions plus three example RE-ARC verifiers, and about
900 output tokens for a 40 to 80 line program. The prices are per-million rates and should be checked
again rather than trusted.

The API bill isn't the real risk. Even the worst case is under $200 on a cheap model. Two other things
are, and both are free to test first.

Generating locally brings the cost close to zero anyway. The rig already runs a 14B coding model
(`scripts/coder_probe.py`). At about 1.5 seconds a program, 10,000 generations is roughly four hours
of card time and no money. The filters are objective checks that run the code, so low precision only
costs throughput, and volume is exactly what a local model is good at. That 14B scored 0 of 120 at
solving ARC tasks, but writing a new generator is a creative task rather than a search, so failing at
one doesn't predict failing at the other.

## Phase 0, before spending anything. Is the number of rules even the lever?

We already have the data to answer this. Train two models on the same number of examples with
different rule diversity, say 8,000 examples drawn from 100 rules against 8,000 drawn from all 402,
and gate both. Each run takes about 2.5 hours, plus 2.2 hours to gate the fine-tuned arm.

If four times the diversity changes nothing, eight times won't either, and the whole direction closes
for free. If it does move something, the slope tells us how many rules are worth buying.

This is the gate, and nothing else should start before it.

## Phase 1, about $2 or one night on the rig. What's the yield?

Yield is the one unknown in the cost table, and it swings the bill tenfold. Generate 100 programs, run
the existing filters and count the survivors. These checks already exist.

- The program runs without raising on the RE-ARC input pools, using `pyexec.py` in a separate process
  that's killed on timeout.
- The output is a valid grid, 1 to 30 cells a side, with colours 0 to 9.
- It isn't the identity, and it isn't a no-op on any instance.
- The input stays visible in the output, so it's a transformation and not a constant.
- The relationship between input and output shapes is the same on every instance.

Those are the same checks that cut the composition survey from 1,440 rules to 402. They're worth
reusing because we tightened them after rendering two tasks and actually looking at them, which is
how we caught a second stage that did nothing on the test instance.

## Phases 2 and 3, if Phase 0 justifies them

Generate at scale, mint examples with `build_ft_corpus.py`, train for about a day per 78,000 examples,
and gate with `rig_ftgate.py` against the stored base-model results. Two weeks end to end, which fits
the calendar.

## What would make me stop

Phase 0 showing flat coverage across rule counts. That would mean the base's ARC ability isn't limited
by how many synthetic rules it has seen, and no amount of generation will fix it.

## Revised after rethinking batching, and a much better target

Batching saves less than it looks like, and it costs the very thing we're buying. Asking for ten
programs per call spreads the prompt across them, from 1,800 tokens each down to 180, which takes
10,000 programs from $16 to $11 on a cheap model, or from $189 to $140 on a frontier one. That's a
rounding error on a two-week project. And ten programs written in one response are correlated with
each other, when diversity is exactly what we're paying for. So batching trades away the thing that
matters to save money that doesn't. Batch for throughput if it helps, but vary the prompt on every
call, seed each with different example verifiers and an explicit list of things to avoid, and don't
batch just to save tokens we don't need to save.

The better idea changes the target. A model that solves ARC-AGI-2 at around 90% can do something far
more useful than invent rules from nothing. It can write a generator for each of the 1,000 public
ARC-AGI-2 training tasks. That would be a RE-ARC for ARC-2, which doesn't exist yet.

Why it's the right target.

- **It fills the exact gap.** RE-ARC implements ARC-1 rules, and our composition work stacks pairs of
  those. Neither looks like ARC-2. The training set does, by construction.
- **The check is objective and free.** A generated program is correct exactly when it reproduces that
  task's own demonstration pairs when you run it. No heuristics, no judgement calls and no rendering
  tasks by hand to spot a no-op. `pyexec.py` already runs code in an isolated process and kills it on
  timeout.
- **It's legitimate.** The ARC-AGI-2 training split is published for training. That's different from
  the public-evaluation contamination we found in another team's checkpoint, and the paper should say
  so plainly.
- **It's cheap.** 1,000 tasks at three attempts each is 3,000 generations. The prompt carries the
  demonstration pairs, so call it 3,000 tokens in and 900 out. That's about $6 on a cheap model and
  $68 on a frontier one.

The hard half isn't the transformation, it's the input sampler. RE-ARC's verifiers come with
generators that produce unlimited fresh inputs. A transformation on its own only gives us the task's
own three or four pairs, which is no more data than we started with. Either the model also writes a
sampler, and we check that it produces inputs the transformation can handle, or we apply verified
transformations to inputs borrowed from other tasks and lean on the filters that reject degenerate
output. The first is better and harder. The second is a fallback that at least multiplies the
instances.

Phase 0 still gates all of this. If training on 402 rules beats training on 100 by nothing, a thousand
ARC-2 rules won't help either, and none of this should be built.

## Measured on 18 August. The cost model above was wrong in both directions

The pilot ran 30 ARC-2 training tasks through deepseek-v4-flash, and three of the assumptions above
didn't survive it.

**The prompt is a third of the size I assumed.** The median prompt is 536 tokens, not the 1,800 I
guessed from instructions plus three example verifiers. The task's own demonstration pairs are the
whole prompt, and they're small.

**V4 Flash is a reasoning model, and that nearly killed the idea.** On a real task it spent 8,000
output tokens thinking and still hadn't written a single character of code. The finish reason was
"length" and the content was empty. Thinking tokens are billed at the output rate. The first pilot came
back 0 for 4 with every program empty, which looked like the model failing but was really the token
cap landing mid-thought. Turning thinking off with `thinking: {"type": "disabled"}` gets a complete
program back in about 600 to 1,000 tokens. The two modes differ by roughly twenty times on the bill.

**Yield is 10%, not the 20% I guessed.** Three of 30 verified on one attempt with thinking off. How it
fails matters more than the rate. Only one of the 27 failures matched even one demonstration pair. The
model either gets the rule or writes something unrelated, so there's no tail of near misses to mine
and no point writing a repair step.

| measured with thinking off | value |
|---|---|
| verified yield on one attempt | 10.0% |
| output tokens per call | 991 median |
| input served from cache | 31% (the 313-token prefix rounds down to 256 in 64-token blocks) |
| cost per call | $0.00118 off-peak |
| **cost per verified program** | **$0.0118 off-peak** |
| all 1,000 tasks with three attempts, off-peak | $3.53, or about $3.19 with early stopping |

**Caching barely matters here, and I put too much weight on it earlier.** Output is 92% of the bill,
so the whole prefix-cache trick saves about 9%. Early stopping, which is already built, saves four
times as much by not making the call at all.

**Running calls in parallel saves money, not just time.** Tokens cost the same however many
connections are open. What parallelism buys is finishing inside the 15-hour off-peak window, from
10:00 to 01:00 UTC. Run one at a time, the full job takes about 19 hours and spills into peak hours at
double the price. Six workers finish it in about three hours, all of it at the discount.

**Still open.** I'm measuring on the same tasks whether thinking mode raises yield enough to justify
roughly twenty times the cost per call. And Phase 0 still gates everything.
