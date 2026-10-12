# Who is above us on the 2026 ARC-AGI-2 board, and what they are doing

Compiled 2026-08-06, from the Kaggle leaderboard API and public sources. Written because the honest
answer to "why is first place so much higher" turned out to change what our own forward section
should say.

## The board has two populations, not one

Pulled live, top of the board with team names:

| rank | team | score |
|---|---|---|
| 1 | nvbanana | **67.50** |
| 2 | rabbithole | **44.86** |
| 3 | Junhua Yang | 37.22 |
| 4 | Nguyen | 34.58 |
| 5 | Bong | 34.44 |
| 6 | Team BlackBox | 33.89 |
| 7 | (tied) | 33.89 |
| 8-10 | (tied) | 33.47 |
| 11-12 | (tied) | 33.19 |
| ~30 | **us** | **32.64**, tied with 21 others |

The top three are isolated singletons separated by 22.6 and 7.6 points. From rank 4 down it is a
dense plateau with entries tied at *identical* decimals: 33.89 twice, 33.47 three times, 32.64
twenty-two times. Ties at exact values are shared code, not convergent independent work. That
plateau, us included, is one public lineage being forked.

So the interesting comparison is not us against first place. It is the public-notebook plateau
against three teams doing original work.

## The plateau is mostly noise

The plateau runs from about 31 to 34.6, roughly 3.5 points wide. Our measured run-to-run standard
deviation on byte-identical code is 1.01 points. The whole plateau is therefore about three and a
half standard deviations of the noise we measured directly, which means a large share of the
ordering inside it is not method quality. Some entries above us are not better, they had a faster
session. We are at 32.64 partly for the same reason.

This is worth stating in the paper because it is a claim only we are positioned to make. Nobody else
has spent a submission re-running one immutable kernel version to price the noise.

## Score quantum, confirmed

Solving for the denominator over all seventeen current-board scores, allowing only the leaderboard's
own two-decimal rounding, gives exactly 720 (and its double). Our 32.64 is 235/720. That reconciles
with the paper's task credit: 240 tasks with one to three test outputs each makes a whole
single-output task worth 1/240 = 0.4167 points, and the finer 0.139 steps are partial credit on
multi-output tasks. The "one task credit = 0.41" figure stands.

## nvbanana is the team we already cite

nvbanana is NVARC renamed, the NVIDIA Kaggle Grandmasters pair (JFPuget and gonedarragh) whose 2025
solution our paper already builds a citation on. Their trajectory:

- 2025 winner at **24.03%**
- mid-2026 at **39%** (Greg Kamradt's leaderboard tracking)
- now **67.50%**

For scale, average individual human performance on ARC-AGI-2 is around 60-66%, and the grand prize
threshold is 85%. The 2025 winning score would now sit far down the plateau.

No 2026 writeup from them yet. Prize eligibility requires open-sourcing, so one is coming. Same for
rabbithole (xiezejian, cookizesong, yuhuake240506) at 44.86.

## What produces scores in that range: execution-verified program evolution

The strongest published open method is imbue's code evolution
(<https://imbue.com/research/2026-02-27-arc-agi-2-evolution/>, 2026-02-27). It evolves Python
functions that map input grid to output grid, and the fitness is dominated by **running the candidate
program on the demonstration pairs**:

- 90% correctness on the demonstration examples, executed, with per-cell partial credit, where the
  identity function scores 0.2 and an exact solution scores 1.0
- 7% a transfer score, a reasoning model judging whether the program generalises rather than
  hard-coding colours
- 3% a simplicity score penalising hard-coded constants and length
- at most 16 iterations, stopping early once two distinct solutions clear a threshold

Reported results, baseline to evolved:

| model | baseline | evolved | cost/task |
|---|---|---|---|
| Kimi K2.5 (open weights) | 12.1% | **34.0%** | $2.67 |
| Gemini 3 Flash | 34.0% | **61.4%** | $2.42 |
| Gemini 3.1 Pro | 88.1% | **95.1%** | $8.71 |

Two things follow for us.

**This is our ceiling theorem's escape route, confirmed at scale by someone else.** Appendix F says
selection over statistics of the adapted model's own likelihoods is bounded by its top-two posterior
mass, and that escaping requires a channel checkable against observable data. Executing a candidate
program against the demonstrations is exactly that channel, and it nearly triples an open-weights
model's score.

**It also explains why our own attempt at that route failed.** v7.5a tried the same escape with a
symbolic object-count prune and scored 27.92, three sigma below the family mean. The difference is
oracle quality, not the idea. imbue's oracle is exact, the program either reproduces the
demonstration pairs or it does not. Ours was a *prediction* of component count from the same AIF
machinery, and our invariant-mining measurement already showed why that fails: with two or three
demonstrations, coincidental regularities are common and ARC-AGI-2 test pairs are built to break
them. The paper's stated precondition, that an external oracle helps only if it is more accurate
than the model's own beliefs about the same property, is what separates the two results. Execution
satisfies the precondition trivially. Prediction does not.

## The caveat that matters for transfer

imbue's numbers are API-based, at $2.42 to $8.71 per task. The Kaggle offline track allows no
internet and roughly $50 per submission across the evaluation set, on the order of $0.20 per task.
That is a factor of twelve to forty. So imbue's method as run is not eligible, and nvbanana's 67.5%
must be something offline-feasible on four L4s in twelve hours.

What that implies is the useful part. The gap between us and the top is not our decoder, our
selector, our adapter rank or our search order, all of which we have closed by direct measurement.
It is that the generator itself needs a verifiable channel, and the open engineering question is how
to get execution-verified synthesis inside a $0.20-per-task offline budget rather than a $2.67 one.

## Consequences for our own work

1. The forward section should name execution-verified program evolution specifically, with imbue's
   numbers, rather than gesturing at "an induction arm". It is the same route the theorem derives,
   and it now has external evidence.
2. The v7.5a negative gets stronger rather than weaker. It is not evidence against the escape route,
   it is evidence for the theorem's precondition on oracle accuracy, and imbue is the positive
   control we could not run ourselves.
3. Confirmed from competition sources: the displayed leaderboard covers about half the test data and
   final standings use the other half. Our 32.64 is a favourable draw on one half and **does not
   carry over**. This kills any argument for spending submission days resampling.

## Sources

- Leaderboard, Kaggle competition API, `arc-prize-2026-arc-agi-2`, pulled 2026-08-06
- <https://imbue.com/research/2026-02-27-arc-agi-2-evolution/>
- <https://arcprize.org/competitions/2026/arc-agi-2>
- <https://x.com/GregKamradt/status/2049872125851529411>
- <https://arxiv.org/html/2601.10904v1> (ARC Prize 2025 technical report)
