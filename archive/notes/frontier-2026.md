# Who is above us on the 2026 ARC-AGI-2 board, and what they are doing

Written on 6 August 2026 from the Kaggle leaderboard API and public sources. I wrote it because the
honest answer to "why is first place so much higher than us" changed what the paper's forward
section should say. The numbers are as of that day. The board has moved a lot since.

## The board has two populations

Here is the top of the board as pulled that day.

| rank | team | score |
|---|---|---|
| 1 | nvbanana | **67.50** |
| 2 | rabbithole | **44.86** |
| 3 | Junhua Yang | 37.22 |
| 4 | Nguyen | 34.58 |
| 5 | Bong | 34.44 |
| 6 | Team BlackBox | 33.89 |
| 7 | (tied) | 33.89 |
| 8 to 10 | (tied) | 33.47 |
| 11 to 12 | (tied) | 33.19 |
| about 30 | **us** | **32.64**, tied with 21 others |

The top three stand alone, 22.6 and then 7.6 points apart. From fourth place down the board is a dense
plateau where entries tie at identical decimals. 33.89 appears twice, 33.47 three times and 32.64
twenty-two times. Exact ties like that come from shared code, not from teams converging
independently. That plateau, us included, is one public lineage being forked over and over.

So the comparison worth making is not us against first place. It is the plateau of public notebooks
against three teams doing original work.

## The plateau is mostly noise

The plateau runs from about 31 to 34.6, roughly three and a half points wide. Our measured run-to-run
standard deviation on byte-identical code was 1.01 points at the time. That makes the whole plateau
about three and a half standard deviations of noise we measured directly, so a good part of the order
inside it says nothing about method. Some entries above us are not better. They had a faster session.
We sit at 32.64 partly for the same reason.

This belongs in the paper because we are the only ones placed to say it. Nobody else spent submissions
re-running one immutable kernel version to price the noise.

## The score quantum

Solving for the denominator across all seventeen scores on the board, allowing only the
leaderboard's two-decimal rounding, gives exactly 720 or its double. Our 32.64 is 235 out of 720. That
matches the paper's task credit. There are 240 tasks with one to three test outputs each, so a whole
single-output task is worth 1/240, or 0.4167 points, and the finer 0.139 steps are partial credit on
tasks with several outputs. The figure of 0.41 points per task credit stands.

## nvbanana is the team we already cite

nvbanana is NVARC under a new name, the NVIDIA Kaggle Grandmasters pair of JFPuget and gonedarragh
whose 2025 solution our paper builds on. Their scores over time tell the story.

- They won 2025 with **24.03%**.
- By mid-2026 they were at **39%**, according to Greg Kamradt's leaderboard tracking.
- They are now at **67.50%**.

For scale, an average person scores somewhere around 60 to 66% on ARC-AGI-2, and the grand prize
needs 85%. The score that won 2025 would now sit well down the plateau.

They have no 2026 writeup yet. Prize eligibility requires open-sourcing, so one will come. The same
goes for rabbithole (xiezejian, cookizesong and yuhuake240506) at 44.86.

## What produces scores in that range

The strongest published open method is imbue's code evolution
(<https://imbue.com/research/2026-02-27-arc-agi-2-evolution/>, 27 February 2026). It evolves Python
functions that turn an input grid into an output grid, and the fitness comes mostly from actually
running each candidate program on the demonstration pairs.

- 90% of the fitness is correctness on the demonstrations, executed, with per-cell partial credit. The
  identity function scores 0.2 and an exact solution scores 1.0.
- 7% is a transfer score, where a reasoning model judges whether the program generalises or just
  hard-codes colours.
- 3% is a simplicity score that penalises hard-coded constants and length.
- It runs for at most 16 iterations and stops early once two distinct solutions clear a threshold.

Here are their reported results.

| model | baseline | evolved | cost per task |
|---|---|---|---|
| Kimi K2.5 (open weights) | 12.1% | **34.0%** | $2.67 |
| Gemini 3 Flash | 34.0% | **61.4%** | $2.42 |
| Gemini 3.1 Pro | 88.1% | **95.1%** | $8.71 |

Two things follow for us.

**This is the escape route from our ceiling theorem, confirmed at scale by someone else.** Appendix F
says that any selection built on the adapted model's own likelihoods is bounded by its top-two
posterior mass, and that getting past the bound needs a signal you can check against observable
data. Running a candidate program against the demonstrations is exactly that kind of signal, and it
nearly triples an open-weights model's score.

**It also explains why our own attempt at that route failed.** v7.5a tried the same escape with a
symbolic prune on object counts and scored 27.92, three standard deviations below the family mean as
we measured it then. The difference is the quality of the check, not the idea. imbue's check is exact,
because the program either reproduces the demonstrations or it does not. Ours was a prediction of the
component count, made by the same AIF machinery, and our invariant-mining work had already shown why
that fails. With only two or three demonstrations, coincidental regularities are common, and
ARC-AGI-2 test pairs are built to break them. The paper states the condition that separates the two
results. An outside check helps only when it is more accurate than the model's own beliefs about the
same property. Execution meets that condition trivially and a prediction does not.

## The caveat that matters

imbue's numbers come from paid APIs at $2.42 to $8.71 a task. The offline Kaggle track allows no
internet and roughly $50 a submission across the evaluation set, which is about $0.20 a task. That is
twelve to forty times less. So imbue's method as they ran it is not eligible here, and whatever
nvbanana does to reach 67.5% has to fit on four L4s in twelve hours with no internet.

That is the useful part. The gap between us and the top is not our decoder, our selector, our adapter
rank or our search order. We closed all of those by direct measurement. The generator itself needs a
signal it can verify, and the open engineering question is how to fit execution-verified synthesis
into a $0.20-a-task offline budget instead of a $2.67 one.

## What this changes for us

1. The forward section should name execution-verified program evolution specifically, with imbue's
   numbers, rather than waving at "an induction arm". It is the route the theorem points to, and it
   now has outside evidence behind it.
2. The v7.5a negative gets stronger, not weaker. It is not evidence against the escape route. It is
   evidence for the theorem's condition on the accuracy of the check, and imbue is the positive
   control we could not run ourselves.
3. Competition sources confirm that the displayed leaderboard covers about half the test data and
   final standings use the other half. Our 32.64 is a lucky draw on one half and will not carry over,
   which kills any case for spending submission days on resampling.

## Sources

- The leaderboard, from the Kaggle competition API for `arc-prize-2026-arc-agi-2`, pulled 6 August 2026
- <https://imbue.com/research/2026-02-27-arc-agi-2-evolution/>
- <https://arcprize.org/competitions/2026/arc-agi-2>
- <https://x.com/GregKamradt/status/2049872125851529411>
- <https://arxiv.org/html/2601.10904v1> (the ARC Prize 2025 technical report)
