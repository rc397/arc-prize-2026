# ARC-AGI-2 generator corpus

128 programs, one per task, for tasks in the public ARC-AGI-2 training split. Each pairs a `solve(I)`
that implements the task's rule with, in 127 of them, a `gen_input(rng)` that samples fresh inputs
from the same family, so the pair can mint new training examples the way RE-ARC does for ARC-1.

## How it was made

A language model was asked, once per training task, for both functions. A program counts as
verified only if `solve` reproduces every demonstration pair of its task under execution, in a
separate process with a hard timeout (`scripts/pyexec.py`). No model judged correctness at any point.

| setting | tasks attempted | verified |
|---|---|---|
| DeepSeek-V4-Flash, reasoning off | 1,000 | 110 |
| DeepSeek-V4-Flash, reasoning effort low, on failures of the first pass | 25 | 17 |
| DeepSeek-V4-Flash, reasoning off, blind retry of failures | 40 | 1 more |

The first pass cost $1.57 in off-peak API charges and the low-effort pass about 2.5 CNY. With more
budget the low-effort setting would have kept working through the 865 first-pass failures it never
attempted, at a rate we measured at 17 of 25, which is the obvious way to grow this.

## Checks, run on the released files

- **Demonstrations.** Re-running every program against its task's demonstrations: 128 of 128 still
  reproduce them.
- **Held-out test outputs.** The training split publishes test outputs that no program saw. 120 of
  128 reproduce them exactly, so passing the demonstrations is a 94% precise signal that a program
  captured the intended rule. The other eight fit their demonstrations with a simpler rule than the
  task intends. They are kept and flagged with `test_outputs_reproduced: false` rather than removed;
  filter on that field if you want only the 120.
- **Samplers.** Of the first 95 programs, 93 mint fresh valid instances and 89 do so reliably,
  meaning at least 8 of 10 random seeds give a new input the program transforms. Draws are often
  close to one another: our minting found about 7 distinct pairs per rule, against roughly 40 for
  RE-ARC, so expect less variety per rule.

## What it did not do for us

Fine-tuning our 4B base on pools from 100 of these generators, against 100 ARC-1 rules at matched
pool depth and an 8,000-example budget, changed nothing on our twelve-task gate: 0 tasks better, 1
worse, 11 tied. That says this base does not benefit at that budget. It says nothing about larger
models or larger budgets, which is where we expect a corpus like this to matter. The paper's section
on synthetic rule diversity has the full design.

## Format

`arc2_generators.jsonl` holds one JSON object per line:

| field | meaning |
|---|---|
| `task` | task ID in the ARC-AGI-2 training split |
| `model`, `setting` | which model and configuration wrote the program |
| `has_sampler` | whether the program defines `gen_input(rng)` |
| `demo_pairs`, `test_pairs` | how many of each the task has |
| `demos_reproduced` | all demonstrations reproduced on re-check |
| `test_outputs_reproduced` | all held-out test outputs reproduced exactly |
| `output_tokens` | tokens the model emitted for this program |
| `program` | the Python source |

`arc2_generators_summary.json` has the counts above. The task grids are not included: pair each
record with `arc-agi_training_challenges.json` and `arc-agi_training_solutions.json` from the
ARC-AGI-2 training split (https://github.com/arcprize/ARC-AGI-2, Apache-2.0). Programs use numpy and
the standard library and none touches files, the network or other processes, but run them sandboxed
anyway, for example with `scripts/pyexec.py`.

To rebuild from the raw generation logs: `python scripts/export_generators.py`.

## License

Released under MIT-0 with the rest of this repository. The programs were generated with
DeepSeek-V4-Flash, whose weights are MIT-licensed, through its API.
