# Context: ARC Prize 2026, single-16GB-GPU entry

Background on this project as of 2026-08-20. Everything is measured unless stated otherwise.

## The project

Two linked entries in ARC Prize 2026. The **Paper Track** is the priority: rubric-judged, one
submission, the writeup is the deliverable (`paper/the-16gb-ceiling.tex`, 77 pages, a living
document that grows with each chapter of work). The **ARC-AGI-2 main track** is an offline Kaggle
notebook, required for eligibility. The repo goes public 1 October.

The organising constraint is a single 16 GB GPU, which is the paper's subject rather than a caveat.

## The system

Base model is NVARC `qwen3_4b_grids15_sft139`: Qwen3-4B with a 16-token vocabulary, prompted at ID
level with ChatML. Not a Claude model.

Inference is: augment (128 train rows from 2 transpose x 4 rot90 x 16 colour permutations, 16 eval
views) -> per-task rank-256 LoRA test-time training (alpha 32, rslora, 9 modules including
`embed_tokens` and `lm_head`, 128 steps, lr 5e-5) -> `turbo_dfs` decode with floor 0.2 ->
`score_kgmon`. Three wall-clock limits bind it: 12 h global, 1800 s per task, 540 s per DFS call.
TTT dominates the cost.

## Where the score sits

| | |
|---|---|
| peak hidden score | 32.64 |
| defended build | v7.2 at 30.97 |
| control | mean 30.31, sd 1.34, n=10 |
| noise floor | 1 task = 0.4167 points |
| levers measured | 33, none moved the hidden score |

Nothing has beaten v7.2. Five distinct builds land on exactly 30.56. The evidence points at the base
model rather than the pipeline: the remaining knobs are all smaller than the instrument can resolve.
At sd 1.42 the leaderboard cannot separate anything below roughly 3.4 points without three runs an
arm.

## The generator corpus

For each of the 1,000 public ARC-AGI-2 *training* tasks, a model writes two functions: `solve(I)`
implementing the rule, and `gen_input(rng)` producing fresh inputs from the same family. A program
counts as verified only if `solve` reproduces every demonstration pair of its task under execution
(`scripts/pyexec.py`, process-isolated with a hard kill timeout). Nothing is judged by a model.

The training split is published for training, which distinguishes this from the evaluation-split
contamination trap documented in `docs/contamination-audit.md`. The result is a "RE-ARC for ARC-2",
which does not otherwise exist: RE-ARC covers ARC-1 rules, and the 402 compositions built earlier
here are pairs of those.

Current state: **128 verified generators, 127 with samplers**, in `experiments/genrules_*.jsonl`
where each line carries the full program under `src`. Total spend about 13 CNY.

Driver is `scripts/gen_rules.py` (`check` / `plan` / `run` / `runc` / `repair` / `report`), an
OpenAI-compatible client with `deepseek`, `openai` and `local` providers.

### What generation measured

All arms below are paired on the same tasks.

| condition | verified | CNY/call |
|---|---|---|
| thinking off | 110/1000 (11%) | 0.0107 |
| blind retry on failures | 2/40 (5%) | 0.0107 |
| repair with traceback | 3/40 (7.5%) | 0.015 |
| effort=low | 17/25 (68%) | 0.117 |
| effort=max, 64k cap | 3/5 | 0.259 |
| Qwen3.8-27B local Q3_K_M | 2/5 | free, 4.4 min/call |

- The effort dial is **not monotone**: low effort beats maximum on yield, price and speed at once.
- These models fail by **not terminating**, not by being wrong. Every call that finished thinking
  produced a verified program; two of five max-effort calls stopped exactly on the cap having
  emitted zero answer tokens. The same pattern appears in a second model family (Qwen at 20k), and
  was independently reported for that model within days of its release.
- **Execution against demonstrations is a 93% precise proxy** for capturing the real rule: 93 of the
  first 100 verified programs reproduce the held-out test pair they never saw. That is what licenses
  a demonstrations-only proof.
- Failures are **bimodal**: 63 of 483 clean failures matched even one demonstration pair, so there is
  no near-miss tail to repair. The same property holds for the 4B solver, at a different scale and
  task.
- Feeding a program its own traceback beat a blind retry 3 to 2 out of 40, which is noise.

### Cost mechanics

DeepSeek's prefix cache is server-side and content-keyed, so parallelism does not lose it: 970 of 970
calls hit 256 cached tokens each under six workers. The instruction block is emitted before the task
grids so the prefix is stable, and `runc` fires one warmup call before the pool starts so workers do
not race a cold cache. Caching is worth about 9%; output is about 92% of the bill.

Rates, verified against a real bill to within 1.5%: V4 Flash peak is $0.014 cache-hit input, $0.44
cache-miss input, $1.32 output per 1M tokens. Off-peak is exactly half and runs 04:00-06:00 and
10:00-01:00 UTC. A CNY balance bills at roughly market, about 7.15 CNY per USD of list price.

## Closed directions

- **Rule count (Phase 0)**: 100 composed ARC-1 rules against 402, equal 8,000-example budgets.
  Identical results, tied 12 of 12 on ARC-AGI-2. Recorded as an inconclusive null rather than a
  refutation, since twelve ties means the gate could not detect an effect in either direction.
- **Greedy fallback**: 30.56 / 29.72 / 29.44, mean 29.91 against control 30.31. Under the noise
  floor. Its earlier appearance was bundled with two other changes in one submission, so this was
  the first time the mechanism was measured alone.
- **Local generation**: worse on accuracy and not fixable on speed. Qwen3.8-27B at Q3_K_M is fully
  GPU-resident at 47.7 tok/s, but enabling the model's own MTP layers for self-speculative decoding
  took it to 6.96 tok/s, a 6.3-fold loss, because 13.6 GiB of resident weights leave 258 MiB and the
  draft context has nowhere to live. Speculation trades memory for bandwidth and there is none to
  trade at this capacity.

## Open questions

**The shape gate.** Phase 0 varied rule *count*. It never varied rule *shape*: every rule held here
is ARC-1, and the new corpus is ARC-2-shaped by construction. `scripts/mint_genrules.py` builds pools
by running each generator process-isolated, then reuses `build_ft_corpus.mint()` unchanged.

There is a known confound. RE-ARC pools carry roughly 40 distinct pairs per rule; these samplers
yield about 7. An earlier figure of "9.5 usable draws per 10 seeds" measured whether draws were
valid, never whether they differed from one another, and largely they do not. So the arms are not
matched on raw material, and a loss for the ARC-2 arm would not separate shape from pool poverty.
Matching requires subsampling the ARC-1 pools and retraining both arms, roughly 9 h on an idle rig.

The outcome gates about 101 CNY: at effort=low the remaining 865 failed tasks would yield roughly
588 more generators, taking the corpus to about 715.

**FAAST** (arXiv 2605.04651) computes adaptation in closed form, `W* = K†V` by pseudoinverse, in one
forward pass with no backprop, roughly 15x cheaper than gradient adaptation. It targets per-task TTT,
which is the binding cost here. Against it: its own limitations section names compositional reasoning
as the weak spot, which is ARC by definition; its language-modelling results are worse than backprop
(15.35 against 13.60 perplexity) and only match with an extra readout component; and it learns only a
linear map from frozen representations. Its precondition is testable without implementing it, by
checking whether ARC transformations are linearly decodable from the base's hidden states.

## Environment notes

GPU work runs on a remote Windows box ("rig", `D:\arc`, RTX 5070 Ti 16 GB, Blackwell sm_120 requiring
CUDA 12.8+). The laptop edits and copies over, and is closed often, so long jobs must survive
disconnect.

Facts that each cost a day:

- The remote shell is PowerShell; scripts go over as base64 UTF-16LE.
- Bare `python` over SSH hits a WindowsApps stub that silently produces nothing; the real interpreter
  must be resolved under `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13*`.
- Only detached *python* survives the SSH session. A `Start-Process` of another executable dies with
  the connection: llama-server was observed loading, binding its port, and vanishing when the ssh
  call returned.
- `Get-Process python` returns nothing while jobs are healthy, because the process name is
  `python3.13`.
- A power cut on 2026-08-18 killed every running job, so long runs checkpoint and resume from disk.
- On Kaggle, `machine_shape: "NvidiaL4"` is required and its absence is silent; `model_sources` must
  read `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` with a capital T, and wrong values
  land in `invalidModelSources` while the push still reports success; and submission needs a
  `kernel_version` where 1 is not a safe default, since Kaggle retains submittable output only for
  recent versions.

## Measurement pitfalls this project hit

- **Inverted symptoms.** Empty model output looked like incapacity and was a token cap landing
  mid-thought. The fix moved cost twentyfold in the same direction as correctness.
- **Small-sample overstatement.** A bimodality figure of 3.7% from 27 failures became 13% at 483. A
  repair rate of 80% at n=5 became 68% at n=25.
- **Mean against median.** A cost projection built on median output tokens understated the bill by
  58%, because a few programs ran twelve times long. Cost scales with the mean.
- **Unprotected `exec`** on model-written code wedged an interpreter for 20 hours.
- **Unstable API shapes.** `balance_infos` carries one entry per currency in unstable order, and
  reading index 0 returned an empty USD entry that would have aborted a fully funded run.
- **Confounded levers.** A mechanism submitted once alongside two other changes is not a measured
  mechanism; isolating one cost three days of a one-per-day submission quota.
- **Displayed numbers are not measurements.** A public notebook showing 33.89 scored 28.06 when
  forked unmodified, which is why unmodified controls run first.
