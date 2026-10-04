# Context for the ARC Prize 2026 single-GPU entry

Background on this project as of 5 October 2026. Everything is measured unless stated otherwise.

## The project

There are two linked entries in ARC Prize 2026. The Paper Track is the priority. It is judged by
rubric on one submission, and the deliverable is the writeup, meaning the 1,500-word
`paper/writeup.md` that Kaggle judges read first and the full paper behind it in
`paper/the-16gb-ceiling.tex`, now 80 pages. The ARC-AGI-2 main track is an offline Kaggle notebook and
is required for eligibility.

The organising constraint is a single 16 GB GPU, which is the paper's subject rather than a caveat.

## The system

The base model is NVARC `qwen3_4b_grids15_sft139`, a Qwen3-4B with a 16-token vocabulary, prompted at
ID level with ChatML. It is not a Claude model.

Inference runs in four stages. Augmentation builds 128 training rows from 2 transposes, 4 rotations
and 16 colour permutations, plus 16 evaluation views. Each task then gets its own rank-256 LoRA
adapter (alpha 32, rslora, nine modules including `embed_tokens` and `lm_head`, 128 steps at a learning
rate of 5e-5). `turbo_dfs` decodes with a probability floor of 0.2, and `score_kgmon` picks the two
attempts. Three wall-clock limits bind it, 12 hours in total, 1,800 seconds per task and 540 seconds
per decoding call. Test-time training dominates the cost.

## Where the score sits

| quantity | value |
|---|---|
| peak hidden score | 32.64, the maximum of twelve draws of one artifact |
| the control series | kernel version 339772104 run twelve times, mean 29.90, sd 1.34, range 28.47 to 32.64 |
| fresh-build control | 31.39 on 31 August, so the build script is not a confound |
| best single-card build | 18.19 (v6.1) |
| noise floor | one task is 0.4167 points, and two draws of one artifact differ by more than that 83% of the time |
| levers measured | 40, none of which moved the hidden score |

Nothing has beaten the reproduction. The evidence points at the base model rather than the pipeline,
because every remaining knob is smaller than the instrument can resolve. At an sd of 1.34 the
leaderboard cannot separate anything below roughly four points without three runs an arm, by an exact
two-sample t at 80% power.

The last arms ran three draws each against the 29.90 control. The greedy fallback alone averaged
29.91, the DFS cache crop 28.70, fair view scheduling 28.61 (29.44, 27.78 and 28.61, t = -2.09,
p = 0.09) and the no-dead-AIF build 29.86. None is distinguishable from the control on its own. All
six draws of the two arms that move decode time sit below it, which was noticed after the fact and is
reported as a pattern rather than a test.

On the public leaderboard of 1 October we sit 61st of 2,310 at 32.64. Tufa Labs leads at 83.06, then
rabbithole at 79.72 and nvbanana at 75.42.

## The generator corpus

For each of the 1,000 public ARC-AGI-2 training tasks, a model writes two functions, `solve(I)`
implementing the rule and `gen_input(rng)` producing fresh inputs from the same family. A program
counts as verified only when `solve` reproduces every demonstration pair of its task under execution,
in a separate process with a hard kill timeout (`scripts/pyexec.py`). No model judges anything.

The training split is published for training, which separates this from the evaluation-split
contamination trap in `docs/contamination-audit.md`. The result is a RE-ARC for ARC-2, which does not
otherwise exist, since RE-ARC covers ARC-1 rules and the 402 compositions built earlier here are pairs
of those.

The corpus holds 128 verified generators, 127 with samplers. It was released on 4 October as
`corpus/arc2_generators.jsonl` with both verification results per program, built by
`scripts/export_generators.py` from the raw `experiments/genrules_*.jsonl` on the rig. Total spend was
about 13 CNY. The driver is `scripts/gen_rules.py`, an OpenAI-compatible client with `deepseek`,
`openai` and `local` providers and the subcommands `check`, `plan`, `run`, `runc`, `repair` and
`report`.

### What generation measured

All arms below are paired on the same tasks.

| condition | verified | CNY per call |
|---|---|---|
| thinking off | 110 of 1,000 (11%) | 0.0107 |
| blind retry on failures | 2 of 40 (5%) | 0.0107 |
| repair with traceback | 3 of 40 (7.5%) | 0.015 |
| effort low | 17 of 25 (68%) | 0.117 |
| effort max, 64k cap | 3 of 5 | 0.259 |
| Qwen3.8-27B local, Q3_K_M | 2 of 5 | free, 4.4 minutes a call |

The effort dial is not monotone. Low effort beats maximum on yield, price and speed at once.

These models fail by not terminating rather than by being wrong. Every call that finished thinking
produced a verified program, and two of five max-effort calls stopped exactly on the cap having
emitted no answer tokens. The same pattern appears in a second model family (Qwen at a 20k cap) and
was reported independently for that model within days of its release.

Execution against demonstrations is a 94% precise proxy for capturing the real rule. Of all 128
verified programs, 120 reproduce the held-out test outputs they never saw, against 93 of the first
100. That is what licenses a proof from demonstrations alone.

Failures are bimodal. Only 63 of 483 clean failures matched even one demonstration pair, so there is
no near-miss tail to repair. The 4B solver shows the same property at a different scale and task.

Feeding a program its own traceback beat a blind retry 3 to 2 out of 40, which is noise.

### Cost mechanics

DeepSeek's prefix cache is server-side and keyed on content, so parallelism does not lose it. All 970
calls under six workers hit 256 cached tokens each. The instruction block comes before the task grids
so the prefix stays stable, and `runc` fires one warmup call before the pool starts so workers do not
race a cold cache. Caching is worth about 9%, and output is about 92% of the bill.

The rates were checked against a real bill to within 1.5%. At peak, V4 Flash costs $0.014 per million
cache-hit input tokens, $0.44 per million cache-miss input tokens and $1.32 per million output tokens.
Off-peak is exactly half and runs from 04:00 to 06:00 and from 10:00 to 01:00 UTC. A CNY balance bills
at roughly the market rate, about 7.15 CNY per USD of list price.

## Closed directions

**Rule count.** Phase 0 trained on 100 composed ARC-1 rules against 402 at equal 8,000-example
budgets. The arms tied 12 of 12 on ARC-AGI-2. This is recorded as an inconclusive null rather than a
refutation, since twelve ties mean the gate could not detect an effect in either direction.

**Greedy fallback.** Three draws scored 30.56, 29.72 and 29.44, a mean of 29.91 against the 29.90
control, under the noise floor. Its first appearance was bundled with two other changes in one
submission, so this was the first time the mechanism was measured alone.

**Rule kind.** The shape gate matched 100 ARC-2 generators against 100 ARC-1 rules at equal pool depth.
It found 0 tasks better, 1 worse and 11 of 12 tied. Rule count and rule kind are both null.

**Training duration.** The same corpus ran from 8k to 96k steps. Validation loss bottoms at 16k and
rises for sixty thousand more while training loss falls to 0.0034, and the arms tie 12 of 12 at every
checkpoint with no grokking signature. The three training-side gates share one twelve-task instrument,
so they are one null measured three ways rather than three confirmations.

**DFS cache crop (v7.7).** The deployed decoder shares one KV cache across sibling branches without
truncation. Cropping is provably correct, bit-identical to a deep copy, and finds 27% more candidates
offline. Three submissions scored 29.72, 27.36 and 29.03, a mean of 28.70. Correctness costs wall
clock.

**Decode budget.** Decoding the same ten tasks at 600, 300 and 150 seconds puts the truth in the pool
6, 4 and 1 times. The throughput bought back cancels almost exactly, so the inherited 540 seconds sits
near a local optimum.

**Fair view scheduling (v7.7b).** This divides the remaining task time evenly over the four view
batches. It scored 29.44, 27.78 and 28.61, a mean of 28.61 with t = -2.09 and p = 0.09, all three
below the mean.

**A fresh build of v7.2.** The rebuild control scored 31.39, which rules out the build script as the
reason the new arms landed below the mean.

**Local generation.** This is worse on accuracy and not fixable on speed. Qwen3.8-27B at Q3_K_M runs
fully on the GPU at 47.7 tokens a second. Turning on the model's own MTP layers for self-speculative
decoding took it to 6.96 tokens a second, a 6.3-fold loss, because 13.6 GiB of resident weights leave
258 MiB and the draft context has nowhere to live. Speculation trades memory for bandwidth, and at this
capacity there is none to trade.

## Open questions

**Buying more generators.** At low effort the remaining failed tasks would yield roughly 588 more
generators for about 101 CNY, taking the corpus to about 715. This was not bought. The shape gate
found that rule kind does not move the base at this scale, so the corpus is a contribution in itself
rather than a route to a better base.

**FAAST** (arXiv 2605.04651) computes adaptation in closed form by pseudoinverse, W* = K†V, in one
forward pass with no backpropagation, roughly 15 times cheaper than gradient adaptation. It targets
per-task test-time training, which is the binding cost here. Three things count against it. Its own
limitations section names compositional reasoning as the weak spot, which is ARC by definition. Its
language-modelling results are worse than backpropagation (15.35 against 13.60 perplexity) and only
match with an extra readout component. And it learns only a linear map from frozen representations.
Its precondition can be tested without implementing it, by checking whether ARC transformations are
linearly decodable from the base's hidden states.

## Where things stand, 5 October

The paper, the writeup and both covers are current and build clean. On 5 October the paper's prose was
brought back to the standard of the June sweep, with no colons, semicolons or dashes in the body and
every conditional recast as a plain statement. The HTML copy of the paper is still an August snapshot with its
headline numbers updated. Regenerating it from the source with make4ht stalled in testing, so it is
either rebuilt properly or retired, and that is Ryan's call.

Every commit's email was rewritten locally from the personal address to
`198507061+rc397@users.noreply.github.com`, and this repository's `user.email` is now that address.
GitHub keeps the old history until Ryan force-pushes, and nobody should run `git pull` before then,
because the old commits would merge back in. Daily replicate resubmissions through
`scripts/rig_daily_replicate.py` are not running. This environment's safety check refuses background
processes on the rig and force pushes, so both need Ryan.

Four things remain before the deadlines, with Kaggle submissions due 2 November and papers 8 November.
The force push comes first, then making the repo public. The PDF needs a public URL. The Kaggle writeup
has to be filed with the triptych as its cover. And the two final submissions have to be chosen,
55261090 at 32.64 and one of the 31.39s.

## Environment notes

GPU work runs on a remote Windows box called the rig (`D:\arc`, an RTX 5070 Ti 16 GB, Blackwell
sm_120, which needs CUDA 12.8 or later). The laptop edits and copies files over and is closed often, so
long jobs must survive a disconnect.

Each of these facts cost a day.

- The remote shell is PowerShell, and scripts go over as base64 UTF-16LE.
- Bare `python` over SSH hits a WindowsApps stub that silently produces nothing. The real interpreter
  has to be resolved under `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13*`.
- Only a detached python process survives the SSH session. A `Start-Process` of anything else dies
  with the connection, and llama-server was seen loading, binding its port and vanishing when the ssh
  call returned.
- `Get-Process python` returns nothing while jobs are healthy, because the process name is
  `python3.13`.
- A power cut on 18 August 2026 killed every running job, so long runs checkpoint and resume from disk.
- In early September a `KAGGLE_API_TOKEN` environment variable appeared on the rig holding a token
  that differs from `~/.kaggle/access_token`. Kaggle 2.x prefers the variable, so every submission
  script failed with "Authentication required". `scripts/rig_submit.py` now drops the variable when
  the file exists, and the variable itself was left alone.
- The rig's GPU is usually busy with Ryan's other projects, which are outside this repo and must not
  be touched. Check the process list before launching anything and wait if the card is busy.
- Kaggle needs `machine_shape` set to `NvidiaL4`, and its absence is silent. The model source must read
  `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` with a capital T, and a wrong value lands in
  `invalidModelSources` while the push still reports success. A submission also needs a
  `kernel_version`, and 1 is not a safe default because Kaggle keeps submittable output only for recent
  versions.

## Measurement pitfalls this project hit

- **Inverted symptoms.** Empty model output looked like incapacity and was a token cap landing
  mid-thought. The fix moved cost twentyfold in the same direction as correctness.
- **Small-sample overstatement.** A bimodality figure of 3.7% from 27 failures became 13% at 483, and a
  repair rate of 80% at n = 5 became 68% at n = 25.
- **Mean against median.** A cost projection built on the median output length understated the bill by
  58%, because a few programs ran twelve times long. Cost scales with the mean.
- **Unprotected `exec`** on model-written code wedged an interpreter for 20 hours.
- **Unstable API shapes.** `balance_infos` carries one entry per currency in no fixed order, and
  reading index 0 returned an empty USD entry that would have stopped a fully funded run.
- **Confounded levers.** A mechanism submitted once alongside two other changes is not a measured
  mechanism, and isolating one cost three days of a one-a-day submission quota.
- **Displayed numbers are not measurements.** A public notebook showing 33.89 scored 28.06 when forked
  unmodified, which is why unmodified controls run first.
