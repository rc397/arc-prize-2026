# Context: ARC Prize 2026, single-16GB-GPU entry

Background on this project as of 2026-10-02. Everything is measured unless stated otherwise.

## The project

Two linked entries in ARC Prize 2026. The **Paper Track** is the priority: rubric-judged, one
submission, the writeup is the deliverable (`paper/the-16gb-ceiling.tex`, 79 pages, a living
document that grows with each chapter of work, plus the 1,500-word `paper/writeup.md` that Kaggle judges read first). The **ARC-AGI-2 main track** is an offline Kaggle
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
| peak hidden score | 32.64, the maximum of twelve draws of one artifact |
| the control series | kernel version 339772104 run twelve times: mean 29.90, sd 1.34, range 28.47 to 32.64 |
| fresh-build control | 31.39 (31 Aug), so the build script is not a confound |
| best single-card build | 18.19 (v6.1) |
| noise floor | 1 task = 0.4167 points; two draws of the same artifact differ by more than that 83% of the time |
| levers measured | 40, none moved the hidden score |

Nothing has beaten the reproduction. The evidence points at the base model rather than the pipeline:
the remaining knobs are all smaller than the instrument can resolve. At sd 1.34 the leaderboard cannot
separate anything below roughly four points without three runs an arm (exact two-sample t, 80% power).
The last arms, each three draws unless noted, against 29.90: greedy fallback alone 29.91; DFS cache
crop 28.70; fair view scheduling 28.61 (29.44, 27.78, 28.61; t=-2.09, p=0.09); no-dead-AIF 29.86.
None is distinguishable from the control on its own, though all six draws of the two arms that move
decode time sit below it (a pattern noticed after the fact, not a pre-registered test).

## The generator corpus

For each of the 1,000 public ARC-AGI-2 *training* tasks, a model writes two functions: `solve(I)`
implementing the rule, and `gen_input(rng)` producing fresh inputs from the same family. A program
counts as verified only if `solve` reproduces every demonstration pair of its task under execution
(`scripts/pyexec.py`, process-isolated with a hard kill timeout). Nothing is judged by a model.

The training split is published for training, which distinguishes this from the evaluation-split
contamination trap documented in `docs/contamination-audit.md`. The result is a "RE-ARC for ARC-2",
which does not otherwise exist: RE-ARC covers ARC-1 rules, and the 402 compositions built earlier
here are pairs of those.

Current state: **128 verified generators, 127 with samplers**, released on 4 October as
`corpus/arc2_generators.jsonl` with both verification results per program, built by
`scripts/export_generators.py` from the raw `experiments/genrules_*.jsonl` on the rig. Total spend
about 13 CNY.

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
- **Execution against demonstrations is a 94% precise proxy** for capturing the real rule: 120 of all
  128 verified programs reproduce the held-out test outputs they never saw (93 of the first 100). That is what licenses
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
- **Greedy fallback**: 30.56 / 29.72 / 29.44, mean 29.91 against control 29.90. Under the noise
  floor. Its earlier appearance was bundled with two other changes in one submission, so this was
  the first time the mechanism was measured alone.
- **Rule kind (the shape gate)**: 100 ARC-2 generators against 100 ARC-1 rules at matched pool depth.
  0 better, 1 worse, 11 of 12 tied. Rule count and rule kind are both null.
- **Training duration**: 8k to 96k steps on the same corpus. Validation loss bottoms at 16k and rises
  for sixty thousand more while training loss falls to 0.0034; tied 12/12 at every checkpoint, no
  grokking signature. The three training-side gates share one twelve-task instrument, so they are one
  null measured three ways rather than three confirmations.
- **DFS cache crop (v7.7)**: the deployed decoder shares one KV cache across sibling branches without
  truncation. Cropping is provably correct, bit-identical to a deepcopy, and finds 27% more candidates
  offline. Submitted three times: 29.72, 27.36, 29.03, mean 28.70. Correctness costs wall clock.
- **Decode budget**: 600/300/150 s on the same ten tasks puts the truth in the pool 6/4/1 times; the
  throughput bought back cancels almost exactly, so the inherited 540 s sits near a local optimum.
- **Fair view scheduling (v7.7b)**: divide the remaining task time evenly over the four view batches.
  29.44, 27.78, 28.61, mean 28.61, t=-2.09, p=0.09. All three below the mean; with the crop, six of
  six draws below, reported as a pattern rather than a test.
- **A fresh build of v7.2 (rebuild control)**: 31.39. Rules out the build script as the reason the
  new arms landed below the mean.
- **Local generation**: worse on accuracy and not fixable on speed. Qwen3.8-27B at Q3_K_M is fully
  GPU-resident at 47.7 tok/s, but enabling the model's own MTP layers for self-speculative decoding
  took it to 6.96 tok/s, a 6.3-fold loss, because 13.6 GiB of resident weights leave 258 MiB and the
  draft context has nowhere to live. Speculation trades memory for bandwidth and there is none to
  trade at this capacity.

## Open questions

**Buying more generators.** At effort=low the remaining failed tasks would yield roughly 588 more
generators for about 101 CNY, taking the corpus to about 715. Not bought: the shape gate found rule
kind does not move the base at this scale, so the corpus is a contribution in itself rather than a
route to a better base.

**FAAST** (arXiv 2605.04651) computes adaptation in closed form, `W* = K†V` by pseudoinverse, in one
forward pass with no backprop, roughly 15x cheaper than gradient adaptation. It targets per-task TTT,
which is the binding cost here. Against it: its own limitations section names compositional reasoning
as the weak spot, which is ARC by definition; its language-modelling results are worse than backprop
(15.35 against 13.60 perplexity) and only match with an extra readout component; and it learns only a
linear map from frozen representations. Its precondition is testable without implementing it, by
checking whether ARC transformations are linearly decodable from the base's hidden states.

## Where things stand, 4 October

The paper, the 1,500-word writeup and both covers are current and build clean. On the public
leaderboard of 1 October we sit 61st of 2,310 at 32.64; the board is led by Tufa Labs at 83.06,
rabbithole at 79.72 and nvbanana at 75.42. The 128 generators are released in `corpus/`.

Every commit's email was rewritten locally from the personal address to
`198507061+rc397@users.noreply.github.com`, and this repository's `user.email` is now that address.
GitHub still holds the old history until Ryan force-pushes; do not `git pull` before then, or the
old commits merge back in. Daily replicate resubmissions (`scripts/rig_daily_replicate.py`) are not
running: this environment's safety check refuses background processes on the rig and force pushes,
so both need Ryan.

Open before the deadlines (Kaggle submissions 2 November, papers 8 November): the force push, then
making the repo public; a public URL for the PDF; filing the Kaggle writeup with the triptych as
the cover; and choosing the two final submissions (55261090 at 32.64 and a 31.39).

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
- A `KAGGLE_API_TOKEN` environment variable appeared on the rig in early September holding a token
  that differs from `~/.kaggle/access_token`; kaggle 2.x prefers the variable, so every submission
  script failed with "Authentication required". `scripts/rig_submit.py` now drops the variable when
  the file exists. The variable itself was left alone.
- The rig's GPU is usually occupied by Ryan's other projects, outside this repo. Do not touch them;
  check the process list before launching anything and wait if the card is busy.
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
