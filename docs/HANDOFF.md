# Handoff: ARC Prize 2026, single-16GB-GPU entry

Context for a model picking this up cold. Written 2026-08-20. Everything here is measured unless it
says otherwise.

## 1. What this is

Two linked entries in ARC Prize 2026:

- **Paper Track** (the priority). Rubric-judged, one submission, the writeup is the deliverable.
  `paper/the-16gb-ceiling.tex`, currently 77 pages, builds clean with `pdflatex` (run twice).
  It is a LIVING document: every chapter of work gets written into it, including negatives.
- **ARC-AGI-2 main track**. Offline Kaggle notebook, no internet. Required for eligibility.

The organising constraint is **one 16 GB GPU**. That is the paper's thesis, not an excuse.

Repo must be public by **1 October** for prize eligibility.

## 2. Scoreboard

| | |
|---|---|
| peak hidden score | 32.64 |
| current defended build | **v7.2 at 30.97** |
| banked control | mean 30.31, sd 1.34, n=10 (draw 11 in flight) |
| noise floor | 1 task = 0.4167 points |
| levers measured and closed | 33 |

**Nothing has beaten v7.2.** Five distinct builds return to exactly 30.56. Treat 30.97 as the thing to
defend and change one factor at a time. The score is set by the base model, not by pipeline knobs.

## 3. Pipeline

Base: NVARC `qwen3_4b_grids15_sft139` (Qwen3-4B, **16-token vocab**, ID-level ChatML prompts).
It is NOT Claude. Details in `DEVLOG.md` and `src/arc/model.py`.

Flow: augment (128 train rows = 2 transpose x 4 rot90 x 16 colour perms; 16 eval views)
-> per-task rank-256 LoRA TTT (alpha 32, rslora, 9 modules incl. `embed_tokens` and `lm_head`,
128 steps, lr 5e-5) -> `turbo_dfs` (floor 0.2) -> `score_kgmon`.

Three wall-clock deadlines: 12 h global, 1800 s/task, 540 s per DFS call. **TTT is the binding
cost.**

## 4. Environment

**Rig** (all GPU work): SSH host `rig`, fallback `rig-ts` over Tailscale.
Windows 11, RTX 5070 Ti 16 GB (Blackwell **sm_120**, needs CUDA 12.8+), project root `D:\arc`.
Laptop edits and scp's; it is closed regularly, so long jobs must survive disconnect.

Hard-won operational facts, each of which cost time:

- Remote shell is PowerShell. Send scripts base64 as UTF-16LE via
  `ssh rig powershell -EncodedCommand <b64>`, and print with `[Console]::WriteLine` to avoid CLIXML
  noise. Set `$env:PYTHONIOENCODING="utf-8"`.
- Bare `python` over SSH resolves to a non-functional WindowsApps stub that silently produces
  nothing. Resolve the real interpreter each time under
  `C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13*\python3.13.exe`.
- **Only detached PYTHON survives the SSH session.** A `Start-Process` of a non-python exe dies with
  the connection (observed: llama-server loaded, bound its port, vanished when the ssh call
  returned). Wrap servers in a python parent that owns them as a child.
- A detached launch makes the launching ssh call hang until timeout even though the job is fine.
  Launch, then poll the log in a separate call.
- `Get-Process python` returns NOTHING while jobs run healthily; the process name is `python3.13`.
  Use `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`, which also gives command lines.
- The rig never idle-sleeps, but Windows Update force-reboots it and a **power cut killed everything
  on 2026-08-18**. Any long run must checkpoint and resume from disk.
- Kaggle token at `~/.kaggle/access_token`, DeepSeek key at `~/.deepseek/api_key`. **Never print
  either.**

**Kaggle submission gotchas**, all of which have burned a day each:

- `machine_shape: "NvidiaL4"` is REQUIRED and its absence is silent; without it you may get a P100
  (sm_60) and the stack dies with "no kernel image available".
- `model_sources` must be `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, capital T.
  Wrong values land in `invalidModelSources` while the push still REPORTS SUCCESS.
- `competition_submit_code` needs `kernel_version`, and **version 1 is not a safe default**. Kaggle
  retains submittable output only for recent versions; `current_version_number` reported 0 while the
  live version was 3. `scripts/rig_submit.py` now walks upward until one is accepted.
- One submission per UTC day. `rig_submit.py list` checks before spending the slot.

## 5. The generator corpus (current work)

**What it is.** For each of the 1,000 public ARC-AGI-2 TRAINING tasks, a model writes `solve(I)` and
`gen_input(rng)`. A program is verified iff `solve` reproduces EVERY demonstration pair under
execution (`scripts/pyexec.py`, process-isolated, hard kill timeout). No model judges anything.

The training split is published for training, which is what makes this legitimate where the
evaluation split would not be. It is a "RE-ARC for ARC-2", which does not otherwise exist.

**State**: **128 verified generators, 127 with samplers**, in `D:\arc\experiments\genrules_*.jsonl`
(each line carries the full program in `src`). Cost so far about 13 CNY.

Driver: `scripts/gen_rules.py`. Commands `check` / `plan` / `run` / `runc` (concurrent) /
`repair` / `report`. Provider-agnostic OpenAI-compatible client (`deepseek`, `openai`, `local`).

**Measured facts about generation**, all paired on the same tasks:

| condition | verified | cost |
|---|---|---|
| thinking off | 110/1000 (11%) | 0.0107 CNY/call |
| blind retry on failures | 2/40 (5%) | 0.0107 |
| repair with traceback | 3/40 (7.5%) | 0.015 |
| **effort=low** | **17/25 (68%)** | **0.117** |
| effort=max, 64k cap | 3/5 | 0.259 |
| Qwen3.8-27B local Q3_K_M | 2/5 | free, 4.4 min/call |

- **The effort dial is not monotone.** Low beats max on yield, price and speed simultaneously.
- **These models fail by not terminating, not by being wrong.** Every call that finished thinking
  produced a verified program. Two of five max-effort calls stopped exactly on the cap with ZERO
  answer tokens. Replicated in a second model family (Qwen at 20k). Forcing termination is the
  intervention, not a limitation.
- **Execution-verification is 93% precise**: 93 of the first 100 verified programs reproduce the
  held-out TEST pair they never saw. This licenses the whole method.
- **Failures are bimodal**: 63 of 483 clean failures matched even one demo pair. No near-miss tail
  to repair. (A 30-task pilot said 1 of 27; the full-scale figure is the honest one.)
- Feeding a program its own traceback beats a blind retry by 3 vs 2 of 40, i.e. by nothing.

**Cache**: DeepSeek's prefix cache is server-side and content-keyed, so parallelism does NOT lose it.
Measured 970/970 calls hitting 256 cached tokens each under 6 workers. The instruction block is
emitted first specifically so this works, and `runc` fires one warmup call before the pool starts so
six workers do not race a cold cache. Caching is worth ~9%; output is ~92% of the bill.

**Pricing** (verified against a real bill to 1.5%): V4 Flash peak $0.014 cache-hit in, $0.44
cache-miss in, $1.32 out per 1M. Off-peak is exactly half and runs 04:00-06:00 and 10:00-01:00 UTC.
CNY balances bill at roughly market (~7.15 CNY per USD of list). `runc` gates every call to off-peak,
refuses to start without `ARC_GEN_CONFIRM=1`, and enforces a spend ceiling from measured tokens.

## 6. Closed directions (do not reopen without new evidence)

- **Phase 0, rule COUNT**: 100 composed ARC-1 rules vs 402, equal 8,000 examples. Identical, tied
  12/12 on ARC-AGI-2. Recorded as an inconclusive null, not a refutation: 12 ties means the gate
  could not detect an effect in either direction.
- **Greedy fallback**: 30.56 / 29.72 / 29.44, mean 29.91 vs control 30.31. Under the noise floor.
- **Local generation**: worse and unfixable. MTP self-speculative decoding took 47.7 -> 6.96 tok/s,
  a 6.3x LOSS, because 13.6 GiB of resident weights leave 258 MiB and the draft context has nowhere
  to live. The 16 GB constraint removes the optimisation that would compensate for it.
- 33 levers total in the paper's ledger, none moved the hidden score.

## 7. Open questions and next actions

**(a) The shape gate. Free, decides a 101 CNY spend.**
Phase 0 varied rule COUNT and found nothing. It never varied rule SHAPE: every rule we hold is
ARC-1 (RE-ARC or pairs composed), none is ARC-2-shaped. The corpus is, by construction.

`scripts/mint_genrules.py` builds pools by running each generator process-isolated, then reuses
`build_ft_corpus.mint()` unchanged.

**KNOWN CONFOUND, must be fixed before trusting the result**: ARC-1 pools carry ~40 distinct pairs
per rule, ours yield ~7 (samplers are valid but low-diversity; an earlier "9.5 usable draws per 10
seeds" measured validity, never whether the draws DIFFERED from each other). So the arms are not
matched on raw material and a loss could be pool poverty rather than shape. Fix: subsample the
ARC-1 pools to ~7 and retrain BOTH arms fresh. About 9 h on an idle rig, no money.

If shape moves the gate, 101 CNY buys the remaining 865 failures at effort=low (~588 more
generators, corpus -> ~715). If it ties, spend nothing and the paper gets a second clean negative.

**(b) FAAST linear probe. Free.**
FAAST (arXiv 2605.04651) computes adaptation in closed form, `W* = K†V` via pseudoinverse, one
forward pass, no backprop, ~15x cheaper than gradient adaptation. It targets exactly our binding
cost, per-task TTT. BUT its own limitations section names compositional reasoning as its weak spot,
which is ARC by definition; its language results are worse than backprop (15.35 vs 13.60 PPL) and
only match with an extra readout; and it can only learn a LINEAR map from frozen representations.

Do not implement it first. Test the precondition: are ARC transformations linearly decodable from
this base's hidden states? Extract intermediate states from demo pairs, fit `W* = K†V`, check
held-out prediction against chance. Hours, no training. Flat probe closes the direction for free.

**(c) Keep drawing the control.** Every replicate tightens sd 1.34, the instrument every comparison
in the paper is measured against.

## 8. Working rules

- **No em-dashes, no emojis.** Professional, natural tone. Minimal comments, and comments explain
  WHY not what.
- Commits: `user.name=rchen`, `user.email=198507061+rc397@users.noreply.github.com`, **no AI/Co-Authored-By trailers**.
- **No agents or subagent swarms** without explicit permission each time.
- **Never launch background monitors or watchers unprompted.**
- Verify, do not trust. This project has a documented PHANTOM VERDICT: a public notebook displaying
  33.89 scored 28.06 when forked unmodified. A number shown beside code is not a measurement of that
  code. Run the unmodified control first.
- Anthropic's terms prohibit using Claude outputs as training data (model distillation). DeepSeek is
  MIT and Qwen is Apache 2.0, which is why they generate the corpus. Claude may design experiments,
  build tooling and analyse results, but must not author artifacts that become training data.
- Report negatives plainly. Most of this project's value is a map of where not to spend.

## 9. Recurring failure modes to watch for in yourself

Every one of these happened here and cost real time:

- **An inverted symptom.** Empty model output read as incapacity; it was a token cap landing
  mid-thought. Check the cap before concluding the model cannot do the task.
- **Small-sample overstatement.** A 30-task pilot's bimodality figure was 3.7%; at full scale 13%.
  A 5-task arm said 80%; at n=25 it was 68%. State n, expect regression.
- **Mean vs median.** A cost projection built on median output tokens understated the bill by 58%
  because a few programs ran 12x long. Cost scales with the MEAN.
- **Unprotected `exec` on generated code** wedged an interpreter for 20 hours. Always use
  `pyexec.run_candidate`, which is process-isolated with a hard kill.
- **Unstable API shapes.** `balance_infos` carries one entry per currency in unstable order; reading
  index 0 got an empty USD entry that would have aborted a funded run.
- **Confounded levers.** A mechanism bundled with two others and submitted once is not a measured
  lever. Establishing that for the greedy fallback cost three days of a one-a-day quota.
