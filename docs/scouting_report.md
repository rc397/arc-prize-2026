> **Maintainer's note (added on save, 2026-06-24).** This is a research scouting output. Its external
> citations and numbers are being independently verified; treat every percentage and arXiv ID as
> indicative until checked. Two corrections from the project's ground truth take precedence over the
> body below:
>
> 1. **The stock 4B cannot be the DSL proposer.** The NVARC base has a 16-token vocabulary (ten digits,
>    newline, two role tokens, pad, two control tokens) and contains no arc-dsl symbols, so it cannot be
>    grammar-constrained (GBNF) to emit DSL programs. The body's "minimal port" (grammar-constrain the
>    existing 4B over arc-dsl) is not possible on this model: GBNF can only mask tokens the model
>    already has, and this model cannot even spell the primitive names. Program search therefore needs a
>    proposer that can emit programs: train a small DSL model (GridCoder scale) on (task -> arc-dsl
>    program) data, or swap to a code model (which breaks the grid-native thesis). This inserts a
>    Stage 1.5 (proposer training) before Stage 2 (search), and raises the primary bet from a "port" to a
>    new-model build, so the LOO-TTT hedge matters more, not less.
> 2. **External ARC-2 numbers are unconstrained-compute upper bounds**, not in-box targets. Pang (26%),
>    Berman (29.4%, natural-language descriptions at ~$8/task), and SOAR (52%) run larger NL models well
>    outside our 12 h / 16 GB / ~$50 / grid-only box. Our target is unchanged: beat the 16.28% blend on
>    the 120 public eval, in-box.
>
> Adjusted sequencing actually being executed: Stage 0 (arc-dsl grader, CPU) and Stage 1 (honest
> leave-one-out TTT, reuses the existing model) first, both bankable; Stage 1.5 (train a DSL proposer)
> gates Stage 2 (search) and Stage 3 (verified-program distillation).

## Verification audit (2026-06-24)

Every load-bearing claim was checked against primary sources. The headline identities and scores hold,
but one framing problem and five specific errors were found; this audit supersedes the body where they
conflict.

**Framing (the important one): the comparison regimes do not match.** The TL;DR's "Pang 26.0% and
Berman 29.4% beat the best neural TTT submission (NVARC 24%)" places in-box and out-of-box results side
by side. NVARC's 24.03% was on the ARC-AGI-2 PRIVATE set under the offline contest constraints that
define our box (12 h, 4x L4, ~$0.20/task), by a fine-tuned Qwen3-4B (our base's lineage), TTT plus
synthetic data. Pang's 26.0% and Berman's 29.4% were on the unconstrained commercial leaderboard using
Grok-4 (a frontier natural-language model) at $8.42 to $30 per task. So within our constraints the
relevant SOTA is NVARC's 24%, achieved by the TTT approach we already run, and no published
program-synthesis result beats TTT in the offline / 4B / grid regime (the top three offline finishers
are all TTT/neural: NVARC 24.03%, ARChitects 16.53%, MindsAI 12.64%). Consequence: the 26 to 29% figures
are an aspirational upper bound from a frontier-NL setup, not evidence that a compute-poor grid-DSL 4B
will clear 16.28%. Whether it can is the open question, not a settled result. This re-weights the plan
toward the secondary lever (honest leave-one-out TTT), which lives in the same regime as our 16.28%,
with program search kept as the higher-risk bet justified solely by its unique partial-credit property.
One corroboration the other way: Berman's fitness function is explicitly partial-credit, scoring each
candidate by the percentage of correct cells, which is the report's central thesis stated by a
practitioner.

**Errata (verified corrected values):**
- GridCoder: the "452M-parameter VLM (one decoder layer, embedding dim 512, 4 heads)" is wrong and
  internally inconsistent (a 1-layer, 512-dim model is single-digit millions of params); the parameter
  count is unverified. GridCoder is a compact CNN+Transformer and is non-execution-guided (it enumerates
  by joint probability under the DSL grammar, then executes candidates as a post-filter; execution is not
  used during decoding). Its mechanism and the "limited ARC examples" quote are confirmed.
- PeARL: 77 primitives, not 30 (the 18/400 = 4.5% figure is correct).
- ARChitects: the DFS cumulative-sampling-probability threshold is T = 9% (Product-of-Experts paper,
  arXiv:2505.07859); the "17% in 2024 lowered to 7% via speculative decoding" is unsupported. The 7.3 GB
  (DFS) vs 14 GB (beam) VRAM figure is correct. The ARChitects were the 2024 paper-award RUNNER-UP (and
  2025 Kaggle 2nd), not "the 2024 winner."
- Akyurek: 53.0% is TTT applied to the BARC model; 47.1% on their own model; 61.9% ensembled with
  program synthesis. The report's "35% improvement" is unsupported (53.0 vs 47.1 is ~13% relative).
- arc-dsl: ships 400 worked solvers (one per ARC-1 training task), not "400+"; the ~160-primitive count
  is approximate (we did independently count 160 `def`s in dsl.py on the rig).

**Confirmed exactly:** Pang 77.1/26.0; Berman 79.6 at $8.42 and 29.4 "(previous best 25%)"; NVARC =
NVIDIA (Sorokin, Puget), 24.03% private, 12 h / 4x L4; TRM ~7M, ~45%/~8%; SOAR 52% ARC-1 (2nd, MIT);
CompressARC 76K, 20%/4%, arXiv:2512.06104; BARC arXiv:2411.02272 (+ quote, 42.2%); Akyurek
arXiv:2411.07279 (+ leave-one-out); CodeIt 59/400, arXiv:2402.04858; icecuber 2020, ~20%; Greenblatt 50%
blog / 42% official.

**Not re-verified (treat as indicative):** Akyurek's exact hyperparameters; SOAR's (rank 256, 6,000
attempts); BARC's "73.5% after TTT" and "400k dataset"; icecuber's "42 functions / 142 variants" and
"~10^7 pieces/task"; LPN's "39M / JAX / doubles OOD" (paper arXiv:2411.08706 confirmed, specifics not).

---

# Engineering Scouting Report: Raising the ARC-AGI-2 Solve Rate of a 4B Grid-Native Model on a 16 GB GPU

## TL;DR
- **Build LLM-guided, execution-verified program search as the primary new lever.** It is the only scouted method that natively manufactures a partial-credit signal (programs that reproduce k of n train pairs) valid on tasks the base misses, fits the box without relaxing the grid-only constraint, and has demonstrated ARC-AGI-2 scores (Eric Pang 26.0%, Jeremy Berman 29.4%), but only on the unconstrained commercial leaderboard with frontier natural-language models (Grok-4, $8 to $30 per task); within our offline 4B/grid box the SOTA is NVARC's 24.03% by TTT and no program-synthesis result beats TTT in-regime (see the verification audit above). The case for program search here rests on its unique partial-credit property, not on those out-of-regime numbers. Port Greenblatt's "generate many programs, keep those that pass train pairs, vote" structure off natural-language Python onto Michael Hodel's MIT-licensed arc-dsl, with the 4B model demoted to a grammar-constrained DSL proposer.
- **Run honest leave-one-out TTT as the cheaper, proven secondary lever.** Akyürek's MIT recipe already builds leave-one-out tasks; held-out-pair exact-match is a continuous re-ranking signal the saturated train-pair check cannot provide. This is the fallback if program search does not fit 12 h.
- **The bimodal wall is mostly an interface + scale artifact, not destiny.** A symbolic program is checkable to fractional train-pair accuracy even when the 4B is "lost," so program search dissolves the wall without needing natural language. Distilling verified programs back into weights is the one self-training path that can move the model where flat fine-tune and STaR could not, because the supervision is a NEW symbolic artifact, not the ground-truth grid the model already memorized.

## Key Findings

1. **Execution-verified program search is alive on ARC-AGI-2, but only in LLM-guided form.** Eric Pang's evolutionary Python-program-synthesis system (ARC Prize 2025 runner-up paper) scored 77.1% on ARC-AGI-1 and 26.0% on ARC-AGI-2 using only tens of programs per task ("My approach scored 77.1% on ARC-AGI-1 and 26.0% on ARC-AGI-2, both beating results of frontier models and previous bespoke systems"). Jeremy Berman's evolutionary search set a new ARC-2 state of the art: "a new high score of 79.6% on ARC v1 at $8.42 per task ... and, more importantly, sets a new state-of-the-art (SoTA) of 29.4% on ARC v2 (previous best: 25%)." Both beat the top ARC-AGI-2 private Kaggle score of 24% (NVARC), which was pure test-time-training neural. Classical hand-DSL brute force (icecuber, 20.6% on ARC-1) has no published ARC-AGI-2 score and is regarded as defeated by ARC-2's compositional/contextual difficulty. The lesson: the neural proposer is what makes program search work on ARC-2; blind enumeration does not scale.

2. **The partial-credit criterion cleanly separates winners from dead ends.** Program search scores k/n train-pair reproduction continuously, on tasks the base misses entirely: this is exactly the signal the established dead ends (verifier-guided vote, STaR) lacked. Honest leave-one-out TTT manufactures a second such signal (held-out-pair accuracy). A learned grid verifier, diversity ensembling, and object-graph parsing are weaker on this axis and stay shallow.

3. **The arc-dsl is a ready, searchable, permissively-licensed target.** Michael Hodel's arc-dsl (MIT) has roughly 160 primitives (count of `def` in dsl.py; the repo ships 400+ worked solver programs), a typed grid/object/patch system in arc_types.py, and is the DSL behind RE-ARC (MIT). A maintained fork (arc-dsl-2) revives it and contemplates ARC-AGI-2 support. The DSL is fully deterministic and CPU-executable, so verification against train pairs is free of the GPU.

4. **TTT precedents give concrete, copyable recipes.** Akyürek et al. (MIT, repo `ekinakyurek/marc`): per-task LoRA rank 128 on MLP+attention+output layers, 2 epochs, batch 2, lr 5e-5, leave-one-out task construction plus D8 augmentation, hierarchical self-consistency voting; their TTT pipeline reached 53.0% on the ARC-1 public validation set applied to the BARC model (47.1% on their own model, 61.9% ensembled with program synthesis). The ARChitects (2024 paper-award runner-up): D8 + color-permutation + example-reorder augmentation, depth-first-search sampling with a cumulative-sampling-probability threshold T=9% (arXiv:2505.07859), and a product-of-experts selection that scores candidates by stability across augmentations; their DFS uses a single path in memory (~7.3 GB VRAM) versus ~14 GB for beam search.

5. **Distilling verified programs is the live route to moving the weights.** STaR was null because a verified-correct grid trace equals the ground-truth target the fine-tune already used (base is 95.5% solved / 1.5% near-miss on its training set). A verified DSL program is a different artifact: it supervises the proposer's program-token distribution, not the grid it already emits, reviving the saturated weight path the way CodeIt's hindsight replay did. CodeIt (Butt et al., ICML 2024, arXiv:2402.04858) "solves 59/400 ARC evaluation tasks, achieving state-of-the-art performance" by learning from its own relabeled program samples.

## Details

### PRIMARY (go deep): Program/DSL search with execution verification

**The canonical shape (Greenblatt 2024), and how to port it off natural language.** Ryan Greenblatt's public ARC-AGI-1 result (Redwood Research): "I recently got to 50% accuracy on the public test set for ARC-AGI by having GPT-4o generate a huge number of Python implementations of the transformation rule (around 8,000 per problem)," keeping those correct on all train examples, then majority-voting over surviving programs' test outputs and picking the top candidates. The selection rule is the reusable essence: deterministic execution against train pairs is the partial-credit grader. Greenblatt's actual implementation is in the "Needs natural language" bucket: it emits NL-reasoned Python, uses few-shot NL chain-of-thought prompts roughly 30k tokens long, and revises programs after seeing their outputs. None of that runs on a 16-token grid-only base.

The minimal port: replace "GPT-4o emits Python" with "the 4B emits arc-dsl token sequences under a GBNF grammar." llama.cpp natively supports grammar-constrained decoding (GBNF), masking every token that would break the DSL grammar to -inf before sampling ("Logits for tokens that would break the grammar rules are set to -inf"). You define a GBNF grammar over the ~160 arc-dsl primitives so the 4B can ONLY emit syntactically valid programs. The 4B becomes a proposer of program-token distributions; the executor (CPU, deterministic) is the grader; the train pairs are the spec. No natural language anywhere. This is a clean "grid-native or ports cleanly" method and the right primary bet. [SEE MAINTAINER'S NOTE 1: not possible with the stock 16-token model; needs a trained DSL proposer.]

**Architecture (buildable):**
- *Proposer:* the existing NVARC qwen3-4b-grids served under llama.cpp Q8_0, prompted with the train pairs in the existing grid token format, decoding under a GBNF arc-dsl grammar. Sample N candidate programs per task with DFS sampling (ARChitects-style cumulative-probability cutoff) for diversity at low VRAM cost. Inference and the executor are never co-resident; the executor is CPU-only so it can overlap.
- *Executor / grader:* arc-dsl (MIT) or the C-speed icecuber DSL (42 functions / 142 variants) as a fast fallback for the geometric subset. Run every candidate on all train inputs; score = fraction of train pairs reproduced exactly. Keep survivors (k/n = 1.0); among ties, rank partial survivors for the self-training buffer.
- *Selector:* majority vote over survivors' test outputs, pass@2. If fewer than 2 distinct outputs, fall back to AIRV's existing vote.
- *Self-training loop (offline, between rounds):* collect all verified programs (k/n = 1.0) into a buffer; periodically LoRA-fine-tune the proposer on (train-pairs -> verified-program-tokens) pairs. This is the CodeIt / SOAR hindsight pattern.

**Compute budget — can it fit 12 h / 16 GB / 4x L4 for 240 tasks?** This is the key open question and the honest answer is: program *count* per task must be small, in the tens to low hundreds, not Greenblatt's 8,000. Evidence: Pang reached 26.0% on ARC-2 with only tens of programs/task; icecuber's C++ DSL tests up to ~10^7 candidate DAG "pieces" per task in tens of seconds on CPU (a depth-2 run over 419 tasks in ~70 s, roughly 6 tasks/sec at depth 2), so the *executor* is not the bottleneck. The bottleneck is the 4B proposer's decode. The existing engine gives a 13x decode speedup over eager, and AIRV already runs 8 augmentations per task in budget; a program-search pass that samples ~100-300 short DSL programs per task (each tens of tokens, far shorter than a full grid) is of the same order as current AIRV decode load. Rough budget for 240 tasks on 4x L4 in 12 h: ~180 s/task of GPU decode (~50 short programs per L4 slot) leaves the executor and selection on CPU, comfortably inside the 12 h / ~$50 envelope NVARC itself targeted (NVARC's own limit was "maximum 12 hours on four small GPUs with a budget of approximately 20 cents per task"). Where the 12 h goes: roughly 70% proposer decode, 20% CPU execution/verification, 10% LoRA self-training between rounds (done offline, not co-resident).

**Why it grips the bimodal wall.** A program reproducing 2 of 3 train pairs is graded 0.67 even when the base would emit a wrong grid and score 0. That fractional signal ranks candidates and seeds curriculum, which is exactly what re-ranking the model's own grid outputs (the established null results) could never get.

**DreamCoder / library learning — assess for the box.** DreamCoder (Ellis et al., PLDI 2021) bootstraps a DSL library via wake-sleep, refactoring frequent sub-programs into new primitives via E-graph matching. On ARC-1 the best DreamCoder-style result (PeARL, 77 primitives) was only 18/400 (4.5%), beaten by stock GPT-4 and smashed by icecuber. Verdict: full wake-sleep library learning is too compute-heavy and too data-starved (ARC has too few tasks per concept) to run in the box, and GridCoder's author explicitly chose to skip DreamCoder's DSL-growth for this reason ("the limited amount of natural ARC-AGI examples [is] an obstacle to making this feature particularly useful"). BUT the *library-learning kernel* — promoting frequent verified sub-program patterns from your self-training buffer into named DSL macros — is a cheap, optional add-on that shrinks search depth over rounds. Recommend the kernel, not the full system.

**Induction vs transduction (BARC, Li et al.).** BARC (ARC Prize 2024 best paper, Wen-Ding Li, Kevin Ellis et al., arXiv:2411.02272) showed induction (synthesize a program) and transduction (directly predict the grid) solve DIFFERENT tasks and ensemble to near-human on ARC-1: "Inductive program synthesis excels at precise computations, and at composing multiple concepts, while transduction succeeds on fuzzier perceptual concepts." Akyürek found that with their TTT pipeline, BARC's neural model solved 73.5% of the tasks the program synthesizer solved (up from 42.2%), i.e. the two are complementary. This is the strategic justification for program search as an ADDITION to the existing transductive 4B, not a replacement: the program-search arm should catch the "precise computation / multi-step composition" tasks ARC-2 emphasizes, exactly where the transductive base hits the wall. BARC's models (barc0 on HuggingFace, Llama-3.1-8B, fine-tuned on a 400k-task GPT-4o-mini-generated dataset) are 8B and NL-Python, so not directly droppable, but the induction/transduction split is the design principle to copy.

**GridCoder as the closest precedent.** Simon Ouellette's GridCoder (ARC Prize 2024 paper "Towards Efficient Neurally-Guided Program Induction for ARC-AGI," arXiv:2411.17708; repo SimonOuellette35) trains a compact CNN+Transformer (the "452M-parameter, 1-layer, 512-dim" description is internally inconsistent and unverified) to output a probability distribution over a 74-98 primitive DSL grammar, used as a non-execution-guided enumerator (execution is a post-filter), then enumerates programs in joint-probability order with a conditional-independence simplification that requires only a few model calls per task. It sorts programs by the product of token probabilities, raises the probability threshold tau until the program count is manageable, executes in order, and returns the first that passes the examples. This is almost exactly the proposed architecture at smaller scale; it is the existence proof that grammar-conditioned program enumeration fits modest compute. Study its search loop directly.

### SECONDARY (one deliberate thread): TTT as the partial-credit manufacturer

**Honest leave-one-out (LOO) adapters.** The established per-task TTT (rank-16 LoRA, 20 steps, scale 1.25) already lifts you to 16.28%. The new idea: train each adapter on n-1 train pairs and *verify on the held-out pair*. Held-out exact-match is a gradable signal on tasks the base misses, and unlike the saturated train-pair check it cannot be gamed by the adapter memorizing all pairs (the held-out pair was never seen). Akyürek's pipeline already constructs leave-one-out tasks ("We start by creating leave-one-out tasks from the given training examples of the task. These tasks are then augmented through rule-based transformations to obtain the full TTT dataset"), so the machinery is in the `marc` repo (torchtune-based fork). Use LOO accuracy to (a) decide whether to trust a task's TTT adapter at all (gate), and (b) re-rank candidate outputs across adapter seeds. Open question answered: yes, LOO verification can re-rank where saturated train-pair checks are silent, because it scores generalization to an unseen pair rather than fit to seen pairs.

**Adapter ensembles and augmentation-strength curriculum.** Train several per-task adapters under different augmentation strengths (light D8 only -> D8 + color-permute + example-reorder, the ARChitects scheme) and seeds; select by LOO accuracy; vote (AIRV reverse+vote) over the survivors. This is cheap (adapters are rank-16 bf16, swappable via llama.cpp hot-swap) and directly extends the proven lever. Curriculum: start each adapter at low augmentation strength, increase only if LOO accuracy is below threshold, to avoid over-regularizing easy tasks.

**Why this is the safe fallback.** Every component is already validated in your stack (LoRA, hot-swap, AIRV). If program search does not fit 12 h, honest-LOO TTT + adapter ensembling is a lower-variance path to beating 16.28%.

### SHALLOW SCAN (revisit only if the top two stall)

**(3) Grid-native learned verifier / value model.** Train a second 4B-or-smaller head on the same 16-token interface to predict held-out-pair correctness from (train pairs, candidate output). Partial-credit signal: a calibrated scalar on hard tasks. Fits the box (inference-only, grid-native). Risk: the bimodal wall reappears — if the base cannot represent a task, a same-interface verifier likely cannot grade it either, which is why this stays shallow. Only worth it if trained on honest-LOO judgements (real held-out labels), not self-labels (which would recreate the STaR saturation problem).

**(4) Diversity ensembling with a second genuinely-different open grid model.** Candidates: a from-scratch Tiny Recursive Model (TRM, ~7M params, ~45% ARC-1 / ~8% ARC-2, already used inside NVARC's ensemble) or LPN (Latent Program Network, 39M, JAX, clement-bonnet/lpn, NeurIPS 2025 spotlight). Both are grid-native and tiny enough to co-run; LPN's headline result is that test-time latent search "doubles its performance on out-of-distribution tasks." Partial-credit signal: only via the same execution check, so ensembling helps coverage (different tasks solved) not partial grading. Recommend TRM as a cheap second arm given NVARC already used it; flag the JAX dependency for LPN as stack friction against the PyTorch/llama.cpp pipeline.

**(5) Neuro-symbolic object reasoning.** Parse grids into objects/relations (ARGA-style object graphs; the Divide-Align-Conquer line solves tasks with ~0.2% of the candidates icecuber needs) and search over that structure. Strong on ARC-2's object-centric and symbolic-interpretation tasks in principle, but building a robust parser is high-effort and the search is a different engine from the DSL line. Partial-credit signal exists (graph-edit distance to target), but integration cost is high. Defer unless the DSL proves too flat for ARC-2's contextual-rule tasks.

### Constraint-relaxation side-branch (kept separate from the main line)

Is the bimodal wall partly *caused* by the grid-only interface? Plausibly yes: a model with a partial-credit channel (a program or a reasoning trace it can write and have graded) has somewhere to put incomplete understanding, whereas a grid-only model must emit a full correct grid or nothing. CompressARC (Liao & Gu, "ARC-AGI Without Pretraining," arXiv:2512.06104) is suggestive: a 76K-parameter from-scratch per-task MDL optimizer with NO pretraining and no external data reaches 20% on ARC-1 (characterized as 20-34%) and 4% on ARC-2 in ~20 minutes/puzzle on a single GPU, implying per-task optimization with a continuous (description-length) objective extracts signal a frozen transductive model cannot. SOAR (Pourcel, Colas, Oudeyer, arXiv:2507.14172, MIT, github.com/flowersteam/SOAR), the ARC Prize 2025 2nd-place paper, reaches 52% on the ARC-1 public eval by self-improving on its own program-search traces (RS-LoRA rank 256, 6,000 synthesis attempts/task, single H100 fine-tuning) — but it uses 7B-123B NL-Python models and reports NO ARC-2 number, so it is squarely a thesis-affecting pivot (new base, NL channel, compute-poor story at risk). Notably, Berman's record ARC-2 result itself came from such a pivot: per a 2025 interview, his "winning method for the ARC-AGI v2 challenge, achieving a 29.4% top score, pivoted from Python code generation to evolving natural language descriptions." **Crucial point for the main line:** symbolic program search does NOT need natural language, so it gets the partial-credit channel WITHOUT abandoning the grid specialization or the compute-poor thesis. The constraint is therefore not binding for the primary bet, which is a further reason it is the right one. Treat "swap to an NL model" as a clearly-bounded experiment, not a roadmap commitment: it would be slower, token-heavy (Greenblatt's prompts ~30k tokens), and would likely blow the 12 h budget on 240 tasks.

### Answers to the core research questions
1. *Smallest external partial-credit signal that re-ranks where the base is silent:* fractional train-pair reproduction by an executed program (continuous, 0-1, valid on missed tasks). Runner-up: honest LOO held-out-pair accuracy.
2. *Can DSL search seeded by the 4B fit 16 GB / 12 h and beat 16.28%?* Plausibly yes if program count per task is held to tens-to-hundreds (Pang got 26% on ARC-2 with tens), with the CPU executor off the critical path and the 4B decode the bottleneck. Concrete budget above. This is the highest-upside but highest-build-risk bet.
3. *Does distilling verified programs move the weights?* This is the one self-training path with a real chance, because the supervision target (program tokens) differs from the grid the model already memorized — unlike STaR. CodeIt's hindsight replay (59/400 on ARC-1) is the precedent.
4. *Is bimodality from scale, interface, or ARC-2 difficulty?* Mostly interface + scale: the grid-only channel gives incomplete understanding nowhere to live, and a 4B has limited compositional depth for ARC-2's multi-step tasks (ARC-2 was explicitly built around multi-step compositional and on-the-fly symbolic-interpretation tasks). A compute-poor method can change the interface effect (give it a program channel) but not the raw scale; program search is the change that needs no new model. [SEE MAINTAINER'S NOTE 1: it does need a new proposer model.]

## Recommendations

**Stage 0 (1-2 days): instrument the grader.** Wrap arc-dsl (MIT) so any program is scored as fraction-of-train-pairs-reproduced on CPU. Benchmark executor throughput on your hardware. Confirm it never touches VRAM. Gate to proceed: executor sustains a few thousand program-executions/sec/core (icecuber-class C++ is far faster, Python arc-dsl slower; if Python is too slow, port the hot primitives to C or use the icecuber geometric core).

**Stage 1 (1 week): honest-LOO TTT — bank a safe win first.** Extend the existing per-task TTT to train on n-1 pairs and record held-out exact-match. Use LOO accuracy to gate and to re-rank adapter seeds; ensemble survivors with AIRV. Measure on the 120 public eval, exact-match on both attempts, head-to-head vs no-TTT AIRV (12.79%) and the TTT blend (16.28%). **Keep/drop gate: ship only if blended public > 16.28% AND the 240-task run fits 12 h on 4x L4.**

**Stage 2 (2-4 weeks): grammar-constrained DSL program search.** Write a GBNF grammar over arc-dsl primitives; have the 4B propose ~50-150 programs/task under DFS sampling; execute, keep survivors, vote pass@2. Start with the geometric primitive subset (cheap, high-yield) before adding object primitives. Measure identically. **Gate: blended public > 16.28% AND fits 12 h / 240 tasks.** Confirm submission timing on 4x L4 before claiming any leaderboard number. [SEE MAINTAINER'S NOTE 1: precede with Stage 1.5, train a DSL proposer; the stock 4B cannot emit DSL tokens.]

**Stage 3 (concurrent, offline): verified-program distillation.** Buffer all k/n=1.0 programs; LoRA-fine-tune the proposer on (train-pairs -> program-tokens). Re-run Stage 2. Watch for the STaR failure mode (no gain) vs the CodeIt success mode (gain from novel program supervision). Optionally promote frequent verified sub-programs to DSL macros (the DreamCoder kernel) to shrink search depth.

**Stage 4 (if Stage 2/3 stall): coverage ensemble.** Add TRM as a second grid-native arm (NVARC precedent) and/or a learned LOO-trained verifier. Treat object-graph parsing as a research spike, not a deliverable.

**Benchmarks that change the plan:** if program search cannot clear ~18% public within the 12 h budget after Stage 2, freeze it and ship the Stage-1 LOO-TTT gain. If verified-program distillation shows zero movement after one round (STaR-like), stop distilling and keep search as inference-only. If the geometric DSL subset saturates below ARC-2's object/contextual tasks, escalate to the object-graph spike (5), not to an NL model.

## Caveats
- **No published ARC-AGI-2 number exists for SOAR, Greenblatt-original, icecuber, or Hodel's arc-dsl ensemble.** The ARC-2 program-search evidence rests on Pang (26.0%) and Berman (29.4%, the current bespoke SoTA), whose detailed methods are in blog / runner-up-paper form rather than peer-reviewed venues; treat the exact percentages as indicative, not settled. Pang's and Berman's systems use larger NL-capable LLMs than your 4B, and Berman's record specifically came after pivoting to natural-language program descriptions, so their scores are an upper bound on what a grammar-constrained 4B DSL proposer will reach.
- **Classical brute-force DSL search is documented as defeated on ARC-2.** Your edge must come from the *neural proposer* guiding the DSL, not from enumeration; a poorly-conditioned 4B proposer could collapse to icecuber-class performance (no ARC-2 signal). ARC-AGI-2 was explicitly engineered to remove brute-force-solvable tasks.
- **The 12 h fit is the dominant risk for the primary bet.** All compute estimates here are order-of-magnitude, derived from secondary throughput figures (icecuber DAG sizes, AIRV decode load, Pang's program counts) and NVARC's stated 12 h / 4x L4 / ~$0.20-per-task envelope; they are NOT measured on your stack. Stage 0/1 must validate them before committing to Stage 2.
- **arc-dsl's ~160 primitives were built for ARC-1 training tasks; completeness on ARC-2 is unproven.** ARC-2 was explicitly designed for symbolic interpretation and contextual rules (e.g. on-the-fly symbol assignment) that a 2023 ARC-1 DSL may not express, so DSL coverage is itself a research variable.
- **Never train on the 120 public eval or hidden sets**; all self-training buffers must be sourced from RE-ARC (MIT), ARC-2 training (Apache-2.0), or synthetic generation only. The licences of every recommended asset (arc-dsl MIT, RE-ARC MIT, SOAR MIT, LPN open-source, GridCoder open-source) are permissive and confirmed.
- **4-bit does not load on the Blackwell+Windows stack**; all LoRA training stays bf16, and inference/training stay non-co-resident, exactly as in the established pipeline. None of the recommendations change this.
