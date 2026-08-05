# Prior-art check: self-supervised TTT on the ARC test input grid

Date of search: 2026-08-05. Scope: does anyone make the unlabeled test input grid a
loss target inside per-task test-time adaptation for ARC, via reconstruction, masked-cell
modelling, denoising or compression?

Throughout, keep two things apart:

- **In-context use.** The test input grid appears in the decode prompt. Universal, in every
  system checked, not the proposal.
- **Loss-target use.** Gradients flow from an objective computed on the test input grid
  itself. This is the proposal.

---

## 1. ARC-specific: is the test input grid ever a loss target during test-time adaptation?

Verdict for the exact proposal: **ABSENT**. Verdict for close neighbours: **ADJACENT, and
one neighbour is an explicit rejection.**

### 1.1 Akyurek, Damani et al., "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning"

<https://arxiv.org/abs/2411.07279>, ICML 2025 (<https://proceedings.mlr.press/v267/akyurek25a.html>)

This is the most important entry, because the authors already wrote down the neighbouring
objective and named its lineage.

TTT data is built by leave-one-out over the demonstrations only. Verbatim construction:

> d_j^ICL = ({(x_k, y_k)}_{k≠j}, x_j, y_j)

Section 3.2 defines three loss variants:

1. test output only, `L_LM(y_test | x_1,y_1,...,x_K,y_K,x_test)`
2. all outputs, adding `sum_k L_LM(y_k | x_1,y_1,...,x_k)` (the variant they ship)
3. inputs and outputs, adding `sum_{k=1..K} L_LM(x_k | x_1,y_1,...,y_{k-1})`

Of variant 3 they say, verbatim:

> "This method, which requires learners to generate task *inputs* as well as outputs, is
> analogous to existing unsupervised TTT objectives (Sun et al., 2020)."

And in the Section 2.2 preliminaries, verbatim:

> "In previous work (e.g., Sun et al., 2020), D_TTT is typically constructed by applying an
> unsupervised objective (e.g., masked autoencoding) to the input x alone."

Two facts follow, and they point in opposite directions.

**Against novelty.** The idea of putting a generative loss on ARC input grids during TTT was
considered, implemented, tied explicitly to the Sun et al. lineage, and ablated. On BIG-Bench
Hard the inputs-and-outputs variant scored 55.9 against 57.8 for all-outputs and 54.4 for
test-output-only. It lost.

**For novelty.** The `x_k` in variant 3 are the **demonstration** inputs of the synthetic
leave-one-out tasks. The real unlabeled test input `x_test` is never a loss target in any of
their three variants, because `D_TTT` contains no pair whose target side is unknown. The
ablation was also reported on BIG-Bench Hard, not on ARC. The ARC ablation figure (Figure 5)
covers FT baseline 5, no transformations 13, direct I/O data 18, shared TTT 22, no
demonstration loss 26, full method 29; the inputs-and-outputs variant does not appear there.
Headline result is 53 percent on ARC public validation with an 8B model.

### 1.2 The ARChitects: an explicit, in-print rejection

2024 writeup: <https://da-fr.github.io/arc-prize-2024/the_architects.pdf>
2025 arXiv, "Product of Experts with LLMs": <https://arxiv.org/abs/2505.07859>
2025 report: <https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>

Verbatim, and this is the single strongest citation a reviewer will reach for:

> "For each task p=((x_i,y_i)_{i=1..k}, x̂), with solution s*_p, we computed gradients only on
> the outputs y_i for i>1 and s*_p. This approach ensures that **the model is never tasked with
> predicting an input grid**, and acknowledges that correctly predicting the first output grid
> is impossible without at least one example."

Their secondary (test-time) training is "time-constrained and focused solely on the evaluation
set, using a LoRA rank of 32 and running for 64 training steps with batch size 1". So the
pipeline that our own stack derives from does not merely omit an input-grid loss, it names the
omission as a design property. 71.6 percent (286.5/400) on ARC-AGI-1 public eval; 16.53 percent
on ARC-AGI-2 in 2025.

Their 2025 system does use masking, but on the wrong side of the pair: a masked-diffusion LM
whose "cross-entropy loss applied only to positions where a token had been masked" operates on
**output** grids during recursive refinement.

### 1.3 NVARC, MindsAI/Tufa, BARC

- **NVARC** (<https://github.com/1ytic/NVARC>), 1st place ARC Prize 2025 at 24.03 percent.
  Described by the ARC Prize 2025 report as a "synthetic-data-driven ensemble of an improved
  Architects-style test-time-trained model and TRM-based components". No self-supervised
  test-input objective in the repository or writeup.
- **MindsAI / Tufa Labs**, 3rd at 12.64 percent. TTFT on demonstration pairs plus AIRV.
- **BARC**, Li et al., "Combining Induction and Transduction for Abstract Reasoning"
  (<https://arxiv.org/abs/2411.02272>). Does **no** test-time training at all.

**Phantom prior art, flagged.** A widely indexed blog post
(<https://omseeth.github.io/blog/2025/MLLM_for_ARC/>) states, verbatim:

> "Following the approach of Sun et al. [45], this strategy introduces an additional
> self-supervised training step before each prediction. Specifically, the input is augmented
> (e.g., through rotations or color permutations), and the model is tasked with recovering the
> original version. If it fails, its weights are updated accordingly."

It attributes this to Li et al. 2024, that is, to BARC. **No such method appears in BARC.** The
BARC abstract and method contain no TTT and no self-supervised adaptation. This looks like a
secondary-source conflation of Sun et al. with the ARC literature, and search engines are now
recycling it into LLM-generated topic pages. Anyone doing this prior-art check quickly will hit
that sentence and wrongly conclude the proposal is published. It is not, at least not there.

### 1.4 ironbar / Omni-ARC: the closest ARC-specific relative, and it stops one step short

<https://ironbar.github.io/arc24/05_Solution_Summary/>
<https://ironbar.github.io/arc24/modeling/Iteration_22_learning_inputs_distribution/>

Omni-ARC is a multi-task model trained on, among others, an `inputs -> input` objective:
generate a new input grid given the task's other input grids. That is a generative model of the
per-task input distribution, which is most of the motivation behind the proposal. Iteration 22
measured it: 6.44 percent with the input-distribution objective against 5.71 percent baseline at
10k steps, weaker than task augmentation at 7.02 percent.

But it never reached test time. Verbatim from the solution summary:

> "Due to the limited submission time test-time fine-tuning was only applied to the canonical
> ARC task of predicting the test outputs."

So: an input-grid generative objective exists in the ARC literature, was measured to give a small
positive effect, and was used only in pretraining, never in per-task adaptation, and never on the
specific unlabeled test input. His 2025 work (<https://ironbar.github.io/arc25/05_Solution_Summary/>)
uses TTT with hindsight relabeling on the BARC induction model, 28.3 against 23.3 percent, which
is pseudo-labelling on generated programs, not self-supervision on the test input.

### 1.5 ARC Prize technical reports

ARC Prize 2024 (<https://arxiv.org/abs/2412.04604>) defines the term the field now uses:

> "TTT, in this context, involves fine-tuning a pretrained LLM on the demonstration pairs of each
> task instance seen at test time, effectively creating a different variant of the base model for
> each task."

ARC Prize 2025 (<https://arxiv.org/html/2601.10904v1>) contains no mention of self-supervised
objectives on unlabeled test inputs. The living survey "The ARC of Progress towards AGI"
(<https://arxiv.org/abs/2603.13372>) likewise lists no ARC system applying masked modelling,
reconstruction or denoising self-supervision to the actual test input grids at inference.

---

## 2. CompressARC (Liao and Gu): the closest relative overall

<https://arxiv.org/abs/2512.06104>, blog
<https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html>,
code <https://github.com/iliao2345/CompressARC>

**What it trains on.** All known grids of the single target puzzle, which explicitly **includes
the test input grid**, and excludes only the answer. Verbatim:

> "We jointly optimize θ, μ, and Σ to minimize the sum of cross-entropies over the known grids
> (5 of them,) ignoring the answer grid."

and

> "training happens only on a single sample: the target inference puzzle itself, with the final
> solution information removed."

**Objective.** A VAE-style description-length bound: KL(N(μ,Σ) || N(0,I)) as the latent code cost
plus cross-entropy reconstruction of the known grids. "A KL divergence penalty keeps N(μ,Σ) close
to N(0,1), as in a VAE." The answer is read out by sampling, with the modal answer across training
taken as the prediction.

**Setup.** A 76K-parameter hand-built equivariant network, randomly initialised per puzzle, no
pretraining, no use of the ARC training set, roughly 20 minutes per puzzle on an RTX 4070.

**Numbers.** 34.75 percent on the ARC-AGI-1 training set, 20 percent on the ARC-AGI-1 evaluation
set (pass@2). The ARC Prize 2025 technical report summarises it as "20-34% on ARC-AGI-1 and 4% on
ARC-AGI-2 without any pretraining or external data". Third place, ARC Prize 2025 paper awards.

**How the proposal differs.** Five ways, and they are real but they are all about staging and
substrate rather than about the core idea:

1. CompressARC has no pretrained model. The proposal adds an objective on top of a pretrained
   LLM's per-task LoRA.
2. In CompressARC the reconstruction objective **is** the solver. The answer is decoded from the
   same fitted model. In the proposal reconstruction is a warm-up whose only job is to hand a
   better-conditioned adapter to a separate supervised transformation objective.
3. CompressARC reconstructs demonstration inputs, demonstration outputs and the test input under
   one joint objective, so it never separates "grid fluency" from "the transformation". The
   proposal separates them deliberately.
4. Architecture is a hand-designed equivariant network with ARC symmetries baked in, not LoRA on
   a transformer over serialized grid tokens.
5. CompressARC is a full-system replacement with a published ARC-AGI-2 number of 4 percent. The
   proposal is an additive component to a pipeline currently near 24 percent.

The honest reading: CompressARC establishes that **fitting a model to reconstruct the test input
grid at test time is a published idea that works** at a modest level. It does not establish that
adding it as an auxiliary to LLM TTT has been tried.

---

## 3. The general TTT lineage, and the terminology collision

The collision is real, it is documented, and it is documented **by the paper that popularised
"TTT" for ARC**.

Canonical lineage, all of which train on the unlabeled **test instance** with a self-supervised
objective:

| Work | URL | Objective on the test instance |
|---|---|---|
| Sun, Wang, Liu, Miller, Efros, Hardt, ICML 2020 | <https://proceedings.mlr.press/v119/sun20b.html> | rotation prediction on the single test image |
| TENT, Wang et al., ICLR 2021 | <https://arxiv.org/abs/2006.10726> | prediction-entropy minimisation |
| TTT++, Liu et al., NeurIPS 2021 | <https://papers.nips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html> | contrastive plus feature-moment alignment |
| MEMO, Zhang et al., NeurIPS 2022 | <https://arxiv.org/abs/2110.09506> | marginal entropy over augmentations of one test input |
| TTT-MAE, Gandelsman et al., NeurIPS 2022 | <https://arxiv.org/abs/2209.07522> | masked autoencoding, 75 percent of patches, per test image |

Sun et al.'s framing, verbatim: "turning a single unlabeled test sample into a self-supervised
learning problem, on which the model parameters are updated before making a prediction."

TTT-MAE is the direct template. The proposal is, in one line, TTT-MAE with grid cells for patches
and a LoRA for the encoder.

**The collision.** In the ARC literature "TTT" means supervised fine-tuning on the demonstration
pairs, which are labelled data supplied inside the task. In the Sun lineage "TTT" means
self-supervised adaptation on the unlabeled test input. These are different procedures sharing a
name. Akyurek et al. inherited the name and flagged the difference themselves in Section 2.2,
quoted in 1.1 above. The ARC Prize 2024 report then froze the ARC-local meaning into the field's
vocabulary with its definition, quoted in 1.5.

**Has anyone applied the Sun-lineage idea to ARC or to few-shot program induction?** Not on ARC,
per sections 1 and 4. Outside ARC, the closest LLM instantiation is TLM, "Test-Time Learning for
Large Language Models" (<https://arxiv.org/abs/2505.20633>), which trains LoRA on unlabeled test
inputs by minimising input perplexity and shows that lower input perplexity lowers output
perplexity. It evaluates on domain, instruction and reasoning benchmarks, and does not touch ARC
or grid tasks. Also adjacent: ReLAT, reconstruction-guided latent reasoning at test time with
temporary LoRA adapters (<https://arxiv.org/abs/2606.06252>).

---

## 4. Masked-autoencoder and masked-cell modelling on ARC grids

This is the item where the answer is closest to "published", but on the wrong grids and at the
wrong stage.

### 4.1 ARC-AGI-2 Technical Report, Lemes de Oliveira et al.

<https://arxiv.org/abs/2603.06590> (Nov 2025), Section 6.6 "UL2 Masked Training", within
Section 6 "Training to Understand".

They run UL2-style denoising on ARC grid tokens. A subset of grid tokens in the **demonstration**
grids is replaced with mask tokens (`<extra_id_0> ... <extra_id_99>`, plus mode tokens
`<task_id_S>`, `<task_id_X>`, `<task_id_R>`) and the model reconstructs them token by token, under
three de-masking modes differing in noise ratio and masking pattern. They describe the motive as
discouraging "overfitting to autoregressive copying patterns" in favour of "latent rule discovery
and structural understanding", and they interleave denoising with standard ARC tasks on a
curriculum. Their contributions list says they use "multi-task learning to enable the model not
only to solve tasks but also to denoise them".

It works: UL2 denoising took them from 25.00 to 27.08 percent on the Kaggle semi-private set, an
absolute gain of 2.08 points, with an ablation section "Effect of Removing UL2 Training".

Two limits that preserve room for the proposal. First, this is **offline** training, not per-task
test-time training. Their own TTT stage is described as constructing "a small adaptation dataset
per task by iterating over demonstrations as pseudo-test pairs", the standard recipe. Second, the
masking is applied to demonstration grids, not to the unlabeled test input of the task under
solution. The document contains no occurrence of "self-supervised" or "self-supervision".

### 4.2 VARC, "ARC Is a Vision Problem!"

<https://arxiv.org/abs/2511.14761> (Hu, Cy, Qiu, Ding, Wang, Zhu, Andreas, He)

Per-task TTT is supervised on demonstrations only. Verbatim: "Given an unseen task T, we have 2-4
sample pairs in D_demo^T. To make test-time training more feasible, we also augment the single
task T into multiple auxiliary tasks", roughly 50 augmented tasks by flips, rotations and colour
permutations, plus a freshly randomly-initialised task token. Loss is the same per-pixel
cross-entropy as offline training. They state "We do not use the inference set D_infer^T from the
training tasks." No self-supervised use of the test input. 54.5 single / 60.4 ensemble on ARC-1
eval, 8.3 / 11.1 on ARC-2.

### 4.3 LoopViT

<https://arxiv.org/abs/2602.02156>

TTT "fine-tune[s] the shared weights θ on the few-shot demonstrations of each specific task",
generating augmented views of the support examples, under plain cross-entropy. No reconstruction
or masked-cell target, and the test input grid is not a training target. Reported 65.8 percent on
ARC-AGI-1 and 14.2 percent on ARC-AGI-2 at 18M parameters.

### 4.4 Others

ViTARC (<https://arxiv.org/abs/2410.06405>) is per-task supervised learning with object-based
positional encodings, relevant to grid perception but not to TTT objectives. The ARChitects'
2025 masked-diffusion decoder masks output grids only. Nothing in this group makes the test input
a self-supervised target.

---

## 5. Are ARC failures perceptual or reasoning?

The proposal's motivation is that per-task grid fluency is worth acquiring. The evidence is real
but contested, and the strongest supporting evidence is from a different regime than ours.

**For the perceptual account.** "Your Reasoning Benchmark May Not Test Reasoning: Revealing
Perception Bottleneck in Abstract Reasoning Benchmarks", Wang, Huang, Zhang, Wang and Ma
(<https://arxiv.org/abs/2512.21329>, Dec 2025, v2 Jan 2026). They report that "approximately 80
percent of model failures stem from perception errors, i.e., failing to properly identify visual
objects", and that decoupling perception from reasoning through a two-stage pipeline lifts
Mini-ARC from 8.05 to 20.13 percent, Bongard-LOGO from 62.00 to 73.00, and ACRE from 22.00 to
34.50. Conclusion quoted: "perception capability is the dominant factor underlying the observed
performance gap."

Caveats that matter for us. The models are GPT-4o, o1, Claude-Sonnet-4.5 and LLaVA-1.5 consuming
**images**; the datasets are Mini-ARC, ACRE and Bongard-LOGO, not ARC-AGI-1 or ARC-AGI-2, which
the authors note are proprietary. A text-grid LLM with per-task TTT already has near-perfect
literal grid access, so the perceptual failure mode being measured is largely an artefact of image
input, and does not transfer cleanly.

**Against the perceptual account.** The ARC-AGI-2 paper itself
(<https://arxiv.org/abs/2505.11831>) attributes difficulty to composition and abstraction, not
perception, listing multi-rule compositional reasoning, multi-step compositional reasoning,
contextual rule application, and in-context symbol definition. It contains no discussion of
grid-parsing or perception bottlenecks. The ARC Prize 2025 results analysis
(<https://arcprize.org/blog/arc-prize-2025-results-analysis>) also does not frame failures as
perception versus reasoning, and instead notes that "Current AI reasoning performance is tied to
model knowledge." The ARC Prize HRM analysis (<https://arcprize.org/blog/hrm-analysis>) finds the
outer refinement loop worth about +13 points while the hierarchical architecture is within roughly
5 points of a plain transformer, again a reasoning-procedure story rather than a perception story.

**Partial support from an unexpected direction.** BARC's central finding is that "transduction
succeeds on fuzzier perceptual concepts" while "inductive program synthesis excels at precise
computations, and at composing multiple concepts". That is direct evidence that perceptual
competence is a separable and separately-improvable axis on ARC, which is the useful half of the
proposal's motivation.

---

## Verdict table

| Claim to check | Verdict | Strongest URL | Implication for the proposal |
|---|---|---|---|
| An ARC system makes the unlabeled TEST INPUT grid a loss target during per-task TTT | **ABSENT** | <https://arxiv.org/abs/2603.13372> (survey lists none) | The precise composition is open |
| Someone put a generative loss on ARC INPUT grids inside TTT and measured it | **PUBLISHED, and negative** | <https://arxiv.org/abs/2411.07279> Sec 3.2 | The nearest measurement went against it, though on BBH not ARC, and on demo inputs not the test input |
| SOTA ARC pipelines deliberately exclude input grids from the loss | **PUBLISHED** | <https://arxiv.org/abs/2505.07859> | "the model is never tasked with predicting an input grid" is an in-print design choice we would be reversing |
| Fitting a model to reconstruct the test input grid at test time solves ARC tasks | **PUBLISHED** | <https://arxiv.org/abs/2512.06104> | CompressARC does exactly this, from scratch, 20 percent ARC-1 eval / 4 percent ARC-2 |
| Masked-cell / denoising modelling of ARC grids helps | **PUBLISHED** | <https://arxiv.org/abs/2603.06590> Sec 6.6 | UL2 denoising gave +2.08 points, but offline and on demonstration grids |
| An input-distribution generative objective helps ARC models | **PUBLISHED, small effect** | <https://ironbar.github.io/arc24/modeling/Iteration_22_learning_inputs_distribution/> | 6.44 vs 5.71 percent, pretraining only, never used at test time |
| Self-supervised objectives on the unlabeled test instance are a canonical method | **PUBLISHED** | <https://arxiv.org/abs/2209.07522> | TTT-MAE is the exact template; the proposal is its ARC port |
| The same objective applied to a pretrained LLM via LoRA on unlabeled test inputs | **PUBLISHED, non-ARC** | <https://arxiv.org/abs/2505.20633> | TLM shows input-perplexity TTT works for LLMs, but never on grids |
| "TTT" means different things in the ARC and Sun lineages | **PUBLISHED, confirmed** | <https://arxiv.org/abs/2411.07279> Sec 2.2 | The collision is acknowledged by the field, not accidental |
| VARC / LoopViT per-task TTT uses the test input self-supervised | **ABSENT** | <https://arxiv.org/abs/2511.14761> | Both are demo-only supervised, so vision-side ARC has not done it either |
| ARC failures are primarily perceptual | **CONTESTED** | <https://arxiv.org/abs/2512.21329> vs <https://arxiv.org/abs/2505.11831> | 80 percent perception claim holds for VLMs on image inputs of Mini-ARC, not for text-grid LLMs |
| A published ARC system does Sun-style augment-and-recover self-supervised TTT | **FALSE CITATION** | <https://omseeth.github.io/blog/2025/MLLM_for_ARC/> | Attributed to BARC, absent from BARC; do not let this sentence end the investigation |

---

## What would be novel

Precisely this and nothing wider: **making the actual unlabeled test input grid a loss target
inside a pretrained LLM's per-task LoRA adaptation, under a masked-cell or reconstruction
objective, staged before or jointly with the supervised demonstration objective.**

Every ingredient is separately published. Masked-cell modelling of ARC grids is published and
positive, but offline and on demonstration grids. Self-supervised adaptation on the unlabeled
test instance is published and canonical, but not on ARC. Test-input reconstruction at test time
is published, but as an entire from-scratch solver rather than a warm-up for a pretrained model.
A generative objective over ARC input distributions is published, but as a pretraining auxiliary
that its own author declined to carry into test-time fine-tuning. The composition of the four
appears to be unclaimed.

Two secondary things would also be new if measured: whether the warm-up ordering matters
(reconstruct first, then learn the transformation, versus one joint loss), and whether the masked-
cell form behaves differently from the autoregressive next-token form that Akyurek ablated.

## The nearest work a reviewer cites against it

In order of danger.

1. **Akyurek et al. Section 3.2.** A reviewer will say this is their `L^all` with the mask moved
   from demonstration inputs to the test input, that they named the Sun et al. connection
   themselves, and that they measured it and it lost by 1.9 points. The defence is narrow but
   accurate: their loss targets are demonstration inputs inside synthetic leave-one-out tasks, the
   real `x_test` is never in `D_TTT`, and the ablation they report is on BIG-Bench Hard, not ARC.
   This defence must be stated up front in any writeup or it will be read as an omission.
2. **CompressARC.** A reviewer will say that reconstructing the test input at test time is exactly
   this system, that it reaches 4 percent on ARC-AGI-2, and will ask why bolting the same objective
   onto a pretrained model should behave differently.
3. **The ARChitects.** Not a competing method but a quoted design decision: "the model is never
   tasked with predicting an input grid." Reversing an explicit choice made by the pipeline you
   build on requires a measurement, not an argument.
4. **ARC-AGI-2 Technical Report Section 6.6.** A reviewer will say masked denoising of ARC grids is
   done and worth 2 points, and will ask whether the per-task, test-input variant adds anything
   beyond moving that same gain around.

## Honest assessment of the terminology collision

The collision is real and it is documented, but it does **not** rescue the proposal as an oversight.
The sequence of events matters. Akyurek et al. knew the Sun lineage, cited it, wrote down the
corresponding loss for ARC, shipped an ablation of it, and dropped it because it underperformed.
That is not a term-mismatch hiding an unexplored idea; that is an idea that was considered under
its correct name and discarded. The collision explains why the ARC field stopped saying
"self-supervised" out loud, not why nobody thought of it.

What survives is a genuinely narrower gap. Nobody has put the loss on the actual unlabeled test
input, nobody has used a masked-cell rather than autoregressive form, nobody has staged it as a
fluency warm-up, and the one adjacent ARC measurement of an input-generative objective
(ironbar's) was mildly **positive** while the one adjacent LLM-TTT measurement (Akyurek's, on a
different benchmark) was mildly negative. Those two point in opposite directions, which is the
only reason the question is still live.

Fair characterisation: **adjacent-negative, not unexplored.** The most likely outcome is a clean
negative result, because the demonstration inputs already carry nearly all of the per-task grid
statistics that the test input would supply (same palette, same object vocabulary, same symmetry
group by construction), so the marginal information in `x_test` is small. The motivating premise,
that ARC failures are perceptual, is also weakest exactly in our regime: the 80 percent perception
figure comes from VLMs reading rendered images of Mini-ARC, and ARC-AGI-2's own authors attribute
difficulty to composition instead. That said, the experiment is cheap, the composition is
publishable either way if the negative is measured carefully against a matched control, and the
proposal is defensible as prior art precisely because the closest published measurement was run on
a different benchmark, a different loss form, and different grids.
