# Prior art for self-supervised training on the ARC test input

Searched on 5 August 2026. The question is whether anyone makes the unlabelled test input grid a
training target during per-task adaptation on ARC, whether by reconstruction, masked-cell modelling,
denoising or compression.

Two uses of the test input need keeping apart throughout.

- **Using it in context.** The test input appears in the decoding prompt. Every system I checked does
  this, and it isn't the proposal.
- **Using it as a loss target.** Gradients flow from an objective computed on the test input grid
  itself. That's the proposal.

## 1. On ARC, is the test input ever a loss target during adaptation?

For the exact proposal, no, it's absent. For close neighbours there's related work, and one neighbour
explicitly rejects the idea.

### 1.1 Akyurek, Damani et al., "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning"

<https://arxiv.org/abs/2411.07279>, ICML 2025 (<https://proceedings.mlr.press/v267/akyurek25a.html>)

This is the most important entry, because the authors already wrote down the neighbouring objective
and named where it comes from.

They build the training data for test time by leaving one demonstration out at a time, using only
the demonstrations. Here's their construction exactly.

> d_j^ICL = ({(x_k, y_k)}_{k≠j}, x_j, y_j)

Section 3.2 defines three versions of the loss.

1. Only the test output, `L_LM(y_test | x_1,y_1,...,x_K,y_K,x_test)`.
2. Every output, adding `sum_k L_LM(y_k | x_1,y_1,...,x_k)`. This is the version they ship.
3. Inputs and outputs, adding `sum_{k=1..K} L_LM(x_k | x_1,y_1,...,y_{k-1})`.

Of the third version they say this.

> "This method, which requires learners to generate task *inputs* as well as outputs, is
> analogous to existing unsupervised TTT objectives (Sun et al., 2020)."

And in the preliminaries in Section 2.2 they say this.

> "In previous work (e.g., Sun et al., 2020), D_TTT is typically constructed by applying an
> unsupervised objective (e.g., masked autoencoding) to the input x alone."

Two things follow, and they point in opposite directions.

**Against novelty.** Putting a generative loss on ARC input grids during adaptation was considered,
built, tied explicitly to Sun et al.'s line of work, and tested. On BIG-Bench Hard the version with
inputs and outputs scored 55.9, against 57.8 for every output and 54.4 for the test output alone. It
lost.

**For novelty.** The inputs in the third version are the demonstration inputs of the synthetic
leave-one-out tasks. The real, unlabelled test input is never a loss target in any of the three
versions, because the training data contains no pair whose target is unknown. The ablation was also
reported on BIG-Bench Hard, not on ARC. The ARC ablation figure, Figure 5, covers the fine-tuned
baseline at 5, no transformations at 13, direct input-output data at 18, shared training at 22, no
demonstration loss at 26 and the full method at 29. The inputs-and-outputs version isn't there. The
headline result is 53% on ARC public validation with an 8B model.

### 1.2 The ARChitects reject it explicitly, in print

Their 2024 writeup is at <https://da-fr.github.io/arc-prize-2024/the_architects.pdf>, the 2025 arXiv
paper "Product of Experts with LLMs" at <https://arxiv.org/abs/2505.07859>, and the 2025 report at
<https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>.

This is the single strongest citation a reviewer will reach for, quoted exactly.

> "For each task p=((x_i,y_i)_{i=1..k}, x̂), with solution s*_p, we computed gradients only on
> the outputs y_i for i>1 and s*_p. This approach ensures that **the model is never tasked with
> predicting an input grid**, and acknowledges that correctly predicting the first output grid
> is impossible without at least one example."

Their training at test time is "time-constrained and focused solely on the evaluation set, using a
LoRA rank of 32 and running for 64 training steps with batch size 1". So the pipeline our own stack
comes from doesn't just leave out a loss on input grids. It names leaving it out as a design choice.
They scored 71.6% (286.5 of 400) on the ARC-AGI-1 public evaluation and 16.53% on ARC-AGI-2 in 2025.

Their 2025 system does use masking, but on the other side of the pair. It's a masked diffusion model
whose "cross-entropy loss applied only to positions where a token had been masked" works on output
grids during recursive refinement.

### 1.3 NVARC, MindsAI and Tufa Labs, and BARC

- **NVARC** (<https://github.com/1ytic/NVARC>) won ARC Prize 2025 with 24.03%. The 2025 report
  describes it as a "synthetic-data-driven ensemble of an improved Architects-style test-time-trained
  model and TRM-based components". There's no self-supervised objective on the test input in the
  repository or the writeup.
- **MindsAI and Tufa Labs** came third with 12.64%. They fine-tune on the demonstration pairs at test
  time and use AIRV.
- **BARC**, Li et al., "Combining Induction and Transduction for Abstract Reasoning"
  (<https://arxiv.org/abs/2411.02272>), does no training at test time at all.

**A citation that points at nothing.** A widely indexed blog post
(<https://omseeth.github.io/blog/2025/MLLM_for_ARC/>) says the following.

> "Following the approach of Sun et al. [45], this strategy introduces an additional
> self-supervised training step before each prediction. Specifically, the input is augmented
> (e.g., through rotations or color permutations), and the model is tasked with recovering the
> original version. If it fails, its weights are updated accordingly."

It credits this to Li et al. 2024, which is BARC. No such method appears in BARC. Its abstract and
method contain no training at test time and no self-supervised adaptation. It looks like a secondary
source mixing up Sun et al. with the ARC literature, and search engines are now recycling it into
topic pages written by LLMs. Anyone checking prior art quickly will hit that sentence and wrongly
decide the proposal is already published. It isn't, at least not there.

### 1.4 ironbar's Omni-ARC is the closest relative on ARC, and it stops one step short

<https://ironbar.github.io/arc24/05_Solution_Summary/>
<https://ironbar.github.io/arc24/modeling/Iteration_22_learning_inputs_distribution/>

Omni-ARC is a multi-task model trained on several objectives, including one that generates a new input
grid from the task's other input grids. That's a generative model of each task's input distribution,
which covers most of the proposal's motivation. Iteration 22 measured it at 6.44% with the
input-distribution objective against a 5.71% baseline at 10,000 steps, which is weaker than task
augmentation at 7.02%.

But it never reached test time. Here's the solution summary, quoted exactly.

> "Due to the limited submission time test-time fine-tuning was only applied to the canonical
> ARC task of predicting the test outputs."

So a generative objective on input grids exists in the ARC literature and was measured to help a
little, but it was only used in pretraining, never in per-task adaptation, and never on the specific
unlabelled test input. His 2025 work (<https://ironbar.github.io/arc25/05_Solution_Summary/>) trains
the BARC induction model at test time with hindsight relabelling, 28.3% against 23.3%, which is
pseudo-labelling generated programs rather than self-supervision on the test input.

### 1.5 The ARC Prize technical reports

The ARC Prize 2024 report (<https://arxiv.org/abs/2412.04604>) defines the term the field now uses.

> "TTT, in this context, involves fine-tuning a pretrained LLM on the demonstration pairs of each
> task instance seen at test time, effectively creating a different variant of the base model for
> each task."

The ARC Prize 2025 report (<https://arxiv.org/html/2601.10904v1>) doesn't mention self-supervised
objectives on unlabelled test inputs. The living survey "The ARC of Progress towards AGI"
(<https://arxiv.org/abs/2603.13372>) also lists no ARC system that applies masked modelling,
reconstruction or denoising to the actual test input grids at inference.

## 2. CompressARC by Liao and Gu is the closest relative overall

<https://arxiv.org/abs/2512.06104>, with a blog post at
<https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html>
and code at <https://github.com/iliao2345/CompressARC>.

**What it trains on.** Every known grid of the one puzzle it's solving, which explicitly includes the
test input and leaves out only the answer.

> "We jointly optimize θ, μ, and Σ to minimize the sum of cross-entropies over the known grids
> (5 of them,) ignoring the answer grid."

> "training happens only on a single sample: the target inference puzzle itself, with the final
> solution information removed."

**The objective.** A bound on description length in the style of a VAE, which is the KL divergence
between N(μ,Σ) and N(0,I) as the cost of the latent code, plus cross-entropy for reconstructing the
known grids. "A KL divergence penalty keeps N(μ,Σ) close to N(0,1), as in a VAE." The answer is read
out by sampling, and the most common answer during training is the prediction.

**The setup.** A hand-built equivariant network of 76,000 parameters, randomly initialized for each
puzzle, with no pretraining and no use of the ARC training set, taking about 20 minutes a puzzle on an
RTX 4070.

**The numbers.** 34.75% on the ARC-AGI-1 training set and 20% on the ARC-AGI-1 evaluation set, at
pass@2. The ARC Prize 2025 report sums it up as "20-34% on ARC-AGI-1 and 4% on ARC-AGI-2 without any
pretraining or external data". It took third place in the 2025 paper awards.

**How the proposal differs.** In five ways. They're real, but they're all about staging and setting
rather than the core idea.

1. CompressARC has no pretrained model. The proposal adds an objective on top of a pretrained LLM's
   per-task LoRA.
2. In CompressARC the reconstruction objective is the solver, and the answer comes out of the same
   fitted model. In the proposal, reconstruction is a warm-up whose only job is to hand a better
   prepared adapter to a separate, supervised objective for the transformation.
3. CompressARC reconstructs the demonstration inputs, demonstration outputs and test input under one
   joint objective, so it never separates fluency with the grids from the transformation. The proposal
   separates them on purpose.
4. Its architecture is a hand-designed equivariant network with ARC's symmetries built in, not a LoRA
   on a transformer reading serialized grid tokens.
5. CompressARC replaces the whole system, with a published ARC-AGI-2 score of 4%. The proposal adds
   one component to a pipeline currently near 24%.

The honest reading is that CompressARC shows fitting a model to reconstruct the test input at test
time is a published idea that works, modestly. It doesn't show that anyone has tried it as an extra
objective for training an LLM at test time.

## 3. The general line of test-time training, and the clash over the name

The clash is real and documented, and it's documented by the very paper that made "TTT" the term for
ARC.

The canonical line of work all trains on the unlabelled test instance with a self-supervised
objective.

| work | link | objective on the test instance |
|---|---|---|
| Sun, Wang, Liu, Miller, Efros and Hardt, ICML 2020 | <https://proceedings.mlr.press/v119/sun20b.html> | predicting rotation on the single test image |
| TENT, Wang et al., ICLR 2021 | <https://arxiv.org/abs/2006.10726> | minimizing the entropy of predictions |
| TTT++, Liu et al., NeurIPS 2021 | <https://papers.nips.cc/paper/2021/hash/b618c3210e934362ac261db280128c22-Abstract.html> | contrastive learning plus matching feature moments |
| MEMO, Zhang et al., NeurIPS 2022 | <https://arxiv.org/abs/2110.09506> | marginal entropy over augmentations of one test input |
| TTT-MAE, Gandelsman et al., NeurIPS 2022 | <https://arxiv.org/abs/2209.07522> | masked autoencoding of 75% of patches, per test image |

Sun et al. put it as "turning a single unlabeled test sample into a self-supervised learning problem,
on which the model parameters are updated before making a prediction."

TTT-MAE is the direct template. In one line, the proposal is TTT-MAE with grid cells in place of
patches and a LoRA in place of the encoder.

**The clash.** In the ARC literature, "TTT" means supervised fine-tuning on the demonstration pairs,
which are labelled data supplied inside the task. In Sun's line of work it means self-supervised
adaptation on the unlabelled test input. They're different procedures with the same name. Akyurek et
al. inherited the name and pointed out the difference themselves in Section 2.2, quoted in 1.1 above.
The ARC Prize 2024 report then fixed the ARC meaning into the field's vocabulary with its definition,
quoted in 1.5.

**Has anyone applied Sun's idea to ARC or to few-shot program induction?** Not on ARC, going by
sections 1 and 4. Outside ARC, the closest version for LLMs is TLM, "Test-Time Learning for Large
Language Models" (<https://arxiv.org/abs/2505.20633>). It trains a LoRA on unlabelled test inputs by
minimizing their perplexity, and shows that lower input perplexity lowers output perplexity. It's
tested on domain, instruction and reasoning benchmarks, and doesn't touch ARC or grids. Also nearby is
ReLAT, reasoning in latent space guided by reconstruction at test time, with temporary LoRA adapters
(<https://arxiv.org/abs/2606.06252>).

## 4. Masked autoencoding and masked-cell modelling on ARC grids

This is where the answer comes closest to "already published", but on the wrong grids and at the wrong
stage.

### 4.1 The ARC-AGI-2 technical report by Lemes de Oliveira et al.

<https://arxiv.org/abs/2603.06590> (November 2025), Section 6.6, "UL2 Masked Training", inside
Section 6, "Training to Understand".

They run UL2-style denoising on ARC grid tokens. Some tokens in the demonstration grids are replaced
with mask tokens (`<extra_id_0>` to `<extra_id_99>`, plus the mode tokens `<task_id_S>`, `<task_id_X>`
and `<task_id_R>`), and the model reconstructs them one by one, in three modes that differ in noise
ratio and masking pattern. They say the motive is to discourage "overfitting to autoregressive copying
patterns" in favour of "latent rule discovery and structural understanding", and they interleave
denoising with ordinary ARC tasks on a curriculum. Their list of contributions says they use
"multi-task learning to enable the model not only to solve tasks but also to denoise them".

It works. UL2 denoising took them from 25.00% to 27.08% on the Kaggle semi-private set, a gain of 2.08
points, and they have an ablation section called "Effect of Removing UL2 Training".

Two limits leave room for the proposal. First, this is offline training, not per-task training at
test time. Their own test-time stage builds "a small adaptation dataset per task by iterating over
demonstrations as pseudo-test pairs", which is the standard recipe. Second, the masking is on the
demonstration grids, not on the unlabelled test input of the task being solved. The words
"self-supervised" and "self-supervision" don't appear anywhere in the document.

### 4.2 VARC, "ARC Is a Vision Problem!"

<https://arxiv.org/abs/2511.14761>, by Hu, Cy, Qiu, Ding, Wang, Zhu, Andreas and He.

Their per-task training at test time is supervised on the demonstrations only.

> "Given an unseen task T, we have 2-4 sample pairs in D_demo^T. To make test-time training more
> feasible, we also augment the single task T into multiple auxiliary tasks"

That gives roughly 50 augmented tasks from flips, rotations and colour permutations, plus a task token
initialized at random. The loss is the same per-pixel cross-entropy as their offline training. They
say "We do not use the inference set D_infer^T from the training tasks." There's no self-supervised
use of the test input. They score 54.5 with one model and 60.4 as an ensemble on the ARC-1
evaluation, and 8.3 and 11.1 on ARC-2.

### 4.3 LoopViT

<https://arxiv.org/abs/2602.02156>

At test time it "fine-tune[s] the shared weights θ on the few-shot demonstrations of each specific
task", generating augmented views of the support examples, under plain cross-entropy. There's no
reconstruction or masked-cell target, and the test input isn't a training target. It reports 65.8% on
ARC-AGI-1 and 14.2% on ARC-AGI-2 with 18 million parameters.

### 4.4 Others

ViTARC (<https://arxiv.org/abs/2410.06405>) learns each task with supervision, using positional
encodings based on objects. It's relevant to how grids are perceived but not to objectives at test
time. The ARChitects' 2025 masked-diffusion decoder only masks output grids. Nothing in this group
makes the test input a self-supervised target.

## 5. Are ARC failures about perception or about reasoning?

The proposal's motivation is that fluency with each task's grids is worth acquiring. The evidence for
that is real but disputed, and the strongest support comes from a setting unlike ours.

**For the perception account.** "Your Reasoning Benchmark May Not Test Reasoning. Revealing Perception
Bottleneck in Abstract Reasoning Benchmarks", by Wang, Huang, Zhang, Wang and Ma
(<https://arxiv.org/abs/2512.21329>, December 2025, revised January 2026), reports that "approximately
80 percent of model failures stem from perception errors, i.e., failing to properly identify visual
objects". Splitting perception from reasoning into two stages lifts Mini-ARC from 8.05% to 20.13%,
Bongard-LOGO from 62.00 to 73.00, and ACRE from 22.00 to 34.50. They conclude that "perception
capability is the dominant factor underlying the observed performance gap."

The caveats matter for us. The models are GPT-4o, o1, Claude-Sonnet-4.5 and LLaVA-1.5 reading images,
and the datasets are Mini-ARC, ACRE and Bongard-LOGO, not ARC-AGI-1 or 2, which the authors note are
proprietary. An LLM reading text grids, trained per task, already sees the grid almost perfectly, so
the perception failures they measure mostly come from reading images and don't carry over cleanly.

**Against the perception account.** The ARC-AGI-2 paper itself (<https://arxiv.org/abs/2505.11831>)
puts the difficulty down to composition and abstraction rather than perception. It lists reasoning
that composes several rules, reasoning in several composed steps, applying rules in context, and
defining symbols in context. It never discusses bottlenecks in reading grids or perceiving them. The
ARC Prize 2025 results analysis (<https://arcprize.org/blog/arc-prize-2025-results-analysis>) doesn't
frame failures as perception against reasoning either, and notes instead that "Current AI reasoning
performance is tied to model knowledge." ARC Prize's analysis of HRM (<https://arcprize.org/blog/hrm-analysis>)
finds the outer refinement loop worth about 13 points, while the hierarchical architecture comes within
about 5 points of a plain transformer. That's again a story about the reasoning procedure, not
perception.

**Partial support from an unexpected place.** BARC's central finding is that "transduction succeeds on
fuzzier perceptual concepts" while "inductive program synthesis excels at precise computations, and at
composing multiple concepts". That's direct evidence that perceptual skill is a separate axis on ARC
that can be improved separately, which is the useful half of the proposal's motivation.

## Verdicts

| claim | verdict | strongest link | what it means for the proposal |
|---|---|---|---|
| An ARC system makes the unlabelled test input a loss target during per-task training | **Absent** | <https://arxiv.org/abs/2603.13372>, where the survey lists none | The exact combination is open |
| Someone put a generative loss on ARC input grids during adaptation and measured it | **Published, and negative** | <https://arxiv.org/abs/2411.07279>, Section 3.2 | The nearest measurement went against it, though on BIG-Bench Hard rather than ARC and on demonstration inputs rather than the test input |
| Top ARC pipelines leave input grids out of the loss on purpose | **Published** | <https://arxiv.org/abs/2505.07859> | "the model is never tasked with predicting an input grid" is a design choice in print that we'd be reversing |
| Fitting a model to reconstruct the test input at test time solves ARC tasks | **Published** | <https://arxiv.org/abs/2512.06104> | CompressARC does exactly this from scratch, at 20% on the ARC-1 evaluation and 4% on ARC-2 |
| Masked-cell or denoising modelling of ARC grids helps | **Published** | <https://arxiv.org/abs/2603.06590>, Section 6.6 | UL2 denoising gave 2.08 points, but offline and on demonstration grids |
| A generative objective on the input distribution helps ARC models | **Published, with a small effect** | <https://ironbar.github.io/arc24/modeling/Iteration_22_learning_inputs_distribution/> | 6.44% against 5.71%, in pretraining only and never at test time |
| Self-supervised objectives on the unlabelled test instance are a standard method | **Published** | <https://arxiv.org/abs/2209.07522> | TTT-MAE is the exact template, and the proposal is its port to ARC |
| The same objective applied to a pretrained LLM through a LoRA on unlabelled test inputs | **Published, outside ARC** | <https://arxiv.org/abs/2505.20633> | TLM shows training on input perplexity at test time works for LLMs, but never on grids |
| "TTT" means different things in the ARC literature and in Sun's | **Published and confirmed** | <https://arxiv.org/abs/2411.07279>, Section 2.2 | The field acknowledges the clash, so it isn't an accident |
| VARC or LoopViT use the test input self-supervised during per-task training | **Absent** | <https://arxiv.org/abs/2511.14761> | Both supervise only on demonstrations, so the vision side of ARC hasn't done it either |
| ARC failures are mainly about perception | **Disputed** | <https://arxiv.org/abs/2512.21329> against <https://arxiv.org/abs/2505.11831> | The 80% perception claim holds for vision-language models reading images of Mini-ARC, not for LLMs reading text grids |
| A published ARC system does Sun-style self-supervised training, augmenting and recovering the input | **A false citation** | <https://omseeth.github.io/blog/2025/MLLM_for_ARC/> | It's credited to BARC and isn't in BARC, so don't let that sentence end the search |

## What would be new

Exactly this and nothing broader. Making the actual unlabelled test input a loss target inside a
pretrained LLM's per-task LoRA adaptation, under a masked-cell or reconstruction objective, run before
the supervised objective on the demonstrations or alongside it.

Every ingredient is published separately. Masked-cell modelling of ARC grids is published and helps,
but offline and on demonstration grids. Self-supervised adaptation on the unlabelled test instance is
published and standard, but not on ARC. Reconstructing the test input at test time is published, but
as a whole solver built from scratch rather than a warm-up for a pretrained model. A generative
objective over ARC input distributions is published, but as an extra objective in pretraining that its
own author chose not to carry into test-time fine-tuning. Nobody seems to have claimed the combination
of all four.

Two smaller things would also be new if measured. One is whether the order of the warm-up matters,
reconstructing first and learning the transformation second, against one joint loss. The other is
whether the masked-cell form behaves differently from the autoregressive next-token form that Akyurek
tested.

## The closest work a reviewer will cite against it

In order of danger.

1. **Akyurek et al., Section 3.2.** A reviewer will say this is their every-output loss with the mask
   moved from the demonstration inputs to the test input, that they named the link to Sun et al.
   themselves, and that they measured it and it lost by 1.9 points. The defence is narrow but accurate.
   Their loss targets are demonstration inputs inside synthetic leave-one-out tasks, the real test input
   is never in their training data, and the ablation they report is on BIG-Bench Hard, not ARC. Any
   writeup has to state this defence up front, or it will read as an omission.
2. **CompressARC.** A reviewer will say reconstructing the test input at test time is exactly this
   system, that it reaches 4% on ARC-AGI-2, and ask why bolting the same objective onto a pretrained
   model should behave any differently.
3. **The ARChitects.** Not a rival method but a quoted design decision, that "the model is never tasked
   with predicting an input grid." Reversing an explicit choice made by the pipeline you build on needs
   a measurement, not an argument.
4. **The ARC-AGI-2 technical report, Section 6.6.** A reviewer will say masked denoising of ARC grids
   has been done and is worth two points, and ask whether the per-task version on the test input adds
   anything beyond moving that same gain around.

## An honest read of the name clash

The clash is real and documented, but it doesn't rescue the proposal as something everyone overlooked.
The order of events matters. Akyurek et al. knew Sun's line of work, cited it, wrote down the matching
loss for ARC, shipped an ablation of it, and dropped it because it did worse. That isn't a mismatch in
terms hiding an unexplored idea. It's an idea that was considered under its right name and discarded.
The clash explains why the ARC field stopped saying "self-supervised" out loud, not why nobody thought
of it.

What's left is a genuinely narrower gap. Nobody has put the loss on the actual unlabelled test input.
Nobody has used a masked-cell form rather than an autoregressive one. Nobody has staged it as a warm-up
for fluency. And the one related ARC measurement of a generative objective on inputs, ironbar's, was
slightly positive, while the one related measurement in LLM test-time training, Akyurek's on a
different benchmark, was slightly negative. Those point in opposite directions, which is the only
reason the question is still open.

The fair summary is that this is a near neighbour of a negative result, not unexplored ground. The
most likely outcome is a clean negative. The demonstration inputs already carry nearly all of each
task's grid statistics that the test input would add, the same palette, the same kinds of objects and
the same symmetries by construction, so the extra information in the test input is small. The
motivating premise, that ARC failures are about perception, is also weakest exactly in our setting.
The 80% perception figure comes from vision-language models reading rendered images of Mini-ARC, and
ARC-AGI-2's own authors put the difficulty down to composition instead. Still, the experiment is
cheap, and the combination is publishable either way if the negative is measured carefully against a
matched control. It also holds up against the prior art precisely because the closest published
measurement was run on a different benchmark, with a different form of loss, on different grids.
