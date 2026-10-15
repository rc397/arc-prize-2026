# Prior art for families of adapters that each leave out one demonstration

Written on 5 August 2026. This is a targeted check for novelty on one specific proposal, not on test-time
training or ensembling in general.

## The proposal

For each ARC task with N demonstration pairs, where N is 2 to 5, train N separate LoRA adapters, with
adapter i trained with demonstration i removed entirely. Then do two things. First, pool the candidate
grids decoded by all N adapters to raise coverage. Second, read whether the adapters agree or disagree
on the test input as a sign that the demonstrations don't pin down the rule.

Two things have to be kept apart throughout, because almost every false positive in this search mixes
them up.

- **Leaving one out to build training samples.** For each pair j, build a sample whose prompt holds the
  other N minus 1 pairs and whose target is pair j. Every sample goes into one adapter's training set.
  This is standard, and our pipeline already does it.
- **Leaving one out to build separate models.** Train N different adapters, each of which never sees one
  pair at all, in any role, prompt or target. This is the proposal.

It's also worth keeping apart changing the order or the view of the evidence, through reordered
examples, the eight symmetries or colour permutations, and changing what the evidence contains, by
deleting a pair. The first is everywhere in ARC. The second is what's being proposed.

## 1. On ARC, are separate adapters ever trained on different subsets of the demonstrations?

### Akyurek et al., "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning" (arXiv 2411.07279)

This is the paper a reviewer reaches for first, and it's the one most likely to be misread as prior art.
It isn't.

Its leave-one-out builds training samples. The paper says of its synthetic training data, "By excluding
the jth example pair from the training examples, we can create the following synthetic task." All those
synthetic tasks go into one training set. On how many adapters, the paper is explicit. "Instead of
learning a single LoRA adapter for all tasks in the test set, we learn an individual task-specific LoRA
adapter for each task. That is, we obtain N different LoRA adapters, where N is the number of test
tasks."

Note the clash of symbols. Their N is the number of tasks. The proposal's N is the number of
demonstrations inside one task. They have one adapter per task, and the proposal has N per task.
Nowhere in the paper is there an ensemble of adapters trained on different subsets of the
demonstrations. Their variety at inference comes from geometric transformations plus a two-stage vote,
the top three within each kind of transformation and then a vote across all of them, all decoded from
the same single adapter.

Sources are <https://arxiv.org/html/2411.07279v1> and <https://ekinakyurek.github.io/papers/ttt.pdf>.

### The ARChitects, who won in 2024 and came second in 2025

In 2024 they trained one LoRA on the examples of many evaluation tasks at once. In 2025, "The test-time
training workflow was changed to run separately for each task, instead of fine-tuning the model on
multiple tasks simultaneously as in the 2024 contest." Either way it's at most one adapter per task,
trained on all the demonstrations. All their variety comes at inference. The product-of-experts score
combines P(aug(s) | aug(p)) over 16 augmented views, and the augmentations are the eight symmetries,
colour permutations and reordering the examples. Reordering is as close as they come to touching the set
of demonstrations, and it's a reshuffle, not a deletion.

Sources are <https://github.com/da-fr/arc-prize-2024>, <https://arxiv.org/html/2505.07859> and
<https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>.

### NVARC, who won in 2025

They use a rank-256 LoRA per task, one adapter per task. Their variety of candidates comes from batched
depth-first decoding and from eight fixed augmentations used for rescoring. Their selector after the
deadline, the number of times the search found a candidate across augmentations plus its mean
log-likelihood, is a mix of frequency and likelihood worked out from one adapter under many views. There's
no splitting of the demonstrations.

Sources are <https://github.com/1ytic/NVARC> and
<https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc>.

### ironbar's Omni-ARC, third in 2024, is the nearest negative result on ARC

This is the closest anyone in the public ARC record has come, and it's worth reading carefully because
it's a partial measurement against the proposal. In his iteration on test-time fine-tuning, he tested
building the training set from subsets, listing "train with n-1 samples" and "train with n-2 samples"
next to the augmentation variants. The n-1 row reports a validation loss of 0.1398 and validation
accuracy of 7.20%. Of going further, he concluded, "Training with n-2 samples did not improve over
training with n-1 samples. This matched my intuition because we are not increasing the number of
tasks, we are just removing input information."

Three things follow. First, this is still one adapter, so it isn't the proposal. Second, it tests how
much evidence a single adapter should be trained on, and the answer went the wrong way. Third, and most
useful, "we are just removing input information" is exactly the objection the proposal has to answer.
It's the null hypothesis that N adapters with less evidence are just N weaker adapters.

Source, <https://ironbar.github.io/arc24/modeling/Iteration_04_test_time_fine-tuning/>.

### Surveys and technical reports

I checked three independent secondary sources for the mechanism, and none contains it.

- "The ARC of Progress towards AGI. A Living Survey of Abstraction and Reasoning" (2026),
  <https://arxiv.org/html/2603.13372v1>. It discusses adaptation at test time at length, and keeping
  several hypotheses appears in its taxonomy ("Multiple hypotheses: Diverse candidates maintained;
  selected via demo consistency"). Leaving out one demonstration pair at a time to produce separate
  models is absent.
- The ARC Prize 2025 technical report, <https://arxiv.org/html/2601.10904v1>. Absent.
- The ARC Prize 2024 technical report, <https://arxiv.org/pdf/2412.04604>. Absent.

Searching the Kaggle discussion forums for the 2024, 2025 and 2026 competitions turned up no thread
proposing or reporting families of adapters trained per subset of demonstrations.

**Verdict for section 1. Absent.** The mechanism as specified isn't in the public ARC record. The
neighbour it's easily confused with, leaving one out to build training samples for a single adapter, is
universal.

## 2. On ARC, is disagreement in an ensemble used as a signal of confidence, routing or ambiguity?

This is where the proposal is least new in spirit, though not in mechanism.

### Land, "Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2" (arXiv 2606.31543, 2026)

Agreement is used directly to signal confidence per task and to route compute. "if early-stage
candidates already show strong agreement (multiple candidates converging on the same output), the
system terminates early." The gap is large. Instances that stopped early scored 97.3% (36 of 37), and
instances that ran the full sweep of 29 candidates scored 70.8% (92 of 130). The paper also reads
disagreement as evidence about the task rather than the model, and blames its one early-stop failure on
a task that really was underdetermined. "All models confidently assume the simpler interpretation
('legend is the same'), while the ground truth requires the more complex one."

The difference from the proposal is where the ensemble's variety comes from. Land's members are
different foundation models working across text, images and code, explicitly "independent candidates
across heterogeneous reasoning modalities", not one model adapted to different slices of the same
evidence. Disagreement there mixes up differences in model ability with ambiguity in the task.
Disagreement across leave-one-out adapters of one base model wouldn't, and that's the proposal's real
argument, and it's a good one.

Source, <https://arxiv.org/html/2606.31543v1>.

### Agreement used for selection in the test-time training lineage

The ARChitects' product of experts and MindsAI's AIRV vote both compute agreement, but agreement across
augmented views of one adapted model, used to rank candidates within a task rather than to characterise
the task. NVARC's selector based on search visits has the same shape. None of these reads out
confidence per task, and none of them varies the evidence.

Sources are <https://arxiv.org/html/2505.07859> and
<https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution>.

### Predicting success per task without disagreement

"Structural Grid Descriptors Predict Within-Task Solver Success on ARC-AGI" (Pendharkar, 2026) predicts
whether a run will succeed from 13 hand-built descriptors of intermediate grids. It reaches an AUC of
0.885 within tasks and saves 33.6% of the beam-search compute while keeping 98.9% of solves. It proves
that confidence signals per task and per run pay for themselves within ARC's compute budget, and it's a
competing baseline, since it gets there from grid structure with no ensemble at all. It doesn't look at
how many demonstrations there are, at underdetermination, or at disagreement in an ensemble.

Source, <https://arxiv.org/html/2606.09026v2>.

**Verdict for section 2.** The general idea is published. The specific instrument has only related
work. Agreement in an ensemble as a signal of confidence and routing on ARC is published, with strong
numbers. Disagreement among models adapted to different evidence from the same task, built on purpose
to detect ambiguity, isn't.

## 3. In machine learning generally, ensembles over subsets when the support set is tiny

### The mechanism has had a name since the 1990s

Query by committee (Seung, Opper and Sompolinsky, 1992) trains several hypotheses and uses their
disagreement as information. Query by bagging (Abe and Mamitsuka, "Query learning strategies using
boosting and bagging", ICML 1998) is the version where the committee's variety comes from training
members on different subsets of the data, which is exactly how the proposal generates its members. The
modAL documentation describes the standard setup. "To create and enhance disagreement among committee
members, the training set can be changed for each estimator (e.g., via bagging) or by varying the
configuration of the estimator."

So the abstract claim, that a committee trained on different evidence carries information a single
model's likelihoods don't, is the founding claim of query by committee, and it's thirty years old. The
proposal can't claim it as a contribution to theory. It can claim the setting.

Sources are
<https://www.semanticscholar.org/paper/Query-Learning-Strategies-Using-Boosting-and-Abe-Mamitsuka/13f3c1bc860eafce2af81d38d885ab9e0a4be01f>
and <https://modal-python.readthedocs.io/en/latest/content/models/Committee.html>.

### The direct evidence against that a reviewer will raise

Lakshminarayanan, Pritzel and Blundell, "Simple and Scalable Predictive Uncertainty Estimation using
Deep Ensembles" (arXiv 1612.01474, NIPS 2017), ran exactly this comparison for deep networks and found
against splitting the data. Bagging made performance worse, and random initialization plus shuffling on
the full dataset was enough. This is the strongest single published prior against the proposal's
premise, and it's a very widely cited default. Any writeup has to take it on directly.

Source, <https://arxiv.org/pdf/1612.01474>.

### And the evidence against that

"Lost in the Folds. When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation" (Kirscher
et al., arXiv 2605.18329, 2026) argues the two aren't interchangeable. Ensembles from cross-validation,
whose members train on different folds, and deep ensembles from different seeds capture different parts
of the uncertainty and complement each other rather than being equivalent. That supports the proposal's
framing in principle. The caveat is that I took this finding from the PDF, and the field is medical
image segmentation, not few-shot induction, so it supports the framing rather than the setting.

Source, <https://arxiv.org/pdf/2605.18329>.

### Ensembling over subsets of the support set specifically

Two papers do this at the level of the prompt, and they're the closest prior art in general machine
learning to the first half of the proposal.

- **DENSE**, Khalifa et al., "Exploring Demonstration Ensembling for In-context Learning" (arXiv
  2308.08780). The demonstrations are split into buckets, the model runs separately on each, and the
  output probabilities are combined. A weighted maximum beat plain concatenation by up to 2.4 points on
  average across 12 tasks on GPT-J. Source, <https://arxiv.org/abs/2308.08780>.
- **MoICL**, Hong et al., "Mixtures of In-Context Learners" (arXiv 2411.02830, ACL 2025). It treats
  subsets of demonstrations as experts and learns a weighting that merges their output distributions.
  It gains on 5 of 7 classification datasets, up to 13% over plain in-context learning, and is notably
  robust to noisy demonstrations (up to 38%) and unbalanced ones (up to 49%). Source,
  <https://arxiv.org/abs/2411.02830>.

Both differ from the proposal in three ways at once. They condition by prompting rather than training
separate adapters, they target classification with plenty of labelled examples rather than grid
induction from 2 to 5 shots, and neither reads disagreement between the subset experts as a signal about
the specification. Both combine, and neither diagnoses.

### The closest published system overall is UG-TTT

"Epistemic Uncertainty for Test-Time Discovery" (Riaz et al., arXiv 2605.11328, 2026) keeps "a small
ensemble of low-rank adapters over a frozen base model", computes "per-token disagreement, quantified as
the mutual information between ensemble predictions and weight hypotheses", and uses it to "isolate
epistemic uncertainty" and find "positions where insufficient coverage leads to adapter divergence
rather than intrinsic problem difficulty."

That's uncomfortably close to the second half of the proposal. The deciding difference is where the
variety comes from. "A nuclear norm regularizer ensures the adapters remain distinct from one another,
thereby preserving the exploration signal throughout training." The adapters see the same data and are
pushed apart by a penalty, with no change to the evidence. It's also reinforcement learning for
scientific discovery, using disagreement as an exploration bonus per token inside policy-gradient
training, and it isn't an ARC system.

Source, <https://arxiv.org/abs/2605.11328>.

### LoRA ensembles for uncertainty

There's a small literature here (arXiv 2310.00035, Balabanov and Linander's arXiv 2402.12264, and
LoRA-Ensemble at arXiv 2405.14438). In all of them the members differ by random initialization and the
randomness of training, not by their training data. They show LoRA ensembles are a cheap and legitimate
approximation of the posterior, which the proposal can lean on for its claim about uncertainty, but none
of them varies the evidence.

**Verdict for section 3.** The general mechanism and its theory are published. The version for tiny
support sets is absent. The mechanism is query by bagging. There's no established name for ensembling
by leaving one out over a support set of 2 to 5, and I found no paper that does it with trained adapters
in that setting.

## 4. In program synthesis, the symbolic version of keeping every consistent hypothesis

The symbolic tradition solved the representation half of this problem twenty-five years ago, which
makes what's actually being proposed clearer.

- **Version space algebra** (Lau and Domingos, ICML 2000, the algorithm at the heart of FlashFill). A
  version space is a compact structure that implicitly represents the set of all programs in the DSL
  consistent with the examples, updated as each new example arrives. The framing says outright that
  "examples, though often easy to provide, are an ambiguous specification of the user's intent", and
  that examples given by a user leave several programs that satisfy them. Sources are
  <https://homes.cs.washington.edu/~pedrod/papers/kcap03b.pdf> and
  <https://www.semanticscholar.org/paper/58b2550804f24d5bb0b8cfe6815623be8760ffe1>.
- **Pragmatic program synthesis** (Pu, Ellis et al., NeurIPS 2020, and "Amortizing Pragmatic Program
  Synthesis with Rankings", arXiv 2407.02499). Given many consistent programs, it uses recursive Bayesian
  reasoning between a speaker and a listener to infer which one the demonstrator meant. That's the
  published answer to "the demonstrations don't pin down the rule, so now what?" Sources are
  <https://proceedings.neurips.cc/paper/2020/file/99c83c904d0d64fbef50d919a5c66a80-Paper.pdf> and
  <https://arxiv.org/html/2407.02499>.
- **Work on ARC that keeps several hypotheses.** Solvers based on description length (Ferré, arXiv
  2112.00848) score competing consistent hypotheses by simplicity. "Solomonoff-Inspired Hypothesis
  Ranking with LLMs for Prediction Under Uncertainty" (arXiv 2512.17145) builds an explicit distribution
  over hypotheses an LLM generates, weighted by simplicity and consistency, applies it to per-cell
  prediction on Mini-ARC, and is better calibrated than Bayesian model averaging because it spreads
  probability rather than concentrating it. BARC's induction arm (arXiv 2411.02272) samples many Python
  programs and keeps the ones consistent with the demonstrations.

Here's how all of this differs from the proposal. Every one of them enumerates or samples the space of
hypotheses and then filters by consistency with all the demonstrations. The evidence stays fixed and the
hypotheses vary. The proposal turns that around. It keeps the learner fixed, varies the evidence, and
infers how wide the space of hypotheses is from how much the learner's output moves. That reversal is
what makes it usable with a neural model that has no space of hypotheses you can enumerate. I found no
symbolic work on ARC that changes the set of demonstrations to estimate how wide the hypothesis space is.

**Verdict for section 4.** Solving ARC with explicit multiple hypotheses is published. Changing the
evidence to probe the width of the hypothesis space is absent.

## 5. Published evidence that ARC demonstrations don't pin down their rules

The evidence is real, but thinner and softer than the proposal would like.

- **ARC Prize's own position.** Part of the reason for pass@2 is this. ARC-AGI uses pass@2 "to account
  for the fact that certain tasks have explicit ambiguity and require two guesses to disambiguate", and
  to catch accidental ambiguity or mistakes in the data. Source, <https://arcprize.org/guide/1>.
- **How ARC-AGI-2 was calibrated against people.** 407 unique participants ran 515 sessions over 1,848
  unique test pairs, with 13,405 attempts. A task stayed in "only if at least two independent
  participants solved one or more sub-pairs within their first two attempts", and every task was solved
  by at least two people, with 2 to 10 per task. Notice what this does and doesn't show. It guarantees a
  task can be solved, not that it has only one answer. Two people can solve a task that still allows a
  second reasonable rule. The technical report doesn't discuss ambiguity or underdetermination
  directly. Source, <https://arxiv.org/html/2505.11831v2>.
- **H-ARC** (Legris et al., arXiv 2409.01374). 1,729 people covered all 800 tasks in the ARC-1 training
  and evaluation sets. The average person scored 73.3% to 77.2%, 98.8% of tasks were solved by at least
  one person, and over 90% by at least three. The failures that remain fit underdetermination, but the
  paper doesn't attribute them to it. Source, <https://arxiv.org/abs/2409.01374>.
- **Johnson, Vong, Lake and Gureckis, "Fast and flexible. Human program induction in abstract reasoning
  tasks"** (arXiv 2103.05823, CogSci 2021). A behavioural study on 40 ARC tasks. Each participant solved
  84% of tasks on average, and 65% of tasks were solved by more than 80% of participants. The complement
  is the interesting number, since roughly a third of tasks show real disagreement between people.
  Source, <https://arxiv.org/abs/2103.05823>.
- **ARCTraj** (arXiv 2511.11079). Records of how people solve tasks show some tasks where everyone
  converges on the same key transformations, and others with a lot of variety, where people pick
  different parts of the grid or apply different operations. The tasks with the most unique solutions
  point to "flexible or ambiguous reasoning spaces."
- **ARC-TGI** (Lehmann et al., arXiv 2603.05099, 2026). The most explicit published statement of the
  failure. "Independent pair sampling can yield ambiguous episodes (e.g., missing critical variation,
  unseen test features, or degenerate shortcuts)", and existing generators "do not reliably guarantee
  that the set of training examples disambiguates the intended rule". Their fix is a design rule they
  call "Disambiguating coverage: ensure training pairs collectively include the variations required to
  infer the rule". The important limit is that this is about generating tasks, and they give no
  estimate of how often the official ARC sets are affected. Source, <https://arxiv.org/html/2603.05099>.
- **A concrete case.** Land documents a task where every model in a varied ensemble confidently picked
  the simpler of two readings, and the right answer needed the more complex one. Source,
  <https://arxiv.org/html/2606.31543v1>.

**The gap.** I found no published study that measures what share of ARC tasks allow more than one rule
consistent with their demonstrations. ARC-TGI names the problem and designs it out of synthetic tasks.
Nobody has measured it on ARC-AGI-1 or ARC-AGI-2, and that measurement is unclaimed in its own right.

**Verdict for section 5.** Related work only. ARC Prize acknowledges underdetermination, ARC-TGI names
it precisely, and it shows up in data on how people solve tasks, but nobody has measured it on the
official sets.

## Verdicts

| claim | verdict | strongest evidence | what it means for novelty |
|---|---|---|---|
| 1a. N separate per-task adapters, each with one demonstration removed, for one ARC task | **Absent** | <https://arxiv.org/html/2411.07279v1> | Not in the ARC record. Akyurek's leave-one-out builds training samples inside one adapter, and his "N adapters" means one per task. Any writeup has to make this distinction in its first paragraph or it will be dismissed as a rediscovery |
| 1b. Pooling candidates across adapters trained on subsets, to raise coverage on ARC | **Absent** | <https://arxiv.org/abs/2308.08780> | The nearest is bucketing demonstrations in DENSE-style in-context learning, which combines probabilities rather than pooling decoded candidates and never trains. The coverage claim hasn't been measured on ARC |
| 1c. Leaving out demonstrations to build training samples and targets | **Published** | <https://arxiv.org/html/2411.07279v1> | Standard since 2024 and already in our pipeline. Claim nothing here |
| 1d. Training a single ARC adapter on fewer demonstrations | **Published, negative** | <https://ironbar.github.io/arc24/modeling/Iteration_04_test_time_fine-tuning/> | Tried with n-1 and n-2. n-2 didn't improve, described as "just removing input information". This is the prior the proposal has to beat |
| 2a. Agreement in an ensemble as a signal of confidence and routing per ARC task | **Published** | <https://arxiv.org/html/2606.31543v1> | Done, with strong numbers, 97.3% on tasks that stopped early against 70.8% on full runs. Don't present routing by agreement as new |
| 2b. Disagreement among models adapted to different evidence from the same task, as a detector of underdetermination | **Absent** | <https://arxiv.org/abs/2605.11328> | UG-TTT ensembles test-time LoRA adapters and reads disagreement as epistemic uncertainty, but it creates variety with a nuclear-norm penalty on identical data, in RL for scientific discovery. Changing the evidence is the open part |
| 3a. A committee built from subsets of data, with disagreement as information | **Published in 1998** | <https://www.semanticscholar.org/paper/13f3c1bc860eafce2af81d38d885ab9e0a4be01f> | This is query by bagging. The theory isn't a contribution, so cite it as the lineage and claim only the setting |
| 3b. Ensembles over subsets of data capture uncertainty that ensembles over seeds don't | **Published, disputed** | <https://arxiv.org/pdf/2605.18329> | It supports the framing, but Lakshminarayanan et al. (<https://arxiv.org/pdf/1612.01474>) found bagging made deep ensembles worse, and that's the default a reviewer holds |
| 3c. Ensembling over subsets of a 2-to-5-shot support set, with trained adapters | **Absent** | <https://arxiv.org/abs/2411.02830> | MoICL and DENSE use subsets as experts by prompting, on classification, with no readout of disagreement. I found no version with trained adapters at N under 10, and it has no accepted name |
| 4. Keeping every hypothesis consistent with the demonstrations, symbolically | **Published** | <https://homes.cs.washington.edu/~pedrod/papers/kcap03b.pdf> | Version space algebra, pragmatic synthesis, description length and Solomonoff ranking all do this. All of them vary hypotheses with the evidence fixed. The proposal varies evidence with the learner fixed, which is the reverse, and that's what makes it work without a space of hypotheses you can enumerate |
| 5. ARC demonstrations don't pin down their rules | **Related work only** | <https://arxiv.org/html/2603.05099> | Acknowledged by ARC Prize in its reasons for pass@2, named precisely by ARC-TGI for generated tasks, and visible in data on how people disagree. Never measured on the official ARC sets |

## What exactly would be new

Three things, most defensible first.

**1. The measurement, not the method.** No published work measures how many ARC tasks allow more than
one rule consistent with their demonstrations. Disagreement between adapters that each leave out one
demonstration is a proposed instrument for that measurement, and the instrument plus the number is a
cleaner contribution than a pipeline improvement. It pairs directly with the competitor's own finding
that final training loss doesn't separate solved tasks from unsolved ones. That result says the adapter
fits the evidence and still fails, and this is the natural next experiment. Framed as diagnosis, the
prior art is thin. Framed as a new ensembling trick, the prior art is thick.

**2. The argument about the setting.** With N from 2 to 5, deleting one demonstration removes 20 to 50%
of the evidence. Every result in the literature comparing bagging with variety from seeds was found
where the subsets are nearly identical and the members converge on nearly the same function, which is
why Lakshminarayanan et al. found bagging useless. No paper I found tests whether that conclusion flips
as the support set shrinks toward the point where the rule stops being identifiable. That's a genuinely
open question, and it can be checked.

**3. The difference between changing the evidence and changing the view.** This is the sharpest point
specific to ARC, and also the sharpest risk. The ARChitects' augmentations already include reordering
the examples, so the pipeline already ensembles over reshuffled demonstrations. A reviewer will ask why
deleting a pair is any different from reordering them. The answer has to be that reordering keeps all
the information in the evidence, so it probes how sensitive the decoder is, while deleting a pair
changes what can be identified, so it probes the specification. That answer is right, but it's an
argument and not a result until pooling the candidates shows leave-one-out adapters producing correct
grids that the adapter trained on everything never produces.

What isn't new, and should be conceded early rather than defended, is the theory. "A family of models
trained on different evidence carries information no single model's likelihoods contain" is the founding
claim of query by committee and the explicit design of query by bagging. Presenting it as a new insight
is the fastest way to lose a reviewer.

## The closest published work a reviewer would cite against it

In the order it's likely to come up.

1. **Akyurek et al., arXiv 2411.07279.** "Leave-one-out training at test time for ARC was published in
   2024." It's the most likely first objection and the easiest to answer, but only if the difference
   between training targets and separate models is made explicitly and early.
   <https://arxiv.org/html/2411.07279v1>
2. **Riaz et al., UG-TTT, arXiv 2605.11328.** "An ensemble of test-time LoRA adapters, with per-token
   disagreement as mutual information isolating epistemic uncertainty, in 2026." It's the closest
   published system in shape. Distinguish it on where the variety comes from (a nuclear-norm penalty on
   identical data, not changed evidence), on granularity (an exploration bonus per token, not a verdict
   on ambiguity per task), and on field. <https://arxiv.org/abs/2605.11328>
3. **Abe and Mamitsuka, ICML 1998, query by bagging.** "The mechanism has had a name for nearly thirty
   years." Cite it first yourself.
4. **Lakshminarayanan et al., arXiv 1612.01474.** "Bagging made performance worse, and random
   initialization is enough." That's the empirical prior against. Answer it with the argument about
   N from 2 to 5, and note that "Lost in the Folds" (arXiv 2605.18329) already disputes the equivalence
   for estimating uncertainty specifically.
5. **Hong et al., MoICL, arXiv 2411.02830, and Khalifa et al., DENSE, arXiv 2308.08780.** "Subsets of
   demonstrations as experts is published." Distinguish on trained adapters against prompting, on the
   2-to-5-shot induction setting, and on diagnosing against combining.
6. **Franzen et al., arXiv 2505.07859.** "Your own pipeline already ensembles over reordered examples."
   That's the most dangerous objection specific to ARC, because it comes from inside the method. See
   point 3 above.
7. **Land, arXiv 2606.31543.** "Agreement in an ensemble as a confidence and routing signal on ARC,
   with better numbers than you'll get." Concede the routing idea entirely and claim only the instrument
   with controlled variety.
