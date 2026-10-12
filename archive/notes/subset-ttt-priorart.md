# Prior-art check: leave-one-demonstration-out adapter families for ARC test-time training

Date: 2026-08-05. Scope: a targeted novelty check on one specific proposal, not on the general area of test-time training or ensembling.

## The proposal being checked

For each ARC task with N demonstration pairs (N is 2 to 5), train N separate per-task LoRA adapters, where adapter i is trained with demonstration i entirely removed from its training evidence. Then (a) union the candidate grids decoded from all N adapters to raise pool coverage, and (b) read cross-adapter agreement or disagreement on the test input as a signal that the demonstrations underdetermine the rule.

Two things must be held apart throughout, because almost every false positive in this search collapses them:

- **LOO as context construction.** For each pair j, build a training sample whose prompt holds the other N-1 pairs and whose target is pair j. Every sample lands in the training set of a single adapter. This is standard and our pipeline already does it.
- **LOO as model construction.** Train N distinct adapters, each of which never sees one pair at all, in any role, prompt or target. This is the proposal.

Also held apart: perturbing the **order** or **view** of the evidence (example reordering, D8 symmetries, color permutations) versus perturbing the **content** of the evidence (deleting a pair). The former is ubiquitous in ARC. The latter is what is being proposed.

---

## 1. ARC-specific: separate per-task adapters trained on different demonstration subsets

### Akyurek et al., "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning" (arXiv 2411.07279)

This is the paper a reviewer reaches for first, and it is the one most at risk of being misread as prior art. It is not.

Its leave-one-out is context construction. The paper states, of the synthetic TTT data: "By excluding the jth example pair from the training examples, we can create the following synthetic task." Those synthetic tasks all go into one training set. On adapter count the paper is explicit: "Instead of learning a single LoRA adapter for all tasks in the test set, we learn an individual task-specific LoRA adapter for each task. That is, we obtain N different LoRA adapters, where N is the number of test tasks."

Note the collision of symbols. Their N is the number of **tasks**. The proposal's N is the number of **demonstrations within one task**. One adapter per task there; N adapters per task here. There is no ensemble over adapters trained on different demonstration subsets anywhere in the paper. Their diversity at inference comes from geometric transformations plus a two-stage hierarchical vote (top-3 within each transformation type, then a global vote), all decoded from the same single adapter.

Source: https://arxiv.org/html/2411.07279v1 and https://ekinakyurek.github.io/papers/ttt.pdf

### The ARChitects (2024 winner, 2025 second place)

2024: a LoRA adapter trained on the evaluation set's examples across multiple tasks simultaneously. 2025: "The test-time training workflow was changed to run separately for each task, instead of fine-tuning the model on multiple tasks simultaneously as in the 2024 contest." Either way, one adapter per task at most, trained on the full demonstration set. Diversity is entirely on the inference side: the product-of-experts score aggregates P(aug(s) | aug(p)) over 16 augmented views, where the augmentations are D8 symmetries, color permutations, and **example reordering**. Reordering is the closest they come to touching the demonstration set, and it is a permutation, not a deletion.

Sources: https://github.com/da-fr/arc-prize-2024, https://arxiv.org/html/2505.07859, https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/

### NVARC (2025 winner)

Per-task LoRA rank-256, one adapter per task. Candidate diversity comes from batched DFS decoding and from 8 fixed augmentations used for rescoring. Their post-deadline selector, score(s) = DFS visit count across augmentations plus mean log-likelihood, is a frequency-plus-likelihood signal computed from one adapter under many views. No demonstration subsetting.

Sources: https://github.com/1ytic/NVARC, https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc

### ironbar / Omni-ARC (2024 third place). The nearest ARC-specific negative.

This is the closest anyone in the public ARC record has come, and it is worth reading carefully because it is a partial measurement against the proposal. In his test-time fine-tuning iteration he ablated the TTT training-set construction along a subset axis, listing variants "train with n-1 samples" and "train with n-2 samples" alongside the augmentation variants. The n-1 row reports val loss 0.1398 and val accuracy 7.20 percent. His conclusion on going further: "Training with n-2 samples did not improve over training with n-1 samples. This matched my intuition because we are not increasing the number of tasks, we are just removing input information."

Three things follow. First, this is still a single adapter, so it is not the proposal. Second, it is an ablation of how much evidence one adapter should be trained on, and the answer trended the wrong way. Third, and most useful, "we are just removing input information" is the exact objection the proposal has to answer: it is the null hypothesis that N reduced-evidence adapters are simply N weaker adapters.

Source: https://ironbar.github.io/arc24/modeling/Iteration_04_test_time_fine-tuning/

### Survey and technical-report sweeps

Three independent secondary sources were checked for the mechanism and none contains it.

- "The ARC of Progress towards AGI: A Living Survey of Abstraction and Reasoning" (2026), https://arxiv.org/html/2603.13372v1. Test-time adaptation is discussed at length and multiple-hypothesis maintenance appears in its taxonomy ("Multiple hypotheses: Diverse candidates maintained; selected via demo consistency"), but leave-one-out over demonstration pairs producing separate models is absent.
- ARC Prize 2025 Technical Report, https://arxiv.org/html/2601.10904v1. Absent.
- ARC Prize 2024 Technical Report, https://arxiv.org/pdf/2412.04604. Absent.

Kaggle discussion sweeps across the 2024, 2025 and 2026 competition forums surfaced no thread proposing or reporting per-demonstration-subset adapter families.

**Verdict for item 1: ABSENT.** The mechanism as specified is not in the public ARC record. The confusable neighbour, LOO context construction inside one adapter, is universal.

---

## 2. ARC-specific: ensemble disagreement as a per-task confidence, routing, or ambiguity signal

This is the item where the proposal is least novel in spirit, though not in mechanism.

### Land, "Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2" (arXiv 2606.31543, 2026)

Agreement is used directly as a per-task confidence and compute-routing signal: "if early-stage candidates already show strong agreement (multiple candidates converging on the same output), the system terminates early." The separation is large. Early-stopped instances scored 97.3 percent accuracy (36 of 37); instances that ran the full 29-candidate sweep scored 70.8 percent (92 of 130). The paper also interprets disagreement as evidence about the task rather than about the model, and its single early-stop failure is attributed to genuine underdetermination: "All models confidently assume the simpler interpretation ('legend is the same'), while the ground truth requires the more complex one."

The difference from the proposal is the source of ensemble diversity. Land's members are heterogeneous foundation models across text, image and code modalities, explicitly "independent candidates across heterogeneous reasoning modalities," not one model adapted to different slices of the same evidence. Disagreement there confounds model-capability differences with task ambiguity. Disagreement across LOO adapters of one base model would not, which is the proposal's actual argument and it is a real one.

Source: https://arxiv.org/html/2606.31543v1

### Agreement-as-selection in the TTT lineage

The ARChitects' product of experts and MindsAI's AIRV vote are both agreement computations, but agreement across augmented **views** of one adapted model, used to rank candidates within a task, not to characterise a task. NVARC's DFS-visit-count selector is the same shape. None of these is a per-task confidence readout and none of them varies the evidence.

Sources: https://arxiv.org/html/2505.07859, https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution

### Per-task success prediction without disagreement

"Structural Grid Descriptors Predict Within-Task Solver Success on ARC-AGI" (Pendharkar, 2026) predicts run success from 13 hand-crafted descriptors of intermediate grids, reaching within-task AUC 0.885 and saving 33.6 percent of beam-search compute at 98.9 percent solve-rate retention. It is an existence proof that per-task and per-run confidence signals pay for themselves in the ARC compute budget, and a competing baseline: it gets there from grid structure, with no ensemble at all. It does not analyse demonstration count, underdetermination, or ensemble disagreement.

Source: https://arxiv.org/html/2606.09026v2

**Verdict for item 2: PUBLISHED for the general idea, ADJACENT for the specific instrument.** Ensemble agreement as an ARC confidence and routing signal is published with strong numbers. Disagreement among models adapted to *different evidence from the same task*, as a purpose-built ambiguity detector, is not.

---

## 3. General ML: subset ensembles in tiny-support-set regimes, and what it is called

### The mechanism has a 1992-1998 name

Query by committee (Seung, Opper and Sompolinsky, 1992) trains multiple hypotheses and uses their disagreement as an information signal. Query by bagging (Abe and Mamitsuka, "Query learning strategies using boosting and bagging", ICML 1998) is the version where committee diversity is manufactured by **training members on different data subsets**, which is precisely the proposal's generator. Standard framing, per the modAL documentation: "To create and enhance disagreement among committee members, the training set can be changed for each estimator (e.g., via bagging) or by varying the configuration of the estimator."

So the abstract claim, that a committee trained on different evidence carries information a single model's likelihoods do not, is the founding claim of query-by-committee and is thirty years old. The proposal cannot claim it as a theoretical contribution. It can claim the regime.

Sources: https://www.semanticscholar.org/paper/Query-Learning-Strategies-Using-Boosting-and-Abe-Mamitsuka/13f3c1bc860eafce2af81d38d885ab9e0a4be01f, https://modal-python.readthedocs.io/en/latest/content/models/Committee.html

### The direct counter-evidence a reviewer will raise

Lakshminarayanan, Pritzel and Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (arXiv 1612.01474, NIPS 2017) ran exactly this comparison for deep nets and found against data subsetting: bagging deteriorated performance, and random initialisation plus data shuffling on the **full** dataset was sufficient. This is the single strongest published prior against the proposal's premise, and it is a highly cited default assumption. Any writeup must confront it directly.

Source: https://arxiv.org/pdf/1612.01474

### The direct counter-counter-evidence

"Lost in the Folds: When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation" (Kirscher et al., arXiv 2605.18329, 2026) argues the two are not interchangeable: cross-validation ensembles (members trained on different data folds) and seed-based deep ensembles capture different components of uncertainty and are complementary rather than equivalent. This supports the proposal's framing at the level of principle. Caveat: the finding was extracted from the PDF and the domain is medical imaging segmentation, not few-shot induction, so it supports the framing rather than the regime.

Source: https://arxiv.org/pdf/2605.18329

### Ensembling over support-set subsets specifically

Two papers do this at the prompt level and are the closest general-ML prior art to part (a).

- **DENSE**, Khalifa et al., "Exploring Demonstration Ensembling for In-context Learning" (arXiv 2308.08780). Demonstrations are split into buckets, the model is run separately on each bucket, and the output probabilities are combined. Weighted max ensembling beat vanilla concatenation by up to 2.4 average points across 12 tasks on GPT-J. Source: https://arxiv.org/abs/2308.08780
- **MoICL**, Hong et al., "Mixtures of In-Context Learners" (arXiv 2411.02830, ACL 2025). Treats subsets of demonstrations as experts and learns a weighting function that merges their output distributions. Gains on 5 of 7 classification datasets, up to +13 percent over ICL, and notably robust to noisy demonstrations (up to +38 percent) and imbalanced ones (up to +49 percent). Source: https://arxiv.org/abs/2411.02830

Both differ from the proposal on three axes at once: they condition by prompting rather than train separate adapters, they target classification with abundant labelled pools rather than 2-to-5-shot grid induction, and neither reads disagreement between subset-experts as a signal about the specification. Both combine, neither diagnoses.

### The closest published system overall: UG-TTT

"Epistemic Uncertainty for Test-Time Discovery" (Riaz et al., arXiv 2605.11328, 2026) maintains "a small ensemble of low-rank adapters over a frozen base model", computes "per-token disagreement, quantified as the mutual information between ensemble predictions and weight hypotheses", and uses it to "isolate epistemic uncertainty" and identify "positions where insufficient coverage leads to adapter divergence rather than intrinsic problem difficulty."

That sentence is close enough to the proposal's part (b) to be uncomfortable. The decisive difference is where diversity comes from: "A nuclear norm regularizer ensures the adapters remain distinct from one another, thereby preserving the exploration signal throughout training." The adapters see the same data and are forced apart by a penalty term. There is no evidence perturbation. It is also RL for scientific discovery, uses disagreement as a per-token exploration bonus inside policy-gradient training, and is not an ARC system.

Source: https://arxiv.org/abs/2605.11328

### LoRA ensembles for uncertainty

A small literature exists (arXiv 2310.00035; Balabanov and Linander, arXiv 2402.12264; LoRA-Ensemble, arXiv 2405.14438). Uniformly, ensemble members differ by random initialisation and training stochasticity, not by training data. They establish that LoRA ensembles are a cheap and legitimate posterior approximation, which the proposal can lean on for its uncertainty claim, but none of them varies the evidence.

**Verdict for item 3: PUBLISHED for the general mechanism and its theory, ABSENT for the tiny-support-set instantiation.** The mechanism is query-by-bagging. There is no established name for leave-one-out ensembling over a support set of size 2 to 5, and no paper found that does it with trained adapters in that regime.

---

## 4. Program synthesis: the symbolic analogue of keeping all consistent hypotheses

The symbolic tradition has solved the representational half of this problem for twenty-five years, which sharpens what is actually being proposed.

- **Version space algebra** (Lau and Domingos, ICML 2000; the algorithmic core of FlashFill). A version space is a compact structure implicitly representing the **set of all programs in the DSL consistent with the examples**, updated as each new example arrives. The framing is explicit that "examples, though often easy to provide, are an ambiguous specification of the user's intent" and that user-specified examples leave multiple programs satisfying the spec. Sources: https://homes.cs.washington.edu/~pedrod/papers/kcap03b.pdf, https://www.semanticscholar.org/paper/58b2550804f24d5bb0b8cfe6815623be8760ffe1
- **Pragmatic program synthesis** (Pu, Ellis et al., NeurIPS 2020; and "Amortizing Pragmatic Program Synthesis with Rankings", arXiv 2407.02499). Given many consistent programs, use recursive Bayesian speaker-listener reasoning to infer which one the demonstrator intended. This is the published answer to "the demos underdetermine the rule, now what". Sources: https://proceedings.neurips.cc/paper/2020/file/99c83c904d0d64fbef50d919a5c66a80-Paper.pdf, https://arxiv.org/html/2407.02499
- **ARC-specific multi-hypothesis work.** MDL and description-length solvers (Ferré, arXiv 2112.00848) score competing consistent hypotheses by simplicity. "Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty" (arXiv 2512.17145) builds an explicit distribution over LLM-generated hypotheses weighted by simplicity and consistency, applies it to Mini-ARC per-cell prediction, and is calibrated better than Bayesian model averaging because it spreads mass rather than concentrating it. BARC's induction arm (arXiv 2411.02272) samples many Python programs and retains those consistent with the demonstrations.

**How this differs from the proposal.** Every one of these enumerates or samples the hypothesis space and then **filters by consistency with the full demonstration set**. Evidence is held fixed and hypotheses are varied. The proposal inverts this: hold the learner fixed and vary the evidence, then infer the width of the hypothesis space from how much the learner's output moves. That inversion is what makes it applicable to a neural transducer that has no enumerable hypothesis space. No ARC symbolic work found perturbs the demonstration set to estimate hypothesis-space width.

**Verdict for item 4: PUBLISHED for explicit multi-hypothesis ARC solving, ABSENT for evidence-perturbation as the hypothesis-width probe.**

---

## 5. Published evidence that ARC tasks are underdetermined by their demonstrations

The evidence is real but thinner and softer than the proposal would want.

- **ARC Prize's own position.** Pass@2 exists in part for this reason: ARC-AGI uses pass@2 "to account for the fact that certain tasks have explicit ambiguity and require two guesses to disambiguate", and to catch unintentional ambiguity or dataset mistakes. Source: https://arcprize.org/guide/1
- **ARC-AGI-2 human calibration.** 407 unique participants, 515 sessions, 1,848 unique task test pairs, 13,405 attempts. A task advanced "only if at least two independent participants solved one or more sub-pairs within their first two attempts", and 100 percent of tasks were solved by at least two humans, with 2 to 10 humans per task. Note what this does and does not establish: it guarantees solvability, not uniqueness. A task can be solved by two humans and still admit a second defensible rule. The technical report does not discuss ambiguity or underdetermination directly. Source: https://arxiv.org/html/2505.11831v2
- **H-ARC** (Legris et al., arXiv 2409.01374). 1,729 humans across all 800 ARC-1 training and evaluation tasks. Average human performance 73.3 to 77.2 percent; 98.8 percent of tasks solved by at least one person; over 90 percent solved by at least three participants. The residual failure mass is consistent with underdetermination but is not attributed to it. Source: https://arxiv.org/abs/2409.01374
- **Johnson, Vong, Lake and Gureckis, "Fast and flexible: Human program induction in abstract reasoning tasks"** (arXiv 2103.05823, CogSci 2021). Behavioural study on 40 ARC tasks: 84 percent of tasks solved per participant on average, and 65 percent of tasks solved by more than 80 percent of participants. The complement is the interesting number. Roughly a third of tasks show substantial human divergence. Source: https://arxiv.org/abs/2103.05823
- **ARCTraj** (arXiv 2511.11079). Human solution trajectories show some tasks with strong convergence on the same key transformations and others with high diversity, participants selecting different substructures or applying distinct operations, with high-uniqueness tasks indicating "flexible or ambiguous reasoning spaces."
- **ARC-TGI** (Lehmann et al., arXiv 2603.05099, 2026). The most explicit published statement of the failure mode: "Independent pair sampling can yield ambiguous episodes (e.g., missing critical variation, unseen test features, or degenerate shortcuts)", and existing generators "do not reliably guarantee that the set of training examples disambiguates the intended rule". Their fix is a design constraint they name "Disambiguating coverage: ensure training pairs collectively include the variations required to infer the rule". Important limitation: this is about task **generation**, and they offer no estimate of how often the official ARC sets are affected. Source: https://arxiv.org/html/2603.05099
- **A concrete instance.** Land documents a task where every model in a heterogeneous ensemble confidently picked the simpler of two readings and the ground truth required the more complex one. Source: https://arxiv.org/html/2606.31543v1

**The gap.** No published study was found that measures what fraction of ARC tasks admit more than one rule consistent with the demonstrations. ARC-TGI names the phenomenon and designs it away in synthetic tasks. Nobody has quantified it on ARC-AGI-1 or ARC-AGI-2. That measurement is itself unclaimed territory.

**Verdict for item 5: ADJACENT.** Underdetermination is acknowledged by ARC Prize, named precisely by ARC-TGI, and visible in human-behaviour data, but never quantified on the official sets.

---

## Verdict table

| Claim | Verdict | Strongest evidence URL | What it means for novelty |
|---|---|---|---|
| 1a. N separate per-task TTT adapters, each with one demonstration removed, for one ARC task | **ABSENT** | https://arxiv.org/html/2411.07279v1 | Not in the ARC record. Akyurek's LOO is context construction inside one adapter and his "N adapters" means one per task. The distinction must be made in the first paragraph of any writeup or it will be dismissed as a rediscovery. |
| 1b. Union of candidate pools across subset-trained adapters to raise ARC coverage | **ABSENT** | https://arxiv.org/abs/2308.08780 | Nearest is DENSE-style demonstration bucketing in ICL, which combines probabilities rather than unioning decoded candidates and never trains. The coverage claim is unmeasured for ARC. |
| 1c. LOO over demos as TTT context/target construction | **PUBLISHED** | https://arxiv.org/html/2411.07279v1 | Standard since 2024. Already in our pipeline. Claim nothing here. |
| 1d. Training a single ARC TTT adapter on a reduced demonstration set | **PUBLISHED (negative)** | https://ironbar.github.io/arc24/modeling/Iteration_04_test_time_fine-tuning/ | Tried as n-1 and n-2 ablations; n-2 did not improve, described as "just removing input information". This is the prior the proposal has to beat. |
| 2a. Ensemble agreement as a per-task ARC confidence and compute-routing signal | **PUBLISHED** | https://arxiv.org/html/2606.31543v1 | Done with strong numbers: 97.3 percent on early-stopped vs 70.8 percent on full-coverage. Do not present agreement-based routing as new. |
| 2b. Disagreement among models adapted to *different evidence from the same task*, as an underdetermination detector | **ABSENT** | https://arxiv.org/abs/2605.11328 | UG-TTT ensembles test-time LoRA adapters and reads disagreement as epistemic uncertainty, but manufactures diversity with a nuclear-norm regulariser on identical data, in RL for scientific discovery. The evidence-perturbation source is the open part. |
| 3a. Committee-from-data-subsets, disagreement as information | **PUBLISHED (1998)** | https://www.semanticscholar.org/paper/13f3c1bc860eafce2af81d38d885ab9e0a4be01f | Query by bagging. The theoretical framing is not a contribution. Cite it as the lineage and claim only the regime. |
| 3b. Data-subset ensembles capture uncertainty that seed ensembles do not | **PUBLISHED, contested** | https://arxiv.org/pdf/2605.18329 | Supports the framing. But Lakshminarayanan et al. (https://arxiv.org/pdf/1612.01474) found bagging deteriorated deep-ensemble performance, and that is the default assumption a reviewer holds. |
| 3c. Ensembling over support-set subsets in a 2-to-5-shot regime, with trained adapters | **ABSENT** | https://arxiv.org/abs/2411.02830 | MoICL and DENSE do subsets-as-experts by prompting, on classification, with no disagreement readout. No trained-adapter version at N under 10 was found. No accepted name exists for it. |
| 4. Keeping all hypotheses consistent with the demos, symbolically | **PUBLISHED** | https://homes.cs.washington.edu/~pedrod/papers/kcap03b.pdf | Version space algebra, pragmatic synthesis, MDL and Solomonoff ranking all do this. All vary hypotheses under fixed evidence. The proposal varies evidence under a fixed learner, which is the inverse and is what makes it work without an enumerable hypothesis space. |
| 5. ARC tasks are underdetermined by their demonstrations | **ADJACENT** | https://arxiv.org/html/2603.05099 | Acknowledged by ARC Prize via the pass@2 rationale, named precisely by ARC-TGI for generated tasks, visible in human divergence data. Never quantified on the official ARC sets. |

---

## What would be novel, precisely

Three things, in descending order of defensibility.

**1. The measurement, not the method.** No published work quantifies how many ARC tasks admit more than one rule consistent with their demonstrations. Cross-adapter disagreement under leave-one-demonstration-out is a *proposed instrument* for that measurement, and the instrument plus the number is a cleaner contribution than the pipeline improvement. This pairs directly with the competitor's own finding that final TTT training loss does not separate solved from unsolved tasks: that result says the adapter fits the evidence and still fails, and this is the natural follow-up experiment. Frame the work as diagnosis and the prior art is thin. Frame it as a new ensembling trick and the prior art is thick.

**2. The regime argument.** With N of 2 to 5, deleting one demonstration removes 20 to 50 percent of the evidence. Every result in the bagging-versus-seed-diversity literature was established where subsets are near-identical and members converge to nearly the same function, which is why Lakshminarayanan et al. found bagging useless. The claim that the conclusion inverts as the support set shrinks toward the identifiability boundary is not something any found paper tests. It is a genuinely open question and it is checkable.

**3. The evidence-perturbation versus view-perturbation distinction.** This is the sharpest ARC-specific point and also the sharpest ARC-specific risk. The ARChitects' augmentation set already includes example reordering, so the pipeline already ensembles over permutations of the demonstration set. A reviewer will ask why deleting a pair is categorically different from reordering them. The answer has to be that reordering preserves the sufficient statistics of the evidence and therefore probes decoder sensitivity, while deletion changes what is identifiable and therefore probes the specification. That answer is correct but it is an argument, not a result, until the coverage-union experiment shows LOO adapters producing correct grids the full-evidence adapter never produces.

What is **not** novel, and should be conceded early rather than defended: the theoretical framing. "A family of models trained on different evidence carries information no single model's likelihoods contain" is the founding claim of query by committee and the explicit design of query by bagging. Presenting it as a new insight is the fastest way to lose a reviewer.

## Nearest published work a reviewer would cite against it

In the order they are likely to come up.

1. **Akyurek et al., arXiv 2411.07279.** "Leave-one-out TTT for ARC, published 2024." The most likely first objection and the easiest to answer, but only if the training-target versus separate-model distinction is made explicitly and early. https://arxiv.org/html/2411.07279v1
2. **Riaz et al., UG-TTT, arXiv 2605.11328.** "Ensemble of test-time LoRA adapters, per-token disagreement as mutual information isolating epistemic uncertainty, 2026." The closest published system by shape. Distinguish on diversity source (nuclear-norm regulariser on identical data, not evidence perturbation), on granularity (per-token exploration bonus, not per-task ambiguity verdict), and on domain. https://arxiv.org/abs/2605.11328
3. **Abe and Mamitsuka, ICML 1998, query by bagging.** "The mechanism has had a name for nearly thirty years." Cite it first yourself.
4. **Lakshminarayanan et al., arXiv 1612.01474.** "Bagging deteriorated performance; random initialisation is sufficient." The empirical prior against. Answer with the N of 2 to 5 regime argument, and note that "Lost in the Folds" (arXiv 2605.18329) already disputes the equivalence for uncertainty estimation specifically.
5. **Hong et al., MoICL, arXiv 2411.02830, and Khalifa et al., DENSE, arXiv 2308.08780.** "Subsets of demonstrations as experts is published." Distinguish on trained adapters versus prompting, on the 2-to-5-shot induction regime, and on diagnosis versus combination.
6. **Franzen et al., arXiv 2505.07859.** "Your own pipeline already ensembles over example reorderings." The most dangerous ARC-specific objection because it comes from inside the method. See point 3 above.
7. **Land, arXiv 2606.31543.** "Ensemble agreement as an ARC confidence and routing signal, with better numbers than you will get." Concede the routing idea entirely and claim only the controlled-diversity instrument.
