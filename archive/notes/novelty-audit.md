# Novelty audit of the public state of the art on ARC-AGI-2

Written on 4 August 2026. It covers the ARC Prize 2025 winners' writeups, the ARChitects lineage,
public research on ARC-AGI-2 from 2025 and 2026, and a check for prior art on six candidate levers. I
treat any score shown beside a notebook as unverified unless it comes from a paper, the final private
leaderboard or ARC Prize's verified semi-private leaderboard. Where I couldn't verify something, I say
so.

For reference, our pipeline is the Sorokin and ARChitects lineage. That means per-task test-time
training of a rank-256 LoRA on a 4B grid model, depth-first decoding with a cumulative-probability
threshold of 0.2, and selection based on augmentations. It scored 30.97 on the hidden set. The 2026
ARC-AGI-2 public leaderboard reads 67.50 for nvbanana, 44.86 for rabbithole, then 37.22, then a dense
band from 33.06 to 34.44 that looks like forks of the public NVARC notebook
(sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue).

## 1. What the winners actually did

### 1.1 The ARChitects in 2024, and the Product-of-Experts paper

This is the base recipe, with exact numbers. The sources are <https://github.com/da-fr/arc-prize-2024>
and the academic version, Franzen, Disselhoff and Hartmann, "Product of Experts with LLMs. Boosting
Performance on ARC Is a Matter of Perspective" (<https://arxiv.org/html/2505.07859>).

- They used Mistral-NeMo-Minitron-8B with the tokenizer and embeddings cut down to a small grid
  vocabulary, trained with Unsloth on one H100 for 98 GPU hours.
- Each task gets its own LoRA trained at test time, which takes 51 seconds a task on an RTX 4090 in
  the paper's setup.
- Decoding is depth first over the token tree, dropping any partial path whose accumulated
  probability falls below a threshold T. The paper's final T is 9%.
- Selection is a product of experts over 16 augmented views, covering the eight symmetries, colour
  permutations and reordered examples. A candidate's score multiplies its probability under each
  augmentation, which amounts to a geometric-mean ensemble of likelihoods.
- The paper only compares depth-first search with beam search, random sampling and greedy decoding.
  Beam search with four beams matches depth first at T of 9% but needs about twice the memory.
  Nowhere does it test best-first or A*-style enumeration.
- The result was 71.6% on the ARC-AGI-1 public evaluation at about two cents a task.

### 1.2 NVARC, first in 2025

They scored 24.03 private, 27.64 public during the competition and 29.72 after the deadline. The
sources are the paper PDF in <https://github.com/1ytic/NVARC> (nvarc_2025.pdf, which I read in full),
the Kaggle writeup at <https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc>, an
interview at <https://trelis.substack.com/p/nvarc-2025-arc-prize-winners> and NVIDIA's blog at
<https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/>.

Here's what they did beyond the ARChitects' base recipe.

1. **Synthetic data in four stages.** First they collected human descriptions of solutions in plain
   language. That meant H-ARC, with attempts from more than 1,700 people, plus 160 BARC descriptions,
   which covered 716 ARC-AGI-2 training puzzles. They reformatted these into a five-part summary
   (how inputs are generated, the solution steps, a summary of the rule, the key insight and the
   concepts) using claude-opus-4 and gpt-oss-120b. Second, they mixed pairs of summaries in the style
   of INSTRUCT-SKILLMIX to make 266,593 harder summaries with gpt-oss-120b. Third, they generated
   Python programs that produce input grids, along with unit tests, and kept programs that produced at
   least 30 distinct grids passing their tests. That accepted 70% of easy mixes and 50% of mixes at the
   evaluation level, for 126,901 programs. Fourth, they generated about 20 programs per puzzle to
   produce output grids, and kept only sets that agreed with themselves, for instance 15 of 20 giving
   identical outputs. The end result was 103,253 verified synthetic puzzles with about 30 pairs each.
2. **Using the public evaluation set in their data on purpose.** They labelled 29 ARC-AGI-2 evaluation
   puzzles by hand, then had an LLM (claude-sonnet-4-5) write summaries for the other 91 evaluation
   puzzles and all 1,000 training puzzles. They say openly that they "leaked" evaluation descriptions
   into the synthetic set and still saw the public leaderboard track their local scores. Their local
   validation was contaminated by design, while the hidden set stayed clean because it doesn't
   overlap.
3. **Encoding grids through the Qwen3 chat template.** Each pair is a user turn with the input grid and
   an assistant turn with the output, so the effective vocabulary is 16 tokens. Those are the ten
   digits, a newline, user, assistant, the two chat markers and padding.
4. **A full fine-tune rather than a LoRA**, of Qwen3-4B-Thinking-2507 on 3.2 million augmented samples.
   Each real puzzle got 256 augmentations and each synthetic one 24 to 32. They used NeMo RL with a
   Megatron backend, packed sequences to 256,000 tokens, and ran on four nodes of eight H100s for 27
   hours. The mix in the paper is MINI-ARC, ConceptARC, RE-ARC, ARC-AGI-2 and their own synthetic data,
   which is about 90% of the samples.
5. **Test-time training** with a LoRA at rank 256 and alpha 32, in bf16, with gradient checkpointing
   and 4-bit quantization both removed, and Flash Attention 2 through Unsloth.
6. **Batched depth-first decoding** that scales almost linearly with batch size. They point out it's
   nondeterministic, because the logits change with what else is in the batch. They tried Thinking
   Machines' batch-invariant operations, which scored better locally but ran 17% slower on Kaggle, so
   they dropped them for the final submission.
7. **A change to rescoring.** Only eight augmentations, but the same eight for every candidate, so the
   scores can be compared across candidates.
8. **Better selection after the deadline.** A candidate's score became the number of times the search
   found it across augmentations, plus its mean log-likelihood. That's a hybrid of frequency and
   likelihood.
9. **TRM as a second source of candidates.** They retrained TRM (batch 3,072, learning rate 3e-4, 3,000
   synthetic plus 1,200 real puzzles, 256 augmentations, 10,000 epochs and 24 hours on eight H100s).
   They then trained it on Kaggle at test time in about two hours, resetting the puzzle-embedding
   table to the mean pretrained embedding and running 2,000 epochs with 4 H cycles and 10 halting
   steps. TRM made 10 attempts per puzzle, which went into the same candidate pool and were scored by
   the Qwen3 model like everything else. TRM alone scored 7.5, or 10.0 after the deadline with 4,000
   epochs. The ensemble was mixed. The 2B went from 21.53 to 22.50 in late tests, the 4B at 10 hours
   stayed at 27.22, and the one ensemble submitted during the competition scored lower, 20.28 against
   21.53.
10. **A 2B result.** Qwen3-VL-2B-Instruct, using only its language model, scored 22.22 public in 6 hours
    10 minutes, against 30 of 120 locally and 29.72 public for the 4B in 12 hours.
11. **A note on measurement from the winners themselves.** They "saw 1-2 points variability in the
    scores across different re-runs of the same notebook."

### 1.3 The ARChitects, second in 2025

They scored 16.53 private and 21.67 public. The source is
<https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>.

- They switched from the autoregressive Minitron-8B to LLaDA-8B, a masked diffusion language model,
  with a 2D variant of RoPE. Grids are placed on a 32 by 32 canvas to assign positions, and the
  padding is removed afterwards.
- The initial fine-tune was a rank-512 LoRA for 175,000 steps, taking 39 hours on eight H100s. At test
  time they trained a rank-32 LoRA for 128 steps per task.
- They sample by recursive masked diffusion. Mask embeddings are added to every position as a soft
  mask, and the model refines iteratively with its logits fed back in, for 102 steps in two rounds with
  a cold restart between them.
- To select, they take either the candidate visited most often during refinement, or the one closest
  to the final predicted distribution.
- A separate LLaDA-8B was fine-tuned only to place delimiter tokens, which makes it a predictor of
  output shape, at 85% accuracy give or take 2.
- Before switching, they upgraded their autoregressive pipeline. They lowered the search cutoff from
  17% to 7%. They added speculative decoding inside the search, "using a simple heuristic to guess the
  next 16 to 32 tokens", for a 4.7 times speedup. They reused the cached keys and values for the
  examples and the challenge input across augmentations, for a 5.8 times speedup in scoring. And they
  raised the number of scoring augmentations from 8 to 32.
- They report about 30.5% on public evaluation against 21.67 on the public leaderboard, and admit
  signs of overfitting to the evaluation set.

### 1.4 MindsAI and Tufa Labs, third in 2025

They scored 12.64 private. The sources are
<https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution>,
<https://github.com/jcole75/arc_2025_mindsai> and <https://arxiv.org/html/2601.10904v1>.

- They fine-tune per task at test time, which they call TTFT, and use AIRV, short for augment,
  infer, reverse the augmentation and vote.
- The 2025 report lists "tokenizer dropout" and "new pretraining techniques", but there's no detailed
  public description. The closest public analogue is regularizing tokenization in the style of BPE
  dropout.
- The repo shows the model family, which is Salesforce CodeT5 in several sizes (codet5-small, 660m and
  a large "arcmega"), pretrained on the mindware/arc-agi-mega dataset of serialized tasks, with TPU
  training scripts included.

### 1.5 Where the 2026 competition stands

- nvbanana, Puget with a new teammate and NVARC's successor, was at 67.50 public on 30 July 2026, up
  from 39 in April. The evidence is <https://x.com/GregKamradt/status/2082936322508116098>,
  <https://x.com/GregKamradt/status/2049872125851529411> and the leaderboard at
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard>. There's no public writeup
  of what changed. The likely core is a bigger version of their flywheel of verified synthetic data,
  since that was what made the difference in 2025 and NVIDIA has kept releasing ARC data.
- rabbithole is at 44.86. I found no public information on their method.
- The band from 33.06 to 34.44 matches forks of the public NVARC submission notebook
  (sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue, linked from <https://github.com/1ytic/NVARC>).
  Our own controls, forking and running two notebooks that displayed 33.89 unchanged, scored 28.06 and
  28.47. That already showed the displayed scores aren't measurements of the published code.
- Several NVARC assets are fully public. These are the solution code and paper
  (github.com/1ytic/NVARC), the 103,000 synthetic puzzles and 3.2 million augmented samples (Kaggle
  datasets linked from the repo), and since March 2026 nvidia/Nemotron-SFT-ARC-AGI-v1 on Hugging Face.
  That last one holds 67,529 wide-coverage and 207,569 deep-coverage multi-turn reasoning traces over
  ARC-style puzzles, seeded from NVARC's augmented puzzles and filtered by rules to runs that reached
  the correct answer, under CC BY 4.0 (<https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1>).

## 2. Recent public research on ARC-AGI-2

I've labelled each score. Verified means ARC Prize's semi-private verified leaderboard, private means
Kaggle's final standings, and claimed means self-reported and unverified.

- **Land, "Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2"**
  (<https://arxiv.org/abs/2606.31543>, June 2026). It generates candidates independently in three
  modes, reasoning in text, images and code, using frontier APIs. A judge model then reads all the
  candidates' reasoning together in one long context and picks the winners. Reading them together
  rescues correct minority answers that a majority vote would kill, which added 7 tasks for 13% of
  the system's cost. It scored 72.9% verified at $38.99 a task (76.1 claimed on public evaluation),
  using API models (GPT-5.2 Pro and Gemini 3 Pro) and no local hardware. They released all the code
  and 7 million lines of run data, and documented negative results too. Prescriptive prompts and
  iterative refinement reduced diversity and hurt.
- **Poetiq** (<https://poetiq.ai/posts/arcagi_verified/>, December 2025). A meta-system that works
  with any model, wrapping Gemini 3 class models in loops that generate, get feedback and refine. It
  scored 54% verified at $30.57 a task, through APIs only. For comparison, Gemini 3 Deep Think on its
  own scored 45.1 verified at $77 a task.
- **Berman, "How I got the highest score on ARC-AGI again, swapping Python for English"**
  (<https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again>, September
  2025). Evolutionary search at test time where the genome is an instruction in plain English. It
  starts with 30 candidates, then runs five rounds revising the top five individually and five rounds
  combining them. Fitness is per-cell accuracy on the demonstrations, and sub-agents apply the
  instructions and return ASCII diffs. It scored 29.4% (verified at the time) with Grok-4, through APIs
  only.
- **The NVARC paper** (see 1.2) scored 24.03 private on four L4s in 12 hours, at about 20 cents a task.
- **The ARChitects' 2025 masked-diffusion solution** (see 1.3) scored 16.53 private on L4s.
- **Jolicoeur-Martineau, "Less is More. Recursive Reasoning with Tiny Networks"**, the TRM paper that
  won the 2025 paper award (<https://arcprize.org/blog/arc-prize-2025-results-analysis>). It's a
  7-million-parameter recursive network that keeps separate states for the answer and a latent, refined
  for up to 16 cycles. It claims 45 on ARC-1 and 8 on ARC-2, trained on augmented training data plus
  data in the style of the evaluation set, on machines with several H100s.
- **"Test-time Adaptation of Tiny Recursive Models"** (<https://arxiv.org/abs/2511.02886>). It
  fine-tunes the whole 7-million-parameter TRM per submission inside Kaggle's limits, for 12,500
  gradient steps after 48 hours of pretraining on four H100s. It claims about 10 on public evaluation
  and scored 6.67 verified on the semi-private set.
- **"Tiny Recursive Models on ARC-AGI-1. Inductive Biases, Identity Conditioning, and Test-Time
  Compute"** (<https://arxiv.org/abs/2512.11847>). An analysis of what actually drives TRM.
- **SOAR, Pourcel et al., "Self-Improving Language Models for Evolutionary Program Synthesis"**
  (<https://arxiv.org/abs/2507.14172>). It alternates evolutionary program search with fine-tuning on
  its own search traces in hindsight, with no DSL. It claims 52% on ARC-1, on a large multi-GPU budget.
- **CompressARC, Liao and Gu, "ARC-AGI Without Pretraining"** (<https://arxiv.org/pdf/2512.06104>).
  It compresses each task by minimum description length. A randomly initialized equivariant network is
  trained by gradient descent at test time to compress the task, with no pretraining and no search. It
  claims 20 on the ARC-1 evaluation and 4 on ARC-2, on one GPU in minutes per task.
- **VARC, Hu et al., "ARC Is a Vision Problem!"** (<https://arxiv.org/abs/2511.14761>). It treats ARC
  as translating one image into another on a canvas, using a plain ViT trained from scratch on ARC
  data plus per-task training at test time. It claims 60.4 on the ARC-1 public evaluation with an 18M
  model, close to the average person. ARC-2 numbers aren't the headline. It got an honourable mention
  in the 2025 awards. A follow-on, LoopViT (<https://arxiv.org/pdf/2602.02156>), applies looped
  transformers to visual ARC.
- **ARCANA** (<https://arxiv.org/html/2607.09059v1>, July 2026). A neurosymbolic loop of four agents.
  One builds scene graphs with slot attention, a conditional VAE proposes programs over a DSL of 47
  primitives, a symbolic executor checks them at several levels, and a reflective step refines them
  with counterfactual credit. It also filters for diversity with a k-DPP and trains a rank-8 LoRA at
  test time. It claims 32.5% on the ARC-AGI-2 semi-private set on four L4s in 12 hours with 48 million
  parameters. That's an extraordinary claim with no independent verification I could find, and it
  isn't on the verified leaderboard. Treat it as sceptically as a score shown beside a notebook until
  someone reproduces it.
- **"Multi-Perspective Transformers in ARC-AGI-2 Challenge"** (<https://arxiv.org/html/2605.01154>). A
  TinyLM of about 20M parameters, with product-of-experts scoring across geometric and colour views,
  test-time training with heavy dropout, and a hard filter that keeps only candidates that "reproduce
  the given training examples exactly". It claims 21.7% on "the evaluation set", which is ambiguous
  and unverified.
- **Surveys and diagnostics.** These are "The ARC of Progress towards AGI. A Living Survey"
  (<https://arxiv.org/html/2603.13372v1>), lewish.io's research review
  (<https://lewish.io/posts/arc-agi-2025-research-review>), which notes nobody has tried specialist
  learned models to guide branching in discrete program search, "Your Reasoning Benchmark May Not Test
  Reasoning" on perception bottlenecks (<https://arxiv.org/pdf/2512.21329>), and ARC Prize's own
  ablation of HRM, which found its gains came mostly from the outer refinement loop and augmentation
  rather than the hierarchical architecture (<https://arcprize.org/blog/hrm-analysis>).
- **The ARC Prize 2025 technical report** (<https://arxiv.org/html/2601.10904v1>). It names the year's
  theme "the refinement loop" and introduces "knowledge overfitting". Even a benchmark that resists
  direct memorization can be overfit when the public training data and the private test are alike and
  models have trained on public ARC data. The report says this is now happening with ARC-AGI-1 and 2,
  but it explicitly doesn't quantify it. That unmeasured gap is the opening for our contamination
  instrument.

## 3. Prior art on the six levers

| lever | verdict | evidence | what it means |
|---|---|---|---|
| a. Global best-first or A* search over the frontier, replacing depth-first token enumeration | Absent for ARC, with related work in general LLM decoding | <https://arxiv.org/html/2505.07859> compares depth first only with beam, greedy and sampling. <https://arxiv.org/pdf/2407.03951> (ULTS, a best-first likelihood tree search) covers NLP tasks only | A chance to publish first. An enumerator for grid decoding that works under any budget and adapts to it, with no ARC precedent to beat |
| b. Hard filtering of candidates by invariants mined from the demonstrations | Related work exists, but not this mechanism | <https://arxiv.org/html/2605.01154> filters candidates by exactly replaying the training pairs. <https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/> has a separate output-shape predictor | Replay filters and shape predictors exist. Mined maps of size, palette and component counts used as necessary conditions on a transductive pool are unpublished |
| c. Reranking one model's pool under a second, independently trained model | Related work, published | The NVARC paper in <https://github.com/1ytic/NVARC> scores TRM's attempts with Qwen3. <https://arxiv.org/abs/2606.31543> uses a frontier judge over candidates from several channels. <https://arxiv.org/html/2603.13372v1> describes Land's judge-based adjudication | A single cross-model scorer exists (NVARC). Selection by agreement between two models, or by a product of independent experts on a local pair, is unpublished in the Kaggle setting |
| d. A meta-learned or pretrained LoRA starting point for per-task training | Absent for ARC, related elsewhere, and the naive version is a published negative | <https://arxiv.org/pdf/2411.07279> gets 71.5 with per-task LoRA against 63 with a shared LoRA on ARC-1. <https://arxiv.org/pdf/2503.22352> and <https://arxiv.org/pdf/2510.10365> cover meta-LoRA in other fields | The published negative is one shared adapter used as is, not a MAML or Reptile-style starting point optimized so that k training steps land well. That exact lever is untried on ARC |
| e. Speculative decoding inside the search | Published in a heuristic form, absent with a model as the draft | <https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/> added speculative decoding to the search "using a simple heuristic to guess the next 16 to 32 tokens", for 4.7 times | Speculation inside the search is claimed. A 2B drafting for a 4B is unpublished, but it's an efficiency gain, not a headline novelty |
| f. Auditing public checkpoints for contamination by an inversion of capability order on fresh data | Absent for ARC, with related statistical work | <https://arxiv.org/pdf/2405.16281> (ConStat, contamination tests based on performance), GSM1k via <https://neurips.cc/media/neurips-2024/Slides/97687.pdf>, <https://arxiv.org/html/2601.10904v1> (knowledge overfitting flagged but unmeasured), <https://arxiv.org/pdf/2510.02386> (how fragile contamination detection is) | No published instrument audits ARC checkpoints. Our measured inversion on sft141 (80 against 10 on public eval, flipping to 65 against 78.3 on fresh RE-ARC, z = 8.0) would be the first, and ConStat and GSM1k give it a statistical frame to cite |

### Notes on each lever

**a. Best-first search over the frontier.** The ARChitects' product-of-experts paper only compares
depth-first search with beam search, random sampling and greedy decoding. NVARC's only decoding
change was batching the same depth-first search. In the general decoding literature, best-first beam
search and ULTS (a best-first tree search guided by uncertainty over LLM tokens) exist, but they were
applied to summarization, translation and code, never to ARC grids. Nothing public runs a global
priority queue over a grid-decoding tree, let alone jointly across augmentations. The real win isn't
the set of candidates, since threshold search already enumerates everything above T. It's behaving
well under any budget. Each task gets depth that adapts to its time, there's no cliff at the threshold
on hard tasks, and it degrades gracefully instead of returning empty pools.

**b. Filtering by invariants.** Program-synthesis work verifies by running candidate programs on the
demonstrations, as in Berman's cell-accuracy fitness, SOAR and Li et al.'s induction. For transductive
pools the closest work is Multi-Perspective Transformers, which keeps only candidates whose model also
reproduces the training outputs exactly, and the ARChitects' separate shape predictor from 2025.
Nobody publishes the specific mechanism of mining necessary conditions from the demonstrations, such
as maps of size, palette and component counts, and applying them as hard constraints on an LLM's
candidate pool after the fact. One caveat comes from our own history. Reweighting the selector and
voting cell by cell are both dead by measurement here, so treat this as a diagnostic first, checking
how often the selected candidate breaks a mined invariant, before betting points on it.

**c. Reranking across models.** NVARC already scored a second generator's candidates, TRM's ten
attempts, under the Qwen3 likelihood. They found the scorer usually failed to pick TRM's unique solves,
catching about one evaluation puzzle on average. That failure is informative. A single scorer leans
toward its own distribution. What's still unpublished in the Kaggle setting is symmetric selection
under two independently trained models, by agreement, product or rank fusion of both models'
product-of-experts scores, which is exactly what a 4B and 2B trained on different data allow. At the
API level, Land's judge shows that comparing candidates together in one context beats scoring each
alone, and no system under Kaggle's limits selects jointly either.

**d. A meta-learned starting point for training.** Akyurek et al. measured 71.5 with a LoRA per task
against 63 with a single shared LoRA on ARC-1, and didn't try a meta-learned starting point. When
MindsAI talk about "dataset-induced meta-learning" (Cole and Osman, Lab42, 2023), they mean training
the base model so that test-time fine-tuning works well, not optimizing where the adapter starts.
Meta-LoRA exists in vision personalization, point clouds and detecting speech deepfakes. A
Reptile-style starting point, averaging the converged per-task adapter changes over training-set
episodes and then training each task from there at test time, is untried on ARC and cheap to test.

**e. Speculative decoding inside the search.** The ARChitects' 2025 report publishes both heuristic
drafts inside the search (4.7 times) and reusing the cache of the shared prefix across augmentations
(5.8 times), with numbers, and both carry straight over to our autoregressive pipeline. Using a model
as the draft inside the search, with the 2B proposing and the 4B checking, is unpublished, but it buys
speed rather than novelty.

**f. Auditing for contamination.** In general, there's ConStat, which detects contamination
statistically by comparing performance against reference models and variants of the benchmark. There's
GSM1k, which rebuilt a benchmark from scratch and measured the drop, up to 13% for families that had
overfit. And a 2026 paper looks at how fragile contamination detection is in reasoning models. Specific
to ARC, the 2025 technical report names knowledge overfitting but doesn't measure it, the ARChitects
admit overfitting to the evaluation set from data derived from it, and NVARC openly built descriptions
derived from the evaluation set into their generator. Nobody has published an audit of public ARC
checkpoints, and nobody uses an inversion of capability order between two models on freshly generated
data as the test. Our measured result, sft141 against the 4B with an interaction of 83.3 points at
z = 8.0, looks like the first of its kind for ARC.

## 4. Directions nobody has exploited that fit our constraints

Our constraints are one submission a day, 12 hours on four L4s with 96 GB, a 4B and 2B grid model
pair, a paper strong on measurement discipline, and a paper track judged by rubric.

1. **Search that works under any budget.** Every decoder on the Kaggle track is threshold depth-first
   search, in 2024, in 2025 and in NVARC's batched version. A global best-first frontier over
   augmentation and prefix states, with an anytime stop, makes the 12-hour budget something you
   actually control. Hard tasks get depth, and easy tasks give time back. There's no ARC precedent
   (section 3a). It also combines with published work we haven't absorbed yet, heuristic speculation
   inside the search and caching the prefix across augmentations, both with published speedups of 4 to
   6 times.
2. **Two independent experts, one of them cheap.** Nobody on the Kaggle track selects with two
   independently trained models. NVARC's single scorer ignoring the other generator's unique solves is
   a published reason to try symmetric selection with two models, and we already have the 4B and 2B.
   The 2B also makes the unpublished version of speculation with a model draft possible.
3. **Choosing among candidates jointly, locally.** Land's verified 72.9 shows the biggest gains in
   selection come from comparing candidates with each other rather than scoring each alone. A local
   version, feeding the top few decoded grids back to the adapted model in one context and scoring
   which one completes the task, has no precedent on the Kaggle track.
4. **Measurement discipline as a contribution.** The winners themselves report one to two points of
   variation between re-runs of identical notebooks, and nondeterminism in batched search. The 2025
   report names knowledge overfitting but can't measure it. Scores displayed beside notebooks provably
   differ from what forks reproduce, as our 33.89 against 28.06 and 28.47 controls showed. A chapter
   that treats the leaderboard as a noisy instrument, with a noise floor, fork controls and a
   contamination audit through inverted ordering on fresh RE-ARC, stands apart from every public
   writeup, and the paper track is judged by rubric.
5. **Known, but not yet used by us.** NVARC's flywheel of verified synthetic data is open source,
   including the code, 103,000 puzzles, 3.2 million augmented samples and the Nemotron reasoning traces
   from March 2026. It's the measured cause of their 2025 win and presumably of nvbanana's climb from
   39 to 67.5. It's replication rather than novelty, but it's the only public lever with an effect
   shown above five points on hidden data, and continuing to fine-tune our 4B on that corpus fits our
   rig. Any claim of novelty should be measured on top of this known lever, or explicitly in place of
   it.

## The top three recommendations

1. **Best-first enumeration for grid decoding that works under any budget.** Replace threshold
   depth-first search with a global priority queue over partial decodes, ordered by cumulative
   log-probability, optionally plus a cheap estimate of the remainder, and shared across
   augmentations. It emits candidates in strictly decreasing probability until the task's time budget
   runs out. For the first experiment, instrument the current search to log the full explored tree on
   the 120 public evaluation tasks. Then replay it offline to simulate best-first order at equal node
   budgets, and count tasks where the correct grid is reachable within the time budget but below the
   0.2 threshold, or the other way round. It costs no GPU time beyond one logging run, and it makes a
   figure for the paper either way.

2. **Symmetric selection with the 4B and 2B pair.** Decode the candidate pool once with the 4B, then
   score it under both the adapted 4B and the independently trained, adapted 2B with the same fixed
   augmentations. Select by the product, or a rank fusion, of the two product-of-experts scores, and
   use NVARC's published single-scorer failure as the baseline to beat. For the first experiment, take
   logged 4B pools from public evaluation plus a fresh RE-ARC control set, and measure how much of the
   oracle gap closes when the 2B scores are added, meaning outputs where the truth was in the pool but
   wasn't selected. Run McNemar's test against the single-scorer selector before any submission to the
   hidden set.

3. **Publish the measurement instruments as results in their own right.** That means the contamination
   audit through inverted capability order on freshly generated RE-ARC data, framed against ConStat and
   GSM1k and applied to public ARC checkpoints. It means the fork-and-run control for scores displayed
   beside notebooks. And it means an account of the noise that reconciles our 0.41-credit noise floor
   with the winners' admitted one to two points between re-runs. For the first experiment, extend the
   existing sft141 audit to two or three more popular public checkpoints, with the inversion statistic
   written down beforehand, so the paper presents an instrument tested on several subjects rather than
   a single anecdote.
