# The 16GB Ceiling

## Test-time training for ARC-AGI-2 on one consumer GPU, and what it takes to trust a number

**Our best submission scored 32.64, the maximum of eleven runs of one immutable artifact whose mean is 29.99 and standard deviation 1.36. Most of this paper is about why we now report it that way.**

### What we built

Our submission reproduces the strongest public lineage here, the ARChitects' 2024 pipeline as ported to ARC-AGI-2 by Ivan Sorokin, on his 4B grid model: a per-task rank-256 LoRA adapter, a depth-first search over completions above 0.2 cumulative probability, and a selector picking two attempts by vote count and augmentation likelihood. We credit the method fully. Our contribution is what the field does not usually publish: the error bar, the controls, the theory of where it stops, and two audits of the numbers everyone is chasing.

The same pipeline prices the constraint in our title. Our best single-card configuration reaches 18.19 on a 16GB RTX 5070 Ti. The identical method on the contest's four L4s averages 29.99. Same base model, same hidden set, same twelve hours. One consumer card costs about twelve points, and that gap belongs to the per-task training, which none of fourteen measured configurations could serve at full rank and context inside 16GB.

Be clear about where that sits: mid-field, far below the leaders. We measured thirty-three levers on top of it. None moved the hidden score, and eleven were closed with no GPU by pricing the idea before building it. The honest summary of our season is that the reproduction was the peak.

### The error bar nobody measures

Every comparison on this leaderboard silently assumes re-running a fixed configuration returns the same score. The 2025 winners mention one to two points of variability in passing. Nobody, including them, measured it. We did, by spending eleven submissions re-running one unchanged artifact, and it demolished our own conclusions. One kernel version, verified identical by script version rather than description, spans **4.17 points**, ten task credits, mean 29.99 and standard deviation **1.36**.

Measured against that, **every build we called a regression falls inside two standard deviations**, as do both public notebooks we ran as controls. We widened this estimate five times, always in the same direction, each widening retiring earlier claims. The mechanism is throughput: the search is deadline-bounded in three places, so how far the machine gets decides where it truncates. Separating two configurations at 80 percent power needs roughly 3.4 points and three runs per arm, so one submission a day permits about eight properly powered comparisons in a season. Single-submission A/B testing, which this field and we both ran, was never capable of the discrimination it claimed.

### Why it works, and exactly where it stops

The base model is bimodal per task. It either internalises the rule during test-time training and the search then finds the answer, or it internalises a wrong one and concentrates its mass there. On our slice, 65 percent of tasks never produce the correct grid under any view.

We prove a theorem about the half of the pipeline everyone tunes. Any selector ranking candidates on statistics derived from the adapted model's own likelihoods is bounded by the expected top-two posterior mass of its beliefs. The reading is Blackwell's: selection is a garbled channel, and no processing recovers what it never carried. On our dumped pools the truth, when it enters at all, is Pareto-dominated by wrong grids on multiplicity and likelihood together, so the bound bites: the attainable ceiling is 18 of 60 outputs and the deployed selector attains 15. Refitting its weights under leave-one-task-out validation scores worse. Cell-wise voting reconstructs zero of the 41 missing truths, because the pool's errors are correlated through the model's beliefs.

The theorem names an escape, reading the task rather than the model, and it fails specifically. Invariants mined from the demonstrations kill eleven of nineteen correct answers, because with two or three pairs coincidental regularities are common and ARC-AGI-2 test pairs are built to break them. Pruning back to invariants that never kill a truth leaves only properties the model already respects, so they kill nothing wrong either. Selectivity and safety are one axis.

### Can the base be improved on this budget?

Everything above changes the pipeline around a fixed base, so we tested the base itself. Three gates, each varying one axis with the others fixed, each with a falsification stated in advance.

**Rule count.** Eight thousand examples from 100 composed rules against the same budget from 402. Identical: tied 12 of 12.

**Rule kind.** Every rule we held was ARC-1 derived, so we built 128 ARC-2-shaped generators (below) and matched 100 against 100 ARC-1 rules at equal pool depth. Zero better, one worse, eleven of twelve tied.

**Training duration.** Eight thousand steps to ninety-six thousand. Validation loss bottoms at 16,000 then rises for sixty thousand more while training loss falls to 0.003, full memorisation. Not one task improved at any checkpoint, and there is no grokking signature.

The duration result retires the obvious objection to the other two, which both stopped at 8,000 steps and might have been undertrained. They were not. We resist calling these three independent confirmations: all share one twelve-task gate that has tied 12 of 12 three times and cannot resolve a single-task effect. The correlation runs through the instrument, not the interventions.

### A generator corpus, and a decoder that got correct and scored worse

Asking a model to write a `solve` and a `gen_input` for each of the 1,000 public ARC-AGI-2 training tasks, verified by execution against that task's own demonstrations, yields **128 verified generators, 98 percent with working samplers, for $1.57**. The verification is a proof, not an opinion, and we can price it: on held-out test pairs those programs never saw, **93 of the first 100 reproduce the answer exactly**, so the free check is a 93 percent precise proxy for the real rule.

Two results there generalise past ARC. Reasoning models can spend an entire output budget without emitting an answer: ours burned 100 percent of 383,997 output tokens on reasoning across twelve tasks and wrote no code, which reads as incapacity and is a cap landing mid-thought. And the effort dial is **not monotone**: low effort beat maximum on yield, price and speed at once, because maximum effort fails by not terminating.

The decoder result we least expected. The deployed depth-first search shares one KV cache across sibling branches with no truncation, so each branch attends over those explored before it. Cropping it is provably correct, bit-identical to a full cache copy, and recovers **27 percent more candidates**. Submitted three times it scored 28.70 against a 29.99 control. Making the search correct made the score worse, because correctness costs wall clock and under a binding deadline those are different objectives. We report it as a defect fixed and a change we did not ship.

### Two audits of the numbers this field runs on

A public notebook displayed 33.89 against our 30.56, and we spent a week producing three quantified explanations of the deficit. Then we forked it unmodified and submitted it: 28.06. The notebook actually titled for 33.89, forked byte-identical, scored 28.47. A displayed score binds to a submission, not the published file, and both are mutable independently. The week was aimed at a number that does not exist. Run the unmodified fork as a control before explaining anything.

The second audit began when a public 2B checkpoint appeared to solve candidate selection perfectly. It greedy-decodes 80 percent of the public evaluation set with no test-time training, while the 4B from the same author manages 10 percent. On 120 fresh RE-ARC instances generated under our own seed after both were published, the ordering reverts: 78 percent against 65, an interaction of 83 points at eight standard errors. A smaller model does not beat a larger sibling eight-fold on harder published data while losing on easier fresh data, so that set is not a valid instrument for that checkpoint. We claim nothing about intent.

### What transfers

The contamination test is a general instrument: establish a capability ordering on data generated after both models were published, measure them on the suspect benchmark, and read contamination off the interaction. It needs no training data, no private split and no author cooperation. The selection theorem applies to any propose-then-rerank pipeline, including best-of-n inference, and says when reranking cannot help and what a verifier must beat first.

One positive result earns its space: masking the training loss on cells a rule leaves unchanged, 88 percent of a median output, made the transformed cells 4.6 times more accurate and the answer worse, while an equal-size random mask matched baseline. The gradient spent on copying is load-bearing.

### Artifacts

The full paper is linked, every figure regenerable from committed scripts, and the attached notebook is the exact build behind our reported submission. It documents its mistakes in place, because the corrections are the method.
