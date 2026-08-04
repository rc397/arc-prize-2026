# The 16GB Ceiling

## Test-time training for ARC-AGI-2 on one consumer GPU, and what it takes to trust a number in this competition

**Submission ID 54974874, scored 30.97 on the hidden set.**

### What we built

Our submission is a reproduction of the strongest public lineage in this competition, the ARChitects' 2024 pipeline as ported to ARC-AGI-2 by Ivan Sorokin, run on his 4B grid model. Each task gets its own LoRA adapter, rank 256 with rslora at 8192 context, trained on sixteen augmented views of the demonstrations. A depth-first search then enumerates every completion whose cumulative probability stays above 0.2, candidates are rescored under fresh augmentations, and a selector picks two attempts by vote count and mean augmentation likelihood. We credit the method fully. Our contribution is everything the field does not usually publish about a method, the error bar, the controls, the theory of where it stops, and two audits of the numbers everyone else is chasing.

The same pipeline prices the constraint in our title. Our best single-card configuration reaches 18.19 on a 16GB RTX 5070 Ti. The identical method on the contest's four L4s reaches 30.97. Same author, same base model, same hidden set, same twelve hours. One consumer card costs about twelve points, and that gap is a property of the per-task training, which cannot be served at full rank and context inside 16GB by any of the fourteen configurations we measured.

### The error bar nobody measures

Every comparison on this leaderboard silently assumes that re-running a fixed configuration returns the same score. The 2025 winners mention one to two points of run-to-run variability in passing in their own writeup. Nobody, including them, had measured it. We did. Three runs of the unchanged pipeline scored 30.56, 30.97 and 30.56. The spread is 0.41, exactly one task credit, with a standard deviation of 0.24. That number converts our season from anecdotes into statistics. Five successive builds that scored below v7.2 sit between 4.7 and 11.2 standard deviations under the lowest baseline run, so each regression is a real effect of the change it carried, and any future lever must move more than 0.4 points before it is distinguishable from a re-run.

### Why it works, and exactly where it stops

The base model is bimodal per task. It either internalises a task's rule during test-time training, and the search then finds the answer with high probability, or it internalises a wrong rule and concentrates its mass there. On our measured slice, 62 percent of tasks never produce the correct grid under any augmented view, and they consume 69 percent of the budget failing.

We prove a theorem about the half of the pipeline everyone tunes. Any selector that ranks candidates using only statistics derived from the adapted model's own likelihoods is bounded by the expected top-two posterior mass of the model's beliefs. The reading is Blackwell's, selection is a garbled channel and no processing recovers information the channel never carried. On our dumped candidate pools the truth, when it enters the pool at all, is Pareto-dominated by wrong grids on multiplicity and likelihood together, so the bound bites. Refitting selector weights under leave-one-task-out validation scores worse than the deployed selector, the headroom is real but not learnable from sixty outputs. Cell-wise voting across candidates reconstructs zero of the 41 missing truths, because the pool's errors are correlated through the model's beliefs.

The theorem names its own two escape routes, and we ran both. Executing task-side structure against the demonstrations, tried as a symbolic object-count prune, scored 27.92, seven credits below baseline at eleven standard deviations. The escape clause has a measured precondition, an external oracle helps only if it is more accurate than the model's own beliefs about the same property, and a noisy oracle is strictly worse than none. The second route, rescoring under a separately trained model, produced our strangest result, below.

### Two audits of the numbers this field runs on

A public notebook displayed 33.89 against our 30.56, and we spent a week explaining the deficit, publishing three quantified explanations internally. Then we forked the notebook unmodified and submitted it. It scored 28.06. The notebook actually titled for 33.89, forked byte-identical with its author's own attachments, scored 28.47. Two public artifacts displaying 33.89-class numbers both land near 28 when executed, below our own unchanged pipeline's 30.56 to 30.97 band. A displayed score binds to a submission, not to the published file, and both are mutable independently. The week of explanations was aimed at a number that does not exist. The remedy is not scepticism, which produced its own wrong conclusion, but running the unmodified fork as a control before explaining anything.

The second audit began when a public 2B checkpoint appeared to solve candidate selection perfectly, placing the truth top-two on 19 of 19 reachable outputs. A perfect score against a ceiling is a symptom. That model assigns the true public evaluation answers a median of 0.0002 nats per token and greedy-decodes 80 percent of them exactly with no test-time training, while the 4B from the same author manages 10 percent. On 120 fresh RE-ARC instances we generated under our own seed after both models were published, each verified by RE-ARC's own verifier, the ordering reverts, the 4B at 78 percent against the 2B's 65. The interaction is 83 points at eight standard errors. A smaller model does not beat a larger sibling eight-fold on harder published data while losing on easier fresh data. The public evaluation set is not a valid measurement instrument for that checkpoint, and any pipeline using it to score or filter against public eval is reading recall. We claim nothing about intent, training on the public split is standard practice when the private split is what scores.

These two results are one finding. The competition's visible numbers, displayed leaderboard scores and public evaluation accuracies, are not the measurements they appear to be, and both failure modes are detectable with cheap controls that we provide.

### What transfers beyond ARC

The contamination test is a general instrument. Establish a capability ordering between models on data generated after they were published, measure the same models on the suspect benchmark, and read contamination off the interaction. It needs no training data access, no private split, and no cooperation from the model's author, only a task generator. The selection theorem applies to any propose-then-rerank pipeline, which includes most best-of-n LLM inference, and says when reranking cannot help and what an external verifier must beat before it does. The replicate control, spending one submission to re-run an unchanged configuration, is the cheapest discipline any competitor can adopt, and ours says the noise here is one task credit.

### What this does for anyone chasing a top score

Negative results with controls are directions not to dig. Colour-permutation restarts do not decorrelate misses, the union of three independently trained adapters equals the best single one, with a same-seed replicate proving the apparent gain was decoder nondeterminism. Selector reweighting, cell-wise voting and attempt diversification are dead by measurement. Noisy symbolic pruning costs seven credits. Displayed public scores are not reproduction targets. And the twelve-point price of one consumer card tells solo entrants where the method ceiling actually is before any engineering starts.

### Artifacts

The full 55-page paper with all proofs, measurements and corrections is attached as the project link, with every figure regenerable from committed scripts and data. The attached notebook is the exact build behind submission 54974874. The paper documents its own mistakes in place, including two wrong explanations we published and retracted, because the corrections are the method.
