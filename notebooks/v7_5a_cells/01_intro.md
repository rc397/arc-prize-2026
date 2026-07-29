# v7 - ARChitects method reproduction (verify the ~31 claim on our model)

This is the public ARChitects 2024-winning pipeline ported onto our grids15_sft139 base. It is **their** method (heavy rank-256 per-task TTT, a DFS decoder, augmentation-product selection). It is reproduced here only to measure its real number on our model. Credit is theirs and the paper attributes it.

**Before running**

1. Attach the grids15_sft139 model and the arc-prize-2026-arc-agi-2 competition data.
2. Set `ARC_MODEL_PATH` and `ARC_DATA_DIR` in the config cell to match the Input panel on the right.
3. Internet ON for this interactive run so Unsloth installs.
4. `ARC_N_EVAL` picks how many eval tasks to score. The final cell prints the score against the eval solutions. Budget roughly 15-30 minutes per task, sharded over the 4 GPUs.

The offline contest submission is a separate step, Unsloth rides in as an attached wheel and the rerun scores the full hidden set.
