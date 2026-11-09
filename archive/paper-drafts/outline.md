# Paper outline (working draft)

Title (working): Efficient ARC-AGI-2 on a Single Consumer GPU: How Far Does TTT + AIRV Go
When You Are Compute-Poor

## Angle (the contribution is the constraint)

Almost all strong ARC-AGI work runs on multi-GPU clusters (the 2025 winners used 8-16x H100 /
GH200). We ask the opposite question: how far does the winning recipe (small grid LLM +
per-task test-time training + Augment-Infer-Reverse-Vote) go on one 16 GB consumer GPU, and
where exactly are the walls? An accounting of what matters, what breaks, and the
accuracy-per-GPU-hour frontier is the contribution, useful to everyone who cannot rent a
cluster. We build on the released NVARC Qwen3-4B grid model and report a reproducible system.

## Findings so far (this repo / DEVLOG)

- Inference engine matters more than anything else on this hardware. HF eager decode is ~13
  tok/s (memory-bandwidth badly underutilized; no CUDA graphs; torch.compile/Triton unavailable
  on native Windows). Converting the custom 16-token-vocab model to GGUF and serving via
  llama.cpp gives ~13x (170 tok/s at Q8_0), but the prebuilt CUDA build is intermittently
  unstable on Blackwell under sustained generation; we engineer around it (disable the prompt
  cache, single stream, self-restarting server).
- No-TTT AIRV baseline: 12.79% (22/172) on the 120-task public eval. Clean, all-candidate run.
- TTT trains cheaply once the few-shot context is capped (~13s/step at full ~3800-token
  context drops to ~2s/step at ~1600 tokens). The wall is generation: 8 augmented decodes per
  task at 13 tok/s in PyTorch make a full run ~7-10h. The PyTorch-train -> fast-engine-generate
  bridge (serving a fresh per-task LoRA) is the core single-GPU efficiency problem; on Linux it
  is dissolved by torch.compile + vLLM dynamic LoRA, on our Windows/Blackwell rig it is not.

## Sections

1. Introduction: ARC-AGI-2; the compute-poor setting; our question and reproducible system.
2. Background: the TTT + AIRV recipe and why each piece is compute-hungry.
3. Constraints of one consumer GPU: Blackwell/Windows toolchain, decode speed, 16 GB VRAM, the
   per-task LoRA serving bridge.
4. System: NVARC Qwen3-4B grids; GGUF conversion of a custom-vocab model; the AIRV pipeline at
   the token-ID level; capped per-task TTT.
5. Results and ablations: no-TTT baseline; TTT lift; sweeps over #augmentations, TTT steps,
   context cap, quantization; the accuracy-per-GPU-hour curve.
6. Novel contribution (decide after the TTT numbers land): candidates below.
7. Discussion: what matters under tight compute; honest negative results (the generation wall);
   what changes with a clean Linux toolchain.

## Novel-contribution candidates (pick after TTT results, target the measured weakness)

- The accuracy-per-GPU-hour frontier itself: a rigorous efficiency study of the recipe, which
  no one publishes because everyone has cluster compute. The constraint becomes the result.
- An efficient-TTT recipe: context capping plus step/rank/target ablations that recover most of
  the lift at a fraction of the cost.
- A stronger selection/verification step than majority voting (the no-TTT failure mode we saw:
  the correct grid is often in the candidate pool but not chosen).

## Status

Baseline done. TTT pipeline built and fixed; lift measurement in progress. Rubric note: accuracy
is one of six paper-track criteria, so a modest score plus a genuinely useful efficiency study
and clear writeup can score well without topping the leaderboard.
