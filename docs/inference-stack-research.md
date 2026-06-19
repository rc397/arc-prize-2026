# Inference Stack Research and Revised Plan (2026-06-19)

Verification tags: [V] read the source directly this session, [E] estimated/extrapolated,
[?] unverified, needs an empirical test.

## Bottom line

Pause the WSL2 + vLLM build. Use llama.cpp on native Windows for generation. Getting vLLM
onto the RTX 5070 Ti took a documented "three-and-a-half bleary-eyed days" building PyTorch
and vLLM from source on Linux [V], while llama.cpp already benches at ~140 tok/s on a 20B
model on this card with no Triton and no WSL [V].

## Root cause of the slowness (confirmed)

- Decode at 13 tok/s is ~10% of the card's memory-bandwidth ceiling (a 4B at ~896 GB/s
  should hit ~130 tok/s) [E]. Prefill is fine (3403 tok/s). The gap is per-token framework
  overhead (36-layer Python dispatch, dynamic cache, no CUDA graphs), the classic slow
  Hugging Face generate problem.
- Standard fix is StaticCache + torch.compile (up to 4x) [V], but torch.compile needs
  Triton and Triton has no native-Windows support (our TritonMissing error) [V]. So native
  PyTorch decode cannot be made fast on this Windows setup; an optimized engine is required.

## Engine comparison

- llama.cpp: ~140 tok/s @20B on the 5070 Ti via llama-cli [V]; native Windows; low setup
  (sm_120 workarounds, issue #22696) [V]. Per-task LoRA is clunky (PEFT->GGUF per task) and
  Qwen3 LoRA conversion has an open bug (#21125) [V]. Best first move.
- vLLM: ~120 tok/s @7B FP4 on the 5070 Ti [V]; Linux/WSL2 only; very high setup (build from
  source, FA3 broken on Blackwell) [V]; clean dynamic LoRA. Only if per-task LoRA forces it.

## TTT economics (the crux)

- TTT is essential: up to 6x accuracy, 53% on ARC-AGI-1 with an 8B model, +27.5 points over
  a fine-tuned baseline [V, Akyurek et al. 2411.07279].
- The winners paid with hardware: ARChitects ran 128 TTT steps/task (rank-32, batch 1) plus
  102 refinement steps, with speculative decoding (4.7x) and prefix caching (5.8x), on
  8xH100 / GH200 / 16xH100 clusters [V]. NVARC ran on Kaggle L4x4 [V].
- The knot: TTT must run in PyTorch (no engine trains), fast generation needs an engine, and
  per-task TTT makes a new LoRA every task that the engine must serve. That serving step is
  the real difficulty on a single consumer GPU. A single global fine-tune throws away most of
  TTT's per-task gain [V], so there is no clean shortcut.

## Staged plan (with validation gates)

- Stage 0 (today): install llama.cpp on the rig (native Windows), convert the NVARC model to
  GGUF, benchmark decode tok/s on one task. Target >=150 tok/s. Tests the key open risk: does
  the 16-token-vocab model convert to GGUF at all? [?]
- Stage 1: fast no-TTT pipeline on llama.cpp (base model, augmented multi-candidate
  generation + voting). Real complete 120-task number in a couple of hours. Validates the
  pipeline at speed.
- Stage 2: add TTT, made tractable: ~40-64 steps, subsample few-shot context to ~1500 tokens,
  drop gradient checkpointing at the shorter length. Decide LoRA serving with data (merge into
  base + re-convert per task, or commit to vLLM+WSL2 only if it pays off). WSL features are
  already enabled, so that door stays open at zero extra cost.
- Stage 3: speculative decoding + prefix caching (the winners' 4.7x and 5.8x), both supported
  by llama.cpp.

## Engineering checklist (anti-traps)

1. Benchmark decode tok/s AND per-task wall-clock on ONE task before any long run.
2. Confirm the inference engine is fast before building the pipeline on it.
3. Always print output lengths (runaway generation to the cap was a silent killer).
4. Measure prefill, decode, TTT-step, and selection costs independently.
5. Size every run to complete; verify task 0 finished correctly before walking away.
6. Keep a known-good fast path (llama.cpp base generation) as fallback.

## Open gaps

- [?] Whether the 16-token-vocab Qwen3 converts to GGUF (Stage 0 tests it; biggest risk).
- [?] Exact llama.cpp decode speed for our specific 4B (extrapolated from 20B).
- [?] Per-task LoRA serving in llama.cpp for this model (empirical, Stage 2).
- NVARC's exact TTT/inference config is gated to interviews, not the written report [V].

## Sources

- vLLM RTX 5070 Ti guide: https://ligma.blog/post1/
- vLLM RTX 5090 issue: https://github.com/vllm-project/vllm/issues/14452
- llama.cpp RTX 5070 Ti eval speed: https://github.com/ggml-org/llama.cpp/issues/15297
- llama.cpp Windows sm_120 workarounds: https://github.com/ggml-org/llama.cpp/issues/22696
- llama.cpp Qwen3 LoRA conversion bug: https://github.com/ggml-org/llama.cpp/issues/21125
- ARChitects 2025 report: https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/
- ARC Prize 2025 technical report: https://arxiv.org/html/2601.10904v1
- Surprising Effectiveness of TTT: https://arxiv.org/abs/2411.07279
- Transformers KV-cache docs: https://huggingface.co/docs/transformers/kv_cache
