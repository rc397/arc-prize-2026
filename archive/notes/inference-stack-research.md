# Inference stack research and a revised plan

Written on 19 June 2026. I tagged claims as I went. [V] means I read the source directly that
session, [E] means estimated or extrapolated, and [?] means unverified and waiting on a test.

## Bottom line

Stop building WSL2 and vLLM. Use llama.cpp on native Windows for generation. Getting vLLM running on
an RTX 5070 Ti took someone a documented "three-and-a-half bleary-eyed days" of building PyTorch and
vLLM from source on Linux [V]. llama.cpp already benchmarks at about 140 tokens a second on a 20B
model on this card, with no Triton and no WSL [V].

## Why decoding is slow

- Decoding at 13 tokens a second uses about 10% of what the card's memory bandwidth allows. A 4B
  model at roughly 896 GB/s should manage about 130 [E]. Prefill is fine at 3,403 tokens a second.
  The gap is per-token framework overhead, meaning Python dispatch across 36 layers, a dynamic cache
  and no CUDA graphs. It's the classic slow Hugging Face generate problem.
- The usual fix is a static cache plus torch.compile, worth up to 4x [V]. But torch.compile needs
  Triton, and Triton has no native Windows support, which is the TritonMissing error we hit [V]. So
  plain PyTorch decoding can't be made fast on this setup, and we need a proper inference engine.

## The engines

- **llama.cpp** runs about 140 tokens a second on a 20B model on the 5070 Ti through llama-cli [V].
  It works natively on Windows and needs little setup beyond a few sm_120 workarounds (issue
  #22696) [V]. Per-task LoRA is awkward, since every adapter has to go from PEFT to GGUF, and
  converting Qwen3 LoRAs has an open bug (#21125) [V]. This is the best first move.
- **vLLM** runs about 120 tokens a second on a 7B model in FP4 on the 5070 Ti [V]. It only runs on
  Linux or WSL2, takes a lot of setup (building from source, and FA3 is broken on Blackwell) [V], and
  handles dynamic LoRA cleanly. Worth it only if per-task LoRA forces our hand.

## The economics of test-time training

- Test-time training matters a lot. It gave up to 6x accuracy, 53% on ARC-AGI-1 with an 8B model, and
  27.5 points over a fine-tuned baseline [V, Akyurek et al., 2411.07279].
- The winners paid for it with hardware. The ARChitects ran 128 training steps per task at rank 32
  and batch 1, plus 102 refinement steps, with speculative decoding (4.7x) and prefix caching (5.8x),
  on clusters of 8 H100s, GH200s and 16 H100s [V]. NVARC ran on Kaggle's four L4s [V].
- Here's the knot. Training has to happen in PyTorch, because no engine trains. Fast generation needs
  an engine. And per-task training makes a new LoRA for every task, which the engine then has to
  serve. That serving step is the real difficulty on one consumer card. A single global fine-tune
  throws away most of the per-task gain [V], so there's no clean shortcut.

## Staged plan, with a check at each step

- **Stage 0, today.** Install llama.cpp on the rig, convert the NVARC model to GGUF and time decoding
  on one task, aiming for at least 150 tokens a second. This tests the biggest open risk, which is
  whether a model with a 16-token vocabulary converts to GGUF at all [?].
- **Stage 1.** Build a fast pipeline without test-time training on llama.cpp, using the base model
  with augmented multi-candidate generation and voting. That gives a real number on all 120 tasks in
  a couple of hours and proves the pipeline runs at speed.
- **Stage 2.** Add test-time training and make it affordable. That means about 40 to 64 steps,
  few-shot context cut to about 1,500 tokens, and no gradient checkpointing at that shorter length.
  Let the data decide how to serve the LoRAs, either merging each into the base and re-converting it
  per task, or moving to vLLM on WSL2 only if that pays off. WSL is already enabled, so that door
  stays open at no cost.
- **Stage 3.** Add speculative decoding and prefix caching, the winners' 4.7x and 5.8x, both of which
  llama.cpp supports.

## Traps to avoid

1. Time decoding speed and the wall-clock for one whole task before starting any long run.
2. Confirm the engine is fast before building the pipeline on top of it.
3. Always print output lengths, because generation silently running to the cap has killed runs before.
4. Measure prefill, decoding, training steps and selection separately.
5. Size every run so it finishes, and check that task 0 came out right before walking away.
6. Keep a known-good fast path, plain llama.cpp generation, as a fallback.

## Still open

- [?] Whether the 16-token Qwen3 model converts to GGUF. Stage 0 tests this, and it's the biggest risk.
- [?] The real llama.cpp decoding speed for our 4B, which I extrapolated from the 20B numbers.
- [?] Serving per-task LoRAs in llama.cpp for this model, to be tested in Stage 2.
- NVARC's exact training and inference settings are only in interviews, not the written report [V].

## Sources

- vLLM on the RTX 5070 Ti, <https://ligma.blog/post1/>
- vLLM RTX 5090 issue, <https://github.com/vllm-project/vllm/issues/14452>
- llama.cpp speed on the RTX 5070 Ti, <https://github.com/ggml-org/llama.cpp/issues/15297>
- llama.cpp Windows sm_120 workarounds, <https://github.com/ggml-org/llama.cpp/issues/22696>
- llama.cpp Qwen3 LoRA conversion bug, <https://github.com/ggml-org/llama.cpp/issues/21125>
- The ARChitects' 2025 report, <https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>
- The ARC Prize 2025 technical report, <https://arxiv.org/html/2601.10904v1>
- The Surprising Effectiveness of Test-Time Training, <https://arxiv.org/abs/2411.07279>
- Transformers KV-cache docs, <https://huggingface.co/docs/transformers/kv_cache>
