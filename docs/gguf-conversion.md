# Converting the NVARC grid model to GGUF for llama.cpp

This records the exact, reproducible steps to run the `sorokin/qwen3_4b_grids15_sft139` model
under llama.cpp. The model ships a custom 16-token WordLevel tokenizer, so the stock converter
needs two small patches. Validated on 2026-06-19 (llama.cpp release b9724).

## Why patches are needed

The tokenizer is WordLevel, not BPE: vocab `{0..9, Ċ(=newline)=10, user=11, assistant=12}`
plus added specials `<|endoftext|>=13, <|im_start|>=14, <|im_end|>=15`, and zero merges.
`convert_hf_to_gguf.py` assumes a Qwen BPE tokenizer, which causes two failures:

1. The pre-tokenizer hash is unrecognized, so the converter raises
   `NotImplementedError: BPE pre-tokenizer was not recognized`. The model's pre-tokenizer
   regex is byte-for-byte the qwen2 regex, so the fix is to map its hash to `"qwen2"`.
2. WordLevel has no merges, so no `tokenizer.ggml.merges` key is written, and the llama.cpp
   loader aborts with `cannot find tokenizer merges in model file`. The fix is to write a
   non-empty merges list. We add two throwaway merges among the `user`/`assistant` tokens;
   they never fire on digit or newline text, and we prompt at the token-ID level anyway.

## Steps (run on the rig)

Engine binary and convert script (matching versions):

```powershell
# prebuilt CUDA 12.4 binary (loads on Blackwell sm_120 via PTX JIT) + cudart, into D:\arc\llamacpp
# source clone for convert_hf_to_gguf.py, into D:\arc\llama.cpp, at the same release tag
git clone --depth 1 --branch b9724 https://github.com/ggml-org/llama.cpp D:\arc\llama.cpp
python -m pip install "D:\arc\llama.cpp\gguf-py" sentencepiece protobuf
```

Patch 1, `conversion/base.py` `get_vocab_base_pre`, before the final `if res is None:` that
precedes the `raise NotImplementedError("BPE pre-tokenizer was not recognized ...")`:

```python
if chkhsh == "3f7ffce2c85ce59d277e21c40359aa7326aaa65660c8166288ec5137c9cc1967":
    res = "qwen2"
```

Patch 2, `conversion/base.py` `_set_vocab_gpt2`, immediately after
`special_vocab.add_to_gguf(self.gguf_writer)`:

```python
if not special_vocab.merges:
    self.gguf_writer.add_token_merges(["user assistant", "assistant user"])
```

Convert and quantize:

```powershell
$model = "D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
python "D:\arc\llama.cpp\convert_hf_to_gguf.py" $model --outfile "D:\arc\models\arc-qwen4b-bf16.gguf" --outtype bf16
& "D:\arc\llamacpp\llama-quantize.exe" "D:\arc\models\arc-qwen4b-bf16.gguf" "D:\arc\models\arc-qwen4b-q8_0.gguf" Q8_0
```

## Measured results (RTX 5070 Ti, all layers on GPU)

| format | size | decode tok/s | prefill tok/s |
|---|---|---|---|
| bf16 | 6.77 GB | 102 | 2344 |
| Q8_0 | 3.60 GB | 170 | 10238 |
| Q4_K_M | 2.02 GB | 244 | 11042 |

Q8_0 is the chosen operating point (near-lossless, 13x the 13 tok/s HF baseline).

## Serving and correctness

Serve with token-ID prompting:

```powershell
D:\arc\llamacpp\llama-server.exe -m D:\arc\models\arc-qwen4b-q8_0.gguf -ngl 99 -c 16384 --host 127.0.0.1 --port 8080 --no-webui
```

POST to `/completion` with `"prompt"` as a token-ID array built by
`src/arc/model.py:build_prompt_ids`. The role and special tokens (user, assistant, im_start,
im_end) cannot be produced by text tokenization, so always prompt with IDs, not text.
`scripts/stage0_check.py` does this and scored 14/15 exact-match on training tasks (greedy,
no TTT), confirming the conversion is faithful.
