"""Build notebooks/arc_prep.ipynb as a SINGLE self-contained code cell (run once on Kaggle, internet
ON) that produces the two offline assets the submission notebook needs: a Linux llama.cpp build and
the Q8_0 GGUF (converted from the attached model with our two custom-vocab patches). One cell so it
cannot be partially run; progress prints show where it is. Reproduces docs/gguf-conversion.md."""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

CELL = r'''
import os, glob, subprocess, shutil, sys, time
t0 = time.time()

print("=== 1. locate the attached model ===", flush=True)
def _has_tok(d):
    return any(os.path.exists(os.path.join(d, t)) for t in ("tokenizer.json", "tokenizer_config.json"))
hits = glob.glob("/kaggle/input/**/config.json", recursive=True)
cands = [h for h in hits if _has_tok(os.path.dirname(h))]
pref = [h for h in cands if "grids15" in h.lower()]
MODEL = os.path.dirname((pref or cands or hits)[0]) if hits else None
print("MODEL dir:", MODEL, flush=True)
assert MODEL, "attach the qwen3-4b-grids15-sft139 model (config.json + safetensors) via Add Input"

print("=== 2. clone llama.cpp b9724 and install deps (needs internet) ===", flush=True)
LCPP = "/kaggle/working/llama.cpp"
if not os.path.isdir(LCPP):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "b9724",
                    "https://github.com/ggml-org/llama.cpp", LCPP], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                os.path.join(LCPP, "gguf-py"), "sentencepiece", "protobuf"], check=True)

print("=== 3. patch the converter for the 16-token vocab ===", flush=True)
base = os.path.join(LCPP, "conversion", "base.py")
s = open(base).read()
if "3f7ffce2c85ce59d277e21c40359aa7326aaa65660c8166288ec5137c9cc1967" not in s:
    needle = 'raise NotImplementedError("BPE pre-tokenizer was not recognized'
    assert needle in s, "patch 1 anchor missing (llama.cpp layout changed)"
    cut = s.rfind("if res is None:", 0, s.index(needle))
    assert cut != -1, "patch 1 anchor (if res is None) missing"
    s = (s[:cut] + 'if chkhsh == "3f7ffce2c85ce59d277e21c40359aa7326aaa65660c8166288ec5137c9cc1967":\n'
         + '            res = "qwen2"\n        ' + s[cut:])
if 'add_token_merges(["user assistant"' not in s:
    anchor = "special_vocab.add_to_gguf(self.gguf_writer)"
    assert anchor in s, "patch 2 anchor missing"
    s = s.replace(anchor, anchor
                  + '\n        if not getattr(special_vocab, "merges", None):'
                  + '\n            self.gguf_writer.add_token_merges(["user assistant", "assistant user"])', 1)
open(base, "w").write(s)
print("patched", flush=True)

print("=== 4. build llama.cpp (CUDA, multi-arch) ===", flush=True)
if not shutil.which("nvcc"):
    raise SystemExit("nvcc not found: this is a CPU-only session and the CUDA build needs it. "
                     "Re-run with any GPU accelerator (this build is ~10 min, about 0.25h quota, the "
                     "GPU is not otherwise used), or attach a prebuilt llama.cpp CUDA dataset.")
subprocess.run(["cmake", "-B", os.path.join(LCPP, "build"), "-S", LCPP, "-DGGML_CUDA=ON",
                "-DLLAMA_CURL=OFF", "-DCMAKE_CUDA_ARCHITECTURES=60;75;89", "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs"], check=True)
subprocess.run(["cmake", "--build", os.path.join(LCPP, "build"), "--config", "Release", "-j",
                "--target", "llama-server", "llama-quantize"], check=True)
BIN = os.path.join(LCPP, "build", "bin")
print("built:", [f for f in os.listdir(BIN) if f.startswith("llama-")], flush=True)

print("=== 5. convert to GGUF and quantize to Q8_0 ===", flush=True)
bf16 = "/kaggle/working/arc-qwen4b-bf16.gguf"
q8 = "/kaggle/working/arc-qwen4b-q8_0.gguf"
subprocess.run([sys.executable, os.path.join(LCPP, "convert_hf_to_gguf.py"), MODEL,
                "--outfile", bf16, "--outtype", "bf16"], check=True)
subprocess.run([os.path.join(BIN, "llama-quantize"), bf16, q8, "Q8_0"], check=True)
os.remove(bf16)
print("GGUF size GB: %.2f" % (os.path.getsize(q8) / 1e9), flush=True)

print("=== 6. stage the offline assets ===", flush=True)
OUT = "/kaggle/working/assets"
os.makedirs(OUT, exist_ok=True)
for f in [q8, os.path.join(BIN, "llama-server"), os.path.join(BIN, "llama-quantize"),
          os.path.join(LCPP, "convert_lora_to_gguf.py"), os.path.join(LCPP, "convert_hf_to_gguf.py")]:
    shutil.copy(f, OUT)
shutil.copytree(os.path.join(LCPP, "conversion"), os.path.join(OUT, "conversion"), dirs_exist_ok=True)
shutil.copytree(os.path.join(LCPP, "gguf-py"), os.path.join(OUT, "gguf-py"), dirs_exist_ok=True)
os.chmod(os.path.join(OUT, "llama-server"), 0o755)
shutil.rmtree(LCPP, ignore_errors=True)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print("  ", f, ("%.1f MB" % (os.path.getsize(p) / 1e6)) if os.path.isfile(p) else "<dir>", flush=True)
print("DONE in %.0f s. Save Version, then in the submission notebook: Add Input -> Notebook Output -> this notebook." % (time.time() - t0), flush=True)
'''

cells = [
md(r"""
# Prep, build the offline assets (run once, internet ON)

This is ONE code cell. It builds a Linux llama.cpp (CUDA) and converts the model to a Q8_0 GGUF with
the two patches the custom 16-token vocab needs, then stages both in `/kaggle/working/assets`. The
submission notebook (run offline) attaches this notebook's output.

How to run, this is the part that has been tripping us up:
1. `+ Add Input` -> Models or Notebooks -> the `qwen3-4b-grids15-sft139` model (config.json +
   safetensors).
2. Settings: Internet ON. Accelerator: any GPU (the build needs nvcc; if you pick None it stops at
   step 4 with a clear message).
3. Click into the code cell and press the Run button (or Shift+Enter), or use Run all. WATCH it run
   live, it takes several minutes and prints "=== 1 ===" through "=== 6 ===". Do NOT use Quick Save,
   that only saves the file without running it (a ~20 second no-op, which is what kept happening).
4. After it prints DONE, then Save Version to commit the output.
"""),

code(CELL),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.x"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "arc_prep.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
json.load(open(out, encoding="utf-8"))
print("wrote", os.path.abspath(out), "cells:", len(cells))
