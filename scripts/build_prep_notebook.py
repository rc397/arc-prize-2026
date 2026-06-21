"""Build notebooks/arc_prep.ipynb, the one-time prep notebook (run on Kaggle with internet ON) that
produces the two offline assets the submission notebook needs: a Linux llama.cpp build and the Q8_0
GGUF (converted from the attached Kaggle model with our two custom-vocab patches). Its /kaggle/working
output is then attached to the offline submission notebook. Reproduces docs/gguf-conversion.md."""
import json, os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# Prep, build the offline assets (run once, internet ON)

ARC-AGI-2 is an offline code competition, so the submission notebook runs with no internet. This
notebook does the one-time setup that needs the network: it builds a Linux llama.cpp (CUDA) and
converts the model to a Q8_0 GGUF with the two patches the custom 16-token vocab needs. Everything
lands in `/kaggle/working`, then you Save Version and attach this notebook's output to the
submission notebook.

Settings: Internet ON, any GPU accelerator (the GPU is only needed so the CUDA toolkit is present
for the build; nothing here stresses it). Takes ~15 minutes.

First, attach the base model: `+ Add Input` -> Models -> `sorokin/qwen3_4b_grids15_sft139`.
"""),

code(r"""
import os, glob, subprocess, shutil, sys
B = '/kaggle/working/build'
os.makedirs(B, exist_ok=True)

# locate the attached Kaggle model (the HF transformers weights)
hits = glob.glob('/kaggle/input/**/config.json', recursive=True)
MODEL = os.path.dirname([h for h in hits if 'qwen3_4b_grids15' in h or 'transformers' in h][0]) if hits else None
print('MODEL dir:', MODEL)
assert MODEL and os.path.exists(os.path.join(MODEL, 'config.json')), 'attach sorokin/qwen3_4b_grids15_sft139 via Add Input'
"""),

md("## 1. Clone llama.cpp at the validated release and apply the two vocab patches"),

code(r"""
LCPP = '/kaggle/working/llama.cpp'
if not os.path.isdir(LCPP):
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'b9724',
                    'https://github.com/ggml-org/llama.cpp', LCPP], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                os.path.join(LCPP, 'gguf-py'), 'sentencepiece', 'protobuf'], check=True)

base = os.path.join(LCPP, 'conversion', 'base.py')
s = open(base).read()
# Patch 1: the custom WordLevel pre-tokenizer hash is unrecognized; its regex is the qwen2 regex.
if '3f7ffce2c85ce59d277e21c40359aa7326aaa65660c8166288ec5137c9cc1967' not in s:
    needle = 'raise NotImplementedError("BPE pre-tokenizer was not recognized'
    assert needle in s, 'patch 1 anchor (the raise) not found, llama.cpp layout changed'
    cut = s.rfind('if res is None:', 0, s.index(needle))
    assert cut != -1, 'patch 1 anchor (if res is None) not found'
    s = (s[:cut]
         + 'if chkhsh == "3f7ffce2c85ce59d277e21c40359aa7326aaa65660c8166288ec5137c9cc1967":\n'
         + '            res = "qwen2"\n        '
         + s[cut:])
# Patch 2: WordLevel has no merges; write a non-empty list (two throwaway role merges).
if 'add_token_merges(["user assistant"' not in s:
    anchor = 'special_vocab.add_to_gguf(self.gguf_writer)'
    assert anchor in s, 'patch 2 anchor not found'
    s = s.replace(anchor, anchor
                  + '\n        if not getattr(special_vocab, "merges", None):'
                  + '\n            self.gguf_writer.add_token_merges(["user assistant", "assistant user"])', 1)
open(base, 'w').write(s)
print('patches applied')
"""),

md("## 2. Build llama.cpp (CUDA, multi-arch so the binary runs on P100 / T4 / L4)"),

code(r"""
if not shutil.which('nvcc'):
    raise SystemExit(
        'nvcc not found: this is a CPU-only session and the CUDA build needs it. Either re-run '
        'with any GPU accelerator (this build is ~15 min, about 0.25h of the 30h quota, and the GPU '
        'is not otherwise used), or attach a prebuilt llama.cpp CUDA dataset and skip this cell.')
subprocess.run(['cmake', '-B', os.path.join(LCPP, 'build'), '-S', LCPP,
                '-DGGML_CUDA=ON', '-DLLAMA_CURL=OFF',
                '-DCMAKE_CUDA_ARCHITECTURES=60;75;89', '-DCMAKE_BUILD_TYPE=Release'], check=True)
subprocess.run(['cmake', '--build', os.path.join(LCPP, 'build'), '--config', 'Release', '-j',
                '--target', 'llama-server', 'llama-quantize'], check=True)
BIN = os.path.join(LCPP, 'build', 'bin')
print('built:', [f for f in os.listdir(BIN) if f.startswith('llama-')])
"""),

md("## 3. Convert the model to GGUF and quantize to Q8_0"),

code(r"""
bf16 = '/kaggle/working/arc-qwen4b-bf16.gguf'
q8 = '/kaggle/working/arc-qwen4b-q8_0.gguf'
subprocess.run([sys.executable, os.path.join(LCPP, 'convert_hf_to_gguf.py'), MODEL,
                '--outfile', bf16, '--outtype', 'bf16'], check=True)
subprocess.run([os.path.join(BIN, 'llama-quantize'), bf16, q8, 'Q8_0'], check=True)
os.remove(bf16)
print('GGUF size GB: %.2f' % (os.path.getsize(q8) / 1e9))
"""),

md("## 4. Stage the offline assets into the notebook output"),

code(r"""
OUT = '/kaggle/working/assets'
os.makedirs(OUT, exist_ok=True)
shutil.copy(q8, OUT)
shutil.copy(os.path.join(BIN, 'llama-server'), OUT)
shutil.copy(os.path.join(BIN, 'llama-quantize'), OUT)
# for TTT (v2): the adapter-to-GGUF converter plus the python it imports
shutil.copy(os.path.join(LCPP, 'convert_lora_to_gguf.py'), OUT)
shutil.copy(os.path.join(LCPP, 'convert_hf_to_gguf.py'), OUT)
shutil.copytree(os.path.join(LCPP, 'conversion'), os.path.join(OUT, 'conversion'), dirs_exist_ok=True)
shutil.copytree(os.path.join(LCPP, 'gguf-py'), os.path.join(OUT, 'gguf-py'), dirs_exist_ok=True)
os.chmod(os.path.join(OUT, 'llama-server'), 0o755)
shutil.rmtree(LCPP, ignore_errors=True)            # keep the output small, drop the source tree
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    sz = os.path.getsize(p) / 1e6 if os.path.isfile(p) else 0
    print('%-28s %s' % (f, ('%.1f MB' % sz) if sz else '<dir>'))
print('\nDONE. Save Version, then in the submission notebook: Add Input -> Notebook Output -> this notebook.')
"""),

md(r"""
## Fallback

If the conversion step ever breaks (a llama.cpp layout change versus release b9724), the reliable
alternative is to upload the already-validated GGUF from the rig
(`D:\arc\models\arc-qwen4b-q8_0.gguf`) as a Kaggle Dataset and skip cells 1-3, the build of
`llama-server` in cell 2 is still needed.
"""),
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
