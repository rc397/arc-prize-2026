"""Keep/drop gate for the self-trained base. Convert qwen3-4b-arc2ft to a Q8_0 GGUF (the rig's
convert_hf_to_gguf is already vocab-patched) and run the no-TTT AIRV eval on the 120 public tasks.
Ship the FT base only if no-TTT(FT) beats the stock 12.79% baseline (and later FT+TTT beats 16.28%).
"""
import os, sys, subprocess

FT    = r"D:\arc\experiments\ft_pilot\qwen3-4b-arc2ft"
BF16  = r"D:\arc\models\arc2ft-bf16.gguf"
Q8    = r"D:\arc\models\arc2ft-q8_0.gguf"
CONV  = r"D:\arc\llama.cpp\convert_hf_to_gguf.py"
QUANT = r"D:\arc\llamacpp\llama-quantize.exe"
EVAL  = r"D:\arc\scripts\run_airv_llamacpp.py"
FTOUT = r"D:\arc\experiments\airv_nottt_ft_eval.json"

print("=== 1. convert FT -> bf16 GGUF ===", flush=True)
subprocess.run([sys.executable, CONV, FT, "--outfile", BF16, "--outtype", "bf16"], check=True)
print("=== 2. quantize -> Q8_0 ===", flush=True)
subprocess.run([QUANT, BF16, Q8, "Q8_0"], check=True)
assert os.path.exists(Q8) and os.path.getsize(Q8) > 3e9, "Q8 GGUF missing or too small"
print("=== 3. GGUF OK %.2f GB; no-TTT AIRV eval on 120 public (baseline 12.79%%) ===" % (os.path.getsize(Q8) / 1e9,), flush=True)
subprocess.run([sys.executable, EVAL, "0", "1", Q8, FTOUT], check=True)
print("=== GATE DONE: the SCORE line above is no-TTT(FT); keep only if > 0.1279 ===", flush=True)
