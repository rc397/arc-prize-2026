import subprocess, os, sys
MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CONVERT = r"D:\arc\llama.cpp\convert_lora_to_gguf.py"
ADIR = r"D:\arc\experiments\loras\135a2760"
OUT = r"D:\arc\models\loras\test2.gguf"
for exe in ["python", sys.executable]:
    if os.path.exists(OUT):
        os.remove(OUT)
    try:
        r = subprocess.run([exe, CONVERT, ADIR, "--base", MODEL, "--outfile", OUT, "--outtype", "f16"],
                           capture_output=True, text=True)
        print("exe=%r rc=%s exists=%s err=%r" % (exe, r.returncode, os.path.exists(OUT), (r.stderr or "")[-120:]))
    except Exception as e:
        print("exe=%r EXCEPTION %r" % (exe, e))
