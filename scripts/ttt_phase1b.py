"""Phase 1b: convert each saved PEFT adapter to a GGUF adapter. Runs as its own process after
phase 1 has exited and freed all VRAM, so the convert (which imports torch) cannot collide with
a loaded model. Writes the task-id to GGUF-path manifest used by phase 2."""
import sys, os, json, subprocess

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CONVERT = r"D:\arc\llama.cpp\convert_lora_to_gguf.py"
ADAPTER_TMP = r"D:\arc\experiments\loras"
GGUF_DIR = r"D:\arc\models\loras"
LIST = r"D:\arc\experiments\adapter_list.json"
MANIFEST = r"D:\arc\experiments\ttt_manifest.json"

os.makedirs(GGUF_DIR, exist_ok=True)
saved = json.load(open(LIST))
manifest = {}
for i, tid in enumerate(saved):
    adir = os.path.join(ADAPTER_TMP, tid)
    gguf = os.path.join(GGUF_DIR, tid + ".gguf")
    if not os.path.exists(gguf):
        with open(os.path.join(ADAPTER_TMP, "convert.log"), "ab") as lf:
            subprocess.run([sys.executable, CONVERT, adir, "--base", MODEL, "--outfile", gguf,
                            "--outtype", "f16"], stdout=lf, stderr=lf,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    ok = os.path.exists(gguf)
    if ok:
        manifest[tid] = gguf
    json.dump(manifest, open(MANIFEST, "w"))
    print("[%d/%d] %s conv_ok=%s" % (i + 1, len(saved), tid, ok), flush=True)
print("MANIFEST %d adapters -> %s" % (len(manifest), MANIFEST), flush=True)
