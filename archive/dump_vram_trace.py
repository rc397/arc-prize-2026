"""Dump vram_trace.csv for fig_phases (B6): run one task's three-phase bridge (train in PyTorch ->
convert to GGUF -> serve on llama.cpp) while a background thread samples nvidia-smi memory.used every
second, tagging each sample by phase. Shows the trainer and engine never co-reside in 16 GB. arg1 = tid."""
import sys, os, json, csv, time, threading, subprocess, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated

TID = sys.argv[1] if len(sys.argv) > 1 else "4a21e3da"
PY = sys.executable
TRAIN = r"D:\arc\b6_train_one.py"
CONVERT = r"D:\arc\llama.cpp\convert_lora_to_gguf.py"
MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
MODEL_GGUF = r"D:\arc\models\arc-qwen4b-q8_0.gguf"
SERVER = r"D:\arc\llamacpp\llama-server.exe"
ADIR = r"D:\arc\experiments\loras_b6\%s" % TID
LORA_GGUF = r"D:\arc\experiments\loras_b6\%s.gguf" % TID
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

samples = []
stop = threading.Event()


def sampler(t0):
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            if out:
                samples.append((time.time() - t0, float(out[0])))
        except Exception:
            pass
        time.sleep(1)


def health():
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def gen(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": 512, "temperature": 0.0, "cache_prompt": False}).encode()
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass


subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
os.makedirs(os.path.dirname(ADIR), exist_ok=True)
t0 = time.time()
th = threading.Thread(target=sampler, args=(t0,))
th.start()
time.sleep(4)  # idle baseline
bounds = []

s = time.time() - t0  # phase 1: train (subprocess exits -> frees GPU)
subprocess.run([PY, TRAIN, TID, ADIR, "20"], check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
bounds.append(("train", s, time.time() - t0))

s = time.time() - t0  # phase 2: convert
subprocess.run([PY, CONVERT, ADIR, "--base", MODEL, "--outfile", LORA_GGUF, "--outtype", "f16"],
               check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
bounds.append(("convert", s, time.time() - t0))

s = time.time() - t0  # phase 3: serve + generate
subprocess.Popen([SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", "16384", "-np", "1", "--host",
                  "127.0.0.1", "--port", "8080", "--no-webui", "--no-cache-prompt", "--lora", LORA_GGUF],
                 creationflags=0x08000000, stdout=open(r"D:\arc\srv.log", "a"), stderr=open(r"D:\arc\srv.err", "a"))
for _ in range(120):
    time.sleep(2)
    if health():
        break
allt, _ = load_consolidated(CH, SOL, n=None)
task = allt[TID]
for _ in range(3):
    gen(build_prompt_ids(task["train"], task["test"][0]["input"]))
bounds.append(("serve", s, time.time() - t0))
subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
time.sleep(3)
stop.set()
th.join()


def phase_of(t):
    for name, a, b in bounds:
        if a <= t <= b:
            return name
    return "idle"


with open(r"D:\arc\experiments\vram_trace.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_seconds", "mem_used_gb", "phase"])
    for t, mb in samples:
        w.writerow(["%.1f" % t, "%.3f" % (mb / 1024.0), phase_of(t)])
print("vram_trace.csv:", len(samples), "samples; bounds:",
      [(n, round(a), round(b)) for n, a, b in bounds], flush=True)
