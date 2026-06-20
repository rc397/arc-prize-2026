"""Phase 2: read the TTT adapter manifest, serve the base model with per-task adapters on
llama.cpp (in VRAM-sized batches), apply each task's adapter via the hot-swap endpoint, and run
AIRV generation. Scores against the eval solutions."""
import sys, os, json, time, subprocess, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission

MODEL_GGUF = r"D:\arc\models\arc-qwen4b-q8_0.gguf"
SERVER = r"D:\arc\llamacpp\llama-server.exe"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
MANIFEST = r"D:\arc\experiments\ttt_manifest.json"
OUT = r"D:\arc\experiments\airv_ttt_eval.json"
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 30
AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def health():
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def stop_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1)


def start_server(lora_paths):
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", "16384", "-np", "1", "--host",
            "127.0.0.1", "--port", "8080", "--no-webui", "--no-cache-prompt",
            "--no-cache-idle-slots", "--lora-init-without-apply"]
    for p in lora_paths:
        args += ["--lora", p]
    subprocess.Popen(args, creationflags=0x08000000,
                     stdout=open(r"D:\arc\srv.log", "a"), stderr=open(r"D:\arc\srv.err", "a"))
    for _ in range(120):
        time.sleep(2)
        if health():
            return True
    return False


def set_adapter(idx, n_loaded):
    body = json.dumps([{"id": j, "scale": (1.0 if j == idx else 0.0)} for j in range(n_loaded)]).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/lora-adapters", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def query(prompt, n_predict=1024, retries=3):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=300).read())
        except Exception:
            if a < retries - 1:
                time.sleep(5)
    return None


def parse(text):
    rows = []
    for line in text.split("\n"):
        s = line.strip()
        if s and all(c in "0123456789" for c in s):
            rows.append([int(c) for c in s])
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None


manifest = json.load(open(MANIFEST))
tasks, sols = load_consolidated(CH, SOL, n=None)
tids = [t for t in tasks if t in manifest]
submission = {}
t0 = time.time()
for b in range(0, len(tids), BATCH):
    batch = tids[b:b + BATCH]
    batch_paths = [manifest[t] for t in batch]
    stop_server()
    if not start_server(batch_paths):
        print("server FAILED for batch at %d" % b, flush=True)
        continue
    for k, tid in enumerate(batch):
        ncands = 0
        try:
            if not health():                      # restart on crash, then re-apply the adapter
                stop_server()
                start_server(batch_paths)
            set_adapter(k, len(batch))
            task = tasks[tid]
            attempts = []
            for ti in range(len(task["test"])):
                cands = []
                for aug in AUGS:
                    at = apply_task(aug, task)
                    r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
                    if r is None:
                        continue
                    g = parse(r.get("content", ""))
                    if g:
                        cands.append((invert_grid(aug, g), 0.0))
                ncands = max(ncands, len(cands))
                a1, a2 = select_top2(cands)
                attempts.append({"attempt_1": a1, "attempt_2": a2})
            submission[tid] = attempts
        except Exception as e:
            submission[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[tid]["test"]]
            print("  task %s ERROR %r" % (tid, str(e)[:100]), flush=True)
        write_submission(submission, tasks, OUT)
        dt = time.time() - t0
        print("[%d/%d] %s cands=%d/8 t=%.0fs avg=%.0fs" % (
            b + k + 1, len(tids), tid, ncands, dt, dt / (b + k + 1)), flush=True)
stop_server()
if sols:
    acc, c, n = score_submission(submission, sols)
    print("SCORE_TTT acc=%.4f correct=%d total=%d (vs 0.1279 no-TTT)" % (acc, c, n), flush=True)
