"""Dump airv_votes.csv for fig_airv_votes (B4): a no-TTT AIRV run that records, per test output, each
of the 8 D4 transforms' agreement with the voted top-1 grid, the top-1 vote count (1-8), and whether
the top-1 answer is correct. Base Q8_0, no adapter. arg1 = N tasks (default 120)."""
import sys, json, csv, time, subprocess, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES

MODEL_GGUF = r"D:\arc\models\arc-qwen4b-q8_0.gguf"
SERVER = r"D:\arc\llamacpp\llama-server.exe"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def health():
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def stop_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1)


def start_server():
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", "16384", "-np", "1", "--host", "127.0.0.1",
            "--port", "8080", "--no-webui", "--no-cache-prompt"]
    subprocess.Popen(args, creationflags=0x08000000,
                     stdout=open(r"D:\arc\srv.log", "a"), stderr=open(r"D:\arc\srv.err", "a"))
    for _ in range(120):
        time.sleep(2)
        if health():
            return True
    return False


def query(prompt, n_predict=1024, retries=3):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": False}).encode()
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
    rows = [[int(c) for c in s.strip()] for s in text.split("\n")
            if s.strip() and all(c in "0123456789" for c in s.strip())]
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None


def key(g):
    return tuple(tuple(r) for r in g) if g else None


tasks, sols = load_consolidated(CH, SOL, n=None)
tids = list(tasks)[:N]
stop_server()
assert start_server(), "server failed to start"
rows = []
t0 = time.time()
for j, tid in enumerate(tids):
    task = tasks[tid]
    for ti in range(len(task["test"])):
        preds = []
        for aug in AUGS:
            at = apply_task(aug, task)
            r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
            g = parse(r.get("content", "")) if r else None
            preds.append(invert_grid(aug, g) if g else None)
        agg = {}
        for g in preds:
            if g:
                agg[key(g)] = agg.get(key(g), 0) + 1
        top1 = max(agg, key=lambda k: agg[k]) if agg else None
        votes = agg[top1] if top1 else 0
        match = [int(key(g) == top1) if g else 0 for g in preds]
        sol = sols[tid][ti] if (tid in sols and ti < len(sols[tid])) else None
        correct = int(top1 is not None and sol is not None and [list(r) for r in top1] == sol)
        rows.append([tid, ti] + match + [votes, correct])
    print("[%d/%d] %s t=%.0fs" % (j + 1, len(tids), tid, time.time() - t0), flush=True)
stop_server()
hdr = ["task_id", "out_idx"] + ["m_" + n for n in D4_NAMES] + ["top1_votes", "correct"]
with open(r"D:\arc\experiments\airv_votes.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(rows)
print("airv_votes.csv:", len(rows), "outputs", flush=True)
