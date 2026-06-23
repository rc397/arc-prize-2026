"""No-TTT AIRV with an expanded augmentation set (8 D4 + 8 color-permuted) for more votes. Runs on
the 84 val tasks (from the val manifest) so it compares directly with no-TTT and TTT on that set.
arg1 N_LIMIT (0 = all 84; small N for the gate)."""
import sys, json, time, math, subprocess, urllib.request
sys.path.insert(0, r"D:\arc")
import numpy as np
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES, random_augmentation
from src.arc.select import select_top2
from src.arc.eval import score_submission

URL = "http://127.0.0.1:8080/completion"
HEALTH = "http://127.0.0.1:8080/health"
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else r"D:\arc\experiments\ttt_manifest_val.json"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
OUT = (r"D:\arc\experiments\airv_colorperm_eval.json" if not TAG
       else r"D:\arc\experiments\airv_colorperm_%s_eval.json" % TAG)
SERVER_ARGS = [r"D:\arc\llamacpp\llama-server.exe", "-m", r"D:\arc\models\arc-qwen4b-q8_0.gguf",
               "-ngl", "99", "-c", "16384", "-np", "1", "--host", "127.0.0.1", "--port", "8080",
               "--no-webui", "--no-cache-prompt", "--no-cache-idle-slots"]
# 8 D4 (identity color) + 8 color-permuted (random D4 + non-background recolor), seeded.
_rng = np.random.default_rng(20260620)
AUGS = [Augmentation(d4=name) for name in D4_NAMES] + [
    random_augmentation(_rng, use_d4=True, recolor=True, keep_background=True, reorder=False)
    for _ in range(8)]


def server_healthy():
    try:
        with urllib.request.urlopen(HEALTH, timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def ensure_server():
    if server_healthy():
        return True
    try:
        subprocess.Popen(SERVER_ARGS, creationflags=0x08000000,
                         stdout=open(r"D:\arc\srv.log", "a"), stderr=open(r"D:\arc\srv.err", "a"))
    except Exception:
        pass
    for _ in range(45):
        time.sleep(2)
        if server_healthy():
            return True
    return False


def stop_server():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1)


def query(prompt_ids, n_predict=1536, retries=4):
    body = json.dumps({"prompt": prompt_ids, "n_predict": n_predict,
                       "temperature": 0.0, "cache_prompt": False, "n_probs": 1}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception:
            if attempt < retries - 1:
                ensure_server()
    return None


def parse_grid(text):
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


def seq_logprob(resp):
    total = 0.0
    for t in resp.get("completion_probabilities") or []:
        lp = t.get("logprob")
        if lp is None and t.get("prob") is not None:
            lp = math.log(max(t["prob"], 1e-9))
        if lp is not None:
            total += lp
    return total


def candidates_for(task, ti):
    cands = []
    for aug in AUGS:
        atask = apply_task(aug, task)
        resp = query(build_prompt_ids(atask["train"], atask["test"][ti]["input"]))
        if resp is None:
            continue
        g = parse_grid(resp.get("content", ""))
        if g is not None:
            cands.append((invert_grid(aug, g), seq_logprob(resp)))
    return cands


def main():
    ensure_server()
    tasks, sols = load_consolidated(CH, SOL, n=None)
    man = json.load(open(MANIFEST))
    tids = [t for t in tasks if t in man]
    if N_LIMIT:
        tids = tids[:N_LIMIT]
    submission = {}
    t0 = time.time()
    for i, tid in enumerate(tids):
        task = tasks[tid]
        attempts = []
        nc = 0
        for ti in range(len(task["test"])):
            cands = candidates_for(task, ti)
            nc = max(nc, len(cands))
            a1, a2 = select_top2(cands)
            attempts.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = attempts
        write_submission(submission, tasks, OUT)
        dt = time.time() - t0
        print("[%d/%d] %s cands=%d/%d t=%.0fs avg=%.0fs" % (
            i + 1, len(tids), tid, nc, len(AUGS), dt, dt / (i + 1)), flush=True)
    stop_server()
    if sols:
        s = {t: sols[t] for t in tids if t in sols}
        acc, c, n = score_submission(submission, s)
        print("SCORE_COLORPERM acc=%.4f correct=%d total=%d (84-task no-TTT 8aug = 16/125 = 0.128)" % (
            acc, c, n), flush=True)


main()
