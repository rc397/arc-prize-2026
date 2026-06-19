import sys, json, time, math, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission

URL = "http://127.0.0.1:8080/completion"
HEALTH = "http://127.0.0.1:8080/health"
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0    # 0 = all tasks
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1    # keep 1: single stream is stable
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
OUT = r"D:\arc\experiments\airv_nottt_eval.json"
SERVER_ARGS = [r"D:\arc\llamacpp\llama-server.exe", "-m", r"D:\arc\models\arc-qwen4b-q8_0.gguf",
               "-ngl", "99", "-c", "16384", "-np", "1", "--host", "127.0.0.1", "--port", "8080",
               "--no-webui", "--no-cache-prompt", "--no-cache-idle-slots"]
AUGS = [Augmentation(d4=name) for name in D4_NAMES]


def server_healthy():
    try:
        with urllib.request.urlopen(HEALTH, timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def ensure_server():
    """Bring the server up (and restart it after an intermittent crash). The eval owns the
    server lifecycle because it is the process that reliably persists."""
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
                ensure_server()  # restart a crashed server, then retry
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
    def one(aug):
        atask = apply_task(aug, task)
        pid = build_prompt_ids(atask["train"], atask["test"][ti]["input"])
        resp = query(pid)
        if resp is None:
            return None
        g = parse_grid(resp.get("content", ""))
        if g is None:
            return None
        return (invert_grid(aug, g), seq_logprob(resp), resp.get("tokens_predicted", -1))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = [r for r in ex.map(one, AUGS) if r is not None]
    cands = [(g, lp) for g, lp, _ in out]
    max_out = max((n for _, _, n in out), default=0)
    return cands, max_out


def main():
    ensure_server()
    tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
    submission = {}
    tids = list(tasks.keys())
    t0 = time.time()
    consec_zero = 0
    for i, tid in enumerate(tids):
        task = tasks[tid]
        attempts = []
        ncands = max_out = 0
        for ti in range(len(task["test"])):
            cands, mo = candidates_for(task, ti)
            ncands, max_out = max(ncands, len(cands)), max(max_out, mo)
            a1, a2 = select_top2(cands)
            attempts.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = attempts
        consec_zero = consec_zero + 1 if ncands == 0 else 0
        dt = time.time() - t0
        print("[%d/%d] %s tests=%d cands=%d/%d max_out=%d t=%.1fs avg=%.1fs" % (
            i + 1, len(tids), tid, len(attempts), ncands, len(AUGS), max_out, dt, dt / (i + 1)),
            flush=True)
        if consec_zero >= 5:
            print("ABORT: 5 consecutive 0-candidate tasks (server unrecoverable).", flush=True)
            write_submission(submission, tasks, OUT)
            sys.exit(2)
    write_submission(submission, tasks, OUT)
    if sols:
        acc, c, n = score_submission(submission, sols)
        print("SCORE acc=%.4f correct=%d total=%d" % (acc, c, n), flush=True)


main()
