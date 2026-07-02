"""Robust rig AIRV validation harness on llama.cpp. This is the ONE harness for rig validation.

Why llama.cpp and not HF eager: generation goes through llama-server, every request has a 300s cap plus
retries plus ensure_server() recovery, so a single task can never hang the whole run (the failure mode
that stalled the HF-eager scripts). It is also ~13x faster and far lighter on VRAM.

Robustness: resumes from a partial OUT (skips done tasks), checkpoints after every task, guards every task
with try/except, writes a heartbeat to eval_status.json (a stale ts means a stall), and aborts after 5
dead tasks.

More-candidates lever: K samples per D4 augmentation at temperature TEMP, all fed to the same
consensus-and-NLL select_top2. K=1 TEMP=0 reproduces the greedy 8-augmentation baseline exactly.

Usage: python run_airv_llamacpp.py [N_LIMIT] [WORKERS] [MODEL] [OUT] [K] [TEMP]
  (N_LIMIT 0 = all tasks, K 1 = greedy baseline)

Launch convention (so a job survives the session and is monitorable):
  Start-Process -FilePath <...PythonSoftwareFoundation.Python.3.13...\\python.exe>
    -ArgumentList "-u","D:\\arc\\run_airv_llamacpp.py",<args> -WorkingDirectory "D:\\arc"
    -RedirectStandardOutput "D:\\arc\\airv.out" -RedirectStandardError "D:\\arc\\airv.err" -WindowStyle Hidden
  Detect the job as python3.13 (not python) or read D:\\arc\\eval_status.json. Do not rely on a
  laptop-side ssh monitor, read eval_status.json and OUT on demand instead.
"""
import sys, os, json, time, math, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission
from src.arc.filter import plausible

URL = "http://127.0.0.1:8080/completion"
HEALTH = "http://127.0.0.1:8080/health"
STATUS = r"D:\arc\eval_status.json"
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0    # 0 = all tasks
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1    # keep 1: single stream is stable
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
MODEL = sys.argv[3] if len(sys.argv) > 3 else r"D:\arc\models\arc-qwen4b-q8_0.gguf"
OUT = sys.argv[4] if len(sys.argv) > 4 else r"D:\arc\experiments\airv_nottt_eval.json"
K = int(sys.argv[5]) if len(sys.argv) > 5 else 1         # samples per augmentation (1 = greedy baseline)
TEMP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0  # sampling temperature when K > 1
FILTER = int(sys.argv[7]) if len(sys.argv) > 7 else 0    # 1 = white-box plausibility filter on candidates
SERVER_ARGS = [r"D:\arc\llamacpp\llama-server.exe", "-m", MODEL,
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


def query(prompt_ids, n_predict=1536, retries=4, temperature=0.0):
    body = json.dumps({"prompt": prompt_ids, "n_predict": n_predict,
                       "temperature": temperature, "top_p": 0.95, "cache_prompt": False, "n_probs": 1}).encode()
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
    """K samples per augmentation. K=1 TEMP=0 is the greedy 8-candidate baseline."""
    def one(aug):
        atask = apply_task(aug, task)
        pid = build_prompt_ids(atask["train"], atask["test"][ti]["input"])
        outs = []
        for _ in range(K):
            resp = query(pid, temperature=TEMP)
            if resp is None:
                continue
            g = parse_grid(resp.get("content", ""))
            if g is None:
                continue
            outs.append((invert_grid(aug, g), seq_logprob(resp), resp.get("tokens_predicted", -1)))
        return outs
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        nested = list(ex.map(one, AUGS))
    out = [r for sub in nested for r in sub]
    cands = [(g, lp) for g, lp, _ in out]
    if FILTER:
        ti_input = task["test"][ti]["input"]
        filtered = [(g, lp) for (g, lp) in cands if plausible(g, task, ti_input)]
        cands = filtered or cands   # mirror the v5 notebook select_top2(f or cs): keep the unfiltered set if the filter empties the list
    max_out = max((n for _, _, n in out), default=0)
    return cands, max_out


def heartbeat(idx, tid, ndone, ntotal):
    try:
        json.dump({"idx": idx, "tid": tid, "ts": time.time(), "done": ndone,
                   "total": ntotal, "K": K, "out": OUT}, open(STATUS, "w"))
    except Exception:
        pass


def main():
    ensure_server()
    tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
    tids = list(tasks.keys())
    submission = {}
    if os.path.exists(OUT):
        try:
            raw = json.load(open(OUT))
            for t, att in raw.items():
                if att and att[0].get("attempt_1") != [[0]]:   # keep real predictions, re-run padding/error defaults
                    submission[t] = att
            print("resume: %d/%d real tasks loaded from %s" % (len(submission), len(tids), OUT), flush=True)
        except Exception:
            submission = {}
    t0 = time.time()
    consec_zero = 0
    for i, tid in enumerate(tids):
        if tid in submission:
            continue
        heartbeat(i, tid, len(submission), len(tids))
        task = tasks[tid]
        ncands = max_out = 0
        try:
            attempts = []
            for ti in range(len(task["test"])):
                cands, mo = candidates_for(task, ti)
                ncands, max_out = max(ncands, len(cands)), max(max_out, mo)
                a1, a2 = select_top2(cands)
                attempts.append({"attempt_1": a1, "attempt_2": a2})
            submission[tid] = attempts
        except Exception as e:
            submission[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]
            print("[%d/%d] %s ERROR %s" % (i + 1, len(tids), tid, repr(e)[:100]), flush=True)
        write_submission(submission, tasks, OUT)   # per-task checkpoint, so a crash never loses progress
        consec_zero = consec_zero + 1 if ncands == 0 else 0
        dt = time.time() - t0
        print("[%d/%d] %s tests=%d cands=%d max_out=%d t=%.1fs" % (
            i + 1, len(tids), tid, len(task["test"]), ncands, max_out, dt), flush=True)
        if consec_zero >= 5:
            print("ABORT: 5 consecutive 0-candidate tasks (server unrecoverable).", flush=True)
            write_submission(submission, tasks, OUT)
            sys.exit(2)
    write_submission(submission, tasks, OUT)
    if sols:
        acc, c, n = score_submission(submission, sols)
        print("SCORE acc=%.4f correct=%d total=%d K=%d TEMP=%s" % (acc, c, n, K, TEMP), flush=True)


main()
