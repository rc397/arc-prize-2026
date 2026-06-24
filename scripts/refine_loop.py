"""Verifier-guided selection (the refinement lever) for ARC-AGI-2.

For each test input we generate an answer under every D4 augmentation, as AIRV does, but instead of a
blind vote we VERIFY each augmentation against the task's own train pairs by honest leave-one-out:
predict each held-out train output from the other pairs, under the same augmentation, and score the
fraction reproduced. The test answer is then chosen from the augmentations whose induced rule actually
reproduces the task (the top fidelity tier); when none reproduce any pair we fall back to the plain
AIRV vote. This is the check-against-train lever the one-shot vote lacks.

Why this is not the earlier dead end: our previous attempt weighted votes by the train fidelity of the
TTT-adapted model, which memorises every train pair and so saturates the check. Here the check runs on
the un-adapted base with the pair under test held out, so a high score means the rule generalises, a
genuinely discriminative signal.

Usage: python refine_loop.py [N_LIMIT] [LOO_MAX] [MODEL] [OUT] [PORT]
Validate on the 120 public eval against the 12.79% no-TTT baseline before stacking on TTT or porting.
"""
import sys
import json
import time
import math
import subprocess
import urllib.request

sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission

N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0       # 0 = all 120
LOO_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3       # leave-one-out checks per aug (cost cap)
MODEL = sys.argv[3] if len(sys.argv) > 3 else r"D:\arc\models\arc-qwen4b-q8_0.gguf"
OUT = sys.argv[4] if len(sys.argv) > 4 else r"D:\arc\experiments\airv_refine_eval.json"
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 8080
ACTIVE_K = int(sys.argv[6]) if len(sys.argv) > 6 else 0   # temperature re-samples on shaky tasks (0 = off)
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
SERVER_ARGS = [r"D:\arc\llamacpp\llama-server.exe", "-m", MODEL, "-ngl", "99", "-c", "16384",
               "-np", "1", "--host", "127.0.0.1", "--port", str(PORT),
               "--no-webui", "--no-cache-prompt", "--no-cache-idle-slots"]
AUGS = [Augmentation(d4=name) for name in D4_NAMES]
SERVER = None


def server_healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def ensure_server():
    global SERVER
    if server_healthy():
        return True
    try:
        SERVER = subprocess.Popen(SERVER_ARGS, creationflags=0x08000000,
                                  stdout=open(r"D:\arc\srv_refine.log", "a"),
                                  stderr=open(r"D:\arc\srv_refine.err", "a"))
    except Exception:
        pass
    for _ in range(45):
        time.sleep(2)
        if server_healthy():
            return True
    return False


def query(prompt_ids, n_predict=1536, retries=4, temperature=0.0):
    body = json.dumps({"prompt": prompt_ids, "n_predict": n_predict, "temperature": temperature,
                       "cache_prompt": False, "n_probs": 1}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                         headers={"Content-Type": "application/json"})
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


def gen(train_pairs, test_input, aug, temp=0.0):
    """Generate one answer for test_input given train_pairs, under augmentation aug, inverted back."""
    task = {"train": train_pairs, "test": [{"input": test_input}]}
    atask = apply_task(aug, task)
    pid = build_prompt_ids(atask["train"], atask["test"][0]["input"])
    resp = query(pid, temperature=temp)
    if resp is None:
        return None, 0.0
    g = parse_grid(resp.get("content", ""))
    if g is None:
        return None, 0.0
    return invert_grid(aug, g), seq_logprob(resp)


def aug_fidelity(train_pairs, aug):
    """Honest leave-one-out under aug: predict each held-out train output from the others; fraction hit."""
    n = len(train_pairs)
    if n < 2:
        return 1.0
    idxs = range(n) if not LOO_MAX else range(min(n, LOO_MAX))
    hit = tot = 0
    for i in idxs:
        ctx = [train_pairs[j] for j in range(n) if j != i]
        pred, _ = gen(ctx, train_pairs[i]["input"], aug)
        tot += 1
        if pred == train_pairs[i]["output"]:
            hit += 1
    return hit / max(1, tot)


def refine_one(task, ti):
    """Per-aug generate + leave-one-out verify; keep the test answers from the top fidelity tier."""
    train = task["train"]
    test_in = task["test"][ti]["input"]
    scored = []
    for aug in AUGS:
        fid = aug_fidelity(train, aug)
        g, lp = gen(train, test_in, aug)
        if g is not None:
            scored.append((g, lp, fid))
    if not scored:
        return None, None, 0.0
    best_fid = max(f for _, _, f in scored)
    pool = ([(g, lp) for g, lp, f in scored if f >= best_fid - 1e-9] if best_fid > 0
            else [(g, lp) for g, lp, _ in scored])
    # ACTIVE: when no augmentation fully reproduces the task, spend extra temperature samples and fold
    # them into the vote (verification-gated search; the 16-token model cannot take natural-language feedback).
    if ACTIVE_K and best_fid < 1.0:
        for k in range(ACTIVE_K):
            g, lp = gen(train, test_in, AUGS[k % len(AUGS)], temp=0.7)
            if g is not None:
                pool.append((g, lp))
    a1, a2 = select_top2(pool)
    return a1, a2, best_fid


def main():
    ensure_server()
    tasks, sols = load_consolidated(CH, SOL, n=(N_LIMIT or None))
    submission = {}
    tids = list(tasks.keys())
    t0 = time.time()
    for i, tid in enumerate(tids):
        task = tasks[tid]
        attempts = []
        fids = []
        for ti in range(len(task["test"])):
            a1, a2, fid = refine_one(task, ti)
            attempts.append({"attempt_1": a1, "attempt_2": a2})
            fids.append(fid)
        submission[tid] = attempts
        dt = time.time() - t0
        print("[%d/%d] %s tests=%d bestfid=%.2f t=%.1fs avg=%.1fs"
              % (i + 1, len(tids), tid, len(attempts), max(fids) if fids else 0.0,
                 dt, dt / (i + 1)), flush=True)
    write_submission(submission, tasks, OUT)
    if sols:
        acc, c, n = score_submission(submission, sols)
        print("SCORE acc=%.4f correct=%d total=%d (no-TTT AIRV baseline 0.1279)" % (acc, c, n), flush=True)


main()
