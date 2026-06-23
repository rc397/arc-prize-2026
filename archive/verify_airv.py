"""E1 verifier prototype (no TTT). For each augmentation, score how well the model reproduces the
task's own train pairs (leave-one-out), then select test candidates by a vote WEIGHTED by that
verify score, instead of a plain majority vote. Candidates from augmentations that cannot even
reproduce the train pairs get downweighted. Generates everything once and scores plain vs verified
on the same candidates, so the lift is isolated. arg1 N (task subset, 0=all)."""
import sys, json, time, subprocess, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated, write_submission
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES
from src.arc.select import select_top2
from src.arc.eval import score_submission

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
SERVER_ARGS = [r"D:\arc\llamacpp\llama-server.exe", "-m", r"D:\arc\models\arc-qwen4b-q8_0.gguf",
               "-ngl", "99", "-c", "16384", "-np", "1", "--host", "127.0.0.1", "--port", "8080",
               "--no-webui", "--no-cache-prompt", "--no-cache-idle-slots"]
AUGS = [Augmentation(d4=n) for n in D4_NAMES]


def healthy():
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3).read()).get("status") == "ok"
    except Exception:
        return False


def ensure_server():
    if healthy():
        return True
    subprocess.Popen(SERVER_ARGS, creationflags=0x08000000,
                     stdout=open(r"D:\arc\srv.log", "a"), stderr=open(r"D:\arc\srv.err", "a"))
    for _ in range(60):
        time.sleep(2)
        if healthy():
            return True
    return False


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
                ensure_server()
    return None


def parse(text):
    rows = [[int(c) for c in s.strip()] for s in text.split("\n")
            if s.strip() and all(c in "0123456789" for c in s.strip())]
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None


def verify_score(atrain):
    """Leave-one-out over the augmented train pairs: fraction the model reproduces exactly."""
    if len(atrain) < 2:
        return 1.0
    ok = 0
    for i in range(len(atrain)):
        ctx = [atrain[j] for j in range(len(atrain)) if j != i]
        r = query(build_prompt_ids(ctx, atrain[i]["input"]))
        g = parse(r.get("content", "")) if r else None
        if g == atrain[i]["output"]:
            ok += 1
    return ok / len(atrain)


def select_weighted(cands):
    """cands: list of (grid, weight). Rank distinct grids by summed weight, then by raw count."""
    agg = {}
    for g, w in cands:
        if not g:
            continue
        k = tuple(tuple(r) for r in g)
        if k not in agg:
            agg[k] = [0.0, 0, g]
        agg[k][0] += w
        agg[k][1] += 1
    if not agg:
        return [[0]], [[0]]
    ranked = sorted(agg.values(), key=lambda v: (v[0], v[1]), reverse=True)
    return ranked[0][2], (ranked[1][2] if len(ranked) > 1 else ranked[0][2])


ensure_server()
tasks, sols = load_consolidated(CH, SOL, n=(N or None))
plain_sub, ver_sub = {}, {}
t0 = time.time()
for idx, tid in enumerate(tasks):
    task = tasks[tid]
    plain = [[] for _ in task["test"]]
    weighted = [[] for _ in task["test"]]
    vscores = []
    for aug in AUGS:
        at = apply_task(aug, task)
        vs = verify_score(at["train"])
        vscores.append(round(vs, 2))
        for ti in range(len(task["test"])):
            r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
            g = parse(r.get("content", "")) if r else None
            if g:
                inv = invert_grid(aug, g)
                plain[ti].append((inv, 0.0))
                weighted[ti].append((inv, vs))
    plain_sub[tid] = [dict(zip(("attempt_1", "attempt_2"), select_top2(plain[ti]))) for ti in range(len(task["test"]))]
    ver_sub[tid] = [dict(zip(("attempt_1", "attempt_2"), select_weighted(weighted[ti]))) for ti in range(len(task["test"]))]
    write_submission(plain_sub, tasks, r"D:\arc\experiments\airv_plain_eval.json")
    write_submission(ver_sub, tasks, r"D:\arc\experiments\airv_verified_eval.json")
    print("[%d/%d] %s verify_per_aug=%s t=%.0fs" % (
        idx + 1, len(tasks), tid, vscores, time.time() - t0), flush=True)
subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
s = {t: sols[t] for t in tasks if t in sols}
ap, cp, npn = score_submission(plain_sub, s)
av, cv, nvn = score_submission(ver_sub, s)
print("PLAIN vote    : %d/%d = %.4f" % (cp, npn, ap), flush=True)
print("VERIFIED vote : %d/%d = %.4f  (delta %+d)" % (cv, nvn, av, cv - cp), flush=True)
