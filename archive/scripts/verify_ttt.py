"""E1 verifier + TTT. Serve the existing per-task val adapters (scale 1.25), and for each
augmentation score how well the ADAPTED model reproduces the task's own train pairs (leave-one-out),
then select test candidates by a vote WEIGHTED by that score. The TTT adapter is trained on the
train pairs, so its verify scores should be far richer than the no-TTT base (which was mostly
all-zero). Reuses loras_val + ttt_manifest_val.json. Compares plain vs verified on the same
candidates. arg1 BATCH (default 30)."""
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
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SCALE = 1.25
MANIFEST = r"D:\arc\experiments\ttt_manifest_val.json"
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
    args = [SERVER, "-m", MODEL_GGUF, "-ngl", "99", "-c", "16384", "-np", "1", "--host", "127.0.0.1",
            "--port", "8080", "--no-webui", "--no-cache-prompt", "--no-cache-idle-slots",
            "--lora-init-without-apply"]
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
    body = json.dumps([{"id": j, "scale": (SCALE if j == idx else 0.0)} for j in range(n_loaded)]).encode()
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
    rows = [[int(c) for c in s.strip()] for s in text.split("\n")
            if s.strip() and all(c in "0123456789" for c in s.strip())]
    if not rows:
        return None
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return rows or None


def verify_score(atrain):
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


manifest = json.load(open(MANIFEST))
tasks, sols = load_consolidated(CH, SOL, n=None)
tids = [t for t in tasks if t in manifest]
plain_sub, ver_sub = {}, {}
t0 = time.time()
for b in range(0, len(tids), BATCH):
    batch = tids[b:b + BATCH]
    stop_server()
    if not start_server([manifest[t] for t in batch]):
        print("server FAILED at %d" % b, flush=True)
        continue
    for k, tid in enumerate(batch):
        try:
            if not health():
                stop_server(); start_server([manifest[t] for t in batch])
            set_adapter(k, len(batch))
            task = tasks[tid]
            plain = [[] for _ in task["test"]]
            weighted = [[] for _ in task["test"]]
            vs_list = []
            for aug in AUGS:
                at = apply_task(aug, task)
                vs = verify_score(at["train"])
                vs_list.append(round(vs, 2))
                for ti in range(len(task["test"])):
                    r = query(build_prompt_ids(at["train"], at["test"][ti]["input"]))
                    g = parse(r.get("content", "")) if r else None
                    if g:
                        inv = invert_grid(aug, g)
                        plain[ti].append((inv, 0.0))
                        weighted[ti].append((inv, vs))
            plain_sub[tid] = [dict(zip(("attempt_1", "attempt_2"), select_top2(plain[ti]))) for ti in range(len(task["test"]))]
            ver_sub[tid] = [dict(zip(("attempt_1", "attempt_2"), select_weighted(weighted[ti]))) for ti in range(len(task["test"]))]
            print("[%d/%d] %s verify=%s t=%.0fs" % (b + k + 1, len(tids), tid, vs_list, time.time() - t0), flush=True)
        except Exception as e:
            plain_sub[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[tid]["test"]]
            ver_sub[tid] = plain_sub[tid]
            print("[%d/%d] %s ERROR %r" % (b + k + 1, len(tids), tid, str(e)[:80]), flush=True)
    write_submission(plain_sub, tasks, r"D:\arc\experiments\airv_ttt_plain_eval.json")
    write_submission(ver_sub, tasks, r"D:\arc\experiments\airv_ttt_verified_eval.json")
stop_server()
s = {t: sols[t] for t in tids if t in sols}
ap, cp, npn = score_submission(plain_sub, s)
av, cv, nvn = score_submission(ver_sub, s)
print("TTT PLAIN vote    : %d/%d = %.4f" % (cp, npn, ap), flush=True)
print("TTT VERIFIED vote : %d/%d = %.4f  (delta %+d)" % (cv, nvn, av, cv - cp), flush=True)
