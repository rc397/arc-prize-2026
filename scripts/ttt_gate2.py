import sys, json, time, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
tasks, sols = load_consolidated(CH, SOL, n=3)
tid = list(tasks.keys())[0]                      # 0934a4d8, the task we trained the adapter on
task = tasks[tid]
prompt = build_prompt_ids(task["train"], task["test"][0]["input"])
sol = sols[tid][0]


def set_lora(scale):
    body = json.dumps([{"id": 0, "scale": scale}]).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/lora-adapters", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def gen():
    body = json.dumps({"prompt": prompt, "n_predict": 512, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return r.get("content", ""), r.get("tokens_predicted", -1), time.time() - t


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


set_lora(0.0)
c0, n0, t0 = gen()
g0 = parse(c0)
set_lora(1.0)
c1, n1, t1 = gen()
g1 = parse(c1)
print("ADAPTER OFF: out_tok=%d  %.1fs  %.1f tok/s  correct=%s" % (n0, t0, n0 / max(t0, 0.01), g0 == sol), flush=True)
print("ADAPTER ON:  out_tok=%d  %.1fs  %.1f tok/s  correct=%s" % (n1, t1, n1 / max(t1, 0.01), g1 == sol), flush=True)
print("adapter_changes_output=%s" % (g0 != g1), flush=True)
