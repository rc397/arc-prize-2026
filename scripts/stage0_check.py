import sys, json, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated

CH = r"D:\arc\data\kaggle\arc-agi_training_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_training_solutions.json"
URL = "http://127.0.0.1:8080/completion"

res = load_consolidated(CH, SOL, n=20)
tasks, sols = res if isinstance(res, tuple) else (res, None)


def query(prompt_ids, n_predict=1500):
    body = json.dumps({"prompt": prompt_ids, "n_predict": n_predict,
                       "temperature": 0.0, "cache_prompt": False}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


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


def solution_for(tid, task):
    if sols and tid in sols:
        return sols[tid][0]
    return task["test"][0].get("output")


tids = list(tasks.keys())[:15]
correct = 0
total = 0
for tid in tids:
    task = tasks[tid]
    pid = build_prompt_ids(task["train"], task["test"][0]["input"])
    try:
        resp = query(pid)
    except Exception as e:
        print("tid=%s ERROR %s" % (tid, e))
        total += 1
        continue
    pred = parse_grid(resp.get("content", ""))
    exp = solution_for(tid, task)
    ok = pred == exp
    correct += int(ok)
    total += 1
    pd = (len(pred), len(pred[0])) if pred else None
    ed = (len(exp), len(exp[0])) if exp else None
    print("tid=%s ok=%s prompt=%d out_tok=%s pred=%s exp=%s" % (
        tid, ok, len(pid), resp.get("tokens_predicted"), pd, ed))
print("RESULT %d/%d correct (Q8_0, greedy, no TTT)" % (correct, total))
