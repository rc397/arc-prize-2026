import sys, json, time, urllib.request
sys.path.insert(0, r"D:\arc")
from src.arc.model import build_prompt_ids
from src.arc.data import load_consolidated

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
tasks, _ = load_consolidated(CH, None, n=3)
task = tasks[list(tasks.keys())[0]]
prompt = build_prompt_ids(task["train"], task["test"][0]["input"])


def set_lora(scale):
    body = json.dumps([{"id": 0, "scale": scale}]).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/lora-adapters", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def gen(n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.0,
                       "cache_prompt": False, "ignore_eos": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r.get("tokens_predicted", -1), time.time() - t


for scale in [0.0, 1.0]:
    set_lora(scale)
    n1, t1 = gen(1)
    n2, t2 = gen(256)
    dec = (t2 - t1) / max(n2 - n1, 1)
    print("scale=%.1f  prefill=%.2fs  decode=%.3fs/tok = %.1f tok/s" % (
        scale, t1, dec, 1.0 / max(dec, 0.001)), flush=True)
