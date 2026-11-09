"""Expert-iteration self-training (STaR) for the ARC-AGI-2 base, run on the rig.

One round:
  harvest : AIRV over the official ARC-2 TRAINING tasks with the current model; keep the tasks the
            model can already reach (oracle any-match vs ground truth) as the curriculum.
  corpus  : mint the leave-one-out corpus from ONLY those self-solved tasks (reuses build_ft_corpus).
  train   : bf16 LoRA distil the base on that curriculum (reuses finetune_base.py).
  gate    : convert + quantize + no-TTT AIRV on the 120 public eval; keep only if > stock 12.79%.

Iterate: the improved model harvests again and reaches more tasks (the STaR loop).

The risk: on the training set a verified-correct solution equals the ground-truth target, so this
shares a failure mode with the continued FT that already failed the gate. The levers that make it a
different experiment are the self-solve CURRICULUM, on-policy iteration, and augmentation multiplicity.
Ship only if it clears the gate; otherwise it goes in the paper as a negative result.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

ROOT = r"D:\arc"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.arc.model import build_prompt_ids                                  # noqa: E402
from src.arc.data import load_consolidated                                  # noqa: E402
from src.arc.augment import Augmentation, apply_task, invert_grid, D4_NAMES  # noqa: E402
from src.arc.select import select_top2                                       # noqa: E402
from build_ft_corpus import official_pools, mint                            # noqa: E402

AUGS = [Augmentation(d4=name) for name in D4_NAMES]
PORT = 8080
SERVER = None
CONV = r"D:\arc\llama.cpp\convert_hf_to_gguf.py"
QUANT = r"D:\arc\llamacpp\llama-quantize.exe"
EVAL = os.path.join(ROOT, "scripts", "run_airv_llamacpp.py")


def server_healthy():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def ensure_server(model):
    global SERVER
    if server_healthy():
        return True
    args = [r"D:\arc\llamacpp\llama-server.exe", "-m", model, "-ngl", "99", "-c", "16384",
            "-np", "1", "--host", "127.0.0.1", "--port", str(PORT),
            "--no-webui", "--no-cache-prompt", "--no-cache-idle-slots"]
    SERVER = subprocess.Popen(args, creationflags=0x08000000,
                              stdout=open(r"D:\arc\srv_star.log", "a"),
                              stderr=open(r"D:\arc\srv_star.err", "a"))
    for _ in range(45):
        time.sleep(2)
        if server_healthy():
            return True
    return False


def stop_server():
    global SERVER
    if SERVER is not None:
        try:
            SERVER.terminate()
        except Exception:
            pass
        SERVER = None
    time.sleep(3)


def seq_logprob(resp):
    total = 0.0
    for t in resp.get("completion_probabilities") or []:
        lp = t.get("logprob")
        if lp is None and t.get("prob") is not None:
            lp = math.log(max(t["prob"], 1e-9))
        if lp is not None:
            total += lp
    return total


def query(prompt_ids, model, n_predict=1536, retries=4):
    body = json.dumps({"prompt": prompt_ids, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False, "n_probs": 1}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception:
            if attempt < retries - 1:
                ensure_server(model)
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


def candidates_for(task, ti, model):
    out = []
    for aug in AUGS:
        atask = apply_task(aug, task)
        pid = build_prompt_ids(atask["train"], atask["test"][ti]["input"])
        resp = query(pid, model)
        if resp is None:
            continue
        g = parse_grid(resp.get("content", ""))
        if g is not None:
            out.append((invert_grid(aug, g), seq_logprob(resp)))
    return out


def harvest(model, challenges, solutions, limit, out_dir):
    """Self-solved = every test output of a task is reproduced by some D4 candidate (oracle any-match)."""
    ensure_server(model)
    tasks, sols = load_consolidated(challenges, solutions, n=(limit or None))
    solved, voted = [], []
    t0 = time.time()
    for i, tid in enumerate(tasks):
        task = tasks[tid]
        ok_oracle, ok_voted = True, True
        for ti in range(len(task["test"])):
            cands = candidates_for(task, ti, model)
            truth = sols[tid][ti]
            if not any(g == truth for g, _ in cands):
                ok_oracle = False
            a1, a2 = select_top2(cands)
            if truth != a1 and truth != a2:
                ok_voted = False
        if ok_oracle:
            solved.append(tid)
        if ok_voted:
            voted.append(tid)
        dt = time.time() - t0
        print("[harvest %d/%d] %s oracle=%s voted=%s n_oracle=%d t=%.0fs avg=%.1fs"
              % (i + 1, len(tasks), tid, ok_oracle, ok_voted, len(solved), dt, dt / (i + 1)),
              flush=True)
    stop_server()
    vset = set(voted)
    near = [t for t in solved if t not in vset]
    json.dump({"solved": solved, "voted": voted, "near_miss": near, "n_tasks": len(tasks)},
              open(os.path.join(out_dir, "harvest.json"), "w"))
    print("HARVEST: oracle %d, voted %d, near-miss %d / %d tasks"
          % (len(solved), len(voted), len(near), len(tasks)), flush=True)
    return solved


def build_corpus(challenges, solved, out_dir, per_task, kmax, max_ctx, val_frac, seed):
    pools = official_pools(challenges)
    keep = {"arc2_" + t: pools["arc2_" + t] for t in solved if "arc2_" + t in pools}
    print("corpus: %d self-solved pools -> minting %d/task" % (len(keep), per_task), flush=True)
    ex = mint(keep, per_task, kmax, max_ctx, seed)
    nval = max(1, int(len(ex) * val_frac))
    cdir = os.path.join(out_dir, "corpus")
    os.makedirs(cdir, exist_ok=True)
    with open(os.path.join(cdir, "val.jsonl"), "w") as f:
        for e in ex[:nval]:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(cdir, "train.jsonl"), "w") as f:
        for e in ex[nval:]:
            f.write(json.dumps(e) + "\n")
    print("corpus: %d train / %d val -> %s" % (len(ex) - nval, nval, cdir), flush=True)
    return cdir


def train(base, cdir, out_dir, rank, epochs, limit, seed):
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "finetune_base.py"),
           "--model", base, "--corpus", cdir, "--out", out_dir,
           "--rank", str(rank), "--epochs", str(epochs), "--seed", str(seed)]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir, "qwen3-4b-arc2ft")   # finetune_base's merged-checkpoint name


def gate(ckpt, out_dir):
    bf16 = os.path.join(out_dir, "star-bf16.gguf")
    q8 = os.path.join(out_dir, "star-q8_0.gguf")
    ftout = os.path.join(out_dir, "airv_nottt_star_eval.json")
    print("=== gate 1: convert -> bf16 GGUF ===", flush=True)
    subprocess.run([sys.executable, CONV, ckpt, "--outfile", bf16, "--outtype", "bf16"], check=True)
    print("=== gate 2: quantize -> Q8_0 ===", flush=True)
    subprocess.run([QUANT, bf16, q8, "Q8_0"], check=True)
    print("=== gate 3: no-TTT AIRV on 120 public (stock baseline 12.79%) ===", flush=True)
    subprocess.run([sys.executable, EVAL, "0", "1", q8, ftout], check=True)
    print("GATE DONE: the SCORE line above is no-TTT(STaR); keep only if > 0.1279", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\arc\models\arc-qwen4b-q8_0.gguf",
                    help="GGUF solver for harvest + gate (the current round's model)")
    ap.add_argument("--base",
                    default=r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1",
                    help="HF base dir to LoRA-distil")
    ap.add_argument("--challenges", default=r"D:\arc\data\kaggle\arc-agi_training_challenges.json")
    ap.add_argument("--solutions", default=r"D:\arc\data\kaggle\arc-agi_training_solutions.json")
    ap.add_argument("--out", default=r"D:\arc\experiments\star_r1")
    ap.add_argument("--harvest-limit", type=int, default=0, help="0 = all training tasks")
    ap.add_argument("--per-task", type=int, default=80)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--max-ctx-tokens", type=int, default=1600)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--train-limit", type=int, default=0, help="cap distil steps for a pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080, help="llama-server port for harvest")
    ap.add_argument("--curriculum", choices=["solved", "voted", "near_miss"], default="solved",
                    help="harvested set to distil on: solved=oracle any-match, near_miss=oracle but not voted")
    ap.add_argument("--skip-harvest", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="harvest + corpus only")
    ap.add_argument("--do-gate", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end plumbing check")
    a = ap.parse_args()
    global PORT
    PORT = a.port
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        a.harvest_limit, a.per_task, a.train_limit = 12, 6, 60

    if not a.skip_harvest:
        harvest(a.model, a.challenges, a.solutions, a.harvest_limit, a.out)
    hv = json.load(open(os.path.join(a.out, "harvest.json")))
    tids = hv.get(a.curriculum) or hv["solved"]
    print("curriculum=%s -> %d tasks (oracle %d, voted %d, near-miss %d)"
          % (a.curriculum, len(tids), len(hv["solved"]), len(hv.get("voted", [])),
             len(hv.get("near_miss", []))), flush=True)
    if not tids:
        print("empty curriculum; nothing to distil", flush=True)
        return
    cdir = build_corpus(a.challenges, tids, a.out, a.per_task, a.kmax,
                        a.max_ctx_tokens, a.val_frac, a.seed)
    if a.skip_train:
        return
    ckpt = train(a.base, cdir, a.out, a.rank, a.epochs, a.train_limit, a.seed)
    if a.do_gate:
        gate(ckpt, a.out)
    print("SELF-TRAIN ROUND DONE -> %s" % ckpt, flush=True)


if __name__ == "__main__":
    main()
