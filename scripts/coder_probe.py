"""Induction-arm disjoint-coverage probe. Does a code LLM solve tasks the 4B provably misses?

A coder GGUF served by llama.cpp writes plain-Python programs (def solve(I) over list-of-list ints).
Each sampled program is executed against the task's train pairs in a separate killable process
(pyexec.run_candidate). A program reproducing every train pair is a survivor, and a task counts solved
when one survivor also reproduces every test output. Only the tasks the 4B-TTT envelope misses are
probed, since only those can be disjoint. The envelope is the task-level union over the recorded eval
dumps, the same definition the archived arc-dsl probe used (that probe scored 0 of 101, and this one
attacks its two named causes, the ARC-1-shaped DSL and the too-small proposer).

Unlike the grid server, the coder server keeps prompt caching ON, because each task's prompt is reused
for all K samples, so the long prefill is paid once.

Usage: python coder_probe.py [N_LIMIT] [K] [OUT] [MODEL] [PORT]
       python coder_probe.py bench          (serve, time one generation, print tok/s, exit)

Launch detached with the real python3.13 interpreter, monitor via eval_status_coder.json and OUT
(resume skips finished tasks on relaunch).
"""
import json
import os
import sys
import time
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pyexec

BASE = os.environ.get("ARC_BASE", r"D:\arc")
CH = os.environ.get("ARC_CH", os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_challenges.json"))
SOL = os.environ.get("ARC_SOL", os.path.join(BASE, "data", "kaggle", "arc-agi_evaluation_solutions.json"))
EXP = os.environ.get("ARC_EXP", os.path.join(BASE, "experiments"))
STATUS = os.path.join(BASE, "eval_status_coder.json")
SERVER_EXE = os.path.join(BASE, "llamacpp", "llama-server.exe")

N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "bench" else 0
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(EXP, "coder_probe.json")
MODEL = sys.argv[4] if len(sys.argv) > 4 else os.path.join(BASE, "models", "qwen2.5-coder-14b-instruct-q4_k_m.gguf")
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 8090

CHAT = "http://127.0.0.1:%d/v1/chat/completions" % PORT
HEALTH = "http://127.0.0.1:%d/health" % PORT
SERVER_ARGS = [SERVER_EXE, "-m", MODEL, "-ngl", "99", "-c", "16384", "-np", "1",
               "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--jinja"]

# The 4B coverage envelope, the union over every recorded AIRV eval (identical to the arc-dsl probe).
FBS = ["airv_ttt_val_s125_eval.json", "airv_ttt_val_s100_eval.json", "airv_ttt_val_s150_eval.json",
       "airv_ttt_val_s85_eval.json", "airv_ttt_val_s70_eval.json", "airv_nottt_eval.json"]

MAX_PROMPT_CHARS = 36000   # ~12k tokens at ~3 chars/token, inside the 16k context with output room

# Default zero-shot system prompt. Set ARC_SYSTEM to a file to swap in a few-shot priming prompt
# (worked example solvers) without touching the rest of the harness.
SYSTEM = (open(os.environ["ARC_SYSTEM"], encoding="utf-8").read()
          if os.environ.get("ARC_SYSTEM") else "You are an expert puzzle solver who writes precise Python.")

INSTRUCTIONS = (
    "Study the transformation from each INPUT grid to its OUTPUT grid. The same single rule maps every "
    "input to its output. Write a Python function solve(I) that applies that rule, where I is the input "
    "grid as a list of rows, each row a list of ints 0-9, and the return value is the output grid in "
    "the same form. The function must reproduce every example above exactly and must generalize to new "
    "inputs of possibly different sizes. Use only the Python standard library and numpy. Reply with one "
    "fenced python code block containing all the code needed, including def solve(I)."
)


def grid_text(g):
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def build_prompt(task, drop_from=None):
    train = task["train"] if drop_from is None else task["train"][:drop_from]
    parts = []
    for i, p in enumerate(train, 1):
        parts.append("Example %d\nINPUT\n%s\nOUTPUT\n%s" % (i, grid_text(p["input"]), grid_text(p["output"])))
    parts.append("Test INPUT (produce its output with your solve function)\n%s" % grid_text(task["test"][0]["input"]))
    parts.append(INSTRUCTIONS)
    return "\n\n".join(parts)


def prompt_for(task):
    """Full prompt, dropping trailing train pairs (never below 2) if the task is context-huge."""
    p = build_prompt(task)
    n = len(task["train"])
    while len(p) > MAX_PROMPT_CHARS and n > 2:
        n -= 1
        p = build_prompt(task, drop_from=n)
    return p, n < len(task["train"])


def extract_programs(text):
    """Every fenced code block that defines solve, else the bare def solve tail. Whole blocks are kept
    so helper functions above solve survive (the archived extractor cut at the first return and would
    truncate real Python)."""
    progs = []
    i = 0
    while True:
        a = text.find("```", i)
        if a < 0:
            break
        nl = text.find("\n", a)
        if nl < 0:
            break
        b = text.find("```", nl)
        if b < 0:
            break
        block = text[nl + 1:b]
        if "def solve" in block:
            progs.append(block.strip())
        i = b + 3
    if not progs and "def solve" in text:
        tail = text[text.index("def solve"):]
        progs.append(tail.split("```")[0].strip())
    return progs


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
                         stdout=open(os.path.join(BASE, "srv_coder.log"), "a"),
                         stderr=open(os.path.join(BASE, "srv_coder.err"), "a"))
    except Exception:
        pass
    for _ in range(90):   # a 9 GB model takes a while to load
        time.sleep(2)
        if server_healthy():
            return True
    return False


def chat(prompt, temperature=0.8, max_tokens=1024, retries=4):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens, "cache_prompt": True,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(CHAT, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            return resp["choices"][0]["message"]["content"], resp.get("timings", {})
        except Exception:
            if attempt < retries - 1:
                ensure_server()
    return None, {}


def fourb_solved(ch, sol):
    solved = set()
    for f in FBS:
        try:
            pred = json.load(open(os.path.join(EXP, f)))
        except Exception:
            continue
        for tid in ch:
            if tid not in pred or tid not in sol:
                continue
            ok = True
            for i, truth in enumerate(sol[tid]):
                p = pred[tid][i] if i < len(pred[tid]) else {}
                if p.get("attempt_1") != truth and p.get("attempt_2") != truth:
                    ok = False
                    break
            if ok:
                solved.add(tid)
    return solved


def heartbeat(idx, tid, done, total):
    try:
        json.dump({"idx": idx, "tid": tid, "ts": time.time(), "done": done, "total": total,
                   "K": K, "out": OUT}, open(STATUS, "w"))
    except Exception:
        pass


def bench():
    if not ensure_server():
        print("BENCH: server failed to start", flush=True)
        sys.exit(1)
    t0 = time.time()
    text, timings = chat("Write a Python function that returns the transpose of a list-of-lists matrix.",
                         temperature=0.0, max_tokens=512)
    dt = time.time() - t0
    if text is None:
        print("BENCH: generation failed", flush=True)
        sys.exit(1)
    tps = timings.get("predicted_per_second")
    ntok = timings.get("predicted_n")
    print("BENCH ok: %s tokens in %.1fs, %s tok/s (server-reported), sample len %d chars"
          % (ntok, dt, ("%.1f" % tps) if tps else "n/a", len(text)), flush=True)


def main():
    ch = json.load(open(CH))
    sol = json.load(open(SOL))
    envelope = fourb_solved(ch, sol)
    missed = [tid for tid in ch if tid not in envelope]
    print("4B envelope solves %d/%d, probing %d missed tasks (only these can be disjoint)"
          % (len(envelope), len(ch), len(missed)), flush=True)
    if N_LIMIT:
        missed = missed[:N_LIMIT]

    results = {}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))
            print("resume: %d tasks already probed" % len(results), flush=True)
        except Exception:
            results = {}

    if not ensure_server():
        print("ABORT: coder server failed to start", flush=True)
        sys.exit(2)

    t0 = time.time()
    consec_dead = 0
    for i, tid in enumerate(missed, 1):
        if tid in results:
            continue
        heartbeat(i, tid, len(results), len(missed))
        task = ch[tid]
        prompt, truncated = prompt_for(task)
        texts = []
        for _ in range(K):
            text, _t = chat(prompt)
            if text:
                texts.append(text)
        if not texts:
            consec_dead += 1
            if consec_dead >= 5:
                print("ABORT: 5 consecutive tasks with zero responses (server unrecoverable)", flush=True)
                sys.exit(2)
        else:
            consec_dead = 0
        progs = []
        for t in texts:
            progs.extend(extract_programs(t))
        uniq = list(dict.fromkeys(progs))
        tr_in = [p["input"] for p in task["train"]]
        tr_out = [p["output"] for p in task["train"]]
        te_in = [t["input"] for t in task["test"]]
        te_out = [g for g in sol[tid]]
        best = 0
        survivors = []
        for src in uniq:
            outs = pyexec.run_candidate(src, tr_in, timeout=4.0)
            if outs is None:
                continue
            npass = sum(1 for a, b in zip(outs, tr_out) if a == b)
            best = max(best, npass)
            if npass == len(tr_out):
                survivors.append(src)
        solved = False
        surv_records = []
        for src in survivors:
            t_outs = pyexec.run_candidate(src, te_in, timeout=4.0)
            surv_records.append({"src": src, "test_outputs": t_outs})
            if t_outs == te_out:
                solved = True
        results[tid] = {"solved": solved, "survivors": len(survivors), "uniq": len(uniq),
                        "responses": len(texts), "best_train": best, "n_train": len(tr_out),
                        "truncated": truncated, "survivor_records": surv_records}
        json.dump(results, open(OUT, "w"))
        print("[%d/%d] %s resp=%d uniq=%d surv=%d best_train=%d/%d SOLVED=%s t=%.0fs"
              % (i, len(missed), tid, len(texts), len(uniq), len(survivors), best, len(tr_out),
                 solved, time.time() - t0), flush=True)

    solved_ids = sorted(t for t, r in results.items() if r.get("solved"))
    near = sum(1 for r in results.values()
               if not r.get("solved") and r.get("best_train", 0) >= max(1, r.get("n_train", 1) - 1)
               and r.get("n_train", 0) > 1)
    print("=== COVERAGE probed=%d solved_disjoint=%d near_miss(>=n-1 train)=%d %s ==="
          % (len(results), len(solved_ids), near, solved_ids), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        bench()
    else:
        main()
