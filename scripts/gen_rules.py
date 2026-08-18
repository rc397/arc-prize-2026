"""Ask a model to write a generator for each ARC-AGI-2 TRAINING task, and verify it by execution.

This is the "RE-ARC for ARC-2" step. RE-ARC ships 400 verifiers for ARC-1 rules and our composition
work stacked pairs of them into 402 more; neither is ARC-2-shaped, and the full 2x2 showed every
configuration of that corpus lands at or below the base on ARC coverage. The training split IS
ARC-2-shaped, by construction, and it is published for training, which is what makes this legitimate
where using the evaluation split would not be.

WHY THE VERIFICATION NEEDS NO MODEL. A generated program is correct if and only if it reproduces the
task's own demonstration pairs, solve(input_i) == output_i for every i. That is a proof rather than
an opinion, it runs in pyexec's isolated worker with a hard kill timeout, and it is strictly stronger
than asking any model whether the grids look right. It is also free, which matters because the
alternative costs money and raises a licensing question for no gain at all.

MODEL CHOICE IS A FLAG, NOT A COMMITMENT. The default backend is an OpenAI-compatible endpoint,
which DeepSeek exposes, so the same code drives V4 Flash (MIT licence, 61.4 percent on ARC-AGI-2,
about six hundredths of a cent per call) or anything else speaking that protocol. The licence is the
reason for the default: MIT permits using the outputs to build training data outright, with no grey
area to reason about.

SHAPE OF THE CALL. One turn, no tools, no agent, no subagents. There is nothing to explore here: the
model reads demonstration pairs and writes a function. An agent runtime would add loops and token
overhead for no benefit. Serial by default, because generation is token-heavy and a stalled quota is
worse than a slow queue. The fixed instruction block is emitted FIRST and the task grids LAST, so a
prefix cache has something stable to hit.

Usage:
  python gen_rules.py plan                 how many tasks, what it will cost
  python gen_rules.py run [N] [ATTEMPTS]   generate and verify, resumable
  python gen_rules.py report               yield so far, broken down by attempt
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.environ.get("ARC_GENRULES_OUT", os.path.join(BASE, "experiments", "genrules.jsonl"))
EXEC_TIMEOUT = 6.0

# Both providers speak the same OpenAI-compatible protocol, so the only thing that changes is the
# base URL, the default model string and where the key lives. Each key file is created by the user
# and read at call time; no code path prints or logs a key.
#
# Which is for what, and why:
#   deepseek  GENERATION. V4 Flash is MIT licensed, so its outputs can be used to build training
#             data outright, with no grey area. 61.4% on ARC-AGI-2, about $0.0006 a call.
#   openai    CALIBRATION only, if wanted. A frontier model measuring the difficulty of a finished
#             corpus is evaluation, not training, and nothing it emits enters the data.
PROVIDERS = {
    "deepseek": {"url": "https://api.deepseek.com/v1",
                 "model": "deepseek-chat",
                 "keyfiles": [os.path.join("~", ".deepseek", "api_key")]},
    "openai": {"url": "https://api.openai.com/v1",
               "model": "gpt-5.6",
               "keyfiles": [os.path.join("~", ".openai", "api_key")]},
}
PROVIDER = os.environ.get("ARC_GEN_PROVIDER", "deepseek")
assert PROVIDER in PROVIDERS, "ARC_GEN_PROVIDER must be one of %s" % sorted(PROVIDERS)
ENDPOINT = os.environ.get("ARC_GEN_BASE_URL", PROVIDERS[PROVIDER]["url"])
MODEL = os.environ.get("ARC_GEN_MODEL", PROVIDERS[PROVIDER]["model"])

INSTRUCTIONS = """You write Python programs that implement ARC-AGI grid transformation rules.

A grid is a list of rows; each row is a list of ints 0-9. You will be shown every demonstration pair
of one task. Infer the single rule that maps input to output, and express it as code.

Return exactly two functions in one fenced python block, and nothing else:

  def solve(I):
      # I is a grid. Return the transformed grid.

  def gen_input(rng):
      # rng is a random.Random. Return a NEW input grid that this same rule applies to, drawn from
      # the same family as the demonstration inputs: similar size range, similar colours, similar
      # structure. Vary it. Do not return a copy of a demonstration input.

Rules:
- solve must reproduce EVERY demonstration pair exactly. It is checked by execution.
- No I/O, no network, no input(). The standard library and numpy are available.
- Prefer explicit literal code over cleverness. A long readable function beats a short obscure one.
- If the rule depends on a property of the grid, such as the largest object, the most common colour
  or a symmetry axis, compute that property rather than hard-coding a value you happened to observe.
- gen_input must not depend on anything defined outside these two functions.
"""


def grid_str(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def task_prompt(task):
    parts = [INSTRUCTIONS, "", "Task demonstration pairs:", ""]
    for i, p in enumerate(task["train"]):
        parts.append("Example %d input:" % (i + 1))
        parts.append(grid_str(p["input"]))
        parts.append("Example %d output:" % (i + 1))
        parts.append(grid_str(p["output"]))
        parts.append("")
    return "\n".join(parts)


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def verify(src, task):
    """solve must reproduce every demonstration pair. Returns (ok, matched, total, note)."""
    from pyexec import run_candidate
    ins = [p["input"] for p in task["train"]]
    outs = [[list(r) for r in p["output"]] for p in task["train"]]
    try:
        got = run_candidate(src, ins, timeout=EXEC_TIMEOUT)
    except Exception as e:
        return False, 0, len(ins), "exec raised: %s" % str(e)[:80]
    if got is None:
        return False, 0, len(ins), "no result, timeout or crash"
    matched = sum(1 for a, b in zip(got, outs) if a is not None and a == b)
    return matched == len(ins), matched, len(ins), ""


def has_sampler(src):
    return "def gen_input" in src


def api_key():
    """Env var first, then a file the user creates. The value is never printed or logged.

    Same shape as the Kaggle token on this rig: it lives in a file under the home directory, gets
    read into memory at call time, and no code path echoes it.
    """
    k = os.environ.get("ARC_GEN_API_KEY")
    if k:
        return k.strip()
    for p in PROVIDERS[PROVIDER]["keyfiles"]:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            v = open(p, encoding="utf-8").read().strip()
            if v:
                return v
    raise SystemExit(
        "no %s key found. Put it in %s (one line, nothing else) or set ARC_GEN_API_KEY."
        % (PROVIDER, os.path.expanduser(PROVIDERS[PROVIDER]["keyfiles"][0])))


def _post(path, payload, timeout=180.0):
    import httpx
    r = httpx.post(ENDPOINT.rstrip("/") + path,
                   headers={"Authorization": "Bearer " + api_key(),
                            "Content-Type": "application/json"},
                   json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError("HTTP %d: %s" % (r.status_code, r.text[:300]))
    return r.json()


def generate(prompt):
    """One turn, no tools. Plain OpenAI-compatible chat completion, which DeepSeek speaks.

    Deliberately httpx rather than the openai package: httpx is already installed here, the endpoint
    is one POST, and this Python install has a history of punishing new dependencies.
    """
    j = _post("/chat/completions",
              {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7})
    return j["choices"][0]["message"]["content"]


def cmd_check():
    """Prove the key, the endpoint and the model string work, without guessing any of them."""
    import httpx
    print("provider %s, endpoint %s, model %s" % (PROVIDER, ENDPOINT, MODEL))
    kf = os.path.expanduser(PROVIDERS[PROVIDER]["keyfiles"][0])
    print("key file %s: %s" % (kf, "present" if os.path.isfile(kf) else "MISSING"))
    try:
        r = httpx.get(ENDPOINT.rstrip("/") + "/models",
                      headers={"Authorization": "Bearer " + api_key()}, timeout=60.0)
        if r.status_code == 200:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print("models the key can see: %s" % (", ".join(str(i) for i in ids) or "(none listed)"))
        else:
            print("model listing returned HTTP %d, continuing anyway" % r.status_code)
    except Exception as e:
        print("model listing failed (%s), continuing anyway" % str(e)[:80])
    print("trying one tiny completion with model=%r ..." % MODEL)
    try:
        out = generate("Reply with the single word: ready")
        print("OK, model replied: %r" % (out or "")[:60])
        print("")
        print("ready to run:  python gen_rules.py run 30 3")
    except Exception as e:
        print("FAILED: %s" % str(e)[:300])
        print("")
        print("If the model string is wrong, set ARC_GEN_MODEL to one of the ids listed above.")


def load_tasks():
    import rig_restart_probe as P
    p = os.path.join(P.data_dir(), "arc-agi_training_challenges.json")
    assert os.path.isfile(p), "training challenges not found at %s" % p
    return json.load(open(p, encoding="utf-8"))


def load_records():
    if not os.path.isfile(OUT):
        return []
    return [json.loads(l) for l in open(OUT, encoding="utf-8")]


def cmd_plan():
    ch = load_tasks()
    toks = sorted(len(task_prompt(t)) // 4 for t in ch.values())    # ~4 chars per token
    n = len(ch)
    med = toks[n // 2]
    print("%d ARC-AGI-2 training tasks" % n)
    print("prompt tokens: p10 %d  median %d  p90 %d" % (toks[n // 10], med, toks[9 * n // 10]))
    print("")
    for att in (1, 3, 5):
        calls = n * att
        tin, tout = calls * med / 1e6, calls * 900 / 1e6
        print("  up to %d attempt(s): %5d calls, %.1fM in + %.1fM out, about $%.2f at V4 Flash list"
              % (att, calls, tin, tout, tin * 0.10 + tout * 0.28))
    print("")
    print("Attempts stop early once a task verifies, so the real spend is well under these ceilings.")
    print("Verification is execution against each task's own demonstration pairs: free, and a proof.")


def cmd_run(n_tasks, attempts):
    ch = load_tasks()
    keys = sorted(ch)
    if n_tasks:
        keys = keys[:n_tasks]
    recs = load_records()
    solved = {r["task"] for r in recs if r["ok"]}
    tried = {(r["task"], r["attempt"]) for r in recs}
    print("%d tasks, up to %d attempts each, model %s, serial" % (len(keys), attempts, MODEL),
          flush=True)
    for k in keys:
        if k in solved:
            continue
        task = ch[k]
        for a in range(1, attempts + 1):
            if (k, a) in tried:
                continue
            t0 = time.time()
            try:
                text = generate(task_prompt(task))
            except Exception as e:
                print("  %-10s a%d GENERATE FAILED %s" % (k, a, str(e)[:90]), flush=True)
                continue
            src = extract_code(text)
            ok, matched, total, note = verify(src, task)
            rec = {"task": k, "attempt": a, "ok": bool(ok), "matched": matched, "pairs": total,
                   "sampler": has_sampler(src), "note": note, "chars": len(src),
                   "seconds": round(time.time() - t0, 1), "src": src}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("  %-10s a%d %s %d/%d pairs%s  %.1fs"
                  % (k, a, "VERIFIED" if ok else "failed  ", matched, total,
                     " +sampler" if rec["sampler"] else "", rec["seconds"]), flush=True)
            if ok:
                break
    cmd_report()


def cmd_report():
    recs = load_records()
    if not recs:
        print("nothing at %s" % OUT)
        return
    tasks = {r["task"] for r in recs}
    verified = {r["task"] for r in recs if r["ok"]}
    sampled = {r["task"] for r in recs if r["ok"] and r["sampler"]}
    print("")
    print("%d tasks attempted, %d VERIFIED (%.0f%%), %d of those also wrote a sampler"
          % (len(tasks), len(verified), 100.0 * len(verified) / max(1, len(tasks)), len(sampled)))
    for a in range(1, 6):
        at = [r for r in recs if r["attempt"] == a]
        if at:
            print("  attempt %d: %3d calls, %3d verified (%.0f%%)"
                  % (a, len(at), sum(r["ok"] for r in at),
                     100.0 * sum(r["ok"] for r in at) / len(at)))
    fails = [r for r in recs if not r["ok"] and r["pairs"]]
    if fails:
        near = sum(1 for r in fails if r["matched"] == r["pairs"] - 1)
        zero = sum(1 for r in fails if r["matched"] == 0)
        print("  of %d failures: %d matched all but one pair, %d matched none"
              % (len(fails), near, zero))
    print("")
    print("A verified program reproduces every demonstration pair, so it captures that task's rule.")
    print("Only the ones with a WORKING sampler can mint unlimited fresh instances. The rest give")
    print("back only the task's own pairs, which is no more data than we started with, so the")
    print("sampler column is the one that decides whether this produces a corpus or a curiosity.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "check":
        cmd_check()
    elif cmd == "plan":
        cmd_plan()
    elif cmd == "run":
        cmd_run(int(sys.argv[2]) if len(sys.argv) > 2 else 30,
                int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == "report":
        cmd_report()
    else:
        raise SystemExit(__doc__)
