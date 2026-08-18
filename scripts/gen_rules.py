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
  python gen_rules.py run [N] [ATTEMPTS]   generate and verify, serial, resumable
  python gen_rules.py runc [N] [ATT] [W]   the full run: concurrent, off-peak gated, balance floor
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
# The corpus the repair pass reads its failures FROM, which need not be where it writes.
SRC_IN = os.environ.get("ARC_GENRULES_SRC", OUT)
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
                 "model": "deepseek-v4-flash",   # confirmed from the live model listing, not guessed
                 "keyfiles": [os.path.join("~", ".deepseek", "api_key")]},
    "openai": {"url": "https://api.openai.com/v1",
               "model": "gpt-5.6",
               "keyfiles": [os.path.join("~", ".openai", "api_key")]},
}
# THINKING IS THE WHOLE BUDGET QUESTION, and it is off by default.
#
# deepseek-v4-flash is a reasoning model. Measured on a real task: it spent 8,000 output tokens on
# reasoning_content and had still not emitted a single character of code (finish_reason "length",
# content empty). Reasoning tokens bill at the OUTPUT rate, which is 92 percent of the bill, so a
# thinking run costs roughly twenty times a non-thinking one and returns nothing at all if the cap
# lands mid-thought. With thinking disabled the same task finishes in 608 tokens and 1,768
# characters of program.
#
# What that trades away is unmeasured, and it is the reason both arms get run: reasoning is
# presumably what makes a model good at inferring an ARC rule, so the arm that matters is the one
# with the better VERIFIED YIELD PER DOLLAR, not the one with the smaller token count.
# The effort dial, which is the part of the space we never explored. The published leaderboard
# reports three settings for this model and we only ever measured the two extremes, off and max.
# Returns at the top are visibly sub-linear there, so the efficient point may well be in the middle.
EFFORT = os.environ.get("ARC_GEN_EFFORT", "").strip().lower()
if EFFORT in ("off", "none", "0", "disabled"):
    EFFORT = ""
THINK = (os.environ.get("ARC_GEN_THINK", "0") not in ("0", "", "off", "false")) or bool(EFFORT)

# Non-thinking programs measured at ~600 tokens, so 4,000 is generous. Thinking needs a different
# order of magnitude, and a cap that lands mid-thought wastes the entire call.
MAX_OUT = int(os.environ.get("ARC_GEN_MAX_OUT", "32000" if THINK else "4000"))

# USD per million tokens, PEAK rate, from api-docs.deepseek.com/quick_start/pricing. Off-peak is
# exactly half of every line and runs 04:00-06:00 and 10:00-01:00 UTC. These replace a $0.10/$0.28
# pair that was taken from a secondary source and was wrong by 4.7x on output, which is the line
# that actually matters: at a 900-token program output is 94 percent of the bill.
RATES = {
    "deepseek-v4-flash": {"hit": 0.014, "miss": 0.44, "out": 1.32},
    "deepseek-v4-pro":   {"hit": 0.044, "miss": 1.32, "out": 3.96},
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
            # utf-8-sig, because a key pasted into Notepad on Windows can arrive with a BOM, and a
            # BOM survives .strip(). That would send a key differing from the real one by three
            # invisible bytes and produce an authentication error nobody would think to debug.
            v = open(p, encoding="utf-8-sig").read().strip()
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

    Returns (text, usage). The usage dict is the point of the pilot: output tokens are 94 percent of
    the bill at these rates, so the measured output length is what decides whether the full run is
    affordable. Estimating it instead of measuring it is how the first cost table came out 7x wrong.
    """
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7,
               "max_tokens": MAX_OUT}
    if EFFORT:
        payload["reasoning_effort"] = EFFORT
    elif not THINK:
        payload["thinking"] = {"type": "disabled"}
    # A non-thinking call answers in seconds; a thinking call can spend tens of thousands of tokens
    # before the first character of the reply, and the default 180s would abort it as a network
    # failure. That would look identical to the model failing, which is the same inverted symptom
    # this pipeline already hit once.
    j = _post("/chat/completions", payload, timeout=1800.0 if THINK else 180.0)
    return j["choices"][0]["message"]["content"], (j.get("usage") or {})


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
        out, _ = generate("Reply with the single word: ready")
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
    """Tolerate a truncated final line.

    Records are appended one per call, so a power cut mid-append leaves a partial line. Parsing
    strictly turns that into a crash on every subsequent run, which would strand a paid corpus
    behind a resume path that cannot start. A cut can only damage the LAST line, so anything
    unparseable earlier is a real problem and is worth saying out loud rather than swallowing.
    """
    if not os.path.isfile(OUT):
        return []
    lines = [l for l in open(OUT, encoding="utf-8") if l.strip()]
    out = []
    for i, l in enumerate(lines):
        try:
            out.append(json.loads(l))
        except ValueError:
            if i == len(lines) - 1:
                print("  (dropped a truncated final record, most likely an interrupted write)")
            else:
                print("  WARNING: record %d of %d is corrupt, not at the end of the file" %
                      (i + 1, len(lines)))
    return out


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
        line = "  up to %d attempt(s): %5d calls, %.1fM in + %.1fM out" % (att, calls, tin, tout)
        for m in ("deepseek-v4-flash", "deepseek-v4-pro"):
            peak = tin * RATES[m]["miss"] + tout * RATES[m]["out"]
            line += ",  %s $%.2f peak / $%.2f off-peak" % (m.split("-")[-1], peak, peak / 2)
        print(line)
    print("")
    print("Assumes 900 output tokens per call and no cache hits, both pessimistic. Off-peak is half")
    print("price and runs 04:00-06:00 and 10:00-01:00 UTC. Run the pilot before trusting any of it.")
    print("")
    print("Attempts stop early once a task verifies, so the real spend is well under these ceilings.")
    print("Verification is execution against each task's own demonstration pairs: free, and a proof.")


ANYTIME = os.environ.get("ARC_GEN_ANYTIME", "0") not in ("0", "", "off", "false")
NOFEEDBACK = os.environ.get("ARC_GEN_NOFEEDBACK", "0") not in ("0", "", "off", "false")
PEAK_UTC = ((1, 4), (6, 10))          # published peak windows; everything else is half price


def peak_ends_in():
    """Seconds until peak pricing ends, or 0 if we are already off-peak."""
    t = time.gmtime()
    now = t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
    for a, b in PEAK_UTC:
        if a * 3600 <= now < b * 3600:
            return b * 3600 - now
    return 0


def wait_offpeak(enabled=True):
    """Block until off-peak. Output is 92 percent of the bill and off-peak halves every line, so
    waiting is worth more than any prompt tweak available to us."""
    if not enabled:
        return
    while True:
        left = peak_ends_in()
        if not left:
            return
        print("  peak pricing, holding %d min for off-peak" % (left // 60 + 1), flush=True)
        time.sleep(min(left + 5, 600))


# CNY per USD of list price, measured against real balance deltas rather than assumed. The 12-call
# thinking arm listed at $0.2551 and billed 1.74 CNY, which is 6.82, not the market 7.1 first
# guessed. Used only to show the balance and the projection in the same units, never to decide
# anything, so an error here misreports rather than misspends.
CNY_PER_USD = 6.82


def balance():
    """Remaining balance as (amount, currency).

    Picks the funded currency rather than balance_infos[0]. The array carries an entry per currency
    and the ORDER IS NOT STABLE: a CNY-funded account returns a 0.00 USD entry too, and reading
    index 0 got the empty one, which would have tripped the spend floor on the first check and
    aborted a run that had plenty of money behind it.
    """
    import httpx
    r = httpx.get(ENDPOINT.rstrip("/").rsplit("/v1", 1)[0] + "/user/balance",
                  headers={"Authorization": "Bearer " + api_key()}, timeout=60.0)
    infos = r.json()["balance_infos"]
    funded = [b for b in infos if float(b["total_balance"]) > 0]
    b = funded[0] if funded else infos[0]
    return float(b["total_balance"]), b["currency"]


# Measured means from the 30-call pilot, thinking off. MEANS, not medians: the median program was
# 991 output tokens but the mean was 1,586, because a handful ran past 12,000 characters. Cost
# scales with the mean, and projecting from the median understated the bill by 58 percent.
FALLBACK_CALIB = (256.0, 581.0, 1586.0)


def calibration():
    """Per-call token means, measured from real records where we have them.

    Prefers whatever this mode has actually produced over any constant written into the file,
    because the constant is only ever right for the run that produced it.
    """
    path = os.environ.get("ARC_GEN_CALIB", "")
    cands = [path] if path else [OUT, os.path.join(os.path.dirname(OUT),
                                                   "genrules_think.jsonl" if THINK
                                                   else "genrules_nothink.jsonl")]
    for c in cands:
        if not c or not os.path.isfile(c):
            continue
        try:
            rs = [json.loads(l) for l in open(c, encoding="utf-8") if l.strip()]
        except ValueError:
            continue
        rs = [r for r in rs if r.get("tok_out")]
        if len(rs) < 5:
            continue
        n = float(len(rs))
        return (sum(r.get("tok_cache_hit") or 0 for r in rs) / n,
                sum((r.get("tok_cache_miss") or r.get("tok_in") or 0) for r in rs) / n,
                sum(r["tok_out"] for r in rs) / n,
                len(rs))
    return FALLBACK_CALIB + (0,)


def usd_of(hit, miss, out):
    """List-price USD for a token triple, at the rate for the model actually being called."""
    rate = RATES.get(MODEL, RATES["deepseek-v4-flash"])
    usd = (hit * rate["hit"] + miss * rate["miss"] + out * rate["out"]) / 1e6
    return usd / 2.0 if not peak_ends_in() else usd


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
            wait_offpeak(not ANYTIME)
            t0 = time.time()
            try:
                text, usage = generate(task_prompt(task))
            except Exception as e:
                print("  %-10s a%d GENERATE FAILED %s" % (k, a, str(e)[:90]), flush=True)
                continue
            src = extract_code(text)
            ok, matched, total, note = verify(src, task)
            rec = {"task": k, "attempt": a, "ok": bool(ok), "matched": matched, "pairs": total,
                   "sampler": has_sampler(src), "note": note, "chars": len(src),
                   "seconds": round(time.time() - t0, 1),
                   "tok_in": usage.get("prompt_tokens"),
                   "tok_out": usage.get("completion_tokens"),
                   "tok_cache_hit": usage.get("prompt_cache_hit_tokens"),
                   "tok_cache_miss": usage.get("prompt_cache_miss_tokens"),
                   "usage_raw": usage,
                   "src": src}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("  %-10s a%d %s %d/%d pairs%s  %.1fs"
                  % (k, a, "VERIFIED" if ok else "failed  ", matched, total,
                     " +sampler" if rec["sampler"] else "", rec["seconds"]), flush=True)
            if ok:
                break
    cmd_report()


def cmd_runc(n_tasks, attempts, workers, floor=2.0):
    """The full run: concurrent ACROSS tasks, serial WITHIN a task.

    Concurrency buys nothing on price -- tokens cost the same however many sockets are open. It buys
    the off-peak WINDOW. The long discount block is 10:00-01:00 UTC, fifteen hours; a serial run of
    roughly 1,750 calls at 40 seconds each is nineteen, so serial spills into peak and pays double
    for the overflow. Six workers bring it to about three hours and the whole run lands inside the
    discount. That, and not the socket count, is the reason for the flag.

    Serial WITHIN a task is not incidental: attempts stop the moment one verifies, and that early
    stop removes far more calls than caching removes cost.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading

    ch = load_tasks()
    keys = sorted(ch)
    if n_tasks:
        keys = keys[:n_tasks]
    recs = load_records()
    solved = {r["task"] for r in recs if r["ok"]}
    tried = {(r["task"], r["attempt"]) for r in recs}
    todo = [k for k in keys if k not in solved]

    amt, cur = balance()
    # Measured on the 30-task pilot, thinking off: 991 output tokens and 665 input at the median,
    # of which 256 cache. Projecting from that beats projecting from a guess, which is how the
    # first estimate came out 4x low.
    # Price the projection at the rate that will ACTUALLY apply. usd_of() reads the clock, and the
    # clock says peak right now, but the gate forces every call into the discount window unless
    # ANYTIME overrides it. Quoting the peak rate for a gated run overstates the bill by exactly 2x.
    hit_m, miss_m, out_m, src_n = calibration()
    per_call = usd_of(hit_m, miss_m, out_m)
    if not ANYTIME and peak_ends_in():
        per_call /= 2.0
    worst = per_call * len(todo) * attempts        # no early stop, every attempt paid for
    # Early stop: a task that verifies makes no further calls. At the measured 10 percent per
    # attempt, the expected number of calls per task is 1 + 0.9 + 0.81 + ... over the attempts.
    likely = per_call * len(todo) * sum(0.90 ** i for i in range(attempts))
    ceiling = float(os.environ.get("ARC_GEN_MAX_USD", "2.30"))

    print("")
    print("  %d tasks (%d already verified), up to %d attempts, %d workers, model %s, thinking %s"
          % (len(keys), len(keys) - len(todo), attempts, workers, MODEL, "ON" if THINK else "off"))
    print("  balance %.2f %s  (about $%.2f of list-price traffic)"
          % (amt, cur, amt / CNY_PER_USD if cur == "CNY" else amt))
    print("  pricing right now: %s%s"
          % ("PEAK" if peak_ends_in() else "off-peak (half price)",
             ", but the gate will hold every call for the discount" if peak_ends_in() and not ANYTIME
             else ""))
    print("  projected: $%.2f likely with early stop, $%.2f worst case  (%.1f / %.1f CNY)"
          % (likely, worst, likely * CNY_PER_USD, worst * CNY_PER_USD))
    print("  per call $%.5f, calibrated from %s"
          % (per_call, ("%d real records" % src_n) if src_n else "the pilot's measured means"))
    print("  hard ceiling: $%.2f, checked every call from measured tokens" % ceiling)
    if ANYTIME:
        print("  !! ARC_GEN_ANYTIME is set: the off-peak gate is DISABLED and this may pay DOUBLE")
    if worst > ceiling:
        print("  note: worst case exceeds the ceiling, the run will stop when the ceiling is hit")
    if os.environ.get("ARC_GEN_CONFIRM") != "1":
        print("")
        print("  NOT STARTING. This spends real money. Re-run with ARC_GEN_CONFIRM=1 to proceed.")
        return
    print("  confirmed, starting", flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    state = {"calls": 0, "ok": 0, "hit": 0, "miss": 0, "out": 0, "usd": 0.0}

    # Warm the prefix cache with ONE call before the workers start. Otherwise the first `workers`
    # calls all race past an empty cache and every one of them pays the cache-miss rate.
    wait_offpeak()
    if not recs:
        try:
            generate(INSTRUCTIONS + " Reply with the single word: ready")
            print("prefix cache warmed", flush=True)
        except Exception as e:
            print("warmup failed (%s), continuing" % str(e)[:80], flush=True)

    def one_task(k):
        task = ch[k]
        for a in range(1, attempts + 1):
            if stop.is_set() or (k, a) in tried:
                continue
            wait_offpeak()
            t0 = time.time()
            try:
                text, usage = generate(task_prompt(task))
            except Exception as e:
                with lock:
                    print("  %-10s a%d GENERATE FAILED %s" % (k, a, str(e)[:90]), flush=True)
                continue
            src = extract_code(text)
            ok, matched, total, note = verify(src, task)
            rec = {"task": k, "attempt": a, "ok": bool(ok), "matched": matched, "pairs": total,
                   "sampler": has_sampler(src), "note": note, "chars": len(src),
                   "seconds": round(time.time() - t0, 1),
                   "tok_in": usage.get("prompt_tokens"),
                   "tok_out": usage.get("completion_tokens"),
                   "tok_cache_hit": usage.get("prompt_cache_hit_tokens"),
                   "tok_cache_miss": usage.get("prompt_cache_miss_tokens"),
                   "usage_raw": usage, "src": src}
            with lock:
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                state["calls"] += 1
                state["ok"] += bool(ok)
                # Spend is accounted from the tokens the API just reported, so the ceiling is
                # enforced on every call with no extra request and no lag behind the meter.
                h = rec["tok_cache_hit"] or 0
                m = rec["tok_cache_miss"] or 0
                if not h and not m:                    # provider did not break the input down
                    m = rec["tok_in"] or 0
                state["hit"] += h
                state["miss"] += m
                state["out"] += rec["tok_out"] or 0
                state["usd"] = usd_of(state["hit"], state["miss"], state["out"])
                n = state["calls"]
                if state["usd"] >= ceiling:
                    print("  -- SPEND CEILING $%.2f reached after %d calls, stopping"
                          % (ceiling, n), flush=True)
                    stop.set()
                print("  [%4d/%d ok] %-10s a%d %s %d/%d%s %.0fs"
                      % (n, state["ok"], k, a, "VERIFIED" if ok else "failed  ", matched, total,
                         " +samp" if rec["sampler"] else "      ", rec["seconds"]), flush=True)
                if n % 20 == 0:
                    print("  -- spent $%.3f (%.2f CNY) over %d calls, %d verified"
                          % (state["usd"], state["usd"] * CNY_PER_USD, n, state["ok"]), flush=True)
                # A prepaid balance and open sockets do not mix without a floor. Checking every 25
                # calls costs one request and bounds the worst case to a few calls of overrun.
                if n % 25 == 0:
                    try:
                        amt2, cur2 = balance()
                        print("  -- balance %.2f %s after %d calls" % (amt2, cur2, n), flush=True)
                        if amt2 < floor:
                            print("  -- BELOW FLOOR, stopping", flush=True)
                            stop.set()
                    except Exception as e:
                        print("  -- balance check failed %s" % str(e)[:60], flush=True)
            if ok:
                break

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one_task, todo))
    cmd_report()


REPAIR_INSTRUCTIONS = """You are repairing a Python program that implements an ARC-AGI grid transformation rule.

A previous attempt at this task is shown below together with what went wrong. Diagnose the fault and
rewrite the program so it reproduces every demonstration pair exactly.

Return exactly two functions in one fenced python block, and nothing else:

  def solve(I):
      # I is a grid, a list of rows of ints 0-9. Return the transformed grid.

  def gen_input(rng):
      # rng is a random.Random. Return a NEW input grid the same rule applies to, from the same
      # family as the demonstration inputs. Vary it. Do not return a demonstration input.

Rules:
- solve must reproduce EVERY demonstration pair. It is checked by execution.
- If the previous attempt raised, fix the fault; do not simply reformat it.
- If the previous attempt ran but produced wrong grids, the rule itself was misread. Reconsider what
  the transformation actually is rather than patching the existing logic.
- No I/O, no network. The standard library and numpy are available.
- gen_input must not depend on anything defined outside these two functions.
"""


def repair_prompt(task, src, err, got, outs):
    """Fixed block first so the prefix caches, task-specific material last."""
    parts = [REPAIR_INSTRUCTIONS, "", "The previous attempt:", "", "```python", src.strip(), "```", ""]
    if err:
        parts += ["It raised:", "", err.strip(), ""]
    else:
        shown = grid_str(got[0]) if (got and got[0]) else "(nothing usable)"
        parts += ["It ran without error but produced the wrong output. On demonstration pair 1 it "
                  "returned:", "", shown, "", "where the correct output is:", "", grid_str(outs[0]), ""]
    parts += ["Task demonstration pairs:", ""]
    for i, p in enumerate(task["train"]):
        parts += ["Example %d input:" % (i + 1), grid_str(p["input"]),
                  "Example %d output:" % (i + 1), grid_str(p["output"]), ""]
    return "\n".join(parts)


def cmd_repair(n_tasks, workers):
    """Second pass over failures, feeding each program its own error back.

    46 percent of failures are programs that raise rather than programs that answer wrongly, and a
    traceback is a far stronger signal than a blank retry: the model already had an approach and lost
    it to a bug. Crashes are repaired first for that reason, near-misses next, and programs that
    matched nothing last, which is also the order of decreasing expected value.
    """
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from pyexec import run_candidate_diag

    ch = load_tasks()
    src_recs = [json.loads(l) for l in open(SRC_IN, encoding="utf-8") if l.strip()]
    solved = {r["task"] for r in src_recs if r["ok"]}
    done = {r["task"] for r in load_records()}
    cands = [r for r in src_recs
             if not r["ok"] and r["task"] not in solved and r["task"] not in done and r.get("src")]
    cands.sort(key=lambda r: (0 if r["note"] else 1, -r["matched"]))
    if n_tasks:
        cands = cands[:n_tasks]

    amt, cur = balance()
    hit_m, miss_m, out_m, src_n = calibration()
    per_call = usd_of(256, miss_m + 1500, out_m)
    if not ANYTIME and peak_ends_in():
        per_call /= 2.0
    ceiling = float(os.environ.get("ARC_GEN_MAX_USD", "0.50"))
    print("")
    print("  %s%s: %d candidates"
          % ("RETRY CONTROL" if NOFEEDBACK else "REPAIR",
             (" effort=%s" % EFFORT) if EFFORT else (" thinking=max" if THINK else " thinking=off"),
             len(cands)))
    print("  REPAIR: %d candidates (%d crashed, %d near-miss), %d workers"
          % (len(cands), sum(1 for r in cands if r["note"]),
             sum(1 for r in cands if not r["note"] and r["matched"] > 0), workers))
    print("  balance %.2f %s, projected $%.2f (%.1f CNY), ceiling $%.2f (%.1f CNY)"
          % (amt, cur, per_call * len(cands), per_call * len(cands) * CNY_PER_USD,
             ceiling, ceiling * CNY_PER_USD))
    if os.environ.get("ARC_GEN_CONFIRM") != "1":
        print("  NOT STARTING. Re-run with ARC_GEN_CONFIRM=1 to proceed.")
        return
    print("  confirmed, starting", flush=True)

    lock = threading.Lock()
    stop = threading.Event()
    st = {"n": 0, "ok": 0, "hit": 0, "miss": 0, "out": 0, "usd": 0.0}

    def one(r):
        if stop.is_set():
            return
        k = r["task"]
        task = ch[k]
        ins = [p["input"] for p in task["train"]]
        outs = [[list(x) for x in p["output"]] for p in task["train"]]
        got, err = run_candidate_diag(r["src"], ins, timeout=EXEC_TIMEOUT)
        # The control arm. Repairing is only worth its higher price if the error feedback beats
        # simply asking again on the same task, and these tasks are selected for difficulty, so a
        # comparison against first-attempt yield on unselected tasks would flatter plain retry.
        # Same candidates, same order, same harness; only the prompt differs.
        prompt = task_prompt(task) if NOFEEDBACK else repair_prompt(task, r["src"], err, got, outs)
        wait_offpeak(not ANYTIME)
        t0 = time.time()
        try:
            text, usage = generate(prompt)
        except Exception as e:
            with lock:
                print("  %-10s REPAIR CALL FAILED %s" % (k, str(e)[:80]), flush=True)
            return
        new_src = extract_code(text)
        ok, matched, total, note = verify(new_src, task)
        rec = {"task": k, "attempt": 90, "ok": bool(ok), "matched": matched, "pairs": total,
               "sampler": has_sampler(new_src), "note": note, "chars": len(new_src),
               "seconds": round(time.time() - t0, 1),
               "was": {"matched": r["matched"], "crashed": bool(r["note"])},
               "tok_in": usage.get("prompt_tokens"), "tok_out": usage.get("completion_tokens"),
               "tok_cache_hit": usage.get("prompt_cache_hit_tokens"),
               "tok_cache_miss": usage.get("prompt_cache_miss_tokens"),
               "usage_raw": usage, "src": new_src}
        with lock:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            st["n"] += 1
            st["ok"] += bool(ok)
            h = rec["tok_cache_hit"] or 0
            m = rec["tok_cache_miss"] or 0
            if not h and not m:
                m = rec["tok_in"] or 0
            st["hit"] += h
            st["miss"] += m
            st["out"] += rec["tok_out"] or 0
            st["usd"] = usd_of(st["hit"], st["miss"], st["out"])
            print("  [%3d/%d ok] %-10s %s %d/%d  (was %s)  %.0fs"
                  % (st["n"], st["ok"], k, "REPAIRED" if ok else "still bad", matched, total,
                     "crash" if r["note"] else "%d/%d" % (r["matched"], r["pairs"]),
                     rec["seconds"]), flush=True)
            if st["n"] % 20 == 0:
                print("  -- spent $%.3f (%.2f CNY), %d repaired of %d"
                      % (st["usd"], st["usd"] * CNY_PER_USD, st["ok"], st["n"]), flush=True)
            if st["usd"] >= ceiling:
                print("  -- CEILING $%.2f reached, stopping" % ceiling, flush=True)
                stop.set()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, cands))
    print("")
    print("  REPAIR DONE: %d attempted, %d repaired (%.1f%%), spent %.2f CNY"
          % (st["n"], st["ok"], 100.0 * st["ok"] / max(1, st["n"]), st["usd"] * CNY_PER_USD))


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
    used = [r for r in recs if r.get("tok_out")]
    if used:
        rate = RATES.get(MODEL, RATES["deepseek-v4-flash"])
        hit = sum(r.get("tok_cache_hit") or 0 for r in used)
        miss = sum(r.get("tok_cache_miss") or 0 for r in used)
        out = sum(r["tok_out"] for r in used)
        if not hit and not miss:                       # provider did not break the input down
            miss = sum(r.get("tok_in") or 0 for r in used)
        peak = (hit * rate["hit"] + miss * rate["miss"] + out * rate["out"]) / 1e6
        reas = sum(((r.get("usage_raw") or {}).get("completion_tokens_details") or {})
                   .get("reasoning_tokens") or 0 for r in used)
        if reas:
            print("")
            print("  of %d output tokens, %d (%.0f%%) were REASONING, billed at the output rate"
                  % (out, reas, 100.0 * reas / max(1, out)))
        outs = sorted(r["tok_out"] for r in used)
        med_out = outs[len(outs) // 2]
        print("")
        print("  MEASURED over %d calls: %d in (%d cached), %d out, median %d out per call"
              % (len(used), hit + miss, hit, out, med_out))
        print("  spent $%.4f at peak rates ($%.4f if run off-peak), $%.5f per call"
              % (peak, peak / 2, peak / max(1, len(used))))
        verified_n = len(verified)
        if verified_n:
            print("  $%.4f per VERIFIED program at peak, $%.4f off-peak"
                  % (peak / verified_n, peak / 2 / verified_n))
        full = peak / max(1, len(used)) * 1000 * 3       # 1000 tasks, 3 attempts, no early stop
        print("  extrapolated FULL run (1000 tasks x3, no early stop): $%.2f peak, $%.2f off-peak"
              % (full, full / 2))
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
    elif cmd == "runc":
        cmd_runc(int(sys.argv[2]) if len(sys.argv) > 2 else 1000,
                 int(sys.argv[3]) if len(sys.argv) > 3 else 3,
                 int(sys.argv[4]) if len(sys.argv) > 4 else 6)
    elif cmd == "repair":
        cmd_repair(int(sys.argv[2]) if len(sys.argv) > 2 else 0,
                   int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif cmd == "report":
        cmd_report()
    else:
        raise SystemExit(__doc__)
