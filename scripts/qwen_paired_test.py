"""Run the five paired tasks through a local Qwen3.8-27B and compare with the DeepSeek arms.

One process does everything, because on this rig a server started by Start-Process over SSH dies
with the session that launched it (observed: model loaded, port bound, then gone the moment the ssh
call returned). A detached PYTHON process survives, so this owns the server as a child and outlives
the connection.

The five tasks are the same ones the DeepSeek arms used, selected the same way (crashes first, in
the same order), so the comparison is paired rather than two samples of a population:

    thinking off, repair with traceback   1 of 5
    effort=low                            4 of 5
    effort=max, 64k cap                   3 of 5

Verification is unchanged: solve() must reproduce every demonstration pair under execution.

Usage: python qwen_paired_test.py
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, BASE)

MODEL = os.path.join(BASE, "models", "qwen38", "Qwen3.8-27B-Q3_K_M.gguf")
SERVER = os.path.join(BASE, "llamacpp_b10488", "llama-server.exe")
PORT = "8080"
SRC_IN = os.path.join(BASE, "experiments", "genrules_nothink.jsonl")
OUT = os.path.join(BASE, "experiments", "genrules_qwen_local.jsonl")
N = int(os.environ.get("QWEN_N", "5"))
BUDGET = os.environ.get("QWEN_THINK_BUDGET", "10000")
CTX = os.environ.get("QWEN_CTX", "24576")
# Speculative decoding. The GGUF already carries the model's MTP layers (blk.64.nextn.*) and the
# server has been reporting them as "unused tensor ... ignoring", so this is weight we downloaded
# and never used. Decode here is memory-bandwidth bound at about 72 percent of what the card can
# do, and no amount of tuning passes that; speculation is the only lever that does, because it
# amortises one weight read over several accepted tokens.
SPEC = os.environ.get("QWEN_SPEC", "")


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def wait_ready(proc, timeout=600):
    import httpx
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            say("server exited early, rc=%s" % proc.returncode)
            return False
        try:
            r = httpx.get("http://127.0.0.1:%s/v1/models" % PORT, timeout=5)
            if r.status_code == 200:
                say("server ready after %.0fs" % (time.time() - t0))
                return True
        except Exception:
            pass
        time.sleep(5)
    say("server did not become ready in %ds" % timeout)
    return False


def main():
    os.environ["ARC_GEN_PROVIDER"] = "local"
    os.environ["ARC_GENRULES_OUT"] = OUT
    os.environ.setdefault("ARC_GEN_MAX_OUT", "20000")

    say("starting llama-server, ctx %s, thinking budget %s, spec %s"
        % (CTX, BUDGET, SPEC or "none"))
    logf = open(os.path.join(BASE, "logs", "qwen_server_child.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(
        [SERVER, "-m", MODEL, "-ngl", "99", "-c", CTX,
         "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "--host", "127.0.0.1", "--port", PORT, "--jinja", "-fa", "on",
         # The whole finding of the DeepSeek arms was that these models fail by not terminating,
         # not by being wrong: every call that finished thinking produced a verified program. This
         # forces the close, and it is a finer instrument than a three-level effort dial because
         # the budget is set in tokens.
         "--reasoning-budget", BUDGET]
        + (["--spec-type", SPEC] if SPEC else []),
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=os.path.dirname(SERVER))
    try:
        if not wait_ready(proc):
            return 1

        import gen_rules as G
        ch = G.load_tasks()
        recs = [json.loads(l) for l in open(SRC_IN, encoding="utf-8") if l.strip()]
        solved = {r["task"] for r in recs if r["ok"]}
        cands = [r for r in recs if not r["ok"] and r["task"] not in solved and r.get("src")]
        cands.sort(key=lambda r: (0 if r["note"] else 1, -r["matched"]))
        cands = cands[:N]
        say("%d paired tasks: %s" % (len(cands), ", ".join(c["task"] for c in cands)))

        ok_n = 0
        for i, c in enumerate(cands, 1):
            k = c["task"]
            task = ch[k]
            t0 = time.time()
            try:
                text, usage = G.generate(G.task_prompt(task))
            except Exception as e:
                say("  %-10s CALL FAILED %s" % (k, str(e)[:120]))
                continue
            dt = time.time() - t0
            src = G.extract_code(text)
            ok, matched, total, note = G.verify(src, task)
            ok_n += bool(ok)
            out_tok = (usage or {}).get("completion_tokens") or 0
            rec = {"task": k, "attempt": 80, "ok": bool(ok), "matched": matched, "pairs": total,
                   "sampler": G.has_sampler(src), "note": note, "chars": len(src),
                   "seconds": round(dt, 1), "tok_in": (usage or {}).get("prompt_tokens"),
                   "tok_out": out_tok, "tok_cache_hit": 0, "tok_cache_miss": 0,
                   "usage_raw": usage, "src": src}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            say("  [%d/%d ok] %-10s %s %d/%d  %d tok  %.0fs  %.1f tok/s"
                % (i, ok_n, k, "VERIFIED" if ok else "failed  ", matched, total,
                   out_tok, dt, out_tok / max(1e-9, dt)))
        say("")
        say("QWEN3.8-27B Q3_K_M LOCAL: %d of %d verified" % (ok_n, len(cands)))
        say("  for comparison on the SAME tasks: off 1/5, low 4/5, max 3/5 (DeepSeek V4 Flash)")
        return 0
    finally:
        say("stopping server")
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        logf.close()


if __name__ == "__main__":
    sys.exit(main())
