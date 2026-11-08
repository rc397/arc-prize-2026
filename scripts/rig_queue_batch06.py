"""The short run: does mixing prevent the forgetting the full corpus caused?

Deliberately the cheapest experiment that can answer it, because GPU hours here cost real money on
my electricity bill and the last cycle was 24 hours of card.

  1 mix      interleave 4,000 composed with 4,000 single-rule examples. CPU, seconds.
  2 train    8,000 steps. The pilot spent 8,000 steps on composed data alone and its gate showed
             ARC candidate generation down 29 percent, so this is an EQUAL-COMPUTE comparison
             against a run we have already measured. About 2.4 hours with the padding fix.
  3 gate     the ft arm only, ~2.3 hours. The base arm is seeded from the full run's records: the
             stock model, the same twelve ARC tasks and eight composed tasks, the same 420-second
             decode budget, the same code. Re-running it would burn 2.3 hours to reproduce numbers
             we already hold, and the gate skips any (arm, task) already in its output file.

Total about 4.7 hours of card, against 24 for the previous cycle.

What each outcome means, fixed in advance. If ARC candidates hold near the base's 36 while composed
coverage still rises, mixing works and the compositional direction is alive at a larger scale. If ARC
candidates collapse the way they did at 47 percent, mixing does not rescue it and the direction is
closed for this base. If nothing moves at all, 8,000 steps is too small to show either, which the
pilot already demonstrated once and would make this uninformative rather than negative.

Usage: python rig_queue_batch06.py [WAIT_PID]
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
LOG = os.path.join(BASE, "experiments", "batch06.log")
EXP = os.path.join(BASE, "experiments")
POLL_S = 30

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
MIXED_DIR = os.path.join(BASE, "_corpus_mixed")
OUT_MODEL = os.path.join(BASE, "models", "comp_mixed")
GATE_OUT = os.path.join(EXP, "ftgate_mixed.jsonl")


def say(msg):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), msg)
    try:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:
        pass
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def interesting(line):
    return "%|" not in line and "it/s]" not in line and "s/it]" not in line


def alive(pid):
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def seed_base_arm():
    """Copy the stock-model records out of the full run's gate so this gate skips that arm."""
    src = os.path.join(EXP, "ftgate_full.jsonl")
    if os.path.isfile(GATE_OUT):
        say("gate output already exists, leaving it alone")
        return
    if not os.path.isfile(src):
        say("no ftgate_full.jsonl to seed from; the base arm will re-run and cost 2.3h")
        return
    rows = [l for l in open(src, encoding="utf-8") if json.loads(l)["arm"] == "base"]
    with open(GATE_OUT, "w", encoding="utf-8") as f:
        f.writelines(rows)
    say("seeded %d base-arm records; the gate will run the ft arm only" % len(rows))


def run(name, args, env_extra=None):
    say("PHASE %s: %s" % (name, " ".join(args)))
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if env_extra:
        env.update(env_extra)
    p = subprocess.Popen([sys.executable, "-u", os.path.join(SCRIPTS, args[0])] + args[1:],
                         cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in p.stdout:
            if interesting(line):
                say("  %s" % line.rstrip())
    except Exception as e:
        say("  [log stream failed: %s %s, the phase keeps running]" % (type(e).__name__, e))
    p.wait()
    say("PHASE %s finished rc=%d in %.0fs" % (name, p.returncode, time.time() - t0))
    return p.returncode == 0


def main(wait_pid):
    if wait_pid:
        say("waiting on pid %d (the single-rule corpus build)" % wait_pid)
        while alive(wait_pid):
            time.sleep(POLL_S)
        say("pid %d exited" % wait_pid)

    if not run("mix", ["mix_corpus.py", os.path.join(BASE, "_corpus_comp"),
                       os.path.join(BASE, "_corpus_single"), MIXED_DIR, "4000", "250"]):
        say("stopping: no mixed corpus")
        return False
    if not run("train", ["finetune_base.py", "--model", STOCK, "--corpus", MIXED_DIR,
                         "--out", OUT_MODEL, "--max-len", "2048", "--epochs", "1",
                         "--eval-every", "2000", "--pad-mult", "256", "--trace", "10"]):
        say("stopping: the gate is meaningless without a checkpoint")
        return False
    seed_base_arm()
    run("gate", ["rig_ftgate.py", os.path.join(OUT_MODEL, "qwen3-4b-arc2ft"), "12", "8", "420"],
        {"ARC_FTGATE_OUT": GATE_OUT})
    say("QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 0) else 1)
