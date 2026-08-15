"""Isolate the mixing effect from the exposure effect.

The 8,000-step mixed run lost none of the base's ARC coverage where the pure-composed pilot lost a
task and 29 percent of its candidates. That is what the anti-forgetting hypothesis predicts, and it
is confounded: the mixed run held 4,000 composed examples where the pilot held 8,000, so "mixing
protected the model" is not separated from "half the exposure did half the damage". Seven of twelve
ARC tasks came back with an identical candidate count and the total was exactly 36 to 36, which looks
more like a model that barely moved than one that moved and was protected.

This holds composed exposure FIXED at the pilot's 8,000 and adds 8,000 single-rule examples on top.
Composed exposure is then identical to the pilot's by construction and the only thing that varies is
the protective half, so the comparison is:

  pilot        8,000 composed                      ARC -1 task, candidates -29%
  this run     8,000 composed + 8,000 single       ?

If ARC is preserved here, mixing is confirmed and the compositional direction is alive at scale. If
ARC degrades the way the pilot's did, the 8,000-step mixed result was exposure and not protection,
and the direction is closed for this base.

Note what it still is not: 16,000 steps against the pilot's 8,000 is twice the compute, which is the
price of holding composed exposure fixed. The equal-compute comparison is the run we already have.

  1 mix    8,000 composed interleaved with 8,000 single-rule
  2 train  16,000 steps, about 4.5 hours
  3 gate   ft arm only, seeded from the full run's base records, about 2.2 hours

Usage: python rig_queue_batch07.py
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
EXP = os.path.join(BASE, "experiments")
LOG = os.path.join(EXP, "batch07.log")

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
MIXED16 = os.path.join(BASE, "_corpus_mixed16")
OUT_MODEL = os.path.join(BASE, "models", "comp_mixed16")
GATE_OUT = os.path.join(EXP, "ftgate_mixed16.jsonl")


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


def seed_base_arm():
    src = os.path.join(EXP, "ftgate_full.jsonl")
    if os.path.isfile(GATE_OUT):
        say("gate output already exists, leaving it alone")
        return
    if not os.path.isfile(src):
        say("no base-arm records to seed from; the base arm will re-run")
        return
    rows = [l for l in open(src, encoding="utf-8") if json.loads(l)["arm"] == "base"]
    with open(GATE_OUT, "w", encoding="utf-8") as f:
        f.writelines(rows)
    say("seeded %d base-arm records; the gate runs the ft arm only" % len(rows))


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


def main():
    if not run("mix", ["mix_corpus.py", os.path.join(BASE, "_corpus_comp"),
                       os.path.join(BASE, "_corpus_single"), MIXED16, "8000", "250"]):
        say("stopping: no mixed corpus")
        return False
    if not run("train", ["finetune_base.py", "--model", STOCK, "--corpus", MIXED16,
                         "--out", OUT_MODEL, "--max-len", "2048", "--epochs", "1",
                         "--eval-every", "4000", "--pad-mult", "256", "--trace", "10"]):
        say("stopping: the gate is meaningless without a checkpoint")
        return False
    seed_base_arm()
    run("gate", ["rig_ftgate.py", os.path.join(OUT_MODEL, "qwen3-4b-arc2ft"), "12", "8", "420"],
        {"ARC_FTGATE_OUT": GATE_OUT})
    say("QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
