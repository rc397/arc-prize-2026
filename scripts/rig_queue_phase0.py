"""Phase 0: is RULE COUNT the lever, or just example count?

Everything in the rule-generation plan rests on one unmeasured assumption: that a base trained on
more DISTINCT rules learns something a base trained on more EXAMPLES of fewer rules does not. The
compositional 2x2 already showed that more examples of the same 402 rules buys nothing on ARC. It
never varied the rule count.

So: two arms, EQUAL example counts, 4x different diversity.

  narrow   8,000 examples drawn from 100 compositions
  wide     8,000 examples drawn from all 402

Both trained for the same 8,000 steps and gated the same way, so the only thing that differs is how
many distinct rules those examples came from. The trainer's --limit shuffles under a fixed seed
before cutting, so each arm gets a fair draw from its own corpus.

Reading, fixed in advance:
  wide clearly better on ARC   rule diversity is the lever, and buying thousands of ARC-2 rules is
                               justified. The slope also says how many are worth buying.
  the two indistinguishable    diversity is not the binding variable at this range, and generating
                               more rules will not help either. The whole pipeline closes for about
                               nine hours of card time and no money.
  narrow better                would be strange and would mean something is wrong with the wide
                               corpus rather than with the hypothesis. Check before believing it.

Roughly 9 hours: two 2.4-hour trainings and two 2.2-hour gates, the gates running their ft arm only
against the banked base records.

Usage: python rig_queue_phase0.py
"""
import json
import os
import random
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
EXP = os.path.join(BASE, "experiments")
LOG = os.path.join(EXP, "phase0.log")

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
SURVEY = os.path.join(EXP, "compose_survey.json")
SURVEY100 = os.path.join(EXP, "compose_survey_100.json")
CHAL = os.path.join(BASE, "data", "kaggle", "arc-agi_training_challenges.json")
REARC = os.path.join(BASE, "re-arc")

ARMS = [
    ("narrow", os.path.join(BASE, "_corpus_r100"), SURVEY100, "160"),
    ("wide", os.path.join(BASE, "_corpus_r402"), SURVEY, "40"),
]


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
        say("  [log stream failed: %s %s]" % (type(e).__name__, e))
    p.wait()
    say("PHASE %s finished rc=%d in %.0fs" % (name, p.returncode, time.time() - t0))
    return p.returncode == 0


def make_survey_100():
    """A 100-composition subset, sampled so the FIRST STAGES stay spread rather than taking a
    prefix, which would concentrate on whichever first stage is most prolific."""
    if os.path.isfile(SURVEY100):
        say("survey_100 already exists")
        return True
    s = json.load(open(SURVEY, encoding="utf-8"))
    kept = s.get("kept") or s.get("examples") or []
    assert kept, "no kept compositions in the survey"
    by_a = {}
    for k in kept:
        by_a.setdefault(k["a"], []).append(k)
    rng = random.Random(11)
    firsts = sorted(by_a)
    rng.shuffle(firsts)
    out, i = [], 0
    while len(out) < 100 and i < 50:
        for a in firsts:
            if len(by_a[a]) > i and len(out) < 100:
                out.append(by_a[a][i])
        i += 1
    say("survey_100: %d compositions over %d distinct first stages (from %d over %d)"
        % (len(out), len({k["a"] for k in out}), len(kept), len(by_a)))
    d = dict(s)
    d["kept"] = out
    json.dump(d, open(SURVEY100, "w"))
    return True


def seed_gate(path):
    src = os.path.join(EXP, "ftgate_full.jsonl")
    if os.path.isfile(path) or not os.path.isfile(src):
        return
    rows = [l for l in open(src, encoding="utf-8") if json.loads(l)["arm"] == "base"]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    say("seeded %d base-arm records into %s" % (len(rows), os.path.basename(path)))


def main():
    if not make_survey_100():
        return False
    for name, corpus, survey, per_pool in ARMS:
        if os.path.isfile(os.path.join(corpus, "train.jsonl")):
            say("corpus %s already built" % name)
        elif not run("corpus-%s" % name,
                     ["build_ft_corpus.py", "--challenges", CHAL, "--rearc-dir", REARC,
                      "--out", corpus, "--compose-only", "--compose-survey", survey,
                      "--compose-per-pool", per_pool, "--max-ctx-tokens", "1600"]):
            say("stopping: corpus %s failed" % name)
            return False

        model = os.path.join(BASE, "models", "phase0_%s" % name)
        if not run("train-%s" % name,
                   ["finetune_base.py", "--model", STOCK, "--corpus", corpus, "--out", model,
                    "--max-len", "2048", "--epochs", "1", "--limit", "8000",
                    "--eval-every", "2000", "--pad-mult", "256", "--resume", "--trace", "6"]):
            say("stopping: training %s failed" % name)
            return False

        gate = os.path.join(EXP, "ftgate_phase0_%s.jsonl" % name)
        seed_gate(gate)
        run("gate-%s" % name,
            ["rig_ftgate.py", os.path.join(model, "qwen3-4b-arc2ft"), "12", "8", "420"],
            {"ARC_FTGATE_OUT": gate})
    say("QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
