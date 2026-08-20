"""Does rule SHAPE matter, when Phase 0 showed rule COUNT does not?

Phase 0 varied how many distinct rules a fixed example budget was drawn from, 100 compositions
against 402, and found nothing: identical on both evaluations, tied 12 of 12 on ARC-AGI-2. It never
varied what KIND of rule. Every rule this project holds is ARC-1, either a RE-ARC verifier or a pair
of them composed. The generator corpus is ARC-2-shaped by construction, since each program was
written against an ARC-AGI-2 training task and verified by reproducing that task's own pairs.

  arc1   7,840 examples, 100 composed ARC-1 rules
  arc2   7,840 examples, 100 verified ARC-2 rules

WHY THE ARC-1 ARM IS REBUILT RATHER THAN REUSED. Phase 0's narrow arm is already trained and would
have been free, but it was built at 160 pairs per pool while the ARC-2 samplers yield 12.8. Reusing
it would confound shape with pool depth by a factor of twelve, and a loss for the ARC-2 arm could
not then be attributed. `--compose-per-pool 13` matches the depth, so rule count, example count,
pool depth, recipe and gate are all held fixed and shape is the only difference.

Reading, fixed in advance:
  arc2 clearly better   ARC-2 shape is the lever Phase 0 missed, and buying the remaining ~588
                        generators for about 101 CNY is justified.
  indistinguishable     shape is not the binding variable either. Nothing more should be bought,
                        and the paper gets a second clean negative alongside the count result,
                        which together are a much stronger ceiling claim than either alone.
  arc1 better           would suggest the ARC-2 corpus is defective rather than that shape hurts.
                        Check the corpus before believing it.

About 9.5 hours: one corpus build, two 2.4-hour trainings, two 2.2-hour gates.

Usage: python rig_queue_shape.py
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
EXP = os.path.join(BASE, "experiments")
LOG = os.path.join(EXP, "shape.log")

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
SURVEY100 = os.path.join(EXP, "compose_survey_100.json")
CHAL = os.path.join(BASE, "data", "kaggle", "arc-agi_training_challenges.json")
REARC = os.path.join(BASE, "re-arc")

ARC2_CORPUS = os.path.join(BASE, "corpus_arc2shape")
ARC1_CORPUS = os.path.join(BASE, "corpus_arc1matched")
LIMIT = "7840"          # what the ARC-2 corpus actually holds; both arms cut to it


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


def seed_gate(path):
    """Reuse the banked base-arm records so each gate runs only its fine-tuned arm."""
    src = os.path.join(EXP, "ftgate_full.jsonl")
    if os.path.isfile(path) or not os.path.isfile(src):
        return
    rows = [l for l in open(src, encoding="utf-8") if json.loads(l)["arm"] == "base"]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    say("seeded %d base-arm records into %s" % (len(rows), os.path.basename(path)))


def main():
    assert os.path.isfile(os.path.join(ARC2_CORPUS, "train.jsonl")), \
        "build the ARC-2 corpus first: python mint_genrules.py 100 8000"
    assert os.path.isfile(SURVEY100), "compose_survey_100.json missing"

    if os.path.isfile(os.path.join(ARC1_CORPUS, "train.jsonl")):
        say("matched ARC-1 corpus already built")
    elif not run("corpus-arc1matched",
                 ["build_ft_corpus.py", "--challenges", CHAL, "--rearc-dir", REARC,
                  "--out", ARC1_CORPUS, "--compose-only", "--compose-survey", SURVEY100,
                  "--compose-per-pool", "13", "--max-ctx-tokens", "1600"]):
        say("stopping: matched ARC-1 corpus failed")
        return False

    for name, corpus in (("arc1", ARC1_CORPUS), ("arc2", ARC2_CORPUS)):
        model = os.path.join(BASE, "models", "shape_%s" % name)
        if not run("train-%s" % name,
                   ["finetune_base.py", "--model", STOCK, "--corpus", corpus, "--out", model,
                    "--max-len", "2048", "--epochs", "1", "--limit", LIMIT,
                    "--eval-every", "2000", "--pad-mult", "256", "--resume", "--trace", "6"]):
            say("stopping: training %s failed" % name)
            return False

        gate = os.path.join(EXP, "ftgate_shape_%s.jsonl" % name)
        seed_gate(gate)
        run("gate-%s" % name,
            ["rig_ftgate.py", os.path.join(model, "qwen3-4b-arc2ft"), "12", "8", "420"],
            {"ARC_FTGATE_OUT": gate})
    say("SHAPE QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
