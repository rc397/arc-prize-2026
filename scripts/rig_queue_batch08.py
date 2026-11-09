"""The last untested cell: full composed exposure WITH full protection. 41 hours.

The 2x2 so far, all against a base of 5 of 20 on ARC and 3 of 8 on composed:

  8k composed, no protection      ARC -1 task, -29% candidates    composed +1, +13%
  67.7k composed, no protection   ARC -2 tasks, -47%              composed +3, +200%
  4k composed, 4k protection      ARC  0, 0%                      composed +1, +8%
  8k composed, 8k protection      ARC  0, -6%                     composed +1, -15%

Protection prevents the damage; the large composed gain needs the large exposure. The cell nobody
has run is both at once, and it is the only one that could show composed gains without ARC loss.
I wanted it run for completeness, knowing the cost.

What it can return: the best plausible outcome is "ARC unchanged, composed
better", and composed is our own synthetic benchmark rather than the target. That is a completed 2x2
rather than a score, and it is worth having in the paper precisely because it closes the direction
with a full grid instead of three of its four corners.

  1 single   regenerate the single-rule corpus large enough to match, ~70k examples. CPU.
  2 mix      67,671 composed interleaved with 67,671 single-rule
  3 train    135,342 steps at about 1.1 s/step, ~41 hours, WITH --resume
  4 gate     ft arm only, seeded from the full run's base records, ~2.2 hours

The resume flag matters here more than anywhere. Windows Update force-rebooted this box twice in ten
days, and a 41-hour run without a checkpoint is a coin flip on losing the lot. The trainer now saves
its adapter and step at every eval and replays the same shuffled order, so a reboot costs at most one
eval interval. If this queue dies, relaunch it: every phase is resumable or a no-op.

Usage: python rig_queue_batch08.py
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
EXP = os.path.join(BASE, "experiments")
LOG = os.path.join(EXP, "batch08.log")

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
SINGLE_BIG = os.path.join(BASE, "_corpus_single_big")
MIXED_FULL = os.path.join(BASE, "_corpus_mixed_full")
OUT_MODEL = os.path.join(BASE, "models", "comp_mixed_full")
GATE_OUT = os.path.join(EXP, "ftgate_mixedfull.jsonl")
N_EACH = 67671


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
    if os.path.isfile(GATE_OUT) or not os.path.isfile(src):
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


def enough(path, n):
    """Skip a phase that has already produced what it was for, so relaunching is cheap."""
    f = os.path.join(path, "train.jsonl")
    if not os.path.isfile(f):
        return False
    with open(f, encoding="utf-8") as fh:
        for i, _ in enumerate(fh, 1):
            if i >= n:
                return True
    return False


def main():
    if enough(SINGLE_BIG, N_EACH):
        say("single-rule corpus already large enough, skipping generation")
    elif not run("single", ["build_ft_corpus.py",
                            "--challenges", os.path.join(BASE, "data", "kaggle",
                                                         "arc-agi_training_challenges.json"),
                            "--rearc-dir", os.path.join(BASE, "re-arc"),
                            "--out", SINGLE_BIG, "--rearc-pool", "60", "--per-task", "55",
                            "--max-ctx-tokens", "1600"]):
        say("stopping: no single-rule corpus")
        return False

    if enough(MIXED_FULL, 2 * N_EACH):
        say("mixed corpus already built, skipping")
    elif not run("mix", ["mix_corpus.py", os.path.join(BASE, "_corpus_comp"), SINGLE_BIG,
                         MIXED_FULL, str(N_EACH), "300"]):
        say("stopping: no mixed corpus")
        return False

    if not run("train", ["finetune_base.py", "--model", STOCK, "--corpus", MIXED_FULL,
                         "--out", OUT_MODEL, "--max-len", "2048", "--epochs", "1",
                         "--eval-every", "10000", "--pad-mult", "256", "--resume",
                         "--trace", "10"]):
        say("stopping: the gate is meaningless without a checkpoint")
        return False

    seed_base_arm()
    run("gate", ["rig_ftgate.py", os.path.join(OUT_MODEL, "qwen3-4b-arc2ft"), "12", "8", "420"],
        {"ARC_FTGATE_OUT": GATE_OUT})
    say("QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
