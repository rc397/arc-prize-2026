"""Does training DURATION move ARC coverage, when rule count and rule shape did not?

Every training-side experiment here has varied the DATA and found nothing. Phase 0 varied rule count,
100 composed rules against 402 at an equal 8,000-example budget, and tied 12 of 12 on ARC-AGI-2. The
shape gate varied rule kind, 100 ARC-1 rules against 100 verified ARC-2 generators at matched pool
depth, and came back 0 better, 1 worse, tied 11 of 12. Neither varied how LONG the model trains.

And the logs say we have never trained to convergence. Validation loss was still falling when both
ARC-1 arms ended:

    shape arc1     0.0752 -> 0.0671 -> 0.0593   still falling at step 6000/7840
    phase0 narrow  0.0636 -> 0.0605 -> 0.0545 -> 0.0537

Independently, the 200M LongT5 ARC-AGI-2 system credits +7.78 points to grokking, 12.08 to 19.86, the
largest single jump in its ablation table and ahead of both traversal conditioning and multi-task
training. Duration is the only training-side lever anyone credits that we have not tested, and it
costs card time rather than money.

ONE VARIABLE. Phase 0's narrow arm is the control and is not re-run: same base, same corpus, same
--limit 8000, same rank 32 / lr 1e-4 / max-len 2048 / pad-mult 256, same 12-task gate. Only the step
count changes, 8,000 to 96,000, and gates at 16k / 48k / 96k give a four-point curve with Phase 0's
banked 8k point.

COVERAGE IS THE CRITERION, NOT LOSS. Phase 0's narrow arm reached validation loss 0.0537 and still
tied 12 of 12, so lower loss on this corpus has never once moved ARC coverage. Loss is logged as
instrumentation and to show whether a late-improvement grokking signature appears at all; it is not
the test.

Reading fixed in advance:
  coverage rises      duration is a real lever, and the curve prices how much more is worth buying
  coverage flat       duration is not the lever either. Three orthogonal training-side variables all
                      null is a far stronger ceiling claim than any one of them
  coverage falls      the base overfits, consistent with continued fine-tuning already measuring
                      11.63 percent against 12.79 stock, and the useful output is where the peak sits

Caveat that belongs in the writeup: twelve tasks have now tied 12/12 twice, so a flat curve means no
effect this instrument can resolve, not no effect.

About 35 hours: 29 h of training at the measured 1.07 s/step, plus three gates at about 2.2 h.

Usage: python rig_queue_grok.py
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
EXP = os.path.join(BASE, "experiments")
LOG = os.path.join(EXP, "grok.log")

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
CORPUS = os.path.join(BASE, "_corpus_r100")      # Phase 0's narrow corpus, unchanged
OUT = os.path.join(BASE, "models", "grok_r100")
LIMIT = "8000"                                    # same 8,000 examples Phase 0 used
EPOCHS = "12"                                     # 12 passes over them = 96,000 steps
SAVE_EVERY = 16000
GATE_AT = (16000, 48000, 96000)


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
    assert os.path.isfile(os.path.join(CORPUS, "train.jsonl")), "no corpus at %s" % CORPUS
    say("duration experiment: %s epochs over %s examples of %s, gates at %s"
        % (EPOCHS, LIMIT, os.path.basename(CORPUS), ", ".join(str(g) for g in GATE_AT)))

    if os.path.isdir(os.path.join(OUT, "adapter_step%d" % max(GATE_AT))):
        say("training already reached the final checkpoint, skipping to gates")
    elif not run("train-96k",
                 ["finetune_base.py", "--model", STOCK, "--corpus", CORPUS, "--out", OUT,
                  "--max-len", "2048", "--epochs", EPOCHS, "--limit", LIMIT,
                  "--eval-every", "2000", "--save-every", str(SAVE_EVERY),
                  "--pad-mult", "256", "--resume", "--trace", "6"]):
        say("stopping: training failed")
        return False

    for step in GATE_AT:
        adapter = os.path.join(OUT, "adapter_step%d" % step)
        if not os.path.isdir(adapter):
            say("no adapter at step %d, skipping (training may have been cut short)" % step)
            continue
        ckpt = os.path.join(OUT, "ckpt_step%d" % step)
        if not run("merge-%d" % step, ["merge_ft_adapter.py", adapter, ckpt]):
            say("merge failed at step %d, skipping its gate" % step)
            continue
        gate = os.path.join(EXP, "ftgate_grok_step%d.jsonl" % step)
        seed_gate(gate)
        run("gate-%d" % step, ["rig_ftgate.py", ckpt, "12", "8", "420"],
            {"ARC_FTGATE_OUT": gate})
    say("GROK QUEUE COMPLETE")
    say("curve: Phase 0 narrow at 8k is the banked fourth point (tied 12/12 on ARC-AGI-2)")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
