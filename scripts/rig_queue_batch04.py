"""The real compositional fine-tune, and its gate.

The pilot trained 8,000 of 78,792 examples and moved val loss 9.8%, which is why its gate came back
null in both directions: too small an intervention to test the hypothesis, not evidence against it.
That was worth accepting when the full corpus cost eleven days. It cost eleven days because every
sequence was a different length and CUDA's caching allocator never reused a block; padding to a
multiple of 256 made the same 8,000 examples take 2.42 hours instead of 26.0, so the full corpus is
about 23 hours and the real experiment is affordable.

  1 finetune   all 78,792 examples under the 2048 cap, one epoch, rank 32, padded. eval-every is set
               to a real value this time so the checkpoint is selected on held-out loss rather than
               being whatever the last step produced.
  2 gate       the resulting checkpoint against the stock base, no TTT, on held-out instances of the
               trained compositions and on ARC-AGI-2. The ARC arm is the veto.

The gate writes to its OWN file. Its resume logic keys on (arm, task), so pointing it at the pilot's
ftgate.jsonl would silently skip all 40 records and report a finished gate that never ran.

Usage: python rig_queue_batch04.py
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
LOG = os.path.join(BASE, "experiments", "batch04.log")

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
FULL = os.path.join(BASE, "models", "comp_full")

PHASES = [
    ("full-finetune", ["finetune_base.py", "--model", STOCK,
                       "--corpus", os.path.join(BASE, "_corpus_comp"),
                       "--out", FULL, "--max-len", "2048", "--epochs", "1",
                       "--eval-every", "5000", "--pad-mult", "256", "--trace", "20"], {}),
    ("full-gate", ["rig_ftgate.py", os.path.join(FULL, "qwen3-4b-arc2ft"), "12", "8", "420"],
     {"ARC_FTGATE_OUT": os.path.join(BASE, "experiments", "ftgate_full.jsonl")}),
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


def main():
    for name, args, extra in PHASES:
        say("PHASE %s: %s" % (name, " ".join(args)))
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.update(extra)
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
        if p.returncode != 0:
            say("stopping: the gate is meaningless without a checkpoint")
            return False
    say("QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
