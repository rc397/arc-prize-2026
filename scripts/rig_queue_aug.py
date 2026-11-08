"""Sequential queue for the augmentation-space experiments. One process, streamed logs, resumable.

The card is single and saturates, so this is a queue and not a fan-out. Every phase runs even if an
earlier one fails, since they share nothing but the GPU and each resumes from its own jsonl.

The point of the queue is that the rig should never be idle. It has been idle for long stretches
this week because a run ended without the next one queued, and idle rig time is the
only resource here that cannot be recovered.

Usage: python rig_queue_aug.py [AUX_TASKS=8] [DECODE_S=600]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
LOG = os.path.join(BASE, "experiments", "aug_queue.log")


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def phase(label, args):
    say("START %s" % label)
    t0 = time.time()
    # stream rather than buffer: a multi-hour phase must show progress, and a queue killed mid-phase
    # must not take its whole log with it, which has happened here before.
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, "-u"] + args, cwd=BASE,
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            f.write(line)
            f.flush()
        rc = p.wait()
    say("END   %s rc=%d in %.0fs" % (label, rc, time.time() - t0))
    return rc == 0


def main():
    t = sys.argv[1] if len(sys.argv) > 1 else "8"
    d = sys.argv[2] if len(sys.argv) > 2 else "600"
    ok = {}
    # my four augmentation ideas, all five arms, in one pass so the base arm is shared
    ok["auxobj"] = phase("auxobj %s" % t, [os.path.join(HERE, "rig_auxobj.py"), t, d])
    say("QUEUE DONE " + " ".join("%s=%s" % kv for kv in ok.items()))
    return all(ok.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
