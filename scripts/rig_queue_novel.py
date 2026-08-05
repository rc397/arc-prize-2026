"""Overnight queue for the two novelty experiments: DFS tree logging, then dual-expert selection.

One python process subprocess-running its phases, the pattern that survives SSH disconnects and
gives each phase its own log. The dual-expert phase runs even if the logger fails, since they share
nothing but the GPU.

Usage: python rig_queue_novel.py [T1_TASKS=16] [T2_TASKS=24]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
LOG = os.path.join(BASE, "experiments", "novel_queue.log")


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def phase(label, args):
    say("START %s" % label)
    t0 = time.time()
    # encoding and errors are not optional here. The default text mode decodes the child's stream
    # with the console codepage, and one byte of unsloth's progress output killed the reader thread
    # and with it a run that was otherwise healthy.
    r = subprocess.run([sys.executable] + args, cwd=BASE,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(r.stdout or "")
        f.write(r.stderr or "")
    say("END   %s rc=%d in %.0fs" % (label, r.returncode, time.time() - t0))
    return r.returncode == 0


def main():
    t2 = sys.argv[1] if len(sys.argv) > 1 else "24"
    t3 = sys.argv[2] if len(sys.argv) > 2 else "12"
    # dualexpert resumes from its own jsonl, so the one completed task is not repeated
    ok2 = phase("dualexpert %s" % t2, [os.path.join(HERE, "rig_dualexpert.py"), t2])
    ok3 = phase("subsetttt %s" % t3, [os.path.join(HERE, "rig_subsetttt.py"), t3])
    say("QUEUE DONE dualexpert=%s subsetttt=%s" % (ok2, ok3))
    return ok2 and ok3


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
