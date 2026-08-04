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
    r = subprocess.run([sys.executable] + args, cwd=BASE,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                       capture_output=True, text=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(r.stdout or "")
        f.write(r.stderr or "")
    say("END   %s rc=%d in %.0fs" % (label, r.returncode, time.time() - t0))
    return r.returncode == 0


def main():
    t1 = sys.argv[1] if len(sys.argv) > 1 else "16"
    t2 = sys.argv[2] if len(sys.argv) > 2 else "24"
    ok1 = phase("dfslog run %s" % t1, [os.path.join(HERE, "rig_dfslog.py"), "run", t1])
    ok2 = phase("dualexpert %s" % t2, [os.path.join(HERE, "rig_dualexpert.py"), t2])
    say("QUEUE DONE dfslog=%s dualexpert=%s" % (ok1, ok2))
    return ok1 and ok2


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
