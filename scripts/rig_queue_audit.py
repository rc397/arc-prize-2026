"""Wait for the GPU to free, then walk the contamination audit's GPU gates, smoke first.

One python process that subprocess-runs its phases, because a pid-chained launcher dies with the
session that launched it and a detached PowerShell orchestrator dies with its SSH connection. Every
phase is gated on the previous one's exit code, and the two smoke phases run six tasks each so that
a broken generate loop costs six minutes rather than two hours.

Usage: python rig_queue_audit.py [WAIT_PID]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
LOG = os.path.join(BASE, "experiments", "audit_queue.log")
AUDIT = os.path.join(HERE, "rig_contamination_audit.py")
WAIT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# (stage, n_tasks, label). The smoke rows come first and are cheap; a failure there costs minutes.
PHASES = [("g2", 6, "smoke: truth-likelihood on six outputs"),
          ("g3", 6, "smoke: greedy decode on six outputs, proves the generate loop runs"),
          ("g2", 0, "full: truth-likelihood over the public evaluation set"),
          ("g3", 0, "full: greedy decode over the public evaluation set"),
          ("g5", 0, "full: greedy decode over the fresh column, then the 2x2")]


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid):
    if not pid:
        return False
    out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    if WAIT_PID:
        say("waiting for pid %d to release the GPU" % WAIT_PID)
        while alive(WAIT_PID):
            time.sleep(60)
        say("pid %d gone" % WAIT_PID)

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for stage, n, label in PHASES:
        args = [sys.executable, AUDIT, stage] + ([str(n)] if n else [])
        say("START %s (%s)" % (stage, label))
        t0 = time.time()
        r = subprocess.run(args, cwd=BASE, env=env, capture_output=True, text=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(r.stdout or "")
            f.write(r.stderr or "")
        say("END   %s rc=%d in %.0fs" % (stage, r.returncode, time.time() - t0))
        if r.returncode != 0:
            say("STOPPED: %s failed, later phases would inherit the fault" % stage)
            return False
    say("ALL PHASES COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
