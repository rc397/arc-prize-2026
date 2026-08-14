"""Overnight queue behind the full-corpus gate.

Two phases, in this order for a reason.

  1 tttab     the cold gate compares the two bases with NO test-time training, which is the right
              way to ask what a base knows and is not the condition we ship. This runs the same
              comparison under per-task TTT and the deployed decode. It is worth running whichever
              way the cold gate lands: if the fine-tune looks damaged cold, this asks whether
              adaptation rebuilds it, and if it looks good cold, the gate's own verdict text demands
              this confirmation before anyone believes it.
  2 maskloss  top up the unchanged-cell loss-masking experiment from 8 tasks to 12. select_tasks is
              monotone, so the existing 24 records stay valid and only the new tasks run. That
              result is already in the paper on 8 tasks and a sign test on 3 discordant tasks; four
              more tasks is the cheapest strengthening available and it fits the tail of a night.

Usage: python rig_queue_batch05.py [WAIT_PID]
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
LOG = os.path.join(BASE, "experiments", "batch05.log")
POLL_S = 60

FULL = os.path.join(BASE, "models", "comp_full", "qwen3-4b-arc2ft")

PHASES = [
    ("tttab", ["rig_tttab.py", FULL, "10", "420"]),
    ("maskloss-topup", ["rig_maskloss.py", "12", "600"]),
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


def alive(pid):
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259


def main(wait_pid):
    if wait_pid:
        say("waiting on pid %d before the queue starts" % wait_pid)
        t0 = time.time()
        while alive(wait_pid):
            time.sleep(POLL_S)
        say("pid %d exited after %.0fs" % (wait_pid, time.time() - t0))
    for name, args in PHASES:
        say("PHASE %s: %s" % (name, " ".join(args)))
        t0 = time.time()
        p = subprocess.Popen([sys.executable, "-u", os.path.join(SCRIPTS, args[0])] + args[1:],
                             cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"), text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in p.stdout:
                if interesting(line):
                    say("  %s" % line.rstrip())
        except Exception as e:
            say("  [log stream failed: %s %s, the phase keeps running]" % (type(e).__name__, e))
        p.wait()
        say("PHASE %s finished rc=%d in %.0fs" % (name, p.returncode, time.time() - t0))
    say("QUEUE COMPLETE")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 0) else 1)
