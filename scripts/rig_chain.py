"""Wait for a running rig job to exit, then start the next one. Keeps the card from going idle.

The card is single and saturates, so work is a queue rather than a fan-out, and the queue only pays
off if the next phase starts when the previous one ends instead of when someone next looks. Idle rig
time is the one resource here that cannot be recovered, and this project has lost whole nights to a
session ending on an observation rather than a launch.

Deliberately one process that waits and then subprocess-runs the follow-on, rather than a chain of
detached launches: a chained waiter and its target die in the same reboot, and then nothing fires and
nothing says so. If this process dies, nothing has started and that is visible.

Usage: python rig_chain.py <PID_TO_WAIT_FOR> <script> [args...]
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
LOG = os.path.join(BASE, "experiments", "chain.log")
POLL_S = 60


def say(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid):
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259          # STILL_ACTIVE
    except Exception:
        return False


def main(pid, args):
    say("waiting on pid %d before starting: %s" % (pid, " ".join(args)))
    t0 = time.time()
    while alive(pid):
        time.sleep(POLL_S)
    say("pid %d exited after %.0fs of waiting, starting follow-on" % (pid, time.time() - t0))
    r = subprocess.run([sys.executable, "-u"] + args, cwd=BASE,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    say("follow-on finished rc=%d" % r.returncode)
    return r.returncode == 0


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]), sys.argv[2:]) else 1)
