"""Wait for the running GPU job to finish, then run the queued experiments in order.

The shape gate owns the card for several hours and two jobs on one 16 GB GPU corrupt each other's
timings, which matters here because peak VRAM is one of the things being measured. So this waits
rather than competing.

It is a PYTHON parent that subprocess-runs each phase, because a detached PowerShell orchestrator
dies with the SSH session that launched it and a pid-chained launcher dies in a Windows Update
reboot without ever firing. Each phase writes its own resumable state, so an interruption costs
wall-clock and not results.

Order is by expected value per hour, given that the ledger already bounds selection at three
outputs and puts the loss in GENERATION:

  1. dfscache_ab   the deployed DFS shares one growing KV cache across sibling branches. Measured
                   on the rig, DynamicCache appends, so every branch attends over the branches
                   before it and the cache grows for the whole search. Generation-side, which is
                   where 65 percent of tasks never produce the truth at all.

Usage: python rig_queue_after.py [WAIT_FOR_PID]
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
LOG = os.path.join(BASE, "experiments", "queue_after.log")

PHASES = [
    ("dfscache_ab", ["rig_dfscache_ab.py", "10", "60", "32", "180"]),
]


def say(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    try:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:
        pass
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def gpu_busy(exclude_pid):
    """Any python process other than this one and the excluded pid holding the card."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return False
    mine = {os.getpid(), exclude_pid}
    pids = set()
    for tok in out.split():
        try:
            pids.add(int(tok))
        except ValueError:
            pass
    return bool(pids - mine)


def main():
    wait_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    say("waiting for the card to free up")
    waited = 0
    while gpu_busy(wait_pid) and waited < 14 * 3600:
        time.sleep(120)
        waited += 120
        if waited % 1800 == 0:
            say("  still waiting, %d min" % (waited // 60))
    say("card free after %d min, starting queue" % (waited // 60))

    for name, args in PHASES:
        say("PHASE %s: %s" % (name, " ".join(args)))
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.Popen([sys.executable, "-u", os.path.join(SCRIPTS, args[0])] + args[1:],
                             cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            if "%|" not in line and "it/s]" not in line:
                say("  %s" % line.rstrip())
        p.wait()
        say("PHASE %s finished rc=%d in %.0fs" % (name, p.returncode, time.time() - t0))
    say("QUEUE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
