"""The queue for Ryan's idea batch. Waits for the card, then works the list without stopping.

One process that subprocess-runs its phases in order, rather than a chain of detached launches.
A chained waiter dies in the same Windows Update reboot that kills its target, and then nothing
fires and nothing says so. If this process dies, nothing has started and that is visible.

Phases, in the order the triage ranked them:

  1 maskloss    unchanged-cell loss masking, three arms. The top pick of the batch: 88 percent of a
                median output is a copy of its input, so that is where the gradient goes today.
  2 dfsfloor    raise the DFS probability floor. Not from the batch at all, it came out of replaying
                the logged trees this afternoon, and it is the only lever here with a measured
                mechanism: 12 of 16 logged decodes ran out of wall clock.
  3 dimprefix   forced output-dimension prefix. A deliberate Gate 3 violation, priced not dismissed.

Every phase writes its own resumable jsonl and skips records it already has, so a reboot costs the
run in flight and nothing else.

Usage: python rig_queue_batch02.py [WAIT_PID]
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
LOG = os.path.join(BASE, "experiments", "batch02.log")
POLL_S = 60

PHASES = [
    ("maskloss", ["rig_maskloss.py", "8", "600"]),
    ("dfsfloor", ["rig_dfsfloor.py", "8", "600"]),
    ("dimprefix", ["rig_dimprefix.py", "6", "600"]),
]


def say(msg):
    """Never let a log line kill the queue.

    It already did once. A detached Start-Process on this box redirects stdout at cp1252 unless
    PYTHONIOENCODING is set on the PARENT, and the first tqdm bar out of the weight loader carries a
    block character cp1252 cannot encode. The queue died on the print, five hours into an empty
    card, while the phase it had launched was still loading. So the console write is forced to ascii
    and the real text goes to the log file, which is opened as utf-8 explicitly.
    """
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
    """Drop progress-bar redraws, which are most of the volume and none of the information."""
    return "%|" not in line and "it/s]" not in line and "s/it]" not in line


def alive(pid):
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259          # STILL_ACTIVE


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
        # streamed line by line rather than captured, so a phase that runs for hours is watchable
        # and a phase killed mid-run still leaves everything it printed
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
