"""The follow-on queue: train on the compositional corpus, then gate the result.

Chained behind the batch-02 queue rather than appended to it, because that process is already
running and will not pick up an edit to its own phase list.

Why this is next. Every one of the 24 measured levers changed the PIPELINE around a fixed model, and
none of them moved the score. This is the first thing in months that changes the model itself, on
data that does not exist anywhere else: 402 two-step rules composed from RE-ARC verifiers, 78,792
examples. The learnability probe cleared its gate at 5 of 14 against 14 of 14 for single rules, which
is the pre-registered "harder than one rule and still learnable" case.

  1 pilot   8,000 examples, one pass, rank 32. A pilot rather than the full corpus because a bad
            fine-tune is discovered at the same cost either way and a good one is worth confirming
            before spending a day on it. The limit is applied AFTER a seeded shuffle, so the sample
            covers the rule set rather than the head of the file.
  2 gate    the checkpoint against the stock base, no TTT, on held-out instances of the trained
            compositions and on ARC-AGI-2 evaluation tasks. The second is the veto.

Usage: python rig_queue_batch03.py [WAIT_PID]
"""
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
LOG = os.path.join(BASE, "experiments", "batch03.log")
POLL_S = 60

STOCK = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
PILOT = os.path.join(BASE, "models", "comp_pilot")

PHASES = [
    ("pilot-finetune", ["finetune_base.py", "--model", STOCK,
                        "--corpus", os.path.join(BASE, "_corpus_comp"),
                        "--out", PILOT, "--limit", "8000", "--max-len", "2048",
                        "--epochs", "1", "--eval-every", "2000"]),
    ("ft-gate", ["rig_ftgate.py", os.path.join(PILOT, "qwen3-4b-arc2ft"), "12", "8", "420"]),
]


def say(msg):
    """Forced to ascii on the console. A tqdm block character in a cp1252 redirect killed the
    batch-02 queue five hours into an empty card; the log file keeps the real text as utf-8."""
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
