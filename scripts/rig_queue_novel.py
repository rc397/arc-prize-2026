"""Overnight queue for the novelty experiments, in the order their results are worth most.

One python process subprocess-running its phases, the pattern that survives SSH disconnects and
gives each phase its own log. Every phase runs even if an earlier one fails, since they share
nothing but the GPU, and each resumes from its own jsonl.

Pre-fitting goes first because it is the only phase testing a method rather than measuring one, and
because the prior-art check puts it at adjacent-negative, which is worth knowing early. Dual-expert
is second because it is the one measurement that speaks to the appendix's second escape route.

Usage: python rig_queue_novel.py [PREFIT_TASKS=10] [DUAL_TASKS=24] [SUBSET_TASKS=12]
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
    #
    # Streaming rather than capture_output is also not optional. Buffering hides a multi-hour
    # phase's progress until it exits, and if the queue dies mid-phase the whole log is lost, which
    # is exactly how one night's subsetttt output disappeared with no record of why it stopped.
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
    t1 = sys.argv[1] if len(sys.argv) > 1 else "8"
    t2 = sys.argv[2] if len(sys.argv) > 2 else "24"
    t3 = sys.argv[3] if len(sys.argv) > 3 else "12"
    # each phase resumes from its own jsonl, so completed cells are never repeated
    ok = {}
    ok["prefit"] = phase("prefit %s" % t1, [os.path.join(HERE, "rig_prefit.py"), t1])
    ok["dualexpert"] = phase("dualexpert %s" % t2, [os.path.join(HERE, "rig_dualexpert.py"), t2])
    ok["subsetttt"] = phase("subsetttt %s" % t3, [os.path.join(HERE, "rig_subsetttt.py"), t3])
    say("QUEUE DONE " + " ".join("%s=%s" % kv for kv in ok.items()))
    return all(ok.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
