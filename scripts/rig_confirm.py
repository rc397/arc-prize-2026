"""Confirmation seed for the deep rank-32 TTT candidate.

deep60r32 scored 27/125 at seed 0, the best local TTT number, but two baseline seeds scored 24 and 22,
so a one-seed result is not distinguishable from a high draw of that variance. This trains the same
60-step rank-32 config at seed 1 and scores it. Near 27 confirms the lift is real, near 23 says it was a
lucky draw.

Waits for the running queue (pid arg) to finish first, since the rig has one GPU. Resumable via the
sweep's per-config manifest, so a reboot mid-train continues rather than restarts.
"""
import subprocess
import sys
import time

PY = sys.executable
SWEEP = r"D:\arc\rig_ttt_sweep.py"
WAIT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def main():
    if WAIT_PID:
        print("waiting for pid %d (the running queue) to finish" % WAIT_PID, flush=True)
        while alive(WAIT_PID):
            time.sleep(120)
        print("queue finished, training confirmation seed", flush=True)
    t0 = time.time()
    # same deep rank-32 config, seed 1
    subprocess.run([PY, "-u", SWEEP, "0", "60", "32", "0.1", "1.25", "deep60r32s1", "2000", "1"])
    print("CONFIRM DONE in %.2fh" % ((time.time() - t0) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
