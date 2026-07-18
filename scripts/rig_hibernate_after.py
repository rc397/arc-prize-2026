"""Hibernate the rig after a job batch (by PID) finishes. Power-saving option 1, resume by power button.

Launched detached alongside a rig job with that job's PID. When the job exits, this waits a short grace
period for final writes and hibernates (S4, near-zero draw, verified available on the rig). Jobs
checkpoint to disk, so nothing state-wise is lost. Wake is a physical power-button press by design, a
hibernated box is off the network so neither SSH nor Tailscale can reach it to wake it remotely.

Attach only to a batch that is the last thing we need for a while, not to every micro-step during active
iteration, or the rig will keep powering down between dependent runs.

Usage: python rig_hibernate_after.py <job_pid> [grace_seconds=90]
"""
import subprocess
import sys
import time

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
GRACE = int(sys.argv[2]) if len(sys.argv) > 2 else 90


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def main():
    if PID:
        while alive(PID):
            time.sleep(60)
    time.sleep(GRACE)
    subprocess.run(["shutdown", "/h"])


if __name__ == "__main__":
    main()
