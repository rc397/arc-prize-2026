"""Keep rig_restart_probe.py running until every cell is recorded.

The probe exits deliberately when a cell faults, because a memory or allocator failure leaves the CUDA
context unreliable and continuing in the same process only produces more misleading failures. That
makes relaunching someone's job, and overnight it should not be mine. This is one python process that
subprocess-runs the probe in a loop, which is the shape the rig notes call for: a detached PowerShell
orchestrator dies with its launching SSH session, a detached python process does not.

It stops when the probe reports every cell done, when no progress is made across two consecutive
attempts (so a cell that faults deterministically cannot loop forever), or at the attempt ceiling.

Usage: python rig_restart_supervise.py [N_TASKS=20] [K_SEEDS=3] [MAX_ATTEMPTS=40]
"""
import json
import os
import subprocess
import sys
import time

BASE = r"D:\arc"
PROBE = os.path.join(BASE, "scripts", "rig_restart_probe.py")
RESULTS = os.path.join(BASE, "experiments", "restart_probe.jsonl")
LOG = os.path.join(BASE, "experiments", "restart_probe.log")
STATUS = os.path.join(BASE, "experiments", "supervisor_status.json")

N_TASKS = sys.argv[1] if len(sys.argv) > 1 else "20"
K_SEEDS = sys.argv[2] if len(sys.argv) > 2 else "3"
MAX_ATTEMPTS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
TARGET = int(N_TASKS) * int(K_SEEDS)


def recorded():
    if not os.path.isfile(RESULTS):
        return 0, 0
    n = ok = 0
    for line in open(RESULTS):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        ok += bool(r.get("ok"))
    return n, ok


def main():
    stale = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before, ok_before = recorded()
        if before >= TARGET:
            print("all %d cells recorded, stopping" % TARGET, flush=True)
            break
        print("=== attempt %d/%d, %d/%d cells recorded (%d ok) ==="
              % (attempt, MAX_ATTEMPTS, before, TARGET, ok_before), flush=True)
        json.dump({"attempt": attempt, "recorded": before, "ok": ok_before,
                   "target": TARGET, "ts": time.time()}, open(STATUS, "w"))

        with open(LOG, "a", encoding="utf-8") as lf:
            lf.write("\n=== supervisor attempt %d ===\n" % attempt)
            lf.flush()
            subprocess.run([sys.executable, "-u", PROBE, N_TASKS, K_SEEDS],
                           stdout=lf, stderr=subprocess.STDOUT, cwd=BASE)

        after, ok_after = recorded()
        # A faulting cell still writes its record, so no growth at all means the probe died before
        # recording anything and relaunching will not help.
        stale = stale + 1 if after == before else 0
        if stale >= 2:
            print("no new records across two attempts, stopping", flush=True)
            break

    n, ok = recorded()
    print("SUPERVISOR DONE: %d/%d recorded, %d ok" % (n, TARGET, ok), flush=True)
    json.dump({"done": True, "recorded": n, "ok": ok, "target": TARGET, "ts": time.time()},
              open(STATUS, "w"))
    subprocess.run([sys.executable, "-u", PROBE, "summary"], cwd=BASE)


if __name__ == "__main__":
    main()
