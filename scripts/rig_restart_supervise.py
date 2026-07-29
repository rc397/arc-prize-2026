"""Keep rig_restart_probe.py running until every cell is recorded, unattended.

The probe exits deliberately when a cell faults, because an allocator or cuBLAS failure leaves the
CUDA context unreliable and continuing in the same process only manufactures more misleading
failures. Relaunching is therefore someone's job, and overnight it should not be a person's. This is
one python process that subprocess-runs the probe in a loop, which is the shape the rig requires: a
detached PowerShell orchestrator dies with its launching SSH session, a detached python process does
not.

Two things it does that the previous version did not. It watches the probe's heartbeat and kills a
subprocess that has gone silent, because nothing read that file before and a hung CUDA call would
have blocked forever. And it stops on lack of progress rather than on a fixed attempt count, so a
cell that faults deterministically cannot spin while a run that is simply slow is left alone.

Usage: python rig_restart_supervise.py [N_TASKS=20] [K_SEEDS=3] [N_REPS=5] [MAX_ATTEMPTS=40]
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
PROBE = os.path.join(BASE, "scripts", "rig_restart_probe.py")
EXP = os.path.join(BASE, "experiments")
RESULTS = os.path.join(EXP, "restart_probe.jsonl")
LOG = os.path.join(EXP, "restart_probe.log")
STATUS = os.path.join(EXP, "supervisor_status.json")
HEARTBEAT = os.path.join(BASE, "eval_status_restart.json")

N_TASKS = sys.argv[1] if len(sys.argv) > 1 else "20"
K_SEEDS = sys.argv[2] if len(sys.argv) > 2 else "3"
N_REPS = sys.argv[3] if len(sys.argv) > 3 else "5"
MAX_ATTEMPTS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
# A cell is bounded by TRAIN_BUDGET + DECODE_BUDGET, about 65 minutes. Silence for twice that means
# the process is wedged rather than slow.
STALE_S = 2 * 60 * 60


def counts():
    ok = started = 0
    if os.path.isfile(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("marker") == "started":
                started += 1
            elif r.get("ok"):
                ok += 1
    return ok, started


def run_once(attempt):
    """Run the probe, killing it if its heartbeat goes stale. Returns its exit code."""
    with open(LOG, "a", encoding="utf-8") as lf:
        lf.write("\n=== supervisor attempt %d ===\n" % attempt)
        lf.flush()
        p = subprocess.Popen([sys.executable, "-u", PROBE, N_TASKS, K_SEEDS, N_REPS],
                             stdout=lf, stderr=subprocess.STDOUT, cwd=BASE)
        while p.poll() is None:
            time.sleep(60)
            try:
                age = time.time() - json.load(open(HEARTBEAT))["ts"]
            except Exception:
                age = 0
            if age > STALE_S:
                lf.write("heartbeat stale for %.0fs, killing\n" % age)
                lf.flush()
                p.kill()
                return -1
        return p.returncode


def main():
    os.makedirs(EXP, exist_ok=True)
    stale = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ok_before, _ = counts()
        json.dump({"attempt": attempt, "ok": ok_before, "ts": time.time()},
                  open(STATUS, "w"))
        print("=== attempt %d/%d, %d cells ok ===" % (attempt, MAX_ATTEMPTS, ok_before), flush=True)
        rc = run_once(attempt)
        ok_after, _ = counts()
        print("    attempt %d exit %s, ok %d -> %d" % (attempt, rc, ok_before, ok_after), flush=True)
        if rc == 0:
            print("probe reported completion", flush=True)
            break
        # A faulting cell still writes its record, so no growth at all means relaunching will not
        # help. Two such attempts in a row is the stop condition.
        stale = stale + 1 if ok_after == ok_before else 0
        if stale >= 2:
            print("no progress across two attempts, stopping", flush=True)
            break

    ok, started = counts()
    print("SUPERVISOR DONE: %d cells ok, %d attempts recorded" % (ok, started), flush=True)
    json.dump({"done": True, "ok": ok, "started": started, "ts": time.time()},
              open(STATUS, "w"))
    subprocess.run([sys.executable, "-u", PROBE, "summary"], cwd=BASE)


if __name__ == "__main__":
    main()
