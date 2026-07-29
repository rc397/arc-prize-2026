"""Run the remaining smoke stages, then the full probe, unattended and gated.

The laptop is going away, so this has to be one detached python process that survives the SSH link
dropping. A detached PowerShell orchestrator does not; a detached python process does, which is the
rule the rig has already taught us once.

The gate matters more than the automation. Each stage exits 0 or 1 with a checklist, and the full run
starts ONLY if every stage passes. s3 additionally projects the full run's wall clock from its own
measured train and decode times and fails itself if that exceeds the ceiling, so the go/no-go on
thirty-odd hours of compute is a check rather than a judgement made in advance by me.

Everything is resumable. Windows Update can force-reboot this box and has killed an overnight run
before, so relaunching this script simply continues: completed smoke stages are skipped via their
marker files, and the probe resumes per cell.

Usage: python rig_queue_restart.py [--no-full]
Watch:  experiments/queue.log and experiments/queue_status.json
Stop:   kill this pid; nothing is left in a bad state.
"""
import json
import os
import subprocess
import sys
import time

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
SCRIPTS = os.path.join(BASE, "scripts")
EXP = os.path.join(BASE, "experiments")
STATUS = os.path.join(EXP, "queue_status.json")
SMOKE = os.path.join(SCRIPTS, "rig_smoke_restart.py")
SUPER = os.path.join(SCRIPTS, "rig_restart_supervise.py")

RUN_FULL = "--no-full" not in sys.argv
STAGES = ["s2", "s3", "s4"]
# generous, since these are bounded by their own budgets; a stage past this is wedged
STAGE_TIMEOUT = {"s2": 3 * 3600, "s3": 3 * 3600, "s4": 3 * 3600}


def note(**kw):
    kw["ts"] = time.time()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f, indent=1)
    os.replace(tmp, STATUS)


def marker(stage):
    return os.path.join(EXP, "smoke_%s_PASSED" % stage)


def run_stage(stage):
    if os.path.isfile(marker(stage)):
        print("[queue] %s already passed, skipping" % stage, flush=True)
        return True
    log = os.path.join(EXP, "smoke_%s.log" % stage)
    print("[queue] running %s, log %s" % (stage, log), flush=True)
    note(phase=stage, state="running")
    with open(log, "w", encoding="utf-8") as lf:
        try:
            rc = subprocess.run([sys.executable, "-u", SMOKE, stage], stdout=lf,
                                stderr=subprocess.STDOUT, cwd=BASE,
                                timeout=STAGE_TIMEOUT[stage]).returncode
        except subprocess.TimeoutExpired:
            print("[queue] %s TIMED OUT" % stage, flush=True)
            note(phase=stage, state="timeout")
            return False
    tail = ""
    try:
        tail = "".join(open(log, encoding="utf-8").readlines()[-3:]).strip()
    except Exception:
        pass
    print("[queue] %s exit %d :: %s" % (stage, rc, tail), flush=True)
    if rc == 0:
        open(marker(stage), "w").write(str(time.time()))
        note(phase=stage, state="passed")
        return True
    note(phase=stage, state="failed", detail=tail)
    return False


def main():
    os.makedirs(EXP, exist_ok=True)
    print("[queue] start, full run %s" % ("ENABLED" if RUN_FULL else "disabled"), flush=True)
    for stage in STAGES:
        if not run_stage(stage):
            print("[queue] STOPPING: %s did not pass. The full run is NOT started." % stage,
                  flush=True)
            note(phase=stage, state="halted", full_run=False)
            return
    print("[queue] all smoke stages passed", flush=True)
    if not RUN_FULL:
        note(phase="smoke", state="all_passed", full_run=False)
        return
    print("[queue] launching the supervised full run", flush=True)
    note(phase="full", state="running")
    log = os.path.join(EXP, "queue_full.log")
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.run([sys.executable, "-u", SUPER, "20", "3", "5", "40"],
                            stdout=lf, stderr=subprocess.STDOUT, cwd=BASE).returncode
    print("[queue] full run exit %d" % rc, flush=True)
    note(phase="full", state="done", rc=rc)


if __name__ == "__main__":
    main()
