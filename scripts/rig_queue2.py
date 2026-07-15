"""Overnight queue 2. Waits for the running sweep to finish, then runs the next experiments.

Chained rather than launched in parallel because the rig has one GPU and two trainers would contend.

Queue, in order of expected value:
  1. Training-context A/B. The base config at 1600 tokens against the 2000 already measured. The paper's
     17.6 was measured at 1600 and this harness scored the same config 19.2 at 2000, but across two
     harnesses, so this is the clean in-harness test of whether context is a real lever or that gap was
     run variance.
  2. Seed-1 adapter set at the base config, the second solver for the ensemble.
  3. Adapter ensemble, seed 0 plus seed 1, voting over the union of both solvers' D4 candidates. This is
     the genuinely untested diversity lever, and it is a different mechanism from the more-candidates
     lever that failed (that was temperature noise from one adapter, this is independently trained
     solvers).
"""
import os
import subprocess
import sys
import time

PY = sys.executable
SWEEP = r"D:\arc\rig_ttt_sweep.py"
ENS = r"D:\arc\rig_ttt_ensemble.py"
EXP = r"D:\arc\experiments"
WAIT_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def run(label, args):
    print("=== RUN %s ===" % label, flush=True)
    t0 = time.time()
    subprocess.run([PY, "-u"] + args)
    print("=== END %s in %.2fh ===" % (label, (time.time() - t0) / 3600.0), flush=True)


def main():
    if WAIT_PID:
        print("waiting for sweep pid %d to finish" % WAIT_PID, flush=True)
        while alive(WAIT_PID):
            time.sleep(120)
        print("sweep finished, starting queue 2", flush=True)

    t_all = time.time()
    # 1. training-context A/B, same config as base but 1600 context
    run("ctx1600", [SWEEP, "0", "20", "16", "0.1", "1.25", "ctx1600", "1600", "0"])
    # 2. second solver for the ensemble, base config, seed 1
    run("base_s1", [SWEEP, "0", "20", "16", "0.1", "1.25", "base_s1", "2000", "1"])
    # 3. the ensemble, seed 0 + seed 1
    run("ens2", [ENS, "1.25", "s0s1",
                 os.path.join(EXP, "sweep_manifest_base.json"),
                 os.path.join(EXP, "sweep_manifest_base_s1.json")])
    print("QUEUE2 DONE in %.2fh" % ((time.time() - t_all) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
