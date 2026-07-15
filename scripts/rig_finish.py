"""Resume everything Windows Update killed, in one process.

The 2026-07-15 23:32 forced reboot (TrustedInstaller, planned OS upgrade) killed the sweep mid-config
and the pid-chained queue2 with it. State survived, base and deep60 have complete manifests and recorded
results, deep60r32 had 74 of 84 adapters. Everything below resumes from the manifests, so re-running a
finished item is cheap (it loads the scores and reprints the RESULT).

Single process rather than a pid-chained pair, since the chaining is the fragile part under a reboot.

Order: finish deep60r32 (closes the depth question), then the training-context A/B (settles whether the
17.6-vs-19.2 gap is a real knob or harness variance, and therefore whether the paper needs a
correction), then the second solver and the adapter ensemble (the one genuinely novel lever left).
"""
import os
import subprocess
import sys
import time

PY = sys.executable
SWEEP = r"D:\arc\rig_ttt_sweep.py"
ENS = r"D:\arc\rig_ttt_ensemble.py"
EXP = r"D:\arc\experiments"

JOBS = [
    ("deep60r32 s1.25", [SWEEP, "0", "60", "32", "0.1", "1.25", "deep60r32", "2000", "0"]),
    ("deep60r32 s1.0",  [SWEEP, "0", "60", "32", "0.1", "1.0",  "deep60r32", "2000", "0"]),
    ("ctx1600",         [SWEEP, "0", "20", "16", "0.1", "1.25", "ctx1600",   "1600", "0"]),
    ("base_s1",         [SWEEP, "0", "20", "16", "0.1", "1.25", "base_s1",   "2000", "1"]),
    ("ens s0+s1",       [ENS, "1.25", "s0s1",
                         os.path.join(EXP, "sweep_manifest_base.json"),
                         os.path.join(EXP, "sweep_manifest_base_s1.json")]),
]


def main():
    t_all = time.time()
    for label, args in JOBS:
        print("=== RUN %s ===" % label, flush=True)
        t0 = time.time()
        try:
            subprocess.run([PY, "-u"] + args)
        except Exception as e:
            print("=== %s CRASHED %r ===" % (label, str(e)[:100]), flush=True)
        print("=== END %s in %.2fh (total %.2fh) ===" %
              (label, (time.time() - t0) / 3600.0, (time.time() - t_all) / 3600.0), flush=True)
    print("FINISH QUEUE DONE in %.2fh" % ((time.time() - t_all) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
