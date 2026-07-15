"""Run the deep-regularized TTT sweep as one detached sequence and report each config's score.

Tests the corner the original sweep never measured, more steps WITH dropout, against the current
20/16/0.1 operating point. Adapters are trained once per (steps, rank, dropout) tag and then scored at
several inference scales, since scale is applied at serve time and needs no retraining. The deeper
adapters are better fit, so they may want less amplification than the 1.25 tuned for the underfit
20-step one, which is why each config is scored at both.

Baseline to beat: 17.6 percent on the 84 trainable tasks (20 steps, rank 16, dropout 0.1, scale 1.25).
The first config re-runs that config as a harness check, it should land near 17.6.
"""
import subprocess
import sys
import time

PY = sys.executable
SCRIPT = r"D:\arc\rig_ttt_sweep.py"

# (tag, steps, rank, dropout, [scales])  N_TASKS=0 means all 84 trainable
CONFIGS = [
    ("base",      "20",  "16", "0.1", ["1.25"]),           # harness check vs the recorded 17.6
    ("deep60",    "60",  "16", "0.1", ["1.25", "1.0"]),    # deep + the regularizer kept on
    ("deep60r32", "60",  "32", "0.1", ["1.25", "1.0"]),    # deep + more capacity, still regularized
]


def main():
    t_all = time.time()
    for tag, steps, rank, dr, scales in CONFIGS:
        for scale in scales:
            print("=== RUN tag=%s steps=%s rank=%s dropout=%s scale=%s ===" % (tag, steps, rank, dr, scale),
                  flush=True)
            t0 = time.time()
            subprocess.run([PY, "-u", SCRIPT, "0", steps, rank, dr, scale, tag])
            print("=== END tag=%s scale=%s in %.2fh (total %.2fh) ===" %
                  (tag, scale, (time.time() - t0) / 3600.0, (time.time() - t_all) / 3600.0), flush=True)
    print("ALL CONFIGS DONE in %.2fh" % ((time.time() - t_all) / 3600.0), flush=True)


if __name__ == "__main__":
    main()
