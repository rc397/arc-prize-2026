"""Validate the rank lever for a greedy-AIRV submission (v7-B).

Runs the real llama.cpp AIRV sweep at rank 64 and rank 128 on all 84 trainable tasks, at the deep
config that produced 18.19 (60 steps, dropout 0.1, scale 1.25, 2000-token context), varying only rank.
Directly comparable to the recorded rank-32 deep-TTT baseline of 27/125. One process runs both
sequentially so a reboot cannot orphan a chained waiter, and rig_ttt_sweep's per-config manifests make
each rank resume from disk.
"""
import os
import subprocess
import sys

BASE = r"D:\arc"
SWEEP = os.path.join(BASE, "scripts", "rig_ttt_sweep.py")

for rank in [64, 128]:
    print("=== rankval rank %d ===" % rank, flush=True)
    subprocess.run([sys.executable, SWEEP, "0", "60", str(rank), "0.1", "1.25",
                    "rankval_r%d" % rank, "2000", "0"], cwd=BASE)
print("=== rankval done ===", flush=True)
