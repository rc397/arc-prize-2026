"""Sweep orchestrator (python, so it survives detachment unlike a detached PowerShell). Runs each
config's three phases as separate subprocesses, so torch (phase 1) fully releases VRAM before
llama.cpp (phase 2) starts. base40 resumes from already-saved adapters."""
import subprocess

S = r"D:\arc\scripts"
N = "15"
configs = [
    ("40", "32", "0.0", "1e-4", "base40"),
    ("20", "16", "0.1", "1e-4", "s20r16d"),
    ("8", "8", "0.05", "1e-4", "s8r8d"),
]
for steps, rank, drop, lr, tag in configs:
    print("=== CONFIG %s (steps=%s rank=%s drop=%s) ===" % (tag, steps, rank, drop), flush=True)
    subprocess.run(["python", S + r"\ttt_phase1.py", N, steps, rank, drop, lr, tag])
    subprocess.run(["python", S + r"\ttt_phase1b.py", tag])
    subprocess.run(["python", S + r"\ttt_phase2.py", "30", "1.0", tag])
print("SWEEP DONE", flush=True)
