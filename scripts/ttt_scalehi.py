"""Follow-up: the scale sweep was still rising at 1.25, so probe higher to find the peak.
Reuses the 84 val adapters (generation only)."""
import subprocess
S = r"D:\arc\scripts"
for s in ["1.4", "1.5", "1.75"]:
    print("=== scale=%s ===" % s, flush=True)
    subprocess.run(["python", S + r"\ttt_phase2.py", "30", s, "val"])
subprocess.run(["python", S + r"\score_all.py"])
print("SCALEHI DONE", flush=True)
