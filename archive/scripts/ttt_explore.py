"""Explore the two open levers: higher adapter scales (1.4/1.5/1.75 on the 84 val adapters, to find
the peak the overnight curve was still climbing toward) and color-perm on the 36 big tasks (does
more voting lift the no-TTT contribution the blend uses there). Reuses ttt_phase2 and airv_colorperm."""
import subprocess
S = r"D:\arc\scripts"
for s in ["1.4", "1.5", "1.75"]:
    print("=== scale=%s ===" % s, flush=True)
    subprocess.run(["python", S + r"\ttt_phase2.py", "30", s, "val"])
print("=== color-perm on 36 big tasks (no-TTT, 16 aug) ===", flush=True)
subprocess.run(["python", S + r"\airv_colorperm.py", "0",
                r"D:\arc\experiments\ttt_manifest_bigttt.json", "big"])
subprocess.run(["python", S + r"\score_all.py"])
print("EXPLORE DONE", flush=True)
