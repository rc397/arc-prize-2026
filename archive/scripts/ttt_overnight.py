"""Overnight queue: adapter-scale sweep -> color-perm AIRV -> big-task TTT -> figure -> summary.
Python so it survives detachment. Each experiment is isolated in try/except; one failing does not
block the rest. Experiments are ordered safest-and-highest-value first."""
import subprocess, traceback

S = r"D:\arc\scripts"
PY = "python"


def run(args):
    print(">>> RUN", " ".join(args), flush=True)
    try:
        subprocess.run([PY] + args)
    except Exception:
        traceback.print_exc()


def kill_servers():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)


# A. adapter-scale sweep (reuse the 84 trained val adapters; generation only)
for s in ["0.5", "0.7", "0.85", "1.25"]:
    print("=== A scale=%s ===" % s, flush=True)
    run([S + r"\ttt_phase2.py", "30", s, "val"])

# B. color-perm AIRV (no-TTT, 16 augs) on the 84 val tasks
kill_servers()
print("=== B colorperm ===", flush=True)
run([S + r"\airv_colorperm.py", "0"])

# C. big-task TTT (the ~36 tasks with maxpair>1000, tight context cap)
kill_servers()
print("=== C big-task TTT ===", flush=True)
run([S + r"\ttt_phase1.py", "0", "20", "16", "0.1", "1e-4", "bigttt", "1000000000", "1000", "600"])
run([S + r"\ttt_phase1b.py", "bigttt"])
run([S + r"\ttt_phase2.py", "30", "1.0", "bigttt"])

# D. figure
kill_servers()
print("=== D figure ===", flush=True)
run([S + r"\make_figure.py"])

# E. score summary
print("=== E score_all ===", flush=True)
run([S + r"\score_all.py"])
print("OVERNIGHT QUEUE DONE", flush=True)
