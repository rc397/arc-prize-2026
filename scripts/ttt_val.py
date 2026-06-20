"""Scale-validation orchestrator: regularized TTT (20 steps, rank 16, dropout 0.1) on all eval
tasks with maxpair <= 1000 (83 tasks), tag 'val'. Python so it survives detachment; phases run as
separate subprocesses so torch frees VRAM before llama.cpp starts."""
import subprocess

S = r"D:\arc\scripts"
subprocess.run(["python", S + r"\ttt_phase1.py", "0", "20", "16", "0.1", "1e-4", "val", "1000"])
subprocess.run(["python", S + r"\ttt_phase1b.py", "val"])
subprocess.run(["python", S + r"\ttt_phase2.py", "30", "1.0", "val"])
print("VAL RUN DONE", flush=True)
