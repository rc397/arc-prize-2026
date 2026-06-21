"""Combine run: TTT adapters (scale 1.25) generated over 8 D4 + 8 color-permuted augmentations,
then a 3-way comparison on the 84 val tasks. The adapter is color-robust (TTT trains with random
recolor), so color-permuted candidates should transfer."""
import subprocess, json, sys
sys.path.insert(0, r"D:\arc")
from src.arc.data import load_consolidated
from src.arc.eval import score_submission

S = r"D:\arc\scripts"
subprocess.run(["python", S + r"\ttt_phase2_cp.py", "30", "1.25", "val"])

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
_, sols = load_consolidated(CH, SOL, n=None)
man = json.load(open(r"D:\arc\experiments\ttt_manifest_val.json"))
s84 = {t: sols[t] for t in man if t in sols}
rows = [("no-TTT (8 aug)", r"D:\arc\experiments\airv_nottt_eval.json"),
        ("color-perm no-TTT (16 aug)", r"D:\arc\experiments\airv_colorperm_eval.json"),
        ("TTT s1.25 (8 aug)", r"D:\arc\experiments\airv_ttt_val_s125_eval.json"),
        ("TTT s1.25 + color-perm (16 aug)", r"D:\arc\experiments\airv_ttt_val_s125_cp_eval.json")]
print("=== combine comparison on 84 val tasks ===", flush=True)
for name, p in rows:
    try:
        a, c, n = score_submission(json.load(open(p)), s84)
        print("%-34s %d/%d = %.4f" % (name, c, n, a), flush=True)
    except Exception as e:
        print("%-34s (missing: %s)" % (name, e), flush=True)
print("COMBINE DONE", flush=True)
