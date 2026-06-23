"""Dump scale_flip.csv for fig_scale_flip (A3): per test output (84 trainable tasks, 125 outputs),
whether it is solved (exact match on either attempt) at no-TTT and each of the 8 adapter scales.
Column sums reproduce the scale curve (per-output counts)."""
import sys, json, csv
sys.path.insert(0, r"D:\arc")
import numpy as np
from src.arc.data import load_consolidated

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
_, sols = load_consolidated(CH, SOL, n=None)
man = json.load(open(r"D:\arc\experiments\ttt_manifest_val.json"))
E = r"D:\arc\experiments"
scales = [("no_ttt", E + r"\airv_nottt_eval.json"), ("s050", E + r"\airv_ttt_val_s50_eval.json"),
          ("s070", E + r"\airv_ttt_val_s70_eval.json"), ("s085", E + r"\airv_ttt_val_s85_eval.json"),
          ("s100", E + r"\airv_ttt_val_s100_eval.json"), ("s125", E + r"\airv_ttt_val_s125_eval.json"),
          ("s140", E + r"\airv_ttt_val_s140_eval.json"), ("s150", E + r"\airv_ttt_val_s150_eval.json"),
          ("s175", E + r"\airv_ttt_val_s175_eval.json")]
subs = {n: json.load(open(p)) for n, p in scales}


def ok(sub, tid, i):
    if tid not in sub or i >= len(sub[tid]):
        return 0
    a = sub[tid][i]
    return int(a["attempt_1"] == sols[tid][i] or a["attempt_2"] == sols[tid][i])


rows = []
for tid in man:
    if tid not in sols:
        continue
    for i in range(len(sols[tid])):
        rows.append([tid, i] + [ok(subs[n], tid, i) for n, _ in scales])
hdr = ["task_id", "out_idx"] + [n for n, _ in scales]
with open(E + r"\scale_flip.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(rows)
arr = np.array([r[2:] for r in rows])
print("scale_flip.csv:", len(rows), "outputs; column sums (= scale curve):",
      dict(zip([n for n, _ in scales], arr.sum(0).tolist())), flush=True)
