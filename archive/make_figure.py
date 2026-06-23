"""Results figure for the paper cover: TTT adapter-scale curve + headline bars. Reads whatever
airv_ttt_val_s*.json exist, scores them on the 84 val tasks. -> paper/cover_draft.png."""
import sys, json, glob, os
sys.path.insert(0, r"D:\arc")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.arc.data import load_consolidated
from src.arc.eval import score_submission

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
_, sols = load_consolidated(CH, SOL, n=None)
man = json.load(open(r"D:\arc\experiments\ttt_manifest_val.json"))
s84 = {t: sols[t] for t in man if t in sols}
nottt = json.load(open(r"D:\arc\experiments\airv_nottt_eval.json"))
an = score_submission(nottt, s84)[0]

pts = []
for f in sorted(glob.glob(r"D:\arc\experiments\airv_ttt_val_s*_eval.json")):
    try:
        sc = int(os.path.basename(f).split("_s")[1].split("_")[0]) / 100.0
        a = score_submission(json.load(open(f)), s84)[0]
        pts.append((sc, a))
    except Exception as e:
        print("skip", f, e, flush=True)
pts.sort()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
if pts:
    ax[0].plot([p[0] for p in pts], [p[1] * 100 for p in pts], "o-", color="#2b6cb0")
    ax[0].axhline(an * 100, ls="--", color="#888", label="no-TTT %.1f%%" % (an * 100))
    ax[0].set_xlabel("adapter scale")
    ax[0].set_ylabel("accuracy on 84 trainable tasks (%)")
    ax[0].set_title("Per-task TTT: adapter-scale sweep")
    ax[0].legend()
ax[1].bar(["no-TTT\n(full 120)", "TTT blended\n(full 120)"], [12.79, 15.70],
          color=["#888", "#2b6cb0"])
ax[1].set_ylabel("accuracy (%)")
ax[1].set_title("ARC-AGI-2 on one 16 GB GPU")
for i, v in enumerate([12.79, 15.70]):
    ax[1].text(i, v + 0.2, "%.2f%%" % v, ha="center")
plt.tight_layout()
os.makedirs(r"D:\arc\paper", exist_ok=True)
plt.savefig(r"D:\arc\paper\cover_draft.png", dpi=140)
print("FIGURE saved paper/cover_draft.png; scale points:", [(s, round(a, 4)) for s, a in pts],
      flush=True)
