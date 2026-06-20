"""Score every output the overnight queue produced, against no-TTT on the matching task set.
One-glance morning summary."""
import sys, json, glob, os
sys.path.insert(0, r"D:\arc")
from src.arc.data import load_consolidated
from src.arc.eval import score_submission

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
_, sols = load_consolidated(CH, SOL, n=None)
manval = json.load(open(r"D:\arc\experiments\ttt_manifest_val.json"))
s84 = {t: sols[t] for t in manval if t in sols}
nottt = json.load(open(r"D:\arc\experiments\airv_nottt_eval.json"))
an, cn, nn = score_submission(nottt, s84)
print("=== ANCHORS on 84 val tasks: no-TTT %d/%d = %.4f ===" % (cn, nn, an))

print("--- A: TTT adapter-scale sweep (84 tasks) ---")
best = (an, "no-TTT")
for f in sorted(glob.glob(r"D:\arc\experiments\airv_ttt_val_s*_eval.json")):
    a, c, n = score_submission(json.load(open(f)), s84)
    tag = os.path.basename(f)
    print("  %-40s %d/%d = %.4f" % (tag, c, n, a))
    if a > best[0]:
        best = (a, tag)
print("  BEST: %.4f (%s)" % best)

cp = r"D:\arc\experiments\airv_colorperm_eval.json"
if os.path.exists(cp):
    a, c, n = score_submission(json.load(open(cp)), s84)
    print("--- B: color-perm no-TTT (16 aug) on 84: %d/%d = %.4f  vs no-TTT 8aug %.4f ---" % (
        c, n, a, an))

bm = r"D:\arc\experiments\ttt_manifest_bigttt.json"
bo = r"D:\arc\experiments\airv_ttt_bigttt_s100_eval.json"
if os.path.exists(bm) and os.path.exists(bo):
    bman = json.load(open(bm))
    sbig = {t: sols[t] for t in bman if t in sols}
    a, c, n = score_submission(json.load(open(bo)), sbig)
    an2, cn2, nn2 = score_submission(nottt, sbig)
    print("--- C: big-task TTT on %d tasks: %d/%d = %.4f  vs no-TTT %d/%d = %.4f ---" % (
        len(bman), c, n, a, cn2, nn2, an2))
