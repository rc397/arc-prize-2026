import json, sys, os
sys.path.insert(0, r"D:\arc")
from src.arc.data import load_consolidated
from src.arc.eval import score_submission

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
tasks, sols = load_consolidated(CH, SOL, n=None)
nottt = json.load(open(r"D:\arc\experiments\airv_nottt_eval.json"))

for tag in ["base40", "s20r16d", "s8r8d"]:
    man_path = r"D:\arc\experiments\ttt_manifest_%s.json" % tag
    sub_path = r"D:\arc\experiments\airv_ttt_%s_s100_eval.json" % tag
    if not (os.path.exists(man_path) and os.path.exists(sub_path)):
        print("%-10s (not finished)" % tag, flush=True)
        continue
    man = json.load(open(man_path))
    sub = json.load(open(sub_path))
    tids = list(man.keys())
    s = {t: sols[t] for t in tids if t in sols}
    at, ct, nt = score_submission(sub, s)
    an, cn, nn = score_submission(nottt, s)
    print("%-10s  TTT %2d/%d = %.3f   no-TTT %2d/%d = %.3f   (%d tasks)" % (
        tag, ct, nt, at, cn, nn, an, len(tids)), flush=True)
