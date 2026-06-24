"""Score no-TTT AIRV and the refine run on exactly the tasks the refine run covered, for a fair compare."""
import sys
import json

sys.path.insert(0, r"D:\arc")
from src.arc.data import load_consolidated
from src.arc.eval import score_submission

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
REFINE = r"D:\arc\experiments\airv_refine_eval.json"
NOTTT = r"D:\arc\experiments\airv_nottt_eval.json"

tids = set(json.load(open(REFINE)).keys())
_, sols = load_consolidated(CH, SOL, n=None)
soln = {t: sols[t] for t in tids}


def sc(path, name):
    sub = json.load(open(path))
    sub = {t: sub[t] for t in tids if t in sub}
    acc, c, n = score_submission(sub, soln)
    print("%-12s %2d/%2d = %.4f  (%d tasks)" % (name, c, n, acc, len(sub)), flush=True)


print("comparing on the %d tasks the refine run covered:" % len(tids), flush=True)
sc(NOTTT, "no-TTT AIRV")
sc(REFINE, "refine")
