import json, sys
sys.path.insert(0, r"D:\arc")
from src.arc.data import load_consolidated
from src.arc.eval import score_submission

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
TAG = sys.argv[1] if len(sys.argv) > 1 else "val"
tasks, sols = load_consolidated(CH, SOL, n=None)
ttt = json.load(open(r"D:\arc\experiments\airv_ttt_%s_s100_eval.json" % TAG))
nottt = json.load(open(r"D:\arc\experiments\airv_nottt_eval.json"))
manifest = json.load(open(r"D:\arc\experiments\ttt_manifest_%s.json" % TAG))

tids = list(manifest.keys())   # only the tasks that actually received TTT
sols5 = {t: sols[t] for t in tids if t in sols}
acc_t, c_t, n_t = score_submission(ttt, sols5)
acc_n, c_n, n_n = score_submission(nottt, sols5)
print("tasks scored:", tids)
print("TTT     : %d/%d = %.4f" % (c_t, n_t, acc_t))
print("no-TTT  : %d/%d = %.4f" % (c_n, n_n, acc_n))
# sanity: are TTT attempts real grids (not the [[0]] default)?
t0 = tids[0]
a1 = ttt[t0][0]["attempt_1"]
print("sample TTT attempt_1 for %s: dims=%s nonzero=%s" % (
    t0, (len(a1), len(a1[0]) if a1 else 0), any(any(v for v in r) for r in a1)))
