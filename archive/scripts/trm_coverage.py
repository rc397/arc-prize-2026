"""Disjoint-coverage of the TRM ARC-2 checkpoint vs our 4B pipeline on the 120 public eval.

Run after trm_eval_only.py produces eval_out_v2/evaluator_ARC_step_*/submission.json. Computes which
tasks TRM solves (pass@2, exact match on either attempt), which our 4B solves (no-TTT and TTT dumps),
and the disjoint set |TRM-solved minus 4B-solved|, the only thing a second transductive arm can add.
"""
import json
import os
import glob

TRM = r"D:\arc\trm"
EXP = r"D:\arc\experiments"
OUR_SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"   # our 120 ARC-2 eval solutions
TRM_SOL = os.path.join(TRM, "kaggle", "combined", "arc-agi_evaluation2_solutions.json")


def norm_sol(raw):
    out = {}
    for tid, v in raw.items():
        grids = []
        for x in v:
            grids.append(x["output"] if isinstance(x, dict) and "output" in x else x)
        out[tid] = grids
    return out


def solved(preds_path, solutions):
    d = json.load(open(preds_path))
    s = set()
    for tid, preds in d.items():
        if tid not in solutions:
            continue
        ok = True
        for i, g in enumerate(solutions[tid]):
            a = preds[i] if i < len(preds) else {}
            if a.get("attempt_1") != g and a.get("attempt_2") != g:
                ok = False
                break
        if ok:
            s.add(tid)
    return s


def main():
    subs = glob.glob(os.path.join(TRM, "eval_out_v2", "evaluator_ARC_step_*", "submission.json"))
    if not subs:
        print("NO TRM submission.json found yet")
        return
    sub = sorted(subs)[-1]
    print("TRM submission:", sub)

    trm_sol = norm_sol(json.load(open(TRM_SOL)))
    our_sol = norm_sol(json.load(open(OUR_SOL)))
    print("task ids  TRM-sol", len(trm_sol), " our-sol", len(our_sol),
          " overlap", len(set(trm_sol) & set(our_sol)))

    trm_solved = solved(sub, trm_sol)
    nottt = solved(os.path.join(EXP, "airv_nottt_eval.json"), our_sol)
    ttt_dumps = [p for p in [
        os.path.join(EXP, "airv_ttt_val_s125_eval.json"),
        os.path.join(EXP, "airv_ttt_val_s125_cp_eval.json"),
        os.path.join(EXP, "airv_colorperm_eval.json"),
    ] if os.path.exists(p)]
    ttt = set()
    for p in ttt_dumps:
        ttt |= solved(p, our_sol)
    ours = nottt | ttt

    print("TRM solved (pass@2):", len(trm_solved), sorted(trm_solved))
    print("4B no-TTT solved:", len(nottt))
    print("4B TTT/colour solved (union of", len(ttt_dumps), "dumps):", len(ttt))
    print("4B union solved:", len(ours))
    print("TRM-only over 4B no-TTT:", sorted(trm_solved - nottt))
    print("TRM-only over 4B union :", sorted(trm_solved - ours))
    print("DISJOINT COVERAGE  |TRM minus 4B-no-TTT| =", len(trm_solved - nottt),
          "  |TRM minus 4B-union| =", len(trm_solved - ours))


if __name__ == "__main__":
    main()
