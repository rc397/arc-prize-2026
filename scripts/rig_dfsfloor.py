"""Raise the DFS probability floor and see whether the returned budget buys candidates.

Two offline replays disagree about how far this goes, and only a run can settle it.

The logged-tree replay (sim_dfs_budget.py) says a floor of 0.45 keeps the truth for all 13 covered
outputs while costing 78 percent of the forward passes, and that 12 of 16 logged decodes hit their
wall clock, so a fifth of the budget comes back as candidates never generated.

The pool replay (sim_prune_replay.py) has more outputs and disagrees above 0.25: at floor 0.30 the
truth survives for 16 of 19 outputs rather than 19, and kgmon top-2 falls from 15 to 14.

Both are static. Neither can see the thing that decides it: the freed passes are spent, and what
they buy is not in any log. This runs it.

  ONE adapter per task, three decodes. Training is identical across arms by construction rather
  than by matching, so every difference is the decode. This also makes the experiment cheap: the
  expensive half runs once.

  The decode budget is held FIXED in wall-clock seconds across arms, which is the whole point. A
  tighter floor that finishes early is not the result; a tighter floor that gets further through the
  batch list in the same seconds is.

Recorded per arm: truth in pool, the selector proxy, pool size, batches completed, and whether the
decode still truncated. If truncation disappears at a tighter floor, the freed budget was real.

Usage: python rig_dfsfloor.py [N_TASKS=8] [DECODE_S=600]
Writes experiments/dfsfloor.jsonl, one record per (task, floor).
"""
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT = os.environ.get("ARC_DFSFLOOR_OUT",
                     os.path.join(BASE, "experiments", "dfsfloor.jsonl"))
FLOORS = [float(x) for x in os.environ.get("ARC_FLOORS", "0.20,0.35,0.45").split(",")]
# Decode is 69 percent of per-task time and hit its cap in every banked row, so it is
# guillotined rather than converged. Sweeping the budget prices the guillotine. Empty
# means use the single budget passed on the command line, which is the old behaviour.
BUDGETS = [int(x) for x in os.environ.get("ARC_DECODE_BUDGETS", "").split(",") if x.strip()]


def main(n_tasks, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["floor"],
                 json.loads(l).get("budget_s", 0))
                for l in open(OUT, encoding="utf-8")}
    print("DFS floor sweep %s on %d tasks, decode %ds each, one adapter per task"
          % (FLOORS, len(keys), decode_s), flush=True)

    budgets = BUDGETS or [decode_s]
    for task in keys:
        if all((task, f, b) in done for f in FLOORS for b in budgets):
            print("skip %s" % task, flush=True)
            continue
        try:
            pz = arc.change_keys([task])
            labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
            rows = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
                formatter=fmt, name="text", max_len=P.MAXSEQ)
            m = P.build_model()
            t0 = time.time()
            tr = P.train_one(m, tok, collator, fmt, rows, deadline=t0 + P.TRAIN_BUDGET)
            train_s = time.time() - t0
            print("%s trained %.0fs loss %.5f" % (task, train_s, tr["train_loss_final"]), flush=True)

            for f, budget in [(f, b) for f in FLOORS for b in budgets]:
                if (task, f, budget) in done:
                    continue
                old = P.DFS_FLOOR
                P.DFS_FLOOR = f
                try:
                    t1 = time.time()
                    pool, trunc, over = P.decode_pool(m, tok, fmt, lifted, pz,
                                                      time.time() + budget)
                finally:
                    P.DFS_FLOOR = old
                graded = P.grade(pool, labels, sorted(labels))
                rec = {"task": task, "floor": f, "stratum": strat[task],
                       "in_pool": sum(1 for g in graded.values() if g["in_pool"]),
                       "selected": sum(1 for g in graded.values() if g["selected_proxy"]),
                       "outputs": len(graded),
                       "n_cands": sum(len(v) for v in pool.values()),
                       "truncated": bool(trunc), "overflow": bool(over),
                       "decode_s": round(time.time() - t1, 1),
                       "budget_s": budget,
                       "final_loss": round(tr["train_loss_final"], 5),
                       "train_s": round(train_s, 1)}
                with open(OUT, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print("  floor %.2f budget %4ds  in_pool %d/%d  sel %d  cands %d  %s  %.0fs"
                      % (f, budget, rec["in_pool"], rec["outputs"], rec["selected"], rec["n_cands"],
                         "TRUNCATED" if rec["truncated"] else "finished", rec["decode_s"]),
                      flush=True)
            del m
        except Exception as e:
            print("FAIL %s %s: %s" % (task, type(e).__name__, e), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    summarize()
    return True


def summarize(path=OUT):
    if not os.path.isfile(path):
        print("nothing at %s" % path)
        return
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    by = {}
    for r in recs:
        by.setdefault(r["task"], {})[r["floor"]] = r
    full = {t: d for t, d in by.items() if all(f in d for f in FLOORS)}
    print("")
    print("%d tasks with every floor (of %d started)" % (len(full), len(by)))
    if not full:
        return
    print("%-8s %8s %6s %8s %10s %9s" % ("floor", "in_pool", "sel", "cands", "truncated", "decode_s"))
    print("-" * 54)
    base = None
    for f in FLOORS:
        ip = sum(full[t][f]["in_pool"] for t in full)
        se = sum(full[t][f]["selected"] for t in full)
        nc = sum(full[t][f]["n_cands"] for t in full)
        tc = sum(1 for t in full if full[t][f]["truncated"])
        ds = sum(full[t][f]["decode_s"] for t in full) / len(full)
        if base is None:
            base = (ip, se)
        print("%-8.2f %8d %6d %8d %6d/%-3d %9.0f" % (f, ip, se, nc, tc, len(full), ds))
    out = sum(full[t][FLOORS[0]]["outputs"] for t in full)
    print("")
    print("out of %d outputs across %d tasks." % (out, len(full)))
    for f in FLOORS[1:]:
        up = sum(1 for t in full if full[t][f]["in_pool"] > full[t][FLOORS[0]]["in_pool"])
        dn = sum(1 for t in full if full[t][f]["in_pool"] < full[t][FLOORS[0]]["in_pool"])
        print("  floor %.2f vs %.2f: better on %d tasks, worse on %d, tied on %d"
              % (f, FLOORS[0], up, dn, len(full) - up - dn))
    print("")
    print("The pool replay predicted truth-in-pool falls above floor 0.25 while the tree replay")
    print("predicted it holds to 0.45. Whichever way this lands, one of them was measuring the")
    print("wrong sample, and the reason to run it was that neither could see the spent budget.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 600) else 1)
