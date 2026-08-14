"""Does test-time training rescue the fine-tuned base, or inherit its damage?

`rig_ftgate.py` deliberately runs with NO test-time training, because the question there is what the
base itself knows and a per-task adapter would paper over a damaged one. That is the right gate and
it is not the deployed condition. Every submission this project has made trains a fresh rank-256
adapter on each task, and a base that decodes worse cold can still adapt better, or worse, than one
that decodes better cold. This project has been wrong about that direction before: local no-TTT gains
have failed to transfer to the deployed pipeline more than once.

So this is the same comparison the gate makes, under the pipeline we actually ship.

  base   stock NVARC checkpoint, per-task TTT, DFS decode
  ft     the compositional fine-tune, identical treatment

One task at a time, both arms, so a task's difficulty cancels in the pairing. The adapter is trained
from scratch in each arm, which is the deployed behaviour, and the decode budget is fixed in
wall-clock seconds so a slower base cannot buy coverage with time.

Reading it, decided in advance:
  ft better than base under TTT     the fine-tune is worth a submission, whatever the cold gate said
  ft equal to base                  not distinguishable at this n; the corpus bought nothing usable
  ft worse than base                the damage survives adaptation and the checkpoint is dead

Usage: python rig_tttab.py <FT_CHECKPOINT_DIR> [N_TASKS=10] [DECODE_S=420]
Writes experiments/tttab.jsonl, one record per (task, arm).
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
OUT = os.environ.get("ARC_TTTAB_OUT", os.path.join(BASE, "experiments", "tttab.jsonl"))


def main(ft_dir, n_tasks, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    assert os.path.isdir(ft_dir), "no checkpoint at %s" % ft_dir
    stock = P.MODEL_DIR
    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(stock)     # vocabulary is identical; the merge may carry none
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8")}
    print("TTT A/B on %d tasks, decode %ds, stock against %s"
          % (len(keys), decode_s, os.path.basename(ft_dir)), flush=True)

    for task in keys:
        for arm, mdir in (("base", stock), ("ft", ft_dir)):
            if (task, arm) in done:
                print("skip %s|%s" % (task, arm), flush=True)
                continue
            try:
                pz = arc.change_keys([task])
                labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
                rows = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
                    formatter=fmt, name="text", max_len=P.MAXSEQ)
                P.MODEL_DIR = mdir
                m = P.build_model()
                t0 = time.time()
                tr = P.train_one(m, tok, collator, fmt, rows, deadline=t0 + P.TRAIN_BUDGET)
                train_s = time.time() - t0
                pool, trunc, over = P.decode_pool(m, tok, fmt, lifted, pz,
                                                  time.time() + decode_s)
                graded = P.grade(pool, labels, sorted(labels))
                rec = {"task": task, "arm": arm, "stratum": strat[task],
                       "final_loss": round(tr["train_loss_final"], 5),
                       "train_steps": tr["train_steps"],
                       "in_pool": sum(1 for g in graded.values() if g["in_pool"]),
                       "selected": sum(1 for g in graded.values() if g["selected_proxy"]),
                       "outputs": len(graded),
                       "n_cands": sum(len(v) for v in pool.values()),
                       "truncated": bool(trunc or over), "train_s": round(train_s, 1)}
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print("%-10s %-5s loss %.5f  in_pool %d/%d  sel %d  cands %3d  train %.0fs"
                      % (task, arm, rec["final_loss"], rec["in_pool"], rec["outputs"],
                         rec["selected"], rec["n_cands"], train_s), flush=True)
                del m
            except Exception as e:
                print("FAIL %s|%s %s: %s" % (task, arm, type(e).__name__, e), flush=True)
            finally:
                P.MODEL_DIR = stock
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
        by.setdefault(r["task"], {})[r["arm"]] = r
    full = {t: d for t, d in by.items() if "base" in d and "ft" in d}
    print("")
    print("%d tasks with both arms (of %d started)" % (len(full), len(by)))
    if not full:
        return
    print("%-6s %8s %6s %8s %10s" % ("arm", "in_pool", "sel", "cands", "mean loss"))
    print("-" * 44)
    for a in ("base", "ft"):
        n = len(full)
        print("%-6s %8d %6d %8d %10.5f"
              % (a, sum(full[t][a]["in_pool"] for t in full),
                 sum(full[t][a]["selected"] for t in full),
                 sum(full[t][a]["n_cands"] for t in full),
                 sum(full[t][a]["final_loss"] for t in full) / n))
    up = sum(1 for t in full if full[t]["ft"]["in_pool"] > full[t]["base"]["in_pool"])
    dn = sum(1 for t in full if full[t]["ft"]["in_pool"] < full[t]["base"]["in_pool"])
    out = sum(full[t]["base"]["outputs"] for t in full)
    print("")
    print("out of %d outputs. ft better on %d tasks, worse on %d, tied on %d"
          % (out, up, dn, len(full) - up - dn))
    print("")
    if max(up, dn) <= 1:
        print("Nothing moves by more than one task, so this does not distinguish the two bases under")
        print("the deployed pipeline in either direction.")
    elif dn > up:
        print("The damage survives test-time training. A per-task adapter does not rebuild what")
        print("continued training on a narrow distribution removed, and the checkpoint is dead.")
    else:
        print("The fine-tune wins under the pipeline we actually ship, which is the condition that")
        print("matters. Next step is a submission, not another probe.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1
                       else os.path.join(BASE, "models", "comp_full", "qwen3-4b-arc2ft"),
                       int(sys.argv[2]) if len(sys.argv) > 2 else 10,
                       int(sys.argv[3]) if len(sys.argv) > 3 else 420) else 1)
