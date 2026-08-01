"""Does the public LB33 notebook's arithmetic patch actually buy search depth?

The notebook titled LB33.89 differs from our best submission in exactly two ways, and one of them is
purely numerical. The aimo lineage we build on computes the search NLL as
    logits.float().cpu().log_softmax(-1)
which materialises a full-vocabulary softmax on the CPU at every node of the tree. The patch computes
the same quantity on the GPU over only the twelve ARC tokens, using
    logsumexp(logits) - logits.index_select(ARC_TOKENS)
which is algebraically identical because log_softmax = logits - logsumexp.

Identical output, so this cannot change what the search would find given unlimited time. But the search
is time-bounded, 540 seconds per call plus a per-task cap, so a faster inner loop explores more of the
tree inside the same budget. That predicts more candidates per second and therefore higher pool
coverage, which is a mechanism for a score difference rather than a correlation with one.

This measures the prediction directly and refuses to take the notebook's displayed score as evidence.
Same adapter, same task, same wall-clock budget, both arithmetics: count what each produces. A patch
that does not produce more candidates per second cannot explain anything, whatever its title claims.

Usage: python rig_dfsbench.py [N_TASKS=4] [DECODE_BUDGET=400]
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P                                          # noqa: E402
from arc_loader import ArcDataset, QwenFormatter                       # noqa: E402
from transformers import AutoTokenizer                                 # noqa: E402

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LB33 = r"D:\arc\_pubnb2\lb33_arc_solver.py"


def main():
    assert os.path.isfile(LB33), "sync the LB33 solver to %s" % LB33
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))

    variants = {"aimo (CPU full-vocab softmax)": P.lift_from_solver(),
                "LB33 (GPU, 12 tokens)": P.lift_from_solver(src_path=LB33)}
    collator = variants["aimo (CPU full-vocab softmax)"][
        "QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)

    keys, _, strat = P.select_tasks(20)
    keys = [k for k in keys if strat[k] == "A"][:N_TASKS]
    print("tasks (reachable stratum): %s" % keys, flush=True)
    print("budget %ds per decode, adapter trained ONCE per task and reused for both arms\n" % BUDGET,
          flush=True)

    totals = {v: {"cands": 0, "secs": 0.0, "inpool": 0, "outputs": 0} for v in variants}
    for key in keys:
        labels = {"%s_%d" % (key, i): np.array(g) for i, g in enumerate(sols[key])}
        m = P.build_model()
        pz = arc.change_keys([key])
        ds = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        t0 = time.time()
        P.train_one(m, tok, collator, fmt, ds, deadline=t0 + P.TRAIN_BUDGET)
        print("%s trained in %.0fs" % (key, time.time() - t0), flush=True)
        for name, lifted in variants.items():
            torch.cuda.empty_cache()
            t1 = time.time()
            pool, trunc, _ = P.decode_pool(m, tok, fmt, lifted, pz, t1 + BUDGET)
            secs = time.time() - t1
            n = sum(len(v) for v in pool.values())
            g = P.grade(pool, labels, sorted(labels))
            hit = sum(1 for v in g.values() if v["in_pool"])
            totals[name]["cands"] += n
            totals[name]["secs"] += secs
            totals[name]["inpool"] += hit
            totals[name]["outputs"] += len(g)
            print("   %-30s %4d cands in %5.1fs  truth-in-pool %d/%d  truncated=%s"
                  % (name, n, secs, hit, len(g), trunc), flush=True)
        del m
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n%-32s %8s %9s %12s %10s" % ("variant", "cands", "seconds", "cands/sec", "in-pool"))
    print("-" * 76)
    for name, t in totals.items():
        print("%-32s %8d %9.1f %12.2f %6d/%d"
              % (name, t["cands"], t["secs"], t["cands"] / max(t["secs"], 1e-9),
                 t["inpool"], t["outputs"]))
    a, b = list(totals.values())
    ra = a["cands"] / max(a["secs"], 1e-9)
    rb = b["cands"] / max(b["secs"], 1e-9)
    print("\nspeed ratio LB33 / aimo: %.2fx candidates per second" % (rb / max(ra, 1e-9)))
    print("\nVERDICT")
    if a["cands"] == 0 and b["cands"] == 0:
        # Both arms empty means the budget bought no completed grid either way, so the ratio is 0/0
        # and carries no information. Say so rather than letting the else-branch below read a
        # vacuous zero as a null result, which is the same defect that made an earlier script
        # report a positive verdict at an AUC of 0.15.
        print("  NO RESULT. Neither arithmetic completed a single grid inside the budget, so this")
        print("  measured nothing about throughput. Candidates-per-second is gated on finishing a")
        print("  whole grid, and on these tasks neither arm does. Re-run on tasks that produce")
        print("  candidates, or instrument node expansions per second instead of completions.")
    elif rb > ra * 1.15:
        print("  The patch does buy search throughput. That is a mechanism for the score difference,")
        print("  and it is a repair rather than a tuning, which is the class that has transferred")
        print("  for us before. Worth a submission slot paired with the 1200s cap.")
    else:
        print("  No meaningful throughput gain, so the patch cannot explain a score difference by")
        print("  this route. Do not spend a slot on it: whatever the title claims, the stated")
        print("  mechanism is not present on our hardware.")
    json.dump({k: v for k, v in totals.items()},
              open(os.path.join(P.EXP, "dfsbench.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
