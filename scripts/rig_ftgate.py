"""Did the compositional fine-tune teach the base anything, and did it break anything?

The composition probe cleared its gate: composed tasks are solvable on 5 of 14 against 14 of 14 for
single rules, which is the pre-registered "harder than one rule and still learnable" outcome, so the
78,792-example corpus is worth training on. This is the gate on the resulting checkpoint, and it has
to answer two questions in that order, because the second one can veto the first.

  composed   held-out INSTANCES of the composed rules the corpus was built from. If this does not
             move, the fine-tune did not learn what it was trained on and nothing else matters. Note
             what it is not: the rules are seen, only the instances are fresh, so this measures
             fitting, not generalisation to unseen compositions.
  arc        ARC-AGI-2 evaluation tasks. This is the one that decides. A fine-tune that learns
             composition and damages the grid prior it was given is a loss, and continued training
             on a narrow distribution is exactly how that happens.

Deliberately NO test-time training. TTT would let a per-task adapter paper over a damaged base, and
the question here is what the base itself knows. A fine-tune that survives this gate still needs a
full TTT A/B before it goes anywhere near a submission.

Both arms run through the SAME pipeline via ARC_MODEL_DIR, rather than through a second harness, so
the comparison is like with like. The tokenizer always comes from the original base directory: the
vocabulary is identical and a merged checkpoint does not necessarily carry one.

A fresh LoRA is attached to both arms and never trained, which is an exact identity because lora_B
initialises to zero. It is there only so the arms take the same code path.

Usage: python rig_ftgate.py <FT_CHECKPOINT_DIR> [N_ARC=12] [N_COMP=8] [DECODE_S=420]
Writes experiments/ftgate.jsonl, one record per (arm, family, task).
"""
import gc
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Overridable, because the resume logic keys on (arm, task) and would silently skip every task
# already recorded against a DIFFERENT checkpoint, reporting a finished gate that never ran.
OUT = os.environ.get("ARC_FTGATE_OUT", os.path.join(BASE, "experiments", "ftgate.jsonl"))
TMP = os.path.join(BASE, "_ftgate")


def main(ft_dir, n_arc, n_comp, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    import rig_composelearn as C
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    assert os.path.isdir(ft_dir), "no checkpoint at %s" % ft_dir
    stock = P.MODEL_DIR                       # captured before anything overrides it
    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(stock)
    fmt = QwenFormatter(tokenizer=tok)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    arc_keys, _, strat = P.select_tasks(n_arc)

    # held-out instances of trained compositions, built exactly as the learnability probe built them
    vs = C.load_verifiers()
    survey = json.load(open(C.SURVEY, encoding="utf-8"))
    kept = survey.get("kept") or survey.get("examples") or []
    assert kept, "no kept compositions in %s" % C.SURVEY
    rng = random.Random(23)                   # a different seed from the probe's 17, so fresh draws
    by_a = {}
    for k in kept:
        by_a.setdefault(k["a"], []).append(k)
    firsts = sorted(by_a)
    rng.shuffle(firsts)
    os.makedirs(TMP, exist_ok=True)
    comp_tasks = []
    for a in firsts:
        if len(comp_tasks) >= n_comp:
            break
        pick = rng.choice(by_a[a])
        pairs = C.build_composed(vs, pick["a"], pick["b"], C.N_DEMO + 1)
        if pairs:
            comp_tasks.append(("comp_%s_%s" % (pick["a"], pick["b"]), pairs))
    print("gating %s: %d arc tasks, %d composed tasks, decode %ds, no TTT"
          % (os.path.basename(ft_dir), len(arc_keys), len(comp_tasks), decode_s), flush=True)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["arm"], json.loads(l)["task"])
                for l in open(OUT, encoding="utf-8")}

    def composed_ds(name, pairs):
        ch = {name: {"train": pairs[:C.N_DEMO], "test": [{"input": pairs[C.N_DEMO]["input"]}]}}
        path = os.path.join(TMP, "%s.json" % name)
        json.dump(ch, open(path, "w"))
        ds = ArcDataset.from_file(path)
        return ds.change_keys([name]), {"%s_0" % name: np.array(pairs[C.N_DEMO]["output"])}

    for arm, mdir in (("base", stock), ("ft", ft_dir)):
        if all((arm, k) in done for k in
               list(arc_keys) + [n for n, _ in comp_tasks]):
            print("skip arm %s, already complete" % arm, flush=True)
            continue
        P.MODEL_DIR = mdir
        m = P.build_model()
        print("%s loaded from %s" % (arm, mdir), flush=True)
        try:
            work = [("arc", k, None) for k in arc_keys] + \
                   [("comp", n, p) for n, p in comp_tasks]
            for family, name, pairs in work:
                if (arm, name) in done:
                    continue
                try:
                    if family == "arc":
                        pz = arc.change_keys([name])
                        labels = {"%s_%d" % (name, i): np.array(g)
                                  for i, g in enumerate(sols[name])}
                    else:
                        pz, labels = composed_ds(name, pairs)
                    t0 = time.time()
                    pool, trunc, over = P.decode_pool(m, tok, fmt, lifted, pz,
                                                      time.time() + decode_s)
                    graded = P.grade(pool, labels, sorted(labels))
                    rec = {"arm": arm, "family": family, "task": name,
                           "stratum": strat.get(name, "-"),
                           "in_pool": sum(1 for g in graded.values() if g["in_pool"]),
                           "selected": sum(1 for g in graded.values() if g["selected_proxy"]),
                           "outputs": len(graded),
                           "n_cands": sum(len(v) for v in pool.values()),
                           "truncated": bool(trunc or over),
                           "decode_s": round(time.time() - t0, 1)}
                    with open(OUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    print("  %-4s %-5s %-22s in_pool %d/%d  cands %3d  %.0fs"
                          % (arm, family, name, rec["in_pool"], rec["outputs"],
                             rec["n_cands"], rec["decode_s"]), flush=True)
                except Exception as e:
                    print("  FAIL %s %s %s: %s" % (arm, name, type(e).__name__, e), flush=True)
        finally:
            P.MODEL_DIR = stock
            del m
            gc.collect()
            torch.cuda.empty_cache()
    summarize()
    return True


def summarize(path=OUT):
    if not os.path.isfile(path):
        print("nothing at %s" % path)
        return
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    print("")
    print("%-6s %-6s %8s %8s %8s %8s" % ("arm", "family", "tasks", "in_pool", "of", "cands"))
    print("-" * 50)
    cells = {}
    for arm in ("base", "ft"):
        for fam in ("comp", "arc"):
            r = [x for x in recs if x["arm"] == arm and x["family"] == fam]
            if not r:
                continue
            cells[(arm, fam)] = (sum(x["in_pool"] for x in r), sum(x["outputs"] for x in r))
            print("%-6s %-6s %8d %8d %8d %8d"
                  % (arm, fam, len(r), sum(x["in_pool"] for x in r),
                     sum(x["outputs"] for x in r), sum(x["n_cands"] for x in r)))
    # Paired counts, not totals. An earlier version of this verdict read the totals and announced
    # both "the fine-tune learned the rules it was trained on" and "it cost grid ability", when each
    # of those rested on ONE task moving by one output. Totals invite that error; sign counts do not.
    paired = {}
    for fam in ("comp", "arc"):
        b = {r["task"]: r for r in recs if r["arm"] == "base" and r["family"] == fam}
        f = {r["task"]: r for r in recs if r["arm"] == "ft" and r["family"] == fam}
        shared = sorted(set(b) & set(f))
        up = sum(1 for t in shared if f[t]["in_pool"] > b[t]["in_pool"])
        dn = sum(1 for t in shared if f[t]["in_pool"] < b[t]["in_pool"])
        bc = sum(b[t]["n_cands"] for t in shared)
        fc = sum(f[t]["n_cands"] for t in shared)
        paired[fam] = (up, dn, len(shared), bc, fc)

    print("")
    print("VERDICT")
    for fam, label in (("comp", "composed"), ("arc", "ARC-AGI-2")):
        if fam not in paired:
            continue
        up, dn, n, bc, fc = paired[fam]
        print("  %-10s better on %d tasks, worse on %d, tied on %d of %d; candidates %d to %d (%+.0f%%)"
              % (label, up, dn, n - up - dn, n, bc, fc, 100.0 * (fc - bc) / max(1, bc)))

    cu, cd = paired.get("comp", (0, 0, 0, 0, 0))[:2]
    au, ad = paired.get("arc", (0, 0, 0, 0, 0))[:2]
    if max(cu, cd, au, ad) <= 1:
        print("")
        print("  Every arm is tied on all but at most one task, so this measures NOTHING in either")
        print("  direction. It did not learn the composed rules and it did not damage the base; a")
        print("  one-task move is not a result at these sample sizes. Read it together with the")
        print("  training curve: if the val loss also barely moved, the conclusion is that")
        print("  the intervention was too small to test the hypothesis, not that the hypothesis is")
        print("  false. Scaling it up is then still an open question and still expensive.")
    elif ad > au:
        print("")
        print("  The fine-tune costs coverage on the real target on more tasks than it gains. That")
        print("  is the veto, and it is the expected failure mode of continued training on a narrow")
        print("  distribution.")
    elif cu > cd and au >= ad:
        print("")
        print("  It learned the composed rules without costing the real target. Confirm with a TTT")
        print("  A/B before believing it, since no-TTT and deployed coverage have disagreed before.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1
                       else os.path.join(BASE, "models", "comp_pilot", "qwen3-4b-arc2ft"),
                       int(sys.argv[2]) if len(sys.argv) > 2 else 12,
                       int(sys.argv[3]) if len(sys.argv) > 3 else 8,
                       int(sys.argv[4]) if len(sys.argv) > 4 else 420) else 1)
