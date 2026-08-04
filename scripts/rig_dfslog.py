"""Log the DFS's explored tree, then replay it offline to ask if visiting order costs us answers.

Every Kaggle-track decode since 2024 is threshold-DFS, and the novelty audit found no ARC precedent
for best-first enumeration (docs/novelty-audit.md, lever a). The cheap first experiment is not to
build the enumerator but to measure whether it could matter: instrument the deployed turbo_dfs to
record every node expansion and every completed-candidate emission, run the stratified task slice
once, and replay the logged tree under best-first order at equal node budgets. If best-first reaches
the same truths in far fewer expansions, the anytime enumerator is worth building; if the orders
tie, the lever dies for one logging run's worth of GPU.

The instrumentation is three text insertions into a copy of the solver source, applied by exact
string match and asserted to land exactly once each, then lifted through the same AST machinery the
probe uses so everything else stays byte-faithful to deployment.

Records per batch, in DFS visit order:
  ("N", pos, [token per lane], [cumulative suffix NLL per lane])   one forward pass
  ("E", pos, lane, score)                                          a completed candidate at EOS

Usage:
  python rig_dfslog.py run [N_TASKS]      on the rig, writes experiments/dfslog/<task>.jsonl
  python rig_dfslog.py replay [DIR]       anywhere, pure python, prints the verdict
"""
import json
import os
import sys

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
OUTDIR = os.path.join(BASE, "experiments", "dfslog")
PAD_SCORE = 1000.0          # the sentinel turbo_dfs assigns to exhausted lanes


# ----------------------------------------------------------------------------- instrumentation

EMIT_ANCHOR = "                if t == EOS_ID:\n" \
              "                    suffixes[i].append((score, [t]))\n"
EMIT_PATCHED = EMIT_ANCHOR + \
              "                    _DFSLOG.append((\"E\", pos, i, float(score)))\n"
STEP_ANCHOR = "        outputs = model(\n"
STEP_PATCHED = "        _DFSLOG.append((\"N\", pos, [int(_bt) for _bt in batch_tokens]," \
               " [float(_bs) for _bs in batch_scores]))\n" + STEP_ANCHOR


def patched_solver_path():
    import rig_restart_probe as P
    src = open(os.path.join(P.V7, "arc_solver.py"), encoding="utf-8").read()
    assert src.count(EMIT_ANCHOR) == 1, "emission anchor not unique: %d" % src.count(EMIT_ANCHOR)
    assert src.count(STEP_ANCHOR) == 1, "step anchor not unique: %d" % src.count(STEP_ANCHOR)
    src = src.replace(EMIT_ANCHOR, EMIT_PATCHED).replace(STEP_ANCHOR, STEP_PATCHED)
    src += "\n_DFSLOG = []\n"
    out = os.path.join(BASE, "_dfslog_solver.py")
    open(out, "w", encoding="utf-8", newline="\n").write(src)
    return out


# ----------------------------------------------------------------------------- run (rig)

def run(n_tasks):
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig_restart_probe as P
    from arc_loader import QwenFormatter
    from transformers import AutoTokenizer
    from collections import defaultdict

    os.makedirs(OUTDIR, exist_ok=True)
    lifted = P.lift_from_solver(src_path=patched_solver_path())
    assert "_DFSLOG" in lifted, "the log list did not survive the lift closure"

    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    enc = P.encoder_for(tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    from arc_loader import ArcDataset
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)
    decode_budget = getattr(P, "DECODE_BUDGET", 800)

    print("logging %d tasks, decode budget %ds, strata %s"
          % (len(keys), decode_budget, {k: strat[k] for k in keys}), flush=True)

    import time
    for key in keys:
        outpath = os.path.join(OUTDIR, "%s.jsonl" % key)
        if os.path.isfile(outpath):
            print("skip %s, already logged" % key, flush=True)
            continue
        labels = {"%s_%d" % (key, i): np.array(g) for i, g in enumerate(sols[key])}
        m = P.build_model()
        pz = arc.change_keys([key])
        train_ds = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        t0 = time.time()
        P.train_one(m, tok, collator, fmt, train_ds, deadline=t0 + P.TRAIN_BUDGET)
        train_s = time.time() - t0
        print("%s trained in %.0fs" % (key, train_s), flush=True)

        base = m.get_base_model()
        base.config.use_cache = True
        base.gradient_checkpointing_disable()
        m.eval()

        multi = pz.split_multi_replies()
        eval_ds = multi.augment(n=2, seed=2).cut_to_len(
            formatter=fmt, name="input", max_len=P.MAXSEQ - fmt.max_new_tokens())
        by_test = defaultdict(list)
        for sk in sorted(eval_ds.keys):
            by_test[sk.split(".")[0].split("_")[1]].append(sk)
        batches = []
        for pairs in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
            for _, sks in by_test.items():
                for p in pairs:
                    h = [sks[o:o + 2] for o in p]
                    batches.extend([h[0] + h[1]] if P.BATCH_VIEWS == 4 else h)

        max_score = -np.log(P.DFS_FLOOR)
        deadline = time.time() + decode_budget
        f = open(outpath, "w", encoding="utf-8")
        truncated = False
        with torch.inference_mode():
            for bi, subkeys in enumerate(batches):
                if not subkeys:
                    continue
                if time.time() > deadline:
                    truncated = True
                    break
                tokens = [enc(eval_ds.get(sk, formatter=fmt)["input"]) for sk in subkeys]
                lifted["_DFSLOG"].clear()
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    dfs_result = P.run_deep(lifted["inference_turbo_dfs"],
                                            m, tokens, fmt.max_new_tokens(), max_score, deadline)
                cands = []
                for lane, beams in dfs_result:
                    sk = subkeys[lane]
                    bk = sk.split(".")[0]
                    for score, seq in beams:
                        arr = fmt.convert_tokens_to_array(seq)
                        if arr is None:
                            continue
                        sol = multi.invert_mod(arr, sk, inv_perm=True)
                        lab = labels.get(bk)
                        cands.append({"lane": lane, "bk": bk, "score": float(score),
                                      "correct": bool(lab is not None
                                                      and np.array_equal(sol, lab))})
                f.write(json.dumps({"task": key, "batch": bi, "subkeys": subkeys,
                                    "prompt_lens": [len(t) for t in tokens],
                                    "records": lifted["_DFSLOG"],
                                    "cands": cands}) + "\n")
                f.flush()
        f.write(json.dumps({"task": key, "meta": True, "train_s": round(train_s, 1),
                            "truncated": truncated, "stratum": strat[key]}) + "\n")
        f.close()
        print("%s logged%s" % (key, " (truncated)" if truncated else ""), flush=True)
        del m
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print("ALL LOGGED", flush=True)
    return True


# ----------------------------------------------------------------------------- replay (anywhere)

def replay_batch(records, cands):
    """Rebuild the explored tree, then count single-lane node expansions to each truth under both
    orders. DFS order is the log itself. Best-first order expands recorded nodes by cumulative NLL,
    restricted to the same explored set, so it measures pure visiting order at equal information."""
    lanes = {}
    nodes = []
    emissions = []
    for rec in records:
        if rec[0] == "N":
            _, pos, toks, scores = rec
            for lane, (t, s) in enumerate(zip(toks, scores)):
                if s >= PAD_SCORE:
                    continue
                stack = lanes.setdefault(lane, [])
                while stack and stack[-1][0] >= pos:
                    stack.pop()
                node = {"lane": lane, "pos": pos, "nll": s, "dfs_i": len(nodes)}
                nodes.append(node)
                stack.append((pos, node))
        else:
            _, pos, lane, score = rec
            emissions.append({"lane": lane, "pos": pos, "score": score,
                              "dfs_i": len(nodes)})
    # match emissions to graded candidates by (lane, score)
    open_c = {}
    for c in cands:
        open_c.setdefault((c["lane"], round(c["score"], 6)), []).append(c["correct"])
    truths_dfs = []
    for e in emissions:
        flags = open_c.get((e["lane"], round(e["score"], 6)))
        if flags:
            e["correct"] = flags.pop(0)
            if e["correct"]:
                truths_dfs.append(e["dfs_i"])
        else:
            e["correct"] = None
    # best-first: expansion order over the same nodes by cumulative NLL (ties by depth, deeper
    # first, since a deeper node at equal NLL is a forced continuation costing nothing extra)
    order = sorted(range(len(nodes)), key=lambda i: (nodes[i]["nll"], -nodes[i]["pos"]))
    bf_rank = {i: r + 1 for r, i in enumerate(order)}
    truths_bf = []
    for e in emissions:
        if e.get("correct"):
            # a truth is emitted once every node on its path is expanded; its path's most expensive
            # recorded node is its own emitter, so its best-first cost is that node's rank
            src = e["dfs_i"] - 1
            truths_bf.append(bf_rank.get(src, len(nodes)))
    return len(nodes), truths_dfs, truths_bf, sum(1 for e in emissions if e["correct"] is None)


def replay(logdir):
    rows = []
    unmatched_total = 0
    for name in sorted(os.listdir(logdir)):
        if not name.endswith(".jsonl"):
            continue
        task = name[:-6]
        n_nodes = 0
        t_dfs, t_bf = [], []
        truncated = False
        for line in open(os.path.join(logdir, name), encoding="utf-8"):
            j = json.loads(line)
            if j.get("meta"):
                truncated = j["truncated"]
                continue
            nn, td, tb, um = replay_batch(j["records"], j["cands"])
            t_dfs += [d + n_nodes for d in td]
            t_bf += [b + n_nodes for b in tb]
            n_nodes += nn
            unmatched_total += um
        rows.append({"task": task, "nodes": n_nodes, "truncated": truncated,
                     "dfs_first_truth": min(t_dfs) if t_dfs else None,
                     "bf_first_truth": min(t_bf) if t_bf else None,
                     "n_truths": len(t_dfs)})

    print("%-10s %8s %6s %12s %12s %8s" % ("task", "nodes", "truths", "DFS-to-truth",
                                           "BF-to-truth", "trunc"))
    print("-" * 62)
    saves, losses = [], 0
    for r in rows:
        print("%-10s %8d %6d %12s %12s %8s"
              % (r["task"], r["nodes"], r["n_truths"],
                 r["dfs_first_truth"] if r["dfs_first_truth"] is not None else "-",
                 r["bf_first_truth"] if r["bf_first_truth"] is not None else "-",
                 "yes" if r["truncated"] else ""))
        if r["dfs_first_truth"] is not None:
            if r["bf_first_truth"] <= r["dfs_first_truth"]:
                saves.append(1.0 - r["bf_first_truth"] / max(r["dfs_first_truth"], 1))
            else:
                losses += 1
    if unmatched_total:
        print("\nWARNING: %d emissions could not be matched to graded candidates" % unmatched_total)
    print("\nVERDICT")
    hit = [r for r in rows if r["dfs_first_truth"] is not None]
    if not hit:
        print("  no truths were found in any logged tree, the replay cannot discriminate orders")
    else:
        import statistics as st
        print("  on %d tasks where the tree contains a truth, best-first reaches the first truth"
              % len(hit))
        print("  with a median saving of %.0f%% of node expansions (losses on %d task%s)."
              % (100 * st.median(saves) if saves else 0.0, losses, "" if losses == 1 else "s"))
        print("  Saving above ~50%% on truncated tasks says order is worth real engineering;")
        print("  near zero says threshold-DFS already visits in nearly the right order and the")
        print("  anytime enumerator dies here, for the cost of one logging run.")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        ok = run(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    else:
        ok = replay(sys.argv[2] if len(sys.argv) > 2 else OUTDIR)
    sys.exit(0 if ok else 1)
