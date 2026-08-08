"""Is the decode batch width free, and does spending it buy candidates?

The deployed decoder splits the sixteen evaluation views into four batches of four, fixed by
hard-coded index offsets, and runs four sequential depth-first searches. Four trees cost the SUM of
their step counts. One tree over all sixteen lanes costs the MAX. Single-token decode is bounded by
reading the weights out of HBM rather than by arithmetic, so a wider batch should cost almost the
same per step, which would make the extra width nearly free.

Measurement says the grouping is possible: over 120 evaluation tasks the sixteen views take exactly
one distinct token length for 39.5 percent of test outputs and exactly two for the rest, never more.
So every task decodes in at most two batches instead of four, and the length gaps are single-digit
token counts from grid separators rather than anything structural.

Why this could be worth points rather than just time. The score is throughput-limited: one immutable
kernel version scored 30.56, 32.64 and 28.47 on three executions, and the only thing that differed
was how far the machine got. Decode is deadline-bounded in three places, so a search that runs out of
budget stops early and the candidates it would have found never exist. Faster decode is therefore not
a convenience, it is coverage.

Two stages, and the second is the one that matters.

  bench   forward-pass time at batch 1, 2, 4, 8, 16 on a real prefix. If time is flat in width the
          bandwidth account holds and the width is free. If it scales linearly the idea is dead here
          and nothing further needs running.
  ab      per task, decode twice under an identical wall budget: the deployed 4-wide pairing, and
          length-grouped batches up to 16 wide. Compare candidates produced and truth-in-pool. This
          is a within-task comparison, so hidden-set run-to-run noise does not enter.

Usage: python rig_widebatch.py bench [N_TASKS=3]
       python rig_widebatch.py ab [N_TASKS=10] [DECODE_S=600]
"""
import gc
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
OUT = os.path.join(BASE, "experiments", "widebatch.jsonl")


def _setup():
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    return P, arc, sols, tok, fmt


def views_for(P, arc, fmt, task):
    """The sixteen evaluation views of a task, grouped by test output, as the deployed code builds
    them. Returned as (base_key, subkey, token_ids) so any batching policy can be applied on top."""
    pz = arc.change_keys([task])
    multi = pz.split_multi_replies()
    ds = multi.augment(n=2, seed=2).cut_to_len(
        formatter=fmt, name="input", max_len=P.MAXSEQ - fmt.max_new_tokens())
    by = defaultdict(list)
    for sk in sorted(ds.keys):
        by[sk.split(".")[0].split("_")[1]].append(sk)
    return multi, ds, by


def bench(n_tasks):
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    P, arc, sols, tok, fmt = _setup()
    enc = P.encoder_for(tok)
    keys, _, _ = P.select_tasks(max(4, n_tasks))
    m = P.build_model()
    base = m.get_base_model()
    base.config.use_cache = True
    base.gradient_checkpointing_disable()
    m.eval()

    task = keys[0]
    _multi, ds, by = views_for(P, arc, fmt, task)
    subkeys = list(by.values())[0]
    toks = enc(ds.get(subkeys[0], fmt)["input"])
    print("bench on %s, prefix %d tokens\n" % (task, len(toks)), flush=True)
    print("%6s %12s %12s %10s" % ("width", "prefill_s", "per_step_ms", "vs width 1"), flush=True)

    base_ms = None
    with torch.inference_mode():
        for w in (1, 2, 4, 8, 16):
            ids = torch.tensor([toks] * w, device="cuda", dtype=torch.long)
            torch.cuda.synchronize()
            t0 = time.time()
            out = m(input_ids=ids, return_dict=True, use_cache=True)
            torch.cuda.synchronize()
            prefill = time.time() - t0
            cache = out.past_key_values
            pos = ids.size(1)
            # time single-token steps, which is what the DFS actually spends its life doing
            step_tok = torch.zeros((w, 1), device="cuda", dtype=torch.long)
            torch.cuda.synchronize()
            t1 = time.time()
            N = 20
            for i in range(N):
                o = m(input_ids=step_tok,
                      position_ids=torch.full((w, 1), pos + i, device="cuda"),
                      past_key_values=cache, return_dict=True, use_cache=True)
                cache = o.past_key_values
            torch.cuda.synchronize()
            ms = (time.time() - t1) / N * 1000
            if base_ms is None:
                base_ms = ms
            print("%6d %12.2f %12.2f %9.2fx" % (w, prefill, ms, ms / base_ms), flush=True)
            del out, cache
            gc.collect()
            torch.cuda.empty_cache()

    print("\nA width that costs the same per step as width 1 is free, and four sequential trees can", flush=True)
    print("become one. A width that scales linearly means the decode is compute-bound here and the", flush=True)
    print("whole idea is dead.", flush=True)
    return True


def decode_with(P, m, tok, fmt, lifted, multi, ds, by, deadline, wide):
    """Decode every view, batching either as the deployed 4-wide pairing or by length up to 16."""
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    enc = P.encoder_for(tok)
    mnt = fmt.max_new_tokens()
    max_score = -np.log(P.DFS_FLOOR)

    batches = []
    if wide:
        # every view of a test output, split only where token length forces it
        for _t, subkeys in by.items():
            groups = defaultdict(list)
            for sk in subkeys:
                groups[len(enc(ds.get(sk, fmt)["input"]))].append(sk)
            batches.extend(groups.values())
    else:
        for pairs in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
            for _t, subkeys in by.items():
                for p in pairs:
                    h = [subkeys[o:o + 2] for o in p]
                    batches.append(h[0] + h[1])

    pool = {sk.split(".")[0]: [] for sk in ds.keys}
    trunc = False
    steps = 0
    with torch.inference_mode():
        for subkeys in batches:
            if not subkeys:
                continue
            if time.time() > deadline:
                trunc = True
                break
            toks = [enc(ds.get(sk, fmt)["input"]) for sk in subkeys]
            groups = defaultdict(list)
            for i, t in enumerate(toks):
                groups[len(t)].append(i)
            for idxs in groups.values():
                if time.time() > deadline:
                    trunc = True
                    break
                steps += 1
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    res = P.run_deep(lifted["inference_turbo_dfs"], m,
                                     [toks[i] for i in idxs], mnt, max_score, deadline)
                for sub_id, beams in res:
                    sk = subkeys[idxs[sub_id]]
                    bk = sk.split(".")[0]
                    for bs, seq in beams:
                        arr = fmt.convert_tokens_to_array(seq)
                        if arr is None:
                            continue
                        pool.setdefault(bk, []).append(
                            {"beam_score": float(bs), "solution": multi.invert_mod(arr, sk, inv_perm=True)})
    return pool, trunc, steps, len(batches)


def ab(n_tasks, decode_s):
    import numpy as np
    import torch
    P, arc, sols, tok, fmt = _setup()
    lifted = P.lift_from_solver()
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {json.loads(l)["task"] for l in open(OUT, encoding="utf-8")}
    print("wide-batch A/B on %d tasks, %ds decode per arm\n" % (len(keys), decode_s), flush=True)

    for task in keys:
        if task in done:
            continue
        pz = arc.change_keys([task])
        labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
        m = P.build_model()
        ds_train = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        t0 = time.time()
        P.train_one(m, tok, collator, fmt, ds_train, deadline=t0 + P.TRAIN_BUDGET)
        train_s = time.time() - t0
        base = m.get_base_model()
        base.config.use_cache = True
        base.gradient_checkpointing_disable()
        m.eval()

        multi, ds, by = views_for(P, arc, fmt, task)
        rec = {"task": task, "stratum": strat[task], "train_s": round(train_s, 1)}
        for arm, wide in (("deployed4", False), ("wide16", True)):
            t1 = time.time()
            pool, trunc, calls, nb = decode_with(P, m, tok, fmt, lifted, multi, ds, by,
                                                 time.time() + decode_s, wide)
            graded = P.grade(pool, labels, sorted(labels))
            rec[arm] = {"cands": sum(len(v) for v in pool.values()),
                        "uniq": len({(bk, tuple(map(tuple, np.asarray(c["solution"]).tolist())))
                                     for bk, cs in pool.items() for c in cs}),
                        "hits": sum(1 for v in graded.values() if v["in_pool"]),
                        "outputs": len(graded), "dfs_calls": calls, "batches": nb,
                        "truncated": trunc, "s": round(time.time() - t1, 1)}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        d, w = rec["deployed4"], rec["wide16"]
        print("%-10s  deployed4 %4d cands %d/%d hits (%d calls)   wide16 %4d cands %d/%d hits (%d calls)"
              % (task, d["cands"], d["hits"], d["outputs"], d["dfs_calls"],
                 w["cands"], w["hits"], w["outputs"], w["dfs_calls"]), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    summarize()
    return True


def summarize():
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    if not rows:
        print("no rows")
        return
    d_c = sum(r["deployed4"]["cands"] for r in rows)
    w_c = sum(r["wide16"]["cands"] for r in rows)
    d_h = sum(r["deployed4"]["hits"] for r in rows)
    w_h = sum(r["wide16"]["hits"] for r in rows)
    n_out = sum(r["deployed4"]["outputs"] for r in rows)
    d_t = sum(r["deployed4"]["truncated"] for r in rows)
    w_t = sum(r["wide16"]["truncated"] for r in rows)
    d_k = sum(r["deployed4"]["dfs_calls"] for r in rows)
    w_k = sum(r["wide16"]["dfs_calls"] for r in rows)
    gained = [r["task"] for r in rows if r["wide16"]["hits"] > r["deployed4"]["hits"]]
    lost = [r["task"] for r in rows if r["wide16"]["hits"] < r["deployed4"]["hits"]]

    print("\n%d tasks, %d outputs, equal wall budget per arm" % (len(rows), n_out))
    print("                     deployed4    wide16")
    print("candidates            %8d  %8d" % (d_c, w_c))
    print("truth in pool         %8d  %8d   of %d outputs" % (d_h, w_h, n_out))
    print("DFS calls             %8d  %8d" % (d_k, w_k))
    print("decodes truncated     %8d  %8d" % (d_t, w_t))
    print("tasks gained by wide: %d %s" % (len(gained), gained))
    print("tasks lost by wide  : %d %s" % (len(lost), lost))
    print("\nVERDICT")
    if w_h > d_h and not lost:
        print("  Wider batching finds %d more truths at the same wall budget and loses none. The" % (w_h - d_h))
        print("  width was free and the search was budget-bound, which is the mechanism the")
        print("  hidden-set variance already implied. Worth a submission.")
    elif w_h == d_h and w_c > d_c * 1.2:
        print("  More candidates (%.2fx) but the same truths. The extra search is finding more of" % (w_c / max(d_c, 1)))
        print("  what the model already believed rather than reaching new hypotheses, so it buys")
        print("  time and not coverage. Still useful if the queue is wall-limited, worthless if not.")
    elif w_h < d_h:
        print("  Wider batching LOSES truths. Batch composition changes the logits, so this is not")
        print("  the semantics-preserving speedup it looks like. Do not ship it.")
    else:
        print("  No difference in coverage. Report the speedup, claim nothing about score.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if mode == "summarize":
        summarize()
        sys.exit(0)
    if mode == "bench":
        sys.exit(0 if bench(int(sys.argv[2]) if len(sys.argv) > 2 else 3) else 1)
    sys.exit(0 if ab(int(sys.argv[2]) if len(sys.argv) > 2 else 10,
                     int(sys.argv[3]) if len(sys.argv) > 3 else 600) else 1)
