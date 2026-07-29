"""Staged smoke tests for rig_restart_probe.py. Cheapest first, each catching a specific fault class.

Five launches of the probe produced zero completed cells, and each fault cost 15 to 60 minutes to find
because the only test available was the full run. Worse, a later audit found defects that would NOT
have crashed: they would have produced a confident wrong number. This ladder costs about two hours of
rig time to de-risk a run of thirty to forty hours, and every stage exits 0 or 1 with a checklist in
rig_biggrid_probe.py's format.

  s0    offline logic and analysis correctness, no GPU        ~1 min
  s0b   data pipeline, token lengths, seed separation         ~5 min
  s1a   tiny random model, three-arm cache equivalence        ~4 min
  s1b   real weights at reduced size, first real decode       ~10 min
  s2    memory-configuration picker on the worst case         ~30 min
  s3    one full cell at the chosen configuration             ~45 min
  s4    endurance across three consecutive cells              ~40 min

Run them in order. Nothing launches the full probe until s0 through s4 pass.

Usage: python rig_smoke_restart.py <stage>
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGE = sys.argv[1] if len(sys.argv) > 1 else "s0"
STAGES = ("s0", "s0b", "s1a", "s1b", "s2", "s3", "s4")
assert STAGE in STAGES, "stage must be one of %s" % (STAGES,)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))
    print("  %-4s %-40s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return bool(ok)


def finish(stage):
    fails = [n for n, ok, _ in CHECKS if not ok]
    print("\n==== %s CHECKLIST ====" % stage.upper())
    for n, ok, d in CHECKS:
        print("%-4s %-40s %s" % ("PASS" if ok else "FAIL", n, d))
    print("%s %s%s" % (stage.upper(), "PASS" if not fails else "FAIL ", ", ".join(fails)))
    sys.exit(0 if not fails else 1)


# --------------------------------------------------------------------------- s0

def stage_s0():
    import numpy as np
    import torch
    import rig_restart_probe as P

    lifted = P.lift_from_solver()
    check("lift.definitions",
          all(k in lifted for k in ("turbo_dfs", "inference_turbo_dfs",
                                    "QwenDataCollatorForCompletionOnlyLM")))
    check("lift.constants_transitive",
          all(k in lifted for k in ("ARC_TOKENS", "PAD_ID", "EOS_ID", "ARC_VOCAB")),
          "hand-listing these is what broke run 3")
    check("lift.branch_injected", "_branch_cache" in lifted["turbo_dfs"].__code__.co_names)
    unpatched = P.lift_from_solver(patch_cache=False)
    check("lift.unpatched_available",
          "_branch_cache" not in unpatched["turbo_dfs"].__code__.co_names,
          "needed for the s1a clone arm")

    check("recursion.limit", sys.getrecursionlimit() > 1131,
          "%d, max_new_tokens is about 931" % sys.getrecursionlimit())

    # the inlined cosine schedule must equal the library's, since both optimizer paths use it
    from transformers import get_cosine_schedule_with_warmup
    dummy = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=P.LR)
    sch = get_cosine_schedule_with_warmup(dummy, int(P.WARMUP_RATIO * 100), 100)
    worst = 0.0
    for i in range(100):
        worst = max(worst, abs(dummy.param_groups[0]["lr"] - P.cosine_lr(i, 100, P.WARMUP_RATIO, P.LR)))
        sch.step()
    check("schedule.matches_library", worst < 1e-12, "max abs diff %.3g" % worst)

    # The collator is the single most likely silent breakage of the training path under
    # transformers 5, and constructing it (which the old preflight did) proves nothing.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    col = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    ids = [11, 1, 2, 15, 12, 3, 4, 15]        # USER .. EOS, ASSISTANT .. EOS
    out = col([{"input_ids": ids}])
    lab = out["labels"][0].tolist()
    # j=0 is the user turn and stays masked; j=1 is the assistant turn, unmasked from start+2 to end+1
    want = [-100, -100, -100, -100, -100, -100, 4, 15]
    check("collator.masks_completion_only", lab == want, "got %s" % lab)
    check("collator.some_supervision", any(x != -100 for x in lab))

    # The shipped tokenizer cannot emit the control ids the collator keys on: it is WordLevel with no
    # merges, and encode("user") returns []. Both turn markers collapse to [im_start, newline], so
    # every label stays masked and the loss is NaN over zero targets. This is the regression guard.
    check("tokenizer.cannot_reach_controls", tok.encode("user") == [],
          "shipped tokenizer drops it, which is why the exact encoder exists")
    enc = P.encoder_for(tok)
    sample = "<|im_start|>user\n12\n34<|im_end|><|im_start|>assistant\n56<|im_end|>"
    got = enc(sample)
    v = tok.get_vocab()
    check("encoder.emits_user_and_assistant", v["user"] in got and v["assistant"] in got,
          "ids %s" % got)
    check("encoder.roundtrips", tok.decode(got) == sample,
          "decode must reproduce the source text exactly")
    lab2 = col([{"input_ids": got}])["labels"][0].tolist()
    check("encoder.gives_supervision", any(x != -100 for x in lab2),
          "%d supervised of %d tokens" % (sum(1 for x in lab2 if x != -100), len(lab2)))

    # an output with no parseable candidate is in_pool=False, not an absent observation
    labels = {"t_0": np.array([[1, 2], [3, 4]]), "t_1": np.array([[5]])}
    g = P.grade({"t_1": [{"beam_score": 1.0, "solution": np.array([[5]])}]}, labels, ["t_0", "t_1"])
    check("grade.empty_pool_counts", "t_0" in g and g["t_0"]["in_pool"] is False
          and g["t_0"]["n_cands"] == 0, "empty keys used to vanish from the comparison")
    check("grade.finds_truth", g["t_1"]["in_pool"] is True)

    cells = P.cell_plan(["a", "b"], 3, 1, {"a": "A", "b": "B"})
    check("plan.cell_count", len(cells) == 7 and ("a", 1, 1) in cells,
          "%d cells, replicate on the reachable task" % len(cells))

    recs = [{"task": "a", "seed": 1, "replicate": 0, "marker": "started"},
            {"task": "a", "seed": 1, "replicate": 0, "ok": True},
            {"task": "a", "seed": 2, "replicate": 0, "marker": "started"},
            {"task": "a", "seed": 2, "replicate": 0, "ok": False},
            {"task": "a", "seed": 3, "replicate": 0, "marker": "started"},
            {"task": "a", "seed": 3, "replicate": 0, "marker": "started"}]
    todo, done, exh = P.plan_todo(cells, recs)
    check("resume.skips_done", ("a", 1, 0) in done and ("a", 1, 0) not in todo)
    check("resume.retries_failed", ("a", 2, 0) in todo,
          "a failed cell used to be skipped forever, defeating the supervisor")
    check("resume.caps_attempts", ("a", 3, 0) in exh and ("a", 3, 0) not in todo,
          "two started markers with no result exhausts the cell")

    # summarize on synthetic input with hand-computed answers
    tmp = os.path.join(P.EXP, "smoke_synth.jsonl")
    os.makedirs(P.EXP, exist_ok=True)
    pat = {"T1": [1, 0, 0], "T2": [0, 0, 0], "T3": [0, 1, 0]}
    with open(tmp, "w") as f:
        for t, bits in pat.items():
            for s, b in enumerate(bits, start=1):
                f.write(json.dumps({
                    "task": t, "seed": s, "replicate": 0, "ok": True, "stratum": "A",
                    "weight": 1.0, "truncated": False, "train_truncated": False,
                    "outputs": {"%s_0" % t: {"in_pool": bool(b), "n_cands": 1,
                                             "selected_proxy": False}}}) + "\n")
        f.write(json.dumps({"task": "T4", "seed": 1, "replicate": 0, "ok": True, "stratum": "A",
                            "weight": 1.0, "outputs": {"T4_0": {"in_pool": True, "n_cands": 1,
                                                                "selected_proxy": False}}}) + "\n")
        f.write(json.dumps({"task": "T1", "seed": 1, "replicate": 1, "ok": True, "stratum": "A",
                            "weight": 1.0, "truncated": True,
                            "outputs": {"T1_0": {"in_pool": False, "n_cands": 0,
                                                 "selected_proxy": False}}}) + "\n")
    res = P.summarize(path=tmp)
    ok = res is not None
    check("summarize.drops_incomplete_task", ok and res["dropped_tasks"] == ["T4"],
          ok and res["dropped_tasks"])
    check("summarize.excludes_truncated", ok and res["excluded_cells"] == 1)
    check("summarize.union", ok and abs(res["union_weighted"] - 2.0 / 3.0) < 1e-9,
          ok and round(res["union_weighted"], 4))
    check("summarize.single", ok and abs(res["single_weighted"][1] - 1.0 / 3.0) < 1e-9)
    check("summarize.gain", ok and abs(res["gain_weighted"] - (2 / 3 - (1 / 3 + 1 / 3 + 0) / 3)) < 1e-9,
          ok and round(res["gain_weighted"], 4))
    check("summarize.no_replicate_is_none", ok and res["net_disagreement"] is None,
          "the truncated replicate was excluded, so the control is absent and flagged")

    keys, weight, strat = P.select_tasks(20)
    from collections import Counter as C
    cnt = C(strat.values())
    check("strata.sample", len(keys) == 20 and cnt["A"] >= 10, dict(cnt))
    check("strata.weights_positive", all(w > 0 for w in weight.values()),
          {s: round(weight[[k for k in keys if strat[k] == s][0]], 2) for s in cnt})
    finish("s0")


# --------------------------------------------------------------------------- s0b

def stage_s0b():
    import numpy as np
    import rig_restart_probe as P
    from transformers import AutoTokenizer
    from arc_loader import ArcDataset, QwenFormatter

    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    keys, _, strat = P.select_tasks(20)

    lens, per_task, texts_by_seed, drops = [], {}, {}, {}
    for k in keys:
        pz = arc.change_keys([k])
        for seed in (1, 2, 3):
            ds = pz.augment(n=16, shfl_keys=True, seed=seed)
            ds = ds.cut_to_len(formatter=fmt, name="text", max_len=P.MAXSEQ)
            rows = ds.as_list(fmt)
            tl = [len(tok(r["text"], return_tensors=None)["input_ids"]) for r in rows]
            lens += tl
            texts_by_seed.setdefault(k, {})[seed] = {r["text"] for r in rows}
            drops.setdefault(k, {})[seed] = len(rows)
            if seed == 1:
                per_task[k] = max(tl) if tl else 0

    a = np.array(lens)
    print("  train-text tokens: n=%d p50=%d p90=%d p100=%d, at-cap=%d"
          % (len(a), np.percentile(a, 50), np.percentile(a, 90), a.max(),
             int((a >= P.MAXSEQ).sum())), flush=True)
    worst = max(per_task, key=per_task.get)
    print("  worst-case task %s at %d tokens (s2/s3 should use this one)"
          % (worst, per_task[worst]), flush=True)
    json.dump({"worst_task": worst, "worst_tokens": int(per_task[worst]),
               "p50": int(np.percentile(a, 50)), "p100": int(a.max()),
               "per_task_max": {k: int(v) for k, v in per_task.items()}},
              open(os.path.join(P.EXP, "smoke_lengths.json"), "w"), indent=1)

    # If the varied axis does not vary, the entire run measures nothing.
    jac = []
    for k, bys in texts_by_seed.items():
        for i in (1, 2):
            for j in range(i + 1, 4):
                inter = len(bys[i] & bys[j])
                uni = len(bys[i] | bys[j])
                jac.append(inter / uni if uni else 1.0)
    check("augment.seeds_differ", max(jac) < 1.0,
          "pairwise Jaccard %.3f..%.3f across %d pairs" % (min(jac), max(jac), len(jac)))
    # Measured near-zero: the eight dihedral views are shared as GEOMETRY, but a colour permutation
    # composes on top of each, so the same view under a different colour map is a different string
    # and the text sets are almost disjoint. That is what the seed actually re-rolls, and it is why
    # the scope of a null result is colour, not geometry.
    # distinct-text ratio: a seed whose colour-permutation sample collided would train on fewer
    # effective examples than its row count suggests
    ratios = [len(bys[s]) / float(drops[k][s]) for k, bys in texts_by_seed.items() for s in bys]
    check("augment.texts_mostly_distinct", min(ratios) >= 0.9,
          "distinct/rows %.3f..%.3f" % (min(ratios), max(ratios)))

    # seed 1 must be byte-stable, since it is the replication target for task_timing.csv
    k0 = keys[0]
    again = arc.change_keys([k0]).augment(n=16, shfl_keys=True, seed=1)
    again = again.cut_to_len(formatter=fmt, name="text", max_len=P.MAXSEQ)
    check("augment.seed1_reproducible",
          {r["text"] for r in again.as_list(fmt)} == texts_by_seed[k0][1])

    # cut_to_len trims the .ex suffix, which can collide two view keys into one on long tasks
    bad, ev_lens = [], []
    for k in keys:
        multi = arc.change_keys([k]).split_multi_replies()
        ev = multi.augment(n=2, seed=2)
        ev = ev.cut_to_len(formatter=fmt, name="input", max_len=P.MAXSEQ - fmt.max_new_tokens())
        by_test = {}
        for sk in ev.keys:
            by_test.setdefault(sk.split(".")[0], []).append(sk)
        for bk, sks in by_test.items():
            if len(sks) != 16 or len(set(sks)) != 16:
                bad.append((bk, len(sks), len(set(sks))))
        for sk in ev.keys:
            ev_lens.append(len(tok(ev.get(sk, fmt)["input"], return_tensors=None)["input_ids"]))
    check("eval_views.sixteen_distinct", not bad,
          "collisions would silently shrink pools on long tasks: %s" % bad[:4])

    # MAXSEQ is a cap, not a length. Knowing the true maximum on BOTH paths says how much of it is
    # dead headroom that could be reclaimed as activation memory at zero fidelity cost.
    ev = np.array(ev_lens)
    need = int(max(a.max(), ev.max() + fmt.max_new_tokens()))
    print("  eval-input tokens: p50=%d p100=%d; decode needs +%d for generation"
          % (np.percentile(ev, 50), ev.max(), fmt.max_new_tokens()), flush=True)
    print("  MAXSEQ actually required: %d (configured %d, dead headroom %d)"
          % (need, P.MAXSEQ, P.MAXSEQ - need), flush=True)
    check("maxseq.never_truncates", need <= P.MAXSEQ,
          "no example is cut at the configured cap")
    json.dump({"required_maxseq": need}, open(os.path.join(P.EXP, "smoke_maxseq.json"), "w"))

    nd = {k: sorted(set(v.values())) for k, v in drops.items() if len(set(v.values())) > 1}
    check("cut_to_len.example_count_seed_independent", not nd,
          "seed-dependent truncation would inflate the decorrelation number: %s" % list(nd)[:3])
    finish("s0b")


# --------------------------------------------------------------------------- s1a

def _manual_branch(cache, pos):
    """Truncate by slicing the layer tensors directly, independent of Cache.crop.

    The point of this arm is to CHECK crop, so it must not call it. transformers 5 dropped
    to_legacy_cache/from_legacy_cache, and a DynamicCache is layers[i].keys / .values shaped
    [batch, heads, seq, dim], so the truncation is a plain slice on a deep copy.
    """
    out = copy.deepcopy(cache)
    for layer in out.layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys[:, :, :pos, :].clone()
            layer.values = layer.values[:, :, :pos, :].clone()
    return out


def stage_s1a():
    import numpy as np
    import torch
    import rig_restart_probe as P
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(P.MODEL_DIR)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    if hasattr(cfg, "head_dim"):
        cfg.head_dim = 32
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to("cuda", torch.bfloat16).eval()
    check("tiny.built", True, "%d layers, hidden %d" % (cfg.num_hidden_layers, cfg.hidden_size))

    crop = P.lift_from_solver()
    manual_ns = P.lift_from_solver()
    manual_ns["_branch_cache"] = _manual_branch
    plain = P.lift_from_solver(patch_cache=False)

    def ref_dfs(prefix, max_new, max_score):
        """Cache-free reference. Only this arm can catch a position_ids error."""
        results = {i: [] for i in range(len(prefix))}

        def walk(row, seq, score):
            if len(seq) > max_new:
                return
            ids = torch.tensor([prefix[row] + seq], device="cuda")
            with torch.no_grad():
                lg = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu()
            nll = score - lg.log_softmax(-1)
            for t in crop["ARC_TOKENS"]:
                s = float(nll[t])
                if s < max_score:
                    if t == crop["EOS_ID"]:
                        results[row].append((round(s, 4), tuple(seq + [t])))
                    elif max_new - len(seq) > 1:
                        walk(row, seq + [t], s)
        for r in range(len(prefix)):
            walk(r, [], 0.0)
        return results

    def run_arm(ns, prefix, max_new, max_score):
        P.CACHE_STATS.update(calls=0, shrinks=0)
        out = ns["inference_turbo_dfs"](model, prefix, max_new, max_score, time.time() + 300)
        got = {i: set() for i in range(len(prefix))}
        for row, beams in out:
            for sc, toks in beams:
                got[row].add((round(float(sc), 4), tuple(toks)))
        return got, dict(P.CACHE_STATS)

    rng = np.random.default_rng(0)

    # ARM 1, decisive. With a threshold loose enough that nothing is ever pruned, all three arms
    # enumerate the SAME tree, so the sequence sets must match exactly and any score difference is
    # pure cache numerics. This tests the cache directly instead of through a hard threshold, which
    # is what a set comparison at the real floor actually tests.
    LOOSE, DEPTH = 50.0, 3
    worst_score_diff, key_mismatch, st = 0.0, 0, {"calls": 0, "shrinks": 0}
    for _ in range(3):
        pre = [[int(x) for x in rng.integers(0, 10, size=6)] for _ in range(2)]
        ref = {r: dict((t, s) for s, t in v) for r, v in ref_dfs(pre, DEPTH, LOOSE).items()}
        got, st = run_arm(crop, pre, DEPTH, LOOSE)
        got = {r: dict((t, s) for s, t in v) for r, v in got.items()}
        for r in ref:
            if set(ref[r]) != set(got.get(r, {})):
                key_mismatch += 1
                continue
            for t, s in ref[r].items():
                worst_score_diff = max(worst_score_diff, abs(s - got[r][t]))
    check("cache.unpruned_tree_identical", key_mismatch == 0,
          "%d row(s) enumerated a different sequence set with nothing pruned" % key_mismatch)
    check("cache.unpruned_scores_agree", worst_score_diff < 5e-3,
          "max |cached - cachefree| = %.2e over the full tree" % worst_score_diff)

    # ARM 2, diagnostic. At the real floor the hard threshold makes set membership numerically
    # fragile: a borderline token flips and a whole subtree appears or vanishes. Disagreement here
    # is expected and is NOT evidence against the port, so it is reported rather than asserted. What
    # would be evidence is arm 1 failing.
    agree_cm = trials = 0
    nodes, thresh_gap = [], []
    for _ in range(6):
        for max_score in (4.0, 6.0, 8.0):
            pre = [[int(x) for x in rng.integers(0, 10, size=6)] for _ in range(2)]
            ref = ref_dfs(pre, 4, max_score)
            gc_, st = run_arm(crop, pre, 4, max_score)
            gm_, _ = run_arm(manual_ns, pre, 4, max_score)
            nodes.append(sum(len(v) for v in ref.values()))
            trials += 1
            agree_cm += int(gc_ == gm_)
            for r, v in ref.items():
                diff = {(s, t) for s, t in v} ^ gc_.get(r, set())
                thresh_gap += [abs(max_score - s) for s, _ in diff]
    check("cache.tree_nonempty", min(nodes) > 0 and max(nodes) >= 4,
          "node counts %d..%d, an empty tree would pass vacuously" % (min(nodes), max(nodes)))
    check("cache.crop_equals_manual", agree_cm == trials,
          "%d/%d trials, crop matches an implementation that never calls crop" % (agree_cm, trials))
    print("  threshold-boundary note: %d set differences vs the cache-free arm, closest to the"
          " floor %.4f nats" % (len(thresh_gap), min(thresh_gap) if thresh_gap else float("nan")),
          flush=True)
    check("cache.branch_actually_fired", st["calls"] > 0 and st["shrinks"] > 0,
          "calls %d shrinks %d, a silent no-op used to be invisible" % (st["calls"], st["shrinks"]))

    P.CACHE_STATS.update(calls=0, shrinks=0)
    try:
        run_arm(plain, [[1, 2, 3, 4, 5, 6]], 4, 6.0)
        unpatched_note = "unpatched decoder did not fault on this tiny model"
    except Exception as e:                                              # noqa: BLE001
        unpatched_note = "unpatched decoder fails as expected: %s" % type(e).__name__
    check("cache.unpatched_reference_noted", True, unpatched_note)
    finish("s1a")


# --------------------------------------------------------------------------- shared cell runner

def _one_cell(P, task, seed, rank, maxseq, max_batches, decode_budget):
    import torch
    from transformers import AutoTokenizer
    from arc_loader import ArcDataset, QwenFormatter
    import numpy as np

    P.RANK, P.MAXSEQ = rank, maxseq
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    lifted = P.lift_from_solver()
    col = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m = P.build_model()
    pz = arc.change_keys([task])
    ds = pz.augment(n=16, shfl_keys=True, seed=seed).cut_to_len(
        formatter=fmt, name="text", max_len=maxseq)
    stats = P.train_one(m, tok, col, fmt, ds, deadline=t0 + P.TRAIN_BUDGET,
                        max_batches=max_batches)
    train_s = time.time() - t0
    t1 = time.time()
    P.CACHE_STATS.update(calls=0, shrinks=0)
    pool, trunc, overflow = P.decode_pool(m, tok, fmt, lifted, pz, t1 + decode_budget)
    graded = P.grade(pool, labels, sorted(labels))
    return {"stats": stats, "train_s": train_s, "decode_s": time.time() - t1,
            "graded": graded, "truncated": trunc, "overflow": overflow,
            "peak_alloc": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "peak_reserved": torch.cuda.max_memory_reserved() / 1024 ** 3,
            "cache": dict(P.CACHE_STATS), "adapter": P.adapter_stats(m), "model": m}


def stage_s1b():
    import rig_restart_probe as P
    # Rank is what makes this cheap, not sequence length. Shrinking MAXSEQ instead sends cut_to_len
    # past the cliff where it has dropped every training pair and fmt_train indexes an empty list,
    # so the cap stays at the production value and the shortest task keeps the stage fast.
    lp = os.path.join(P.EXP, "smoke_lengths.json")
    assert os.path.isfile(lp), "run stage s0b first"
    per = json.load(open(lp))["per_task_max"]
    task = min(per, key=per.get)
    print("  shortest task %s at %d tokens, rank 16, cap %d" % (task, per[task], P.MAXSEQ),
          flush=True)
    r = _one_cell(P, task, 1, rank=16, maxseq=P.MAXSEQ, max_batches=8,
                  decode_budget=420)
    s = r["stats"]
    check("s1b.train_finite", s["train_loss_final"] == s["train_loss_final"]
          and s["train_loss_final"] > 0, "final %.5f over %d steps"
          % (s["train_loss_final"], s["train_steps"]))
    check("s1b.loss_decreased", s["train_loss_last10"] <= s["train_loss_mean"],
          "last10 %.5f vs mean %.5f" % (s["train_loss_last10"], s["train_loss_mean"]))
    tot = sum(v["n_cands"] for v in r["graded"].values())
    check("s1b.pool_nonempty", tot > 0, "%d candidates across %d outputs"
          % (tot, len(r["graded"])))
    check("s1b.cache_fired", r["cache"]["shrinks"] > 0, r["cache"])
    check("s1b.reserved_headroom", r["peak_reserved"] < 15.0,
          "peak reserved %.2f GB" % r["peak_reserved"])
    print("  timings: train %.0fs decode %.0fs" % (r["train_s"], r["decode_s"]), flush=True)
    finish("s1b")


# --------------------------------------------------------------------------- s2

def stage_s2():
    """Pick the memory configuration empirically, on the worst-case example, in historical order."""
    import numpy as np
    import torch
    import rig_restart_probe as P
    from transformers import AutoTokenizer
    from arc_loader import ArcDataset, QwenFormatter

    lp = os.path.join(P.EXP, "smoke_lengths.json")
    assert os.path.isfile(lp), "run stage s0b first, it identifies the worst-case task"
    L = json.load(open(lp))
    task = L["worst_task"]

    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    lifted = P.lift_from_solver()
    col = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    arc = ArcDataset.from_file(os.path.join(P.data_dir(), "arc-agi_evaluation_challenges.json"))
    rows = arc.change_keys([task]).augment(n=16, shfl_keys=True, seed=1).cut_to_len(
        formatter=fmt, name="text", max_len=P.MAXSEQ).as_list(fmt)
    toks = sorted(([tok(r["text"], return_tensors=None)["input_ids"] for r in rows]), key=len)
    p50, p100 = toks[len(toks) // 2], toks[-1]
    # Optimizer state allocates on step 1, so a probe that runs the longest example first passes and
    # the real run dies later. Replay the order that actually broke: warm up, then the worst case,
    # then alternate to expose fragmentation rather than a single clean peak.
    order = [p50, p50, p100, p50, p100, p50]
    print("  worst task %s, p50 %d tokens, p100 %d tokens" % (task, len(p50), len(p100)), flush=True)

    # Fidelity order: keep rank as high as fits. bf16 adapters come first because they cost nothing
    # semantically (it is what unsloth does) and halve both the parameter and gradient terms, and
    # the first pass showed every fp32 arm reserving past the card and thrashing into host memory.
    candidates = [("r256 bf16 adamw", 256, "adamw", True),
                  ("r256 bf16 offload", 256, "adamw_offload", True),
                  ("r128 bf16 adamw", 128, "adamw", True),
                  ("r64 bf16 adamw", 64, "adamw", True),
                  ("r32 bf16 adamw", 32, "adamw", True)]
    picked, results = None, []
    for name, rank, optim, bf16 in candidates:
        P.OPTIM, P.RANK, P.ADAPTER_BF16 = optim, rank, bf16
        rec = {"config": name, "optim": optim, "rank": rank, "adapter_bf16": bf16}
        m = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            g0 = torch.cuda.memory_allocated()
            m = P.build_model()
            rec.update(P.adapter_stats(m))
            # where the bytes actually go, so a failure names its own cause instead of needing
            # arithmetic after the fact
            rec["weights_gb"] = round((torch.cuda.memory_allocated() - g0) / 1024 ** 3, 2)
            base = m.get_base_model()
            base.config.use_cache = False
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
            m.train()
            params = [p for p in m.parameters() if p.requires_grad]
            opt, _ = P.make_optimizer(params)
            from torch.nn.attention import sdpa_kernel, SDPBackend
            t0 = time.time()
            for ids in order:
                batch = {k: v.to("cuda") for k, v in col([{"input_ids": ids}]).items()}
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    loss = m(**batch).loss
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, P.GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
            rec["s_step"] = round((time.time() - t0) / len(order), 2)
            rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            rec["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            # Memory alone is not the test. Windows lets torch reserve past physical VRAM and spill
            # to host rather than raising, so a config can "fit" and still crawl. Require both.
            fits = rec["peak_reserved_gb"] <= 14.5
            fast = rec["s_step"] <= 15.0
            rec["verdict"] = "OK" if (fits and fast) else ("SPILL" if not fits else "SLOW")
            del opt
        except Exception as e:                                          # noqa: BLE001
            rec["verdict"] = "OOM" if P.is_memory_error(e) else "ERROR"
            rec["error"] = "%s %s" % (type(e).__name__, str(e).replace("\n", " ")[:120])
        finally:
            del m
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
        results.append(rec)
        print("  %-20s %-6s reserved %-6s alloc %-6s weights %-5s trainable %-7s s/step %-7s %s"
              % (name, rec["verdict"], rec.get("peak_reserved_gb", "-"),
                 rec.get("peak_alloc_gb", "-"), rec.get("weights_gb", "-"),
                 rec.get("trainable_m", "-"), rec.get("s_step", "-"),
                 rec.get("error", "")), flush=True)
        if picked is None and rec["verdict"] == "OK":
            picked = rec
    json.dump({"picked": picked, "all": results, "task": task},
              open(os.path.join(P.EXP, "smoke_memcfg.json"), "w"), indent=1)
    check("s2.a_config_fits", picked is not None,
          picked and "%s at %.2f GB reserved" % (picked["config"], picked["peak_reserved_gb"]))
    check("s2.headroom", picked is not None and picked["peak_reserved_gb"] <= 14.5,
          picked and "%.2f GB of 16" % picked["peak_reserved_gb"])
    check("s2.step_time_sane", picked is not None and picked["s_step"] <= 15.0,
          picked and "%.2f s/step, 128 steps is %.0f min per cell"
          % (picked["s_step"], 128 * picked["s_step"] / 60.0))
    check("s2.adapter_dtype_bf16", picked is not None and "bfloat16" in str(picked.get("param_dtype")),
          picked and picked.get("param_dtype"))
    if picked and picked["rank"] < 256:
        print("  NOTE: rank %d, below the deployed 256. State this as a limitation: a weaker adapter"
              " may have a different seed-correlation structure." % picked["rank"], flush=True)
    finish("s2")


# --------------------------------------------------------------------------- s3, s4

def _load_cfg(P):
    p = os.path.join(P.EXP, "smoke_memcfg.json")
    assert os.path.isfile(p), "run stage s2 first"
    cfg = json.load(open(p))["picked"]
    assert cfg, "s2 found no configuration that fits"
    P.OPTIM, P.RANK = cfg["optim"], cfg["rank"]
    return cfg


def stage_s3():
    import csv
    import rig_restart_probe as P
    cfg = _load_cfg(P)
    L = json.load(open(os.path.join(P.EXP, "smoke_lengths.json")))
    task = L["worst_task"]
    print("  config %s rank %d optim %s on worst task %s"
          % (cfg["config"], P.RANK, P.OPTIM, task), flush=True)
    r = _one_cell(P, task, 1, rank=P.RANK, maxseq=P.MAXSEQ, max_batches=None,
                  decode_budget=P.DECODE_BUDGET)
    s = r["stats"]
    check("s3.train_completed", not s["train_truncated"], "%d steps" % s["train_steps"])
    check("s3.reserved_under_15", r["peak_reserved"] < 15.0, "%.2f GB" % r["peak_reserved"])
    check("s3.decode_completed", not r["truncated"] and not r["overflow"])
    check("s3.every_output_graded", len(r["graded"]) > 0
          and all("n_cands" in v for v in r["graded"].values()), len(r["graded"]))
    check("s3.cache_fired", r["cache"]["shrinks"] > 0, r["cache"])
    rows = {x["task"]: x for x in csv.DictReader(open(
        os.path.join(P.REPO, "paper", "data", "task_timing.csv")))}
    kag = float(rows[task]["train_loss"]) if task in rows else None
    check("s3.loss_plausible_vs_kaggle", kag is None or s["train_loss_final"] < 10 * kag,
          "probe %.5f vs kaggle %.5f" % (s["train_loss_final"], kag or float("nan")))
    per_cell = r["train_s"] + r["decode_s"] + 40
    print("  projection: %.0fs per cell, 65 cells = %.1f hours"
          % (per_cell, 65 * per_cell / 3600.0), flush=True)
    check("s3.projection_under_48h", 65 * per_cell / 3600.0 < 48,
          "%.1f h" % (65 * per_cell / 3600.0))
    finish("s3")


def stage_s4():
    import rig_restart_probe as P
    _load_cfg(P)
    keys, _, _ = P.select_tasks(20)
    task = keys[0]
    peaks, steps = [], []
    for i in range(3):
        r = _one_cell(P, task, 1, rank=P.RANK, maxseq=P.MAXSEQ, max_batches=12,
                      decode_budget=300)
        peaks.append(r["peak_reserved"])
        steps.append(r["train_s"])
        print("  cell %d: reserved %.2f GB train %.0fs" % (i + 1, peaks[-1], steps[-1]), flush=True)
        del r
    drift = max(peaks) - min(peaks)
    check("s4.no_memory_drift", drift < 0.2, "%.3f GB across three cells" % drift)
    check("s4.no_slowdown", steps[-1] < 1.15 * steps[0],
          "cell3 %.0fs vs cell1 %.0fs" % (steps[-1], steps[0]))
    finish("s4")


if __name__ == "__main__":
    print("=== SMOKE %s ===" % STAGE.upper(), flush=True)
    {"s0": stage_s0, "s0b": stage_s0b, "s1a": stage_s1a, "s1b": stage_s1b,
     "s2": stage_s2, "s3": stage_s3, "s4": stage_s4}[STAGE]()
