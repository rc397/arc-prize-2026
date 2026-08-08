"""Is another task's adapter worth starting from? The gate that decides the meta-initialisation idea.

The retrieval half of the cross-task proposal now has support: an embedding of a task's
demonstrations separates same-rule from different-rule instances at AUC 0.88 against 0.68 for a
surface-statistics baseline. That says similar tasks can be FOUND. It says nothing about whether
finding them is worth anything, which is the claim the whole design rests on.

So this measures the claim directly, and skips retrieval and fusion entirely. If starting a task's
test-time training from another adapter does not beat starting from zero, then no retrieval rule and
no fusion scheme can rescue it, and the terabyte adapter bank and the online SVDs never need
building.

RE-ARC makes this a strong kill test rather than a weak one. Instances from one generator share their
transformation exactly, so a same-generator warm start is the most favourable case that will ever
exist. Real retrieval would return a merely SIMILAR task and could only do worse. A null here
therefore closes the idea; a positive here is necessary but still not sufficient, and would need
re-testing across genuinely different tasks.

Three arms per task, on identical data and an identical step budget:

  cold      the deployed configuration, adapter initialised as usual.
  warm_same initialised from an adapter trained on a DIFFERENT INSTANCE OF THE SAME GENERATOR.
  warm_diff initialised from an adapter trained on an instance of ANOTHER generator.

The third arm is not optional and it is the same reasoning that made the pre-fitting result readable.
Any warm start changes the optimisation landscape, so cold-versus-warm alone cannot separate "a
related task's structure helps" from "any non-zero initialisation helps". Reading: warm_same above
cold with warm_diff at cold means transfer follows the rule, which is the claim. Both above cold
means the benefit is warm-starting in general and has nothing to do with retrieval, which kills the
retrieval half while leaving a cheaper lever. Neither above cold closes the route.

Measured per arm: final training loss, loss after a fixed small number of steps (the quantity that
matters, since the budget is 240 seconds and convergence speed is the proposed benefit), and
truth-in-pool from the deployed decoder.

Usage: python rig_adapterxfer.py [N_GENERATORS=6] [DECODE_S=600] [PREFIT_DONORS=1]
Writes experiments/adapterxfer.jsonl.
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
POOLS = os.path.join(BASE, "experiments", "rearc_pools", "tasks")
OUT = os.path.join(BASE, "experiments", "adapterxfer.jsonl")
DECODE_S = int(os.environ.get("ARC_XFER_DECODE", "600"))
N_DEMO = 4
EARLY_STEPS = 16          # loss here is the convergence-speed claim, not the final loss


def build_tasks(n_gen, rng):
    """Two instances per generator: one donates an adapter, one is solved. Same rule, disjoint pairs."""
    files = sorted(f for f in os.listdir(POOLS) if f.endswith(".json"))
    rng.shuffle(files)
    out = []
    for fn in files:
        pairs = json.load(open(os.path.join(POOLS, fn), encoding="utf-8"))
        if len(pairs) < 3 * (N_DEMO + 1):
            continue
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        take = lambda s: [pairs[j] for j in idx[s:s + N_DEMO + 1]]
        target, donor = take(0), take(N_DEMO + 1)
        out.append({"gen": fn[:-5],
                    "target": {"train": target[:N_DEMO], "test": [target[N_DEMO]]},
                    "donor": {"train": donor[:N_DEMO], "test": [donor[N_DEMO]]}})
        if len(out) >= n_gen:
            break
    return out


def as_arc(P, name, spec, tmpdir):
    """Wrap a {train, test} spec as an ArcDataset the deployed code accepts, plus its answer."""
    from arc_loader import ArcDataset
    ch = {name: {"train": spec["train"],
                 "test": [{"input": spec["test"][0]["input"]}]}}
    p = os.path.join(tmpdir, "%s.json" % name)
    json.dump(ch, open(p, "w"))
    return ArcDataset.from_file(p), [spec["test"][0]["output"]]


def lora_state(m):
    import torch
    return {k: v.detach().clone() for k, v in m.named_parameters() if "lora_" in k}


def load_lora(m, state):
    import torch
    with torch.no_grad():
        for k, v in m.named_parameters():
            if k in state:
                v.copy_(state[k])


def main(n_gen, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import QwenFormatter
    from transformers import AutoTokenizer

    rng = random.Random(11)
    tasks = build_tasks(n_gen, rng)
    # two is the mechanical minimum, since warm_diff only needs one generator that is not this one.
    # A run worth reporting needs six or more; this bound exists so a smoke test can be small.
    assert len(tasks) >= 2, ("need at least two generators with pools of %d pairs or more, found %d"
                             % (3 * (N_DEMO + 1), len(tasks)))
    tmpdir = os.path.join(BASE, "_xfer")
    os.makedirs(tmpdir, exist_ok=True)

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)

    done = set()
    if os.path.isfile(OUT):
        done = {json.loads(l)["gen"] for l in open(OUT, encoding="utf-8")}
    print("adapter transfer over %d generators, decode %ds, early loss at %d steps"
          % (len(tasks), decode_s, EARLY_STEPS), flush=True)

    def train_ds_for(arc_ds, key):
        return arc_ds.change_keys([key]).augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)

    # donor adapters: one per generator, trained on the donor instance of that generator
    donors = {}
    for t in tasks:
        gname = "donor_" + t["gen"]
        ds, _ = as_arc(P, gname, t["donor"], tmpdir)
        m = P.build_model()
        P.train_one(m, tok, collator, fmt, train_ds_for(ds, gname),
                    deadline=time.time() + P.TRAIN_BUDGET)
        donors[t["gen"]] = lora_state(m)
        print("  donor adapter trained for %s" % t["gen"], flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    gens = [t["gen"] for t in tasks]
    for ti, t in enumerate(tasks):
        if t["gen"] in done:
            continue
        tname = "target_" + t["gen"]
        ds, answer = as_arc(P, tname, t["target"], tmpdir)
        labels = {"%s_0" % tname: np.array(answer[0])}
        other = gens[(ti + 1) % len(gens)]
        rec = {"gen": t["gen"], "donor_same": t["gen"], "donor_diff": other}

        for arm in ("cold", "warm_same", "warm_diff"):
            m = P.build_model()
            if arm == "warm_same":
                load_lora(m, donors[t["gen"]])
            elif arm == "warm_diff":
                load_lora(m, donors[other])
            tds = train_ds_for(ds, tname)
            t0 = time.time()
            # ONE training run, exactly as deployed, and the early loss is read off its own curve.
            # An earlier version called train_one twice, once capped at EARLY_STEPS and once to the
            # full budget, which restarts the optimizer and the LR schedule and makes the second a
            # fresh warmup rather than a continuation.
            full = P.train_one(m, tok, collator, fmt, tds, deadline=time.time() + P.TRAIN_BUDGET)
            curve = full.get("losses") or []
            early = {"train_loss_last10": (sum(curve[:EARLY_STEPS]) / len(curve[:EARLY_STEPS]))
                     if curve else 0.0}
            train_s = time.time() - t0
            base = m.get_base_model()
            base.config.use_cache = True
            base.gradient_checkpointing_disable()
            m.eval()
            pool, trunc, _ = P.decode_pool(m, tok, fmt, lifted, ds.change_keys([tname]),
                                           time.time() + decode_s)
            graded = P.grade(pool, labels, sorted(labels))
            rec[arm] = {"early_loss": round(float(early.get("train_loss_last10", 0.0)), 5),
                        "final_loss": round(float(full.get("train_loss_last10", 0.0)), 5),
                        "cands": sum(len(v) for v in pool.values()),
                        "hits": sum(1 for v in graded.values() if v["in_pool"]),
                        "outputs": len(graded), "truncated": trunc,
                        "train_s": round(train_s, 1)}
            print("    %-10s %-9s early %.5f  final %.5f  %3d cands  hit %d/%d"
                  % (t["gen"], arm, rec[arm]["early_loss"], rec[arm]["final_loss"],
                     rec[arm]["cands"], rec[arm]["hits"], rec[arm]["outputs"]), flush=True)
            del m
            gc.collect()
            torch.cuda.empty_cache()

        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    summarize()
    return True


def summarize():
    import statistics as st
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    if not rows:
        print("no rows")
        return
    arms = ("cold", "warm_same", "warm_diff")
    print("\n%d generators" % len(rows))
    print("%-11s %10s %10s %8s %8s" % ("arm", "early_loss", "final_loss", "hits", "cands"))
    agg = {}
    for a in arms:
        e = st.mean(r[a]["early_loss"] for r in rows)
        f = st.mean(r[a]["final_loss"] for r in rows)
        h = sum(r[a]["hits"] for r in rows)
        c = sum(r[a]["cands"] for r in rows)
        agg[a] = (e, f, h, c)
        print("%-11s %10.5f %10.5f %8d %8d" % (a, e, f, h, c))
    n_out = sum(r["cold"]["outputs"] for r in rows)
    print("(%d outputs total)" % n_out)

    ce, _, ch, cc = agg["cold"]
    se, _, sh, sc = agg["warm_same"]
    de, _, dh, dc = agg["warm_diff"]

    # A verdict needs something to have been decoded. Zero candidates everywhere means the decode
    # budget or the harness failed, not that warm starts do nothing, and this summariser printed
    # "the route is closed" on exactly that once already.
    if cc + sc + dc == 0:
        print("\nNO VERDICT")
        print("  Every arm produced zero candidates, so the coverage comparison is empty. That is a")
        print("  budget or harness failure, not a result about warm starts. Raise the decode budget")
        print("  and re-run before reading anything into the loss columns.")
        return
    if len(rows) < 5:
        print("\nNO VERDICT")
        print("  Only %d generators. Too few to separate three arms; run six or more." % len(rows))
        return

    print("\nVERDICT")
    faster_same = se < ce * 0.9
    faster_diff = de < ce * 0.9
    if sh > ch and dh <= ch:
        print("  Warm-starting from the SAME rule adds coverage (%d vs %d) while an unrelated" % (sh, ch))
        print("  adapter does not (%d). Transfer follows the rule, which is the claim the" % dh)
        print("  retrieval design needs. Next question is whether it survives merely-similar tasks.")
    elif sh > ch and dh > ch:
        print("  Both warm starts add coverage (%d and %d against %d), so the benefit is" % (sh, dh, ch))
        print("  warm-starting in general and not relatedness. That kills retrieval and the")
        print("  adapter bank, and leaves a much cheaper lever: one fixed initialisation.")
    elif faster_same and not faster_diff:
        print("  No coverage gain, but same-rule warm starts converge faster (%.5f vs %.5f at" % (se, ce))
        print("  %d steps) and unrelated ones do not (%.5f). Worth something only if the budget" % (EARLY_STEPS, de))
        print("  binds on training rather than decode, which our timing log says it does not.")
    else:
        print("  No coverage gain and no convergence advantage. Starting from another task's")
        print("  adapter is worth nothing even when the rule is IDENTICAL, which is the most")
        print("  favourable case retrieval could ever return. The cross-task transfer route is")
        print("  closed, and neither the adapter bank nor the fusion maths needs building.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
                       int(sys.argv[2]) if len(sys.argv) > 2 else DECODE_S) else 1)
