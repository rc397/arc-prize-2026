"""Mask the training loss on cells the rule does not change. Ryan's idea, and the best of the batch.

Measured today: among the 67.5 percent of evaluation tasks whose transformation preserves grid
shape, 88 percent of a median output is a verbatim copy of its input. The collator supervises every
output token equally, so most of the gradient goes to reproducing cells nothing happened to. This
masks those positions and leaves supervision only where the rule acts.

It is a better-aimed idea than the background re-weighting cluster it came in with, because it
assumes nothing about which colour is background, and today's other measurement showed that
assumption to be wrong on most tasks anyway: colour 0 has a median share of zero.

The obvious way it fails is the reason for the third arm and for the second reading below. Removing
the supervision that teaches copying may produce an adapter that transforms correctly and corrupts
everything it was supposed to leave alone, which would show up as a pool full of near-misses rather
than as a lower loss.

  base       the deployed objective, every output token supervised.
  unchanged  labels set to -100 on output cells identical to the corresponding input cell, in
             demonstration pairs whose shapes match. Other pairs are untouched.
  random     labels set to -100 on a RANDOM subset of the same size, per span. This is the arm that
             separates "less supervision" from "supervision on the right cells", and without it a
             win means nothing: any masking reduces the token count and changes the effective
             learning rate.

Step budget is identical across arms, as in rig_auxobj. Training loss is NOT comparable across arms
because it averages over a different number of supervised tokens; the comparison is coverage and
the error split, not loss.

Two readings, both recorded:
  in_pool / selected   the usual coverage numbers.
  copy_err / chg_err   for the top-ranked candidate, the cell error rate split into cells the truth
                       copies from the test input and cells the truth changes. If masking works,
                       chg_err falls. If it corrupts copying, copy_err rises. Either is a result.

Alignment between the label positions and the grid cells is done at token level and is easy to get
silently wrong, so it is checkable without a GPU first:

  python rig_maskloss.py check          alignment only, CPU, prints spans found and cells masked
  python rig_maskloss.py [N=9] [DEC=600]  the experiment
Writes experiments/maskloss.jsonl, one record per (task, arm).
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
OUT = os.path.join(BASE, "experiments", "maskloss.jsonl")
STEPS = int(os.environ.get("ARC_MASK_STEPS", "32"))


class MaskingCollator:
    """Wraps the deployed completion-only collator and blanks selected label positions.

    The prompt is a flat alternation of user and assistant turns, one demonstration pair each:
    <|im_start|>user\\n GRID <|im_end|> <|im_start|>assistant\\n GRID <|im_end|>. The collator marks
    an assistant span from the assistant token plus two through the closing <|im_end|>, so within a
    span the digit tokens are exactly the output grid in row-major order and the newline tokens are
    its row breaks. The input grid is the preceding user span, read the same way.

    Nothing is masked unless both grids parse and their shapes match, and a row is left completely
    alone if masking would leave it with no supervised token at all, since a batch with zero targets
    trains on NaN.
    """

    def __init__(self, inner, tokenizer, mode, seed=0):
        v = tokenizer.get_vocab()
        self.inner = inner
        self.mode = mode
        self.user = v["user"]
        self.assistant = v["assistant"]
        self.eos = v["<|im_end|>"]
        self.nl = next(i for t, i in v.items() if tokenizer.convert_tokens_to_string([t]) == "\n")
        self.digit = {v[str(d)]: d for d in range(10)}
        self.rng = random.Random(seed)
        self.stats = {"rows": 0, "spans": 0, "aligned": 0, "cells": 0, "masked": 0, "skipped": 0}

    def _grid(self, ids, lo, hi):
        """Token span to (rows of values, rows of positions), or (None, None) if it is not a grid."""
        vals, poss, rv, rp = [], [], [], []
        for p in range(lo, hi):
            t = ids[p]
            if t in self.digit:
                rv.append(self.digit[t])
                rp.append(p)
            elif t == self.nl:
                if not rv:
                    return None, None
                vals.append(rv)
                poss.append(rp)
                rv, rp = [], []
            else:
                return None, None
        if rv:
            vals.append(rv)
            poss.append(rp)
        if not vals or any(len(r) != len(vals[0]) for r in vals):
            return None, None
        return vals, poss

    def _turns(self, ids):
        starts = [i for i, t in enumerate(ids) if t in (self.user, self.assistant)]
        ends = [i for i, t in enumerate(ids) if t == self.eos]
        return list(zip(starts, ends))

    def mask_row(self, ids, labels):
        turns = self._turns(ids)
        picks = []
        for j in range(1, len(turns), 2):
            self.stats["spans"] += 1
            (as_, ae) = turns[j]
            (us_, ue) = turns[j - 1]
            out_v, out_p = self._grid(ids, as_ + 2, ae)
            in_v, _ = self._grid(ids, us_ + 2, ue)
            if out_v is None or in_v is None:
                continue
            if len(in_v) != len(out_v) or len(in_v[0]) != len(out_v[0]):
                continue
            self.stats["aligned"] += 1
            same = [out_p[r][c] for r in range(len(out_v)) for c in range(len(out_v[0]))
                    if out_v[r][c] == in_v[r][c]]
            self.stats["cells"] += sum(len(r) for r in out_v)
            if self.mode == "random":
                flat = [p for row in out_p for p in row]
                same = self.rng.sample(flat, min(len(same), len(flat)))
            picks += same
        if not picks:
            return 0
        supervised = int((labels != -100).sum())
        if len(picks) >= supervised:            # never leave a batch with nothing to train on
            self.stats["skipped"] += 1
            return 0
        for p in picks:
            labels[p] = -100
        self.stats["masked"] += len(picks)
        return len(picks)

    def __call__(self, examples):
        batch = self.inner(examples)
        if self.mode == "base":
            return batch
        ids_all, lab_all = batch["input_ids"], batch["labels"]
        for i in range(ids_all.shape[0]):
            self.stats["rows"] += 1
            self.mask_row(ids_all[i].tolist(), lab_all[i])
        return batch


def split_error(pred, truth, test_in):
    """Cell error rate on the cells the truth copies from the input, and on the cells it changes."""
    if pred is None:
        return None, None
    p = [list(r) for r in pred]
    t = [list(r) for r in truth]
    x = [list(r) for r in test_in]
    if len(p) != len(t) or len(p[0]) != len(t[0]):
        return None, None
    if len(x) != len(t) or len(x[0]) != len(t[0]):
        return None, None
    copy_n = copy_e = chg_n = chg_e = 0
    for r in range(len(t)):
        for c in range(len(t[0])):
            if t[r][c] == x[r][c]:
                copy_n += 1
                copy_e += int(p[r][c] != t[r][c])
            else:
                chg_n += 1
                chg_e += int(p[r][c] != t[r][c])
    return (copy_e / copy_n if copy_n else None, chg_e / chg_n if chg_n else None)


def check():
    """CPU only: does the token-level alignment actually find the grids?"""
    import numpy as np
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    inner = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    keys, _, _ = P.select_tasks(6)
    enc = P.encoder_for(tok)

    for mode in ("unchanged", "random"):
        col = MaskingCollator(inner, tok, mode, seed=1)
        for task in keys:
            pz = arc.change_keys([task])
            rows = pz.augment(n=4, shfl_keys=True, seed=1).cut_to_len(
                formatter=fmt, name="text", max_len=P.MAXSEQ).as_list(fmt)
            for r in rows:
                b = {"input_ids": enc(r["text"])[:P.MAXSEQ]}
                batch = inner([b])
                before = int((batch["labels"] != -100).sum())
                col.stats["rows"] += 1
                col.mask_row(batch["input_ids"][0].tolist(), batch["labels"][0])
                after = int((batch["labels"] != -100).sum())
                assert after > 0, "a row lost every supervised token"
                assert after <= before
        s = col.stats
        print("%-10s rows %4d  assistant spans %5d  aligned %5d (%.0f%%)  cells %7d  "
              "masked %7d (%.0f%% of cells)  rows skipped %d"
              % (mode, s["rows"], s["spans"], s["aligned"],
                 100.0 * s["aligned"] / max(1, s["spans"]), s["cells"], s["masked"],
                 100.0 * s["masked"] / max(1, s["cells"]), s["skipped"]))
    print("")
    print("Alignment is the whole experiment: if 'aligned' is near zero the mask never fires, and")
    print("if 'masked' is near the 88 percent the grids say, the two arms differ by exactly the")
    print("thing they are meant to differ by.")
    return True


def main(n_tasks, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    inner = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    challenges = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8")}
    print("unchanged-cell loss masking on %d tasks, %d steps every arm, decode %ds"
          % (len(keys), STEPS, decode_s), flush=True)

    def run(task, arm):
        if (task, arm) in done:
            print("skip %s|%s" % (task, arm), flush=True)
            return
        pz = arc.change_keys([task])
        labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
        rows = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ).as_list(fmt)
        col = MaskingCollator(inner, tok, arm, seed=abs(hash(task)) % 10 ** 6)

        m = P.build_model()
        t0 = time.time()
        tr = P.train_one(m, tok, col, fmt, rows, deadline=t0 + P.TRAIN_BUDGET, max_batches=STEPS)
        train_s = time.time() - t0
        pool, trunc, over = P.decode_pool(m, tok, fmt, lifted, pz, time.time() + decode_s)
        graded = P.grade(pool, labels, sorted(labels))

        # the failure mode this experiment exists to catch, on the candidate the selector would pick
        errs = []
        for bk, g in graded.items():
            idx = int(bk.rsplit("_", 1)[1])
            cands = pool.get(bk, [])
            if not cands:
                continue
            best = min(cands, key=lambda c: c["beam_score"])["solution"]
            ce, xe = split_error(best.tolist(), labels[bk].tolist(),
                                 challenges[task]["test"][idx]["input"])
            if ce is not None:
                errs.append((ce, xe))
        rec = {"task": task, "arm": arm, "stratum": strat[task],
               "final_loss": round(tr["train_loss_final"], 5), "steps": tr["train_steps"],
               "masked": col.stats["masked"], "cells": col.stats["cells"],
               "aligned": col.stats["aligned"], "spans": col.stats["spans"],
               "in_pool": sum(1 for g in graded.values() if g["in_pool"]),
               "selected": sum(1 for g in graded.values() if g["selected_proxy"]),
               "outputs": len(graded), "n_cands": sum(len(v) for v in pool.values()),
               "copy_err": round(sum(e[0] for e in errs) / len(errs), 4) if errs else None,
               "chg_err": round(sum(e[1] for e in errs if e[1] is not None)
                                / max(1, sum(1 for e in errs if e[1] is not None)), 4)
               if errs else None,
               "shape_matched_outputs": len(errs),
               "truncated": bool(trunc or over), "train_s": round(train_s, 1)}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("%-10s %-10s loss %.5f  masked %d/%d  in_pool %d/%d  sel %d  copy_err %s  chg_err %s"
              % (task, arm, rec["final_loss"], rec["masked"], rec["cells"], rec["in_pool"],
                 rec["outputs"], rec["selected"], rec["copy_err"], rec["chg_err"]), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    for task in keys:
        for arm in ("base", "unchanged", "random"):
            try:
                run(task, arm)
            except Exception as e:
                print("FAIL %s|%s %s: %s" % (task, arm, type(e).__name__, e), flush=True)
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
    arms = ("base", "unchanged", "random")
    full = {t: d for t, d in by.items() if all(a in d for a in arms)}
    print("")
    print("%d tasks with all three arms (of %d started)" % (len(full), len(by)))
    if not full:
        return
    print("%-12s %8s %8s %8s %9s %9s" % ("arm", "in_pool", "sel", "cands", "copy_err", "chg_err"))
    print("-" * 58)
    for a in arms:
        ip = sum(full[t][a]["in_pool"] for t in full)
        se = sum(full[t][a]["selected"] for t in full)
        nc = sum(full[t][a]["n_cands"] for t in full)
        ce = [full[t][a]["copy_err"] for t in full if full[t][a]["copy_err"] is not None]
        xe = [full[t][a]["chg_err"] for t in full if full[t][a]["chg_err"] is not None]
        print("%-12s %8d %8d %8d %9s %9s"
              % (a, ip, se, nc,
                 "%.3f" % (sum(ce) / len(ce)) if ce else "-",
                 "%.3f" % (sum(xe) / len(xe)) if xe else "-"))
    out = sum(full[t]["base"]["outputs"] for t in full)
    print("")
    print("out of %d outputs. Paired sign counts, unchanged vs base:" % out)
    up = sum(1 for t in full if full[t]["unchanged"]["in_pool"] > full[t]["base"]["in_pool"])
    dn = sum(1 for t in full if full[t]["unchanged"]["in_pool"] < full[t]["base"]["in_pool"])
    print("  in_pool better on %d tasks, worse on %d, tied on %d" % (up, dn, len(full) - up - dn))
    rup = sum(1 for t in full if full[t]["random"]["in_pool"] > full[t]["base"]["in_pool"])
    rdn = sum(1 for t in full if full[t]["random"]["in_pool"] < full[t]["base"]["in_pool"])
    print("  random control better on %d, worse on %d, tied on %d" % (rup, rdn, len(full) - rup - rdn))
    print("")
    print("Read the control first. If unchanged and random move together, the effect is the token")
    print("count, not the choice of tokens, and the idea is not what produced it.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0 if check() else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 9,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 600) else 1)
