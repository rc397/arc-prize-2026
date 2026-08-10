"""Make the model declare the output's dimensions before drawing it. A deliberate Gate 3 violation.

Gate 3 says do not fight the 16-token vocabulary: the base was pretrained to emit digit grids and
nothing else, and every proposal that invents syntax has to teach it from four demonstrations during
test-time training. It is also the deepest constraint in the system and the one nothing we have
tried has attacked, which is why this is priced rather than dismissed.

Of the four vocabulary-breaking ideas in Ryan's batch this is the cheapest and the best motivated.
The decoder currently has no way to know how tall the output is until it has finished emitting it,
so every branch of the search is committing to a shape implicitly, one row at a time. A prefix makes
that commitment explicit and checkable.

The syntax stays inside the vocabulary, which is what makes it affordable: four digits and a
newline, zero-padded, so 12 rows by 7 columns is "1207". No new token, no space character (which the
encoder cannot represent), five extra tokens per output grid.

  base   the deployed format.
  dim    every assistant turn is "HHWW\\n" then the grid. The parser reads the prefix and REJECTS a
         candidate whose grid does not match its own declaration, so a branch that loses track of
         its shape is discarded rather than submitted.

The rejection is the mechanism worth measuring and also the way this can lose: it is a filter, and
every filter measured so far kills truths at about the rate it kills wrongs. The difference is that
this one compares the model against its own stated intention rather than against a mined invariant,
so it can only kill candidates that are internally inconsistent.

Expect it to fail. The base has never seen a leading dimension line, and one bad prefix corrupts a
whole grid. What it buys either way is a number for the cost of teaching this base new in-vocabulary
syntax during TTT, which every other idea in that family also has to pay.

Usage: python rig_dimprefix.py [N_TASKS=6] [DECODE_S=600]
Writes experiments/dimprefix.jsonl, one record per (task, arm).
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
OUT = os.path.join(BASE, "experiments", "dimprefix.jsonl")
STEPS = int(os.environ.get("ARC_DIM_STEPS", "32"))
STATS = {"parsed": 0, "rejected": 0, "unparseable": 0}


def make_formatter(QwenFormatter, convert_grid_to_string, is_valid_solution):
    import numpy as np

    class DimFormatter(QwenFormatter):
        """The deployed formatter with a zero-padded HHWW line in front of every output grid."""

        @staticmethod
        def _prefix(grid):
            g = np.asarray(grid)
            return "%02d%02d\n" % (g.shape[0], g.shape[1])

        def fmt_reply(self, reply):
            return self._prefix(reply[0]) + convert_grid_to_string(reply[0]) + "<|im_end|>"

        def fmt_train(self, train, last_is_challenge=False):
            test = None
            if last_is_challenge:
                test = train[-1]
                train = train[:-1]
            text = ""
            for x in train:
                text += ("<|im_start|>user\n" + convert_grid_to_string(x["input"])
                         + "<|im_end|><|im_start|>assistant\n" + self._prefix(x["output"])
                         + convert_grid_to_string(x["output"]) + "<|im_end|>")
            if test is not None:
                text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
            return text

        def convert_tokens_to_array(self, tokens, limit_rows=30):
            """Parse HHWW then the grid, and refuse anything that contradicts its own header."""
            if len(tokens) < 2:
                return None
            text = self.tokenizer.decode(tokens[:-1])
            lines = text.strip().split("\n")
            if len(lines) < 2 or len(lines[0]) != 4 or not lines[0].isdigit():
                STATS["unparseable"] += 1
                return None
            h, w = int(lines[0][:2]), int(lines[0][2:])
            try:
                by_rows = [row for row in
                           [[int(x) for x in line if x.isdigit()] for line in lines[1:]] if row]
                if len(by_rows) > limit_rows:
                    by_rows = by_rows[:limit_rows]
                array = np.array(by_rows, dtype=int)
            except Exception:
                STATS["unparseable"] += 1
                return None
            if not is_valid_solution(array):
                STATS["unparseable"] += 1
                return None
            if array.shape != (h, w):
                STATS["rejected"] += 1
                return None
            STATS["parsed"] += 1
            return array

    return DimFormatter


def check():
    """CPU only: does the new syntax encode in a 16-token vocabulary and parse back exactly?

    The two ways this dies before it ever reaches the GPU are an unencodable character (the vocab
    has no space, so the header had to be zero-padded rather than separated) and a parser that
    disagrees with the writer. Both are checkable here for nothing.
    """
    import numpy as np
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter, convert_grid_to_string, is_valid_solution
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    DimFormatter = make_formatter(QwenFormatter, convert_grid_to_string, is_valid_solution)
    fmt = DimFormatter(tokenizer=tok)
    enc = P.encoder_for(tok)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    keys, _, _ = P.select_tasks(6)

    v = tok.get_vocab()
    eos = v["<|im_end|>"]
    n_ok = n_grid = 0
    for task in keys:
        pz = arc.change_keys([task])
        rows = pz.augment(n=2, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ).as_list(fmt)
        for r in rows:
            ids = enc(r["text"])          # raises on any character outside the 16-token vocabulary
            n_ok += 1
        # round trip: every output grid must survive fmt_reply then convert_tokens_to_array
        for x in arc.queries[task]["train"]:
            g = np.array(x["output"])
            toks = enc(fmt.fmt_reply([g]))
            assert toks[-1] == eos
            back = fmt.convert_tokens_to_array(toks)
            assert back is not None and np.array_equal(back, g), "round trip failed on %s" % task
            n_grid += 1
    print("%d rows encoded with no unencodable character" % n_ok)
    print("%d output grids round tripped through the header exactly" % n_grid)
    print("parser stats: parsed %d, rejected on shape %d, unparseable %d"
          % (STATS["parsed"], STATS["rejected"], STATS["unparseable"]))
    print("")
    print("Prompt cost: %d extra tokens per output grid, which is the whole price of this syntax."
          % 5)
    return True


def main(n_tasks, decode_s):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter, convert_grid_to_string, is_valid_solution
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    DimFormatter = make_formatter(QwenFormatter, convert_grid_to_string, is_valid_solution)
    fmts = {"base": QwenFormatter(tokenizer=tok), "dim": DimFormatter(tokenizer=tok)}
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8")}
    print("dimension prefix on %d tasks, %d steps every arm, decode %ds"
          % (len(keys), STEPS, decode_s), flush=True)

    def run(task, arm):
        if (task, arm) in done:
            print("skip %s|%s" % (task, arm), flush=True)
            return
        fmt = fmts[arm]
        pz = arc.change_keys([task])
        labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
        rows = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        for k in STATS:
            STATS[k] = 0
        m = P.build_model()
        t0 = time.time()
        tr = P.train_one(m, tok, collator, fmt, rows,
                         deadline=t0 + P.TRAIN_BUDGET, max_batches=STEPS)
        train_s = time.time() - t0
        pool, trunc, over = P.decode_pool(m, tok, fmt, lifted, pz, time.time() + decode_s)
        graded = P.grade(pool, labels, sorted(labels))
        rec = {"task": task, "arm": arm, "stratum": strat[task],
               "final_loss": round(tr["train_loss_final"], 5), "steps": tr["train_steps"],
               "in_pool": sum(1 for g in graded.values() if g["in_pool"]),
               "selected": sum(1 for g in graded.values() if g["selected_proxy"]),
               "outputs": len(graded), "n_cands": sum(len(v) for v in pool.values()),
               "parsed": STATS["parsed"], "rejected": STATS["rejected"],
               "unparseable": STATS["unparseable"],
               "truncated": bool(trunc or over), "train_s": round(train_s, 1)}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("%-10s %-6s loss %.5f  in_pool %d/%d  sel %d  cands %d  parsed %d rejected %d bad %d"
              % (task, arm, rec["final_loss"], rec["in_pool"], rec["outputs"], rec["selected"],
                 rec["n_cands"], rec["parsed"], rec["rejected"], rec["unparseable"]), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    for task in keys:
        for arm in ("base", "dim"):
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
    full = {t: d for t, d in by.items() if "base" in d and "dim" in d}
    print("")
    print("%d tasks with both arms (of %d started)" % (len(full), len(by)))
    if not full:
        return
    print("%-8s %8s %6s %8s %8s %9s %7s" % ("arm", "in_pool", "sel", "cands", "parsed", "rejected",
                                            "bad"))
    print("-" * 60)
    for a in ("base", "dim"):
        print("%-8s %8d %6d %8d %8d %9d %7d"
              % (a, sum(full[t][a]["in_pool"] for t in full),
                 sum(full[t][a]["selected"] for t in full),
                 sum(full[t][a]["n_cands"] for t in full),
                 sum(full[t][a].get("parsed", 0) for t in full),
                 sum(full[t][a].get("rejected", 0) for t in full),
                 sum(full[t][a].get("unparseable", 0) for t in full)))
    up = sum(1 for t in full if full[t]["dim"]["in_pool"] > full[t]["base"]["in_pool"])
    dn = sum(1 for t in full if full[t]["dim"]["in_pool"] < full[t]["base"]["in_pool"])
    print("")
    print("out of %d outputs. dim better on %d tasks, worse on %d, tied on %d"
          % (sum(full[t]["base"]["outputs"] for t in full), up, dn, len(full) - up - dn))
    bad = sum(full[t]["dim"].get("unparseable", 0) for t in full)
    par = sum(full[t]["dim"].get("parsed", 0) for t in full)
    if par + bad:
        print("")
        print("Syntax cost: %d of %d completions in the dim arm failed to produce a well formed"
              % (bad, par + bad))
        print("header at all. That number is the price of teaching this base new in-vocabulary")
        print("syntax during TTT, and every other proposal in that family pays it too.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0 if check() else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 600) else 1)
