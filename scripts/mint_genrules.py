"""Mint a training corpus from the verified ARC-2 generators, for the SHAPE test.

Phase 0 varied how MANY rules the model saw and found nothing: 100 composed ARC-1 rules tied 402 of
them, 12 of 12 on ARC-AGI-2. It never varied what KIND of rule. Every rule we hold is ARC-1, either
a RE-ARC verifier or a pair of them composed, and none is ARC-2-shaped. This corpus is, by
construction, since each generator was written against an ARC-AGI-2 training task and verified by
reproducing that task's own demonstration pairs.

So the arms are matched on everything Phase 0 held fixed and differ only in shape:

    narrow (already trained)  8,000 examples, 100 composed ARC-1 rules
    this one                  8,000 examples, 100 verified ARC-2 rules

Rule count and example budget are identical, which is why 100 are sampled here rather than all 128.

EVERY GENERATOR RUNS IN ITS OWN PROCESS WITH A HARD TIMEOUT. These programs are model-written and at
least one of them contains a loop that does not terminate; running them in-process cost 20 hours of a
wedged interpreter earlier in this project before anyone noticed.

Usage: python mint_genrules.py [N_RULES] [N_EXAMPLES]
"""
import glob
import json
import multiprocessing as mp
import os
import random
import sys

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, BASE)

EXP = os.path.join(BASE, "experiments")
SKIP = ("genrules_deadrun_capped.jsonl", "genrules_qwen_local.jsonl", "genrules_think.jsonl")
POOL_N = 60          # generated pairs wanted per rule, on top of the task's own pairs
GEN_TIMEOUT = 25.0   # per generator, for the whole pool


def valid(g):
    if not isinstance(g, list) or not (1 <= len(g) <= 30):
        return False
    w = None
    for row in g:
        if not isinstance(row, list) or not (1 <= len(row) <= 30):
            return False
        if w is None:
            w = len(row)
        elif len(row) != w:
            return False
        for c in row:
            if not isinstance(c, int) or not (0 <= c <= 9):
                return False
    return True


def _pool_worker(q, src, n_want, seed):
    try:
        import os as _os
        import sys as _sys
        dn = open(_os.devnull, "w")
        _sys.stdout = dn
        _sys.stderr = dn
        ns = {}
        for alias, mod in (("np", "numpy"), ("numpy", "numpy"), ("math", "math"),
                           ("itertools", "itertools"), ("collections", "collections"),
                           ("re", "re"), ("copy", "copy"), ("functools", "functools"),
                           ("random", "random")):
            try:
                ns[alias] = __import__(mod)
            except Exception:
                pass
        exec(src, ns)
        gi, sv = ns.get("gen_input"), ns.get("solve")
        pairs, seen = [], set()
        for s in range(n_want * 3):
            if len(pairs) >= n_want:
                break
            try:
                g = [list(r) for r in gi(random.Random(seed + s))]
                if not valid(g):
                    continue
                key = repr(g)
                if key in seen:
                    continue
                o = [list(r) for r in sv([list(r) for r in g])]
                if not valid(o) or o == g:      # identity output teaches nothing
                    continue
                seen.add(key)
                pairs.append({"input": g, "output": o})
            except Exception:
                continue
        q.put(pairs)
    except Exception:
        q.put([])


def build_pool(src, n_want, seed, timeout=GEN_TIMEOUT):
    q = mp.Queue()
    p = mp.Process(target=_pool_worker, args=(q, src, n_want, seed))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return []
    try:
        return q.get(timeout=1.0)
    except Exception:
        return []


def load_generators():
    best = {}
    for path in glob.glob(os.path.join(EXP, "genrules_*.jsonl")):
        if os.path.basename(path) in SKIP:
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ok") and r.get("src") and r.get("sampler"):
                best[r["task"]] = r
    return best


def main():
    n_rules = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    import build_ft_corpus as B

    gens = load_generators()
    print("%d verified generators with a sampler" % len(gens), flush=True)
    keys = sorted(gens)
    random.Random(42).shuffle(keys)
    keys = keys[:n_rules]
    print("sampled %d to match the narrow arm's rule count" % len(keys), flush=True)

    ch = json.load(open(os.path.join(BASE, "data", "kaggle",
                                     "arc-agi_training_challenges.json"), encoding="utf-8"))
    pools, dead, total_pairs = {}, 0, 0
    for i, k in enumerate(keys, 1):
        own = [{"input": p["input"], "output": p["output"]} for p in ch[k]["train"]]
        fresh = build_pool(gens[k]["src"], POOL_N, seed=1000 + i)
        pool = own + fresh
        if len(pool) < 2:
            dead += 1
            continue
        pools["arc2gen_" + k] = pool
        total_pairs += len(pool)
        if i % 20 == 0:
            print("  %d/%d rules, %d pairs so far" % (i, len(keys), total_pairs), flush=True)
    print("%d usable pools (%d dead), %d pairs, mean %.1f per pool"
          % (len(pools), dead, total_pairs, total_pairs / max(1, len(pools))), flush=True)

    per_task = max(1, int(n_examples / max(1, len(pools)) * 1.4))    # mint dedupes, so overshoot
    ex = B.mint(pools, per_task=per_task, kmax=6, max_ctx_tokens=1600, seed=7)
    ex = ex[:n_examples]
    print("minted %d examples" % len(ex), flush=True)

    out = os.path.join(BASE, "corpus_arc2shape")
    os.makedirs(out, exist_ok=True)
    n_val = max(1, int(len(ex) * 0.02))
    with open(os.path.join(out, "val.jsonl"), "w", encoding="utf-8") as f:
        for e in ex[:n_val]:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as f:
        for e in ex[n_val:]:
            f.write(json.dumps(e) + "\n")
    print("wrote %s: %d train, %d val" % (out, len(ex) - n_val, n_val), flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
