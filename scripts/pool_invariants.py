"""Lever A2: mined-invariant candidate filtering, measured offline against the dumped pool.

Each invariant is a predicate on (test input, candidate grid) mined from the task's own train
pairs and kept only if it held on every pair and the task has at least 2 pairs. A candidate
violating any kept invariant is killed. The filter reads the task, not the model's beliefs, so
the Appendix-F selection ceiling does not bind it. Measured in this order:
(a) false-kill rate on the 19 in-pool truths, the kill condition; any invariant type that
    false-kills is dropped and both before/after numbers are reported,
(b) the 4 leaked outputs (truth in pool, kgmon top-2 misses): blockers killed, truths recovered,
(c) kgmon top-2 attainment before vs after filtering over all 60 outputs, must not decrease,
(d) truth-absent outputs whose entire pool dies, a fallback-routing count only.
Invariant PARAMETERS are mined per task from train pairs and never cross tasks. The one
cross-task fitted choice is which invariant TYPES survive pruning, so that choice is also
scored leave-one-task-out: prune on the other tasks' false kills, evaluate on the held-out task.
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = r"C:\Users\rchen\AppData\Local\Temp\claude\D--ARC-AGI-2-3\8cbe1c19-90e9-462d-be4f-d7a77df62f8c\scratchpad"
POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "paper", "data", "pool_dump.json")
CHAL = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRATCH, "arc-agi_evaluation_challenges.json")
SOLS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(SCRATCH, "arc-agi_evaluation_solutions.json")


def kgmon(c):
    return len(c["beam"]) - float(np.mean([np.mean(a) for a in c["aug"]]))


def palette(a):
    return frozenset(np.unique(a).tolist())


def counts(a):
    return np.bincount(a.ravel(), minlength=10)


def bg(a):
    return int(counts(a).argmax())


def n_components(a):
    b = bg(a)
    h, w = a.shape
    seen = np.zeros((h, w), bool)
    n = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or a[i, j] == b:
                continue
            n += 1
            col = a[i, j]
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and a[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return n


def mine(pairs):
    """Return [(type_name, predicate)] where predicate(test_input, candidate) -> bool (True = pass).
    Every invariant emitted held on all train pairs; callers guarantee len(pairs) >= 2."""
    inv = []
    outs = [o for _, o in pairs]

    if all(o.shape == i.shape for i, o in pairs):
        inv.append(("shape_eq_input", lambda t, c: c.shape == t.shape))
    if len({o.shape for o in outs}) == 1:
        s = outs[0].shape
        inv.append(("shape_fixed", lambda t, c, s=s: c.shape == s))
    fh = {Fraction(o.shape[0], i.shape[0]) for i, o in pairs}
    fw = {Fraction(o.shape[1], i.shape[1]) for i, o in pairs}
    if len(fh) == 1 and len(fw) == 1 and (fh, fw) != ({Fraction(1)}, {Fraction(1)}):
        rh, rw = next(iter(fh)), next(iter(fw))

        def scaled(t, c, rh=rh, rw=rw):
            eh, ew = t.shape[0] * rh, t.shape[1] * rw
            if eh.denominator != 1 or ew.denominator != 1:
                return True
            return c.shape == (int(eh), int(ew))
        inv.append(("shape_scaled", scaled))

    if all(palette(o) <= palette(i) for i, o in pairs):
        inv.append(("palette_subset", lambda t, c: palette(c) <= palette(t)))
    if len({palette(o) for o in outs}) == 1:
        p = palette(outs[0])
        inv.append(("palette_fixed", lambda t, c, p=p: palette(c) == p))
    add = frozenset().union(*(palette(o) - palette(i) for i, o in pairs))
    if add and all(palette(o) <= palette(i) | add for i, o in pairs):
        inv.append(("palette_additive", lambda t, c, add=add: palette(c) <= palette(t) | add))

    seen_cols = sorted(frozenset().union(*(palette(i) | palette(o) for i, o in pairs)))
    cons = [k for k in seen_cols if all(counts(i)[k] == counts(o)[k] for i, o in pairs)]
    if cons:
        inv.append(("colour_count_conserved",
                    lambda t, c, cons=tuple(cons): all(counts(c)[k] == counts(t)[k] for k in cons)))

    if all(bg(o) == bg(i) for i, o in pairs):
        inv.append(("background_preserved", lambda t, c: bg(c) == bg(t)))

    nc = [(n_components(i), n_components(o)) for i, o in pairs]
    if all(a == b for a, b in nc):
        inv.append(("components_eq_input", lambda t, c: n_components(c) == n_components(t)))
    if len({b for _, b in nc}) == 1:
        k = nc[0][1]
        inv.append(("components_fixed", lambda t, c, k=k: n_components(c) == k))

    if all(o.shape == i.shape for i, o in pairs):
        imm = [k for k in sorted(frozenset().union(*(palette(i) for i, _ in pairs)))
               if all(np.all(o[i == k] == k) for i, o in pairs)]
        if imm:
            def immutable(t, c, imm=tuple(imm)):
                if c.shape != t.shape:
                    return True
                m = np.isin(t, imm)
                return bool(np.all(c[m] == t[m]))
            inv.append(("immutable_colours", immutable))

    if len({o.shape[0] % 2 for o in outs}) == 1:
        p = outs[0].shape[0] % 2
        inv.append(("row_parity", lambda t, c, p=p: c.shape[0] % 2 == p))
    if len({o.shape[1] % 2 for o in outs}) == 1:
        p = outs[0].shape[1] % 2
        inv.append(("col_parity", lambda t, c, p=p: c.shape[1] % 2 == p))
    if all(np.array_equal(o, o[:, ::-1]) for o in outs):
        inv.append(("mirror_lr", lambda t, c: np.array_equal(c, c[:, ::-1])))
    if all(np.array_equal(o, o[::-1]) for o in outs):
        inv.append(("mirror_ud", lambda t, c: np.array_equal(c, c[::-1])))
    return inv


def evaluate(pool, chal, invs, drop=frozenset()):
    """Per output key: which invariant types each candidate violates, after dropping types in drop."""
    rec = {}
    for bk, cands in pool.items():
        task, idx = bk.rsplit("_", 1)
        t_in = np.array(chal[task]["test"][int(idx)]["input"])
        rows = []
        for c in cands:
            g = np.array(c["grid"])
            viol = [n for n, pred in invs[task] if n not in drop and not pred(t_in, g)]
            rows.append((c, viol))
        rec[bk] = rows
    return rec


def attained_after(rec):
    n = 0
    for rows in rec.values():
        alive = [c for c, viol in rows if not viol]
        n += int(any(c["correct"] for c in sorted(alive, key=kgmon, reverse=True)[:2]))
    return n


def false_kills(rec):
    """(per-type counter of correct-grid kills, list of (key, violated types)) over in-pool truths."""
    per_type, killed = {}, []
    for bk, rows in rec.items():
        for c, viol in rows:
            if c["correct"] and viol:
                killed.append((bk, viol))
                for v in viol:
                    per_type[v] = per_type.get(v, 0) + 1
    return per_type, killed


def main():
    pool = json.load(open(POOL))
    chal = json.load(open(CHAL))
    sols = json.load(open(SOLS))
    for bk, cands in pool.items():
        task, idx = bk.rsplit("_", 1)
        tg = sols[task][int(idx)]
        assert all(c["correct"] == (c["grid"] == tg) for c in cands), bk
    tasks = sorted({k.rsplit("_", 1)[0] for k in pool})

    invs = {}
    for t in tasks:
        pairs = [(np.array(p["input"]), np.array(p["output"])) for p in chal[t]["train"]]
        invs[t] = mine(pairs) if len(pairs) >= 2 else []
    n_inv = [len(invs[t]) for t in tasks]
    print("tasks: %d   invariants mined: %d   per task min/median/max: %d/%d/%d   tasks with 0: %d"
          % (len(tasks), sum(n_inv), min(n_inv), int(np.median(n_inv)), max(n_inv),
             sum(1 for n in n_inv if n == 0)))

    truth_in = sorted(bk for bk, cands in pool.items() if any(c["correct"] for c in cands))
    absent = sorted(bk for bk in pool if bk not in truth_in)
    before = {bk: sorted(pool[bk], key=kgmon, reverse=True) for bk in pool}
    attained_pre = sum(1 for bk in pool if any(c["correct"] for c in before[bk][:2]))
    leaked = [bk for bk in truth_in if not any(c["correct"] for c in before[bk][:2])]
    print("outputs: %d   truth in pool: %d   kgmon top-2 attained: %d   leaked: %s"
          % (len(pool), len(truth_in), attained_pre, " ".join(leaked)))

    print("\n[a] FALSE-KILL RATE (kill condition: correct grids killed on the %d in-pool truths)" % len(truth_in))
    rec_all = evaluate(pool, chal, invs)
    fk_type, fk_keys = false_kills(rec_all)
    wrong_kills = {}
    for rows in rec_all.values():
        for c, viol in rows:
            if not c["correct"]:
                for v in viol:
                    wrong_kills[v] = wrong_kills.get(v, 0) + 1
    mined_per_type = {}
    for t in tasks:
        for n, _ in invs[t]:
            mined_per_type[n] = mined_per_type.get(n, 0) + 1
    print("%-24s %6s %11s %11s" % ("invariant", "tasks", "wrong-kills", "false-kills"))
    for n in sorted(mined_per_type):
        print("%-24s %6d %11d %11d" % (n, mined_per_type[n], wrong_kills.get(n, 0), fk_type.get(n, 0)))
    for bk, viol in fk_keys:
        print("   FALSE KILL %-14s violates %s" % (bk, ",".join(viol)))
    bad = frozenset(fk_type)
    print("false-killed correct grids BEFORE pruning: %d of %d   attained after filter: %d"
          % (len(fk_keys), len(truth_in), attained_after(rec_all)))
    print("pruned invariant types: %s" % (", ".join(sorted(bad)) if bad else "none"))

    rec = evaluate(pool, chal, invs, drop=bad)
    _, fk_after = false_kills(rec)
    print("false-killed correct grids AFTER pruning: %d of %d" % (len(fk_after), len(truth_in)))

    print("\n[b] LEAKED OUTPUTS (payoff: truth in pool, kgmon top-2 missed it)")
    recovered = []
    for bk in leaked:
        rows = rec[bk]
        order = before[bk]
        tpos = next(i for i, c in enumerate(order) if c["correct"])
        blockers = order[:tpos]
        kill_map = {id(c): bool(viol) for c, viol in rows}
        nk = sum(1 for c in blockers if kill_map[id(c)])
        alive = [c for c, viol in rows if not viol]
        now = any(c["correct"] for c in sorted(alive, key=kgmon, reverse=True)[:2])
        recovered.append((bk, now))
        print("   %-14s truth rank %2d, blockers %2d, blockers killed %2d, truth top-2 after: %s"
              % (bk, tpos + 1, len(blockers), nk, "YES" if now else "no"))
    n_rec = sum(1 for _, now in recovered if now)

    print("\n[c] OVERALL kgmon top-2: before %d, after filter %d (of %d)"
          % (attained_pre, attained_after(rec), len(pool)))

    wiped = sum(1 for bk in absent if all(viol for _, viol in rec[bk]))
    print("\n[d] truth-absent outputs with ENTIRE pool killed (fallback-routable): %d of %d"
          % (wiped, len(absent)))

    # Leave-one-task-out applies only to the pruning choice. The invariants themselves are mined
    # per task from that task's train pairs, so no information from any other task (or from this
    # 60-output slice) enters them; that is why this lever differs in kind from selector weight
    # refitting, which fit shared weights on the slice and collapsed under LOO (14 vs 15). The
    # residual cross-task fit is WHICH types get pruned, so LOO re-prunes per fold: drop types
    # that false-kill on the other tasks, then score the held-out task with that pruned set.
    loo_att = 0
    loo_fk = 0
    for t in tasks:
        fk_other = {v for bk, viol in fk_keys if bk.rsplit("_", 1)[0] != t for v in viol}
        keys_t = [bk for bk in pool if bk.rsplit("_", 1)[0] == t]
        rec_t = evaluate({bk: pool[bk] for bk in keys_t}, chal, invs, drop=frozenset(fk_other))
        loo_att += attained_after(rec_t)
        loo_fk += len(false_kills(rec_t)[1])
    print("\n[LOO] leave-one-task-out over the pruning choice: attained %d (in-sample after %d, "
          "kgmon baseline %d), false kills %d" % (loo_att, attained_after(rec), attained_pre, loo_fk))

    print("\n================ VERDICT ================")
    print("recovered leaked outputs: %d of %d" % (n_rec, len(leaked)))
    print("false kills after pruning: %d" % len(fk_after))
    print("attained: %d -> %d" % (attained_pre, attained_after(rec)))
    print("LOO attained %d vs baseline %d, LOO false kills %d" % (loo_att, attained_pre, loo_fk))
    ok = n_rec >= 2 and len(fk_after) == 0
    print("pre-registered bar (+2 recovered, 0 false kills): %s" % ("MET" if ok else "NOT MET"))


if __name__ == "__main__":
    main()
