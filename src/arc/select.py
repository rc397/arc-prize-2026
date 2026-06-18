"""Candidate selection: aggregate augmented candidates by consensus and confidence."""


def _key(g):
    return tuple(tuple(int(v) for v in r) for r in g) if g else ()


def select_top2(candidates):
    """candidates: list of (grid, logprob). Rank distinct grids by vote count, then mean
    log-probability, and return the two best distinct grids (attempt_1, attempt_2)."""
    agg = {}
    for g, lp in candidates:
        if not g:
            continue
        k = _key(g)
        if k not in agg:
            agg[k] = [0, 0.0, g]
        agg[k][0] += 1
        agg[k][1] += lp
    if not agg:
        return [[0]], [[0]]
    ranked = sorted(agg.values(), key=lambda v: (v[0], v[1] / v[0]), reverse=True)
    a1 = ranked[0][2]
    a2 = ranked[1][2] if len(ranked) > 1 else a1
    return a1, a2
