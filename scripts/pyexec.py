"""Process-isolated execution of generated plain-Python ARC programs, with a hard kill timeout.

A generated program can contain an infinite loop, and a thread cannot be killed in Python, so each
candidate runs in its own process that is terminated on timeout (the same pattern as the archived
arc-dsl executor). This module imports no torch, so spawning a worker stays cheap.

The candidate source must define solve(I) taking a grid as a list of rows of ints and returning the
transformed grid. The worker execs the source with normal builtins, so the program may import from the
standard library or numpy. Outputs are normalized to lists of lists of plain ints (numpy arrays and
tuples flatten to the canonical form) and size-guarded, so a runaway grid cannot blow up the queue.
"""
import multiprocessing as mp

MAX_DIM = 64   # ARC grids are at most 30x30; anything past this is a runaway, not an answer


def _norm(g):
    rows = []
    for r in g:
        row = [int(v) for v in r]
        if not row or len(row) > MAX_DIM:
            return None
        rows.append(row)
    if not rows or len(rows) > MAX_DIM:
        return None
    return rows


def _worker(q, src, inputs):
    try:
        import os
        import sys
        devnull = open(os.devnull, "w")
        sys.stdout = devnull   # generated code may print in a loop; keep the parent log clean
        sys.stderr = devnull
        # Pre-populate the namespace with the modules the prompt invites the model to use. Models
        # routinely write np.array without an import line, assuming the harness supplies it, and a
        # bare namespace turns that into a NameError that is indistinguishable from wrong logic.
        # Scoring a near-correct program as zero because of a missing import corrupts any
        # partial-credit fitness measured on top of this executor. Programs that do import are
        # unaffected, since their own import simply rebinds the same name.
        ns = {}
        for _alias, _mod in (("np", "numpy"), ("numpy", "numpy"), ("math", "math"),
                             ("itertools", "itertools"), ("collections", "collections"),
                             ("re", "re"), ("copy", "copy"), ("functools", "functools")):
            try:
                ns[_alias] = __import__(_mod)
            except Exception:
                pass
        exec(src, ns)
        f = ns["solve"]
        outs = []
        for I in inputs:
            o = f([list(r) for r in I])
            outs.append(_norm(o))
        q.put(outs)
    except Exception:
        q.put(None)


def run_candidate(src, inputs, timeout=4.0):
    """Run solve(I) for each I in inputs in a separate process. Return the list of normalized outputs
    (an unusable output becomes None in place), or None if the program errors or exceeds the timeout.
    The worker process is terminated on timeout."""
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(q, src, list(inputs)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return None
    try:
        return q.get(timeout=1.0)
    except Exception:
        return None

def _worker_diag(q, src, inputs):
    """Same execution as _worker, but reports WHY it failed instead of discarding it.

    run_candidate collapses every failure to None, which is the right contract for scoring: a
    program either reproduces the pairs or it does not. Repairing one needs the opposite, the
    traceback that scoring throws away, so this is additive rather than a change to that contract.
    """
    import traceback
    try:
        import os
        import sys
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
        ns = {}
        for _alias, _mod in (("np", "numpy"), ("numpy", "numpy"), ("math", "math"),
                             ("itertools", "itertools"), ("collections", "collections"),
                             ("re", "re"), ("copy", "copy"), ("functools", "functools")):
            try:
                ns[_alias] = __import__(_mod)
            except Exception:
                pass
        exec(src, ns)
        f = ns["solve"]
        outs = []
        for I in inputs:
            outs.append(_norm(f([list(r) for r in I])))
        q.put(("ok", outs))
    except Exception:
        q.put(("err", traceback.format_exc(limit=4)[-700:]))


def run_candidate_diag(src, inputs, timeout=4.0):
    """Return (outputs_or_None, error_text). error_text is "" when the program ran."""
    q = mp.Queue()
    p = mp.Process(target=_worker_diag, args=(q, src, list(inputs)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return None, "TIMEOUT after %.1fs (probable infinite loop)" % timeout
    try:
        kind, payload = q.get(timeout=1.0)
    except Exception:
        return None, "worker produced no result"
    return (payload, "") if kind == "ok" else (None, payload)
