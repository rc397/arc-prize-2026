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
        ns = {}
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
