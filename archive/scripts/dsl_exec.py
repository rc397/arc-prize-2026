"""Process-isolated arc-dsl execution with a hard kill timeout.

A generated program can contain an infinite loop, and a thread cannot be killed in Python, so the
thread-based timeout used before would hang the run. Here each candidate runs in its own process that
is terminated on timeout. This module imports no torch, so spawning a worker stays cheap. Heavy imports
(dsl, constants) happen inside the worker at call time, not at module import.
"""
import sys
import multiprocessing as mp

ARC_DSL = r"D:\arc\arc-dsl"


def _ns():
    sys.path.insert(0, ARC_DSL)
    import dsl as _d
    import constants as _c
    ns = {k: getattr(_d, k) for k in dir(_d) if not k.startswith("__")}
    ns.update({k: getattr(_c, k) for k in dir(_c) if not k.startswith("__")})
    return ns


def _worker(q, src, inputs):
    try:
        ns = _ns()
        exec(src, ns)
        f = ns["solve"]
        q.put([f(I) for I in inputs])
    except Exception:
        q.put(None)


def run_candidate(src, inputs, timeout=2.0):
    """Run solve(I) for each I in inputs in a separate process. Return the list of outputs, or None on
    error or timeout. The worker process is terminated if it exceeds the timeout."""
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
