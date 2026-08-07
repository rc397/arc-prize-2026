"""How many generated programs died on a missing import rather than on wrong logic?

The executor ran generated code in a bare namespace while the prompt invited the model to use numpy.
A program writing np.array without an import line raises NameError, the executor returns None, and
the program scores exactly the same as one whose rule is nonsense. That is tolerable for an
exact-match probe and fatal for a partial-credit one, because the whole question about evolution is
whether a gradient exists between wrong and right.

This quantifies the damage on freshly sampled programs, scoring each under the old bare namespace
and the new pre-populated one, so the difference is attributable to the namespace and nothing else.

Usage: python rig_importaudit.py [N_TASKS=12] [K=4]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_BASE", r"D:\arc")
for _p in (HERE, BASE, os.path.dirname(HERE)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)
import multiprocessing as mp
import pyexec
import coder_probe as C


def _bare_worker(q, src, inputs):
    """The executor exactly as it was before the fix: an empty namespace."""
    try:
        import os as _os
        import sys as _sys
        dn = open(_os.devnull, "w")
        _sys.stdout = dn
        _sys.stderr = dn
        ns = {}
        exec(src, ns)
        f = ns["solve"]
        q.put([pyexec._norm(f([list(r) for r in I])) for I in inputs])
    except Exception:
        q.put(None)


def run_bare(src, inputs, timeout=4.0):
    q = mp.Queue()
    p = mp.Process(target=_bare_worker, args=(q, src, list(inputs)))
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


USES_NP = re.compile(r"\bnp\s*\.")
IMPORTS_NP = re.compile(r"^\s*(import\s+numpy|from\s+numpy\s+import)", re.M)


def main(n_tasks, k):
    challenges = json.load(open(C.CH))
    keys = sorted(challenges)[:n_tasks]
    C.ensure_server()

    n_prog = 0
    naked_np = 0
    bare_none = 0
    fixed_none = 0
    rescued = 0
    print("import audit over %d tasks, K=%d" % (len(keys), k), flush=True)

    for tid in keys:
        task = challenges[tid]
        ins = [p["input"] for p in task["train"]]
        for _ in range(k):
            text, _t = C.chat(C.prompt_for(task)[0])
            for src in C.extract_programs(text or ""):
                n_prog += 1
                naked = bool(USES_NP.search(src)) and not IMPORTS_NP.search(src)
                naked_np += int(naked)
                b = run_bare(src, ins)
                f = pyexec.run_candidate(src, ins)
                bare_none += int(b is None)
                fixed_none += int(f is None)
                if b is None and f is not None:
                    rescued += 1
        print("  %s  programs %d  naked-np %d  bare-fail %d  fixed-fail %d  rescued %d"
              % (tid, n_prog, naked_np, bare_none, fixed_none, rescued), flush=True)

    def pct(a):
        return 100.0 * a / n_prog if n_prog else 0.0

    print("\n%d programs" % n_prog)
    print("use np. without importing numpy : %d (%.1f%%)" % (naked_np, pct(naked_np)))
    print("failed to run, OLD bare namespace: %d (%.1f%%)" % (bare_none, pct(bare_none)))
    print("failed to run, NEW namespace     : %d (%.1f%%)" % (fixed_none, pct(fixed_none)))
    print("RESCUED by the namespace fix     : %d (%.1f%%)" % (rescued, pct(rescued)))
    print("\nVERDICT")
    if not n_prog:
        print("  No programs generated. Harness problem, not a result.")
    elif rescued == 0:
        print("  The namespace was not the bottleneck. The old executor's failures are real")
        print("  failures and the prior negative stands unqualified on this point.")
    else:
        print("  %.1f%% of programs ran only after the fix. Every one of them scored zero under the"
              % pct(rescued))
        print("  old executor for a reason unrelated to its logic, so any partial-credit fitness")
        print("  measured on the old harness understates the surface by at least that much.")
    json.dump({"programs": n_prog, "naked_np": naked_np, "bare_none": bare_none,
               "fixed_none": fixed_none, "rescued": rescued},
              open(os.path.join(BASE, "experiments", "importaudit.json"), "w"), indent=1)
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 4) else 1)
