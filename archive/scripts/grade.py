"""Stage 0 grader for the program-search line: score an arc-dsl program against a task's train pairs.

A program is a callable f(I)->O composed of arc-dsl primitives (e.g. solvers.solve_<id>). The grader
runs it on each pair and returns the fraction of outputs reproduced exactly. CPU-only, never touches the
GPU. That continuous k/n score is the partial-credit signal the program-search line is built on.

Validation: run arc-dsl's reference solvers against the ARC-1 training tasks (each should reproduce its
own task fully) and benchmark executor throughput.

Usage: python grade.py [ARC1_TRAIN_DIR]
"""
import os
import sys
import json
import time

ARC_DSL = r"D:\arc\arc-dsl"
sys.path.insert(0, ARC_DSL)
import solvers  # noqa: E402  (pulls dsl/constants/arc_types via ARC_DSL on the path)


def grade(program, pairs):
    """Fraction of (input, output) grid pairs the program reproduces exactly; exceptions count as misses."""
    if not pairs:
        return 0.0
    hit = 0
    for inp, out in pairs:
        try:
            if program(inp) == out:
                hit += 1
        except Exception:
            pass
    return hit / len(pairs)


def load_arc1(train_dir):
    """ARC-1 per-task JSONs -> {tid: [(input_grid, output_grid), ...]} as immutable tuple grids."""
    ast = lambda g: tuple(tuple(r) for r in g)
    tasks = {}
    for fn in os.listdir(train_dir):
        if fn.endswith(".json"):
            d = json.load(open(os.path.join(train_dir, fn)))
            tasks[fn[:-5]] = [(ast(e["input"]), ast(e["output"])) for e in d["train"] + d["test"]]
    return tasks


def main():
    train_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\arc\arc1\data\training"
    tasks = load_arc1(train_dir)
    print("loaded %d ARC-1 training tasks from %s" % (len(tasks), train_dir), flush=True)
    n_full = n_have = execs = 0
    partial = 0.0
    t0 = time.time()
    for tid, pairs in tasks.items():
        fn = getattr(solvers, "solve_%s" % tid, None)
        if fn is None:
            continue
        n_have += 1
        execs += len(pairs)
        s = grade(fn, pairs)
        partial += s
        if s == 1.0:
            n_full += 1
    dt = time.time() - t0
    print("reference solvers found: %d" % n_have, flush=True)
    print("solvers reproducing their whole task: %d/%d (%.1f%%)"
          % (n_full, n_have, 100.0 * n_full / max(1, n_have)), flush=True)
    print("mean fractional grade across solvers: %.3f" % (partial / max(1, n_have)), flush=True)
    print("throughput: %d program-executions in %.2fs = %.0f exec/s (CPU, no GPU)"
          % (execs, dt, execs / max(1e-9, dt)), flush=True)


if __name__ == "__main__":
    main()
