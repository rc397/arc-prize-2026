"""Build the DSL-proposer training set: serialized ARC-1 train grids -> arc-dsl program.

Each of the 400 arc-dsl reference solvers is a straight-line DSL program for one ARC-1 training task.
The proposer learns to emit that program from the task's train pairs. We hold out 40 tasks to test
generalization (the go/no-go for program search). Outputs JSONL of {tid, prompt, completion} plus a
val-tasks file (grids) for the execute-and-verify validation harness.

Run on the rig (pure Python, no GPU): python dsl_data.py
"""
import os
import sys
import json
import inspect
import random

ARC_DSL = r"D:\arc\arc-dsl"
ARC1_TRAIN = r"D:\arc\arc1\data\training"
OUT = r"D:\arc\experiments"
N_VAL = 40

sys.path.insert(0, ARC_DSL)
import solvers  # noqa: E402


def grid_str(g):
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def make_prompt(train_pairs):
    lines = ["# Map each INPUT grid to its OUTPUT grid by composing arc-dsl primitives.",
             "# Train examples:"]
    for i, (inp, out) in enumerate(train_pairs, 1):
        lines.append("# example %d" % i)
        lines.append("# INPUT")
        lines.append("# " + grid_str(inp).replace("\n", "\n# "))
        lines.append("# OUTPUT")
        lines.append("# " + grid_str(out).replace("\n", "\n# "))
    lines.append("")
    return "\n".join(lines)


def main():
    rng = random.Random(0)
    tids = []
    for fn in sorted(os.listdir(ARC1_TRAIN)):
        if fn.endswith(".json") and hasattr(solvers, "solve_%s" % fn[:-5]):
            tids.append(fn[:-5])
    rng.shuffle(tids)
    val = set(tids[:N_VAL])

    train_rows, val_rows, val_tasks = [], [], {}
    for tid in tids:
        d = json.load(open(os.path.join(ARC1_TRAIN, tid + ".json")))
        train_pairs = [(e["input"], e["output"]) for e in d["train"]]
        src = inspect.getsource(getattr(solvers, "solve_%s" % tid))
        src = src.replace("def solve_%s(I):" % tid, "def solve(I):", 1).strip() + "\n"
        row = {"tid": tid, "prompt": make_prompt(train_pairs), "completion": src}
        if tid in val:
            val_rows.append(row)
            val_tasks[tid] = {"train": d["train"], "test": d["test"]}
        else:
            train_rows.append(row)

    with open(os.path.join(OUT, "dsl_train.jsonl"), "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "dsl_val.jsonl"), "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    json.dump(val_tasks, open(os.path.join(OUT, "dsl_val_tasks.json"), "w"))

    print("train %d  val %d" % (len(train_rows), len(val_rows)), flush=True)
    print("=== sample prompt ===", flush=True)
    print(train_rows[0]["prompt"][:400], flush=True)
    print("=== sample completion ===", flush=True)
    print(train_rows[0]["completion"], flush=True)


if __name__ == "__main__":
    main()
