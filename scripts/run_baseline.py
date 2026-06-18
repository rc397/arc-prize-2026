"""No-TTT baseline: run the NVARC grid model greedily over the ARC-AGI-2 eval set
and score it. Validates the I/O format and gives the pre-TTT baseline number.

Run on the rig, e.g.:
  python scripts/run_baseline.py --model-dir <dir> --data-dir <eval_dir> --n 10
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc import model as M   # noqa: E402
from arc import eval as E    # noqa: E402


def load_eval(data_dir, n=None):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if n:
        files = files[:n]
    challenges, solutions = {}, {}
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        d = json.load(open(f, encoding="utf-8"))
        challenges[tid] = d
        if all("output" in t and t["output"] for t in d["test"]):
            solutions[tid] = [t["output"] for t in d["test"]]
    return challenges, solutions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    t0 = time.time()
    model, tok = M.load_model(args.model_dir)
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    challenges, solutions = load_eval(args.data_dir, args.n)
    print(f"[data] {len(challenges)} tasks, {len(solutions)} with solutions", flush=True)

    def dims(g):
        if not g:
            return (0, 0)
        w = len(g[0])
        return (-1, -1) if any(len(r) != w for r in g) else (len(g), w)

    submission = {}
    shape_ok = shape_tot = 0
    t1 = time.time()
    for i, (tid, task) in enumerate(challenges.items()):
        preds = []
        for j, tp in enumerate(task["test"]):
            ids = M.build_prompt_ids(task["train"], tp["input"])
            gen = M.greedy_generate(model, ids, args.max_new_tokens)
            g = M.ids_to_grid(gen)
            preds.append({"attempt_1": g, "attempt_2": g})
            if tid in solutions:
                shape_tot += 1
                shape_ok += int(dims(g) == dims(solutions[tid][j]))
        submission[tid] = preds
        if i < 5:
            a = preds[0]["attempt_1"]
            gt0 = solutions[tid][0] if tid in solutions else []
            print(f"[{i} {tid}] pred={dims(a)} gt={dims(gt0)} "
                  f"exact={bool(tid in solutions and E.grids_equal(a, gt0))}", flush=True)
    dt = time.time() - t1
    print(f"[infer] {dt:.1f}s total, {dt / max(1, len(challenges)):.2f}s/task", flush=True)

    if solutions:
        acc, c, t = E.score_submission(submission, solutions)
        sr = shape_ok / max(1, shape_tot)
        print(f"[score] exact {c}/{t} = {acc:.4f} | shape {shape_ok}/{shape_tot} = {sr:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "experiments",
                            f"baseline_n{len(challenges)}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(submission, open(out_path, "w"))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
