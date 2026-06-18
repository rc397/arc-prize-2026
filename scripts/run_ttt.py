"""Phase 2: per-task test-time training plus augmented voting, scored on the eval set.

For each task: fit a fresh LoRA adapter (TTT), generate one candidate per augmentation
(inverting each back to the original frame), then take the two most common grids as the
attempts. Uses the official Kaggle consolidated files (challenges plus solutions).

Run on the rig, e.g.:
  python scripts/run_ttt.py --model-dir <dir> --challenges <eval_challenges.json> \
      --solutions <eval_solutions.json> --n 5
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc import augment as A   # noqa: E402
from arc import data as D      # noqa: E402
from arc import eval as E      # noqa: E402
from arc import model as M     # noqa: E402
from arc import ttt as T       # noqa: E402


def _key(g):
    return tuple(tuple(int(v) for v in r) for r in g) if g else ()


def predict(peft_model, task, test_input, rng, n_aug, max_new_tokens):
    augs = [A.Augmentation()] + [A.random_augmentation(rng, n_examples=len(task["train"]))
                                 for _ in range(n_aug - 1)]
    cands = []
    for aug in augs:
        ta = A.apply_task(aug, {"train": task["train"], "test": [{"input": test_input}]})
        prompt = M.build_prompt_ids(ta["train"], ta["test"][0]["input"])
        g = M.ids_to_grid(M.greedy_generate(peft_model, prompt, max_new_tokens))
        cands.append(A.invert_grid(aug, g) if g else g)
    return cands


def top2(cands):
    counts = Counter(_key(g) for g in cands if g)
    top = counts.most_common(2)
    unkey = lambda k: [list(r) for r in k]
    a1 = unkey(top[0][0]) if top else [[0]]
    a2 = unkey(top[1][0]) if len(top) > 1 else a1
    return a1, a2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-aug", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    model, tok = M.load_model(args.model_dir)
    peft_model, init = T.make_adapter(model, rank=args.rank, alpha=args.rank)
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    ch, sol = D.load_consolidated(args.challenges, args.solutions, n=args.n)
    print(f"[data] {len(ch)} tasks, {len(sol)} with solutions", flush=True)

    submission = {}
    t1 = time.time()
    for i, (tid, task) in enumerate(ch.items()):
        rng = np.random.default_rng(args.seed + i)
        tt = time.time()
        loss = T.test_time_train(peft_model, init, task, rng, steps=args.steps, lr=args.lr)
        preds = []
        for tp in task["test"]:
            a1, a2 = top2(predict(peft_model, task, tp["input"], rng,
                                  args.n_aug, args.max_new_tokens))
            preds.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = preds
        ok = sol.get(tid) and E.grids_equal(preds[0]["attempt_1"], sol[tid][0])
        print(f"[{i} {tid}] loss={loss:.3f} t={time.time() - tt:.0f}s correct1={bool(ok)}",
              flush=True)
    print(f"[infer] {time.time() - t1:.1f}s", flush=True)

    if sol:
        acc, c, t = E.score_submission(submission, sol)
        print(f"[score] exact {c}/{t} = {acc:.4f}")

    out = args.out or os.path.join(os.path.dirname(__file__), "..", "experiments",
                                   f"ttt_n{len(ch)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    D.write_submission(submission, ch, out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
