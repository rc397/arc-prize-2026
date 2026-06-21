"""Phase 2: per-task test-time training plus batched augmented voting, scored on eval.

For each task: fit a fresh LoRA adapter (TTT), generate one candidate per augmentation in a
single batched call, invert each back to the original frame, then select the two best
distinct grids by consensus and model confidence. Uses the official Kaggle consolidated
files (challenges plus solutions).

Run on the rig, e.g.:
  python scripts/run_ttt.py --model-dir <dir> --challenges <eval_challenges.json> \
      --solutions <eval_solutions.json> --n 5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc import augment as A   # noqa: E402
from arc import data as D      # noqa: E402
from arc import eval as E      # noqa: E402
from arc import model as M     # noqa: E402
from arc import select as S    # noqa: E402
from arc import ttt as T       # noqa: E402


def aug_set(n_train, n_aug, rng):
    base = [A.Augmentation(d4=name) for name in A.D4_NAMES]   # 8 geometric views
    if n_aug <= len(base):
        return base[:n_aug]
    extra = [A.random_augmentation(rng, n_examples=n_train) for _ in range(n_aug - len(base))]
    return base + extra


def predict(peft_model, task, test_input, augs, max_new_tokens, do_sample, temperature):
    prompts = []
    for aug in augs:
        ta = A.apply_task(aug, {"train": task["train"], "test": [{"input": test_input}]})
        prompts.append(M.build_prompt_ids(ta["train"], ta["test"][0]["input"]))
    gens = M.batched_generate(peft_model, prompts, max_new_tokens,
                              do_sample=do_sample, temperature=temperature)
    cands = []
    for aug, (gen, lp) in zip(augs, gens):
        g = M.ids_to_grid(gen)
        cands.append((A.invert_grid(aug, g) if g else g, lp))
    return cands


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
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
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
        ttt_t = time.time() - tt
        augs = aug_set(len(task["train"]), args.n_aug, rng)
        gt = time.time()
        preds = []
        for tp in task["test"]:
            cands = predict(peft_model, task, tp["input"], augs,
                            args.max_new_tokens, args.do_sample, args.temperature)
            a1, a2 = S.select_top2(cands)
            preds.append({"attempt_1": a1, "attempt_2": a2})
        gen_t = time.time() - gt
        submission[tid] = preds
        ok = sol.get(tid) and E.grids_equal(preds[0]["attempt_1"], sol[tid][0])
        print(f"[{i} {tid}] loss={loss:.3f} ttt={ttt_t:.0f}s gen={gen_t:.0f}s correct1={bool(ok)}",
              flush=True)
        torch.cuda.empty_cache()
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
