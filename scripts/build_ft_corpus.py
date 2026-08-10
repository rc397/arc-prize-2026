"""Build the continued-fine-tune corpus from official ARC-AGI-2 training tasks + RE-ARC.

Each source reduces to a per-task POOL of input/output pairs. A training example is minted by
sampling k pairs from a pool into a synthetic task and running the same construction inference uses:
a random D4 + colour-permutation augmentation, a leave-one-out target, the few-shot context capped to
a token budget, and the loss masked to the answer tokens. This mirrors `src/arc/ttt.py:_example`
(inlined here so the corpus build pulls only the prompt/augment code, not the training stack), which
is the thing that guarantees train/inference prompt parity.

Output: `train.jsonl` / `val.jsonl`, each line `{"ids": [...], "labels": [...]}`. The 120 public eval
is never loaded (we read only the official TRAINING challenges; RE-ARC is ARC-1-derived and disjoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.arc import augment as A                                  # noqa: E402
from src.arc.model import IM_END, build_prompt_ids, grid_to_ids   # noqa: E402


def make_example(task, rng, max_ctx_tokens):
    """Mirrors src/arc/ttt.py:_example: augment, leave-one-out target, cap context, mask the prompt."""
    n = len(task["train"])
    aug = A.random_augmentation(rng, n_examples=n)
    pairs = A.apply_task(aug, task)["train"]
    t = int(rng.integers(0, n))
    ctx_idx = [i for i in range(n) if i != t]
    rng.shuffle(ctx_idx)
    chosen = []
    for i in ctx_idx:
        if chosen and len(build_prompt_ids([pairs[j] for j in chosen + [i]],
                                           pairs[t]["input"])) > max_ctx_tokens:
            break
        chosen.append(i)
    prompt = build_prompt_ids([pairs[i] for i in chosen], pairs[t]["input"])
    target = grid_to_ids(pairs[t]["output"]) + [IM_END]
    return prompt + target, [-100] * len(prompt) + target


def rearc_pools(rearc_dir, out_dir, n_examples, seed):
    """Run RE-ARC's generate_dataset (it writes tasks/{key}.json), then read each as a pool."""
    tasks_dir = os.path.join(out_dir, "tasks")
    if not (os.path.isdir(tasks_dir) and os.listdir(tasks_dir)):
        sys.path.insert(0, rearc_dir)        # RE-ARC uses flat `from dsl import *`
        cwd = os.getcwd()
        os.chdir(rearc_dir)
        try:
            import main as rearc              # re-arc/main.py
            rearc.generate_dataset(path=out_dir, seed=seed, n_examples=n_examples)
        finally:
            os.chdir(cwd)
    pools = {}
    for fn in sorted(os.listdir(tasks_dir)):
        if fn.endswith(".json"):
            pools["rearc_" + fn[:-5]] = json.loads(open(os.path.join(tasks_dir, fn)).read())
    return pools


def official_pools(challenges_path):
    """Official ARC-2 TRAINING tasks: pool = train pairs + test pairs (training tasks carry outputs)."""
    ch = json.loads(open(challenges_path, encoding="utf-8").read())
    pools = {}
    for tid, task in ch.items():
        pool = list(task.get("train", []))
        for t in task.get("test", []):
            if "output" in t:
                pool.append({"input": t["input"], "output": t["output"]})
        if len(pool) >= 2:
            pools["arc2_" + tid] = pool
    return pools


def composed_pools(rearc_dir, tasks_dir, survey_path, per_pool):
    """Pools for TWO-STEP rules, verify_B(verify_A(x)), from the pairs that survived the survey.

    The base was already pretrained on ARC-1-shaped synthetic data, and our first continued
    fine-tune on more of the same came back a null. ARC-AGI-2's stated difficulty is compositional,
    so the question these pools exist to answer is whether data the base has NOT effectively seen
    moves it where more of the same did not.

    Consistency is free because both stages are deterministic functions. The survey already filtered
    for non-degeneracy: B must run on A's output, B must act, A must stay visible, the pair must not
    be an identity, the output must be well formed.
    """
    cwd = os.getcwd()
    sys.path.insert(0, rearc_dir)
    os.chdir(rearc_dir)
    try:
        import verifiers
    finally:
        os.chdir(cwd)
    vs = {n[len("verify_"):]: getattr(verifiers, n)
          for n in dir(verifiers) if n.startswith("verify_")}

    survey = json.loads(open(survey_path, encoding="utf-8").read())
    kept = survey.get("kept") or survey.get("examples") or []
    pools, dropped = {}, 0
    for k in kept:
        a, b = k["a"], k["b"]
        p = os.path.join(tasks_dir, "%s.json" % a)
        if not os.path.isfile(p):
            continue
        src = json.loads(open(p, encoding="utf-8").read())
        pool = []
        for e in src[:per_pool * 3]:
            try:
                y = vs[b](vs[a](e["input"]))
            except Exception:
                continue
            y = [list(r) for r in y]
            if not y or not y[0] or len(y) > 30 or len(y[0]) > 30:
                continue
            pool.append({"input": [list(r) for r in e["input"]], "output": y})
            if len(pool) >= per_pool:
                break
        if len(pool) >= 2:
            pools["comp_%s_%s" % (a, b)] = pool
        else:
            dropped += 1
    print("composed pools: %d built, %d dropped for too few instances" % (len(pools), dropped),
          flush=True)
    return pools


def mint(pools, per_task, kmax, max_ctx_tokens, seed):
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for pool in pools.values():
        if len(pool) < 2:
            continue
        for _ in range(per_task):
            k = int(rng.integers(2, min(kmax, len(pool)) + 1))
            idx = rng.choice(len(pool), size=k, replace=False)
            task = {"train": [pool[int(i)] for i in idx], "test": []}
            try:
                ids, labels = make_example(task, rng, max_ctx_tokens)
            except Exception:
                continue
            h = hashlib.blake2b(bytes(ids), digest_size=16).digest()
            if h in seen:
                continue
            seen.add(h)
            out.append({"ids": ids, "labels": labels})
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True, help="official ARC-2 TRAINING challenges json")
    ap.add_argument("--rearc-dir", required=True, help="path to a clone of michaelhodel/re-arc")
    ap.add_argument("--out", required=True, help="corpus output dir")
    ap.add_argument("--rearc-pool", type=int, default=120, help="RE-ARC pairs generated per task")
    ap.add_argument("--per-task", type=int, default=200, help="examples minted per pool")
    ap.add_argument("--kmax", type=int, default=6, help="max pairs in a synthetic task")
    ap.add_argument("--max-ctx-tokens", type=int, default=1600)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end validation run")
    ap.add_argument("--compose-survey", default="",
                    help="path to compose_survey.json; adds two-step composed pools")
    ap.add_argument("--compose-per-pool", type=int, default=40,
                    help="instances generated per composed rule")
    ap.add_argument("--compose-only", action="store_true",
                    help="build the corpus from composed pools ALONE, so a fine-tune on it is a "
                         "clean test of compositional data rather than a mixture")
    a = ap.parse_args()

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        a.rearc_pool, a.per_task = 4, 3

    if a.compose_only:
        assert a.compose_survey, "--compose-only needs --compose-survey"
        tasks_dir = os.path.join(a.rearc_dir, "..", "experiments", "rearc_pools", "tasks")
        tasks_dir = os.environ.get("ARC_REARC_TASKS", os.path.abspath(tasks_dir))
        pools = composed_pools(a.rearc_dir, tasks_dir, a.compose_survey, a.compose_per_pool)
    else:
        print("generating RE-ARC pools (pool=%d)..." % a.rearc_pool, flush=True)
        pools = rearc_pools(a.rearc_dir, os.path.join(a.out, "rearc"), a.rearc_pool, a.seed)
        print("RE-ARC pools:", len(pools), flush=True)
        op = official_pools(a.challenges)
        print("official ARC-2 pools:", len(op), flush=True)
        pools.update(op)
        if a.compose_survey:
            tasks_dir = os.environ.get(
                "ARC_REARC_TASKS",
                os.path.abspath(os.path.join(a.rearc_dir, "..", "experiments", "rearc_pools", "tasks")))
            pools.update(composed_pools(a.rearc_dir, tasks_dir, a.compose_survey, a.compose_per_pool))
    if a.smoke:
        pools = {k: pools[k] for k in list(pools)[:20]}

    print("minting from %d pools (per_task=%d)..." % (len(pools), a.per_task), flush=True)
    ex = mint(pools, a.per_task, a.kmax, a.max_ctx_tokens, a.seed)
    nval = max(1, int(len(ex) * a.val_frac))
    val, train = ex[:nval], ex[nval:]
    with open(os.path.join(a.out, "train.jsonl"), "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(a.out, "val.jsonl"), "w") as f:
        for e in val:
            f.write(json.dumps(e) + "\n")
    lens = [len(e["ids"]) for e in ex] or [0]
    print("DONE %d train / %d val in %.0fs (ctx min/med/max %d/%d/%d) -> %s"
          % (len(train), len(val), time.time() - t0,
             min(lens), int(np.median(lens)), max(lens), a.out), flush=True)


if __name__ == "__main__":
    main()
