"""Does the round-trip signal separate a correct grid from a wrong one at all.

Before spending on real candidates this asks the cheapest version of the question. For each evaluation
test output we take the true grid and four distractors, score each one by the round-trip delta, and check
whether the true grid wins. If it cannot beat a colour swap or another task's answer, the idea is dead and
nothing downstream is worth building.

A pass here is necessary rather than sufficient. Distractors chosen this way are further from the truth
than the near-misses a real decoder produces, so a clean separation only earns the right to run the harder
test against an actual candidate pool.

The score is the same construction throughout. Inject the pair (test input, candidate) as an extra
demonstration, then measure how much easier or harder it makes predicting a held-out real demonstration,
averaged over held-out choices. Negative means the candidate behaved like a true example of the rule.

Usage: python rig_roundtrip_gate.py [N_TASKS]
"""
import json
import os
import sys
import time

BASE = r"D:\arc"
sys.path.insert(0, BASE)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = os.path.join(BASE, "experiments", "roundtrip_gate.json")
STATUS = os.path.join(BASE, "eval_status_roundtrip.json")
MODEL_DIR = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                         "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
DATA = os.path.join(BASE, "data", "kaggle")
MAX_PROMPT = 12000

import torch                                                          # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                # noqa: E402
from transformers import AutoModelForCausalLM                         # noqa: E402

from src.arc.model import build_prompt_ids, grid_to_ids, IM_END       # noqa: E402


def heartbeat(phase, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "ts": time.time(), "extra": extra}, f)
    os.replace(tmp, STATUS)


@torch.inference_mode()
def target_nll(model, prompt_ids, target_grid):
    tgt = grid_to_ids(target_grid) + [IM_END]
    ids = prompt_ids + tgt
    if len(ids) > MAX_PROMPT:
        return None
    x = torch.tensor([ids], device="cuda")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        logits = model(input_ids=x).logits
    lp = torch.log_softmax(logits[0, len(prompt_ids) - 1:-1].float(), dim=-1)
    tt = torch.tensor(tgt, device="cuda")
    return float(-lp.gather(1, tt.unsqueeze(1)).mean())


def swap_two_colours(grid):
    present = sorted({v for row in grid for v in row})
    if len(present) < 2:
        return [row[:] for row in grid]
    a, b = present[0], present[1]
    return [[b if v == a else a if v == b else v for v in row] for row in grid]


def distractors(true_out, test_in, other_out):
    """One correct grid and four wrong ones, each wrong in a different way."""
    cands = [("true", true_out), ("identity", test_in), ("colour-swap", swap_two_colours(true_out))]
    t = [list(r) for r in zip(*true_out)]
    cands.append(("transpose", t))
    if other_out is not None:
        cands.append(("other-task", other_out))
    return cands


def main():
    challenges = json.load(open(os.path.join(DATA, "arc-agi_evaluation_challenges.json")))
    solutions = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))
    keys = sorted(challenges)[:N_TASKS]
    heartbeat("load")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()

    out, t0, wins, total = {}, time.time(), 0, 0
    for n, tid in enumerate(keys):
        task, sols = challenges[tid], solutions[tid]
        train = task["train"]
        other_tid = keys[(n + 1) % len(keys)]
        for sub, (test, true_out) in enumerate(zip(task["test"], sols)):
            bk = "%s_%d" % (tid, sub)
            heartbeat("scoring", "%s %d/%d" % (bk, n + 1, len(keys)))

            base_nll, folds = {}, []
            for i in range(len(train)):
                others = [p for j, p in enumerate(train) if j != i]
                if not others:
                    continue
                v = target_nll(model, build_prompt_ids(others, train[i]["input"]), train[i]["output"])
                if v is not None:
                    base_nll[i] = v
                    folds.append(i)
            if not folds:
                continue

            recs = []
            for name, grid in distractors(true_out, test["input"], solutions[other_tid][0]):
                deltas = []
                for i in folds:
                    others = [p for j, p in enumerate(train) if j != i]
                    injected = others + [{"input": test["input"], "output": grid}]
                    v = target_nll(model, build_prompt_ids(injected, train[i]["input"]),
                                   train[i]["output"])
                    if v is not None:
                        deltas.append(v - base_nll[i])
                if deltas:
                    recs.append({"kind": name, "delta": sum(deltas) / len(deltas)})

            if len(recs) >= 2:
                best = min(recs, key=lambda r: r["delta"])
                total += 1
                wins += int(best["kind"] == "true")
                out[bk] = recs
                print("%-14s best=%-12s true_delta=%+.4f  running %d/%d  %.1f min" % (
                    bk, best["kind"], next(r["delta"] for r in recs if r["kind"] == "true"),
                    wins, total, (time.time() - t0) / 60), flush=True)
            with open(OUT, "w") as f:
                json.dump(out, f, indent=1)

    print("\nTRUE GRID RANKED FIRST ON %d OF %d OUTPUTS" % (wins, total), flush=True)
    heartbeat("done")


if __name__ == "__main__":
    main()
