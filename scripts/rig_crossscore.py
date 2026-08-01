"""Score the dumped candidate pool under a second belief state, and ask if it re-ranks the truth.

Appendix F's ceiling theorem says any selector built from the adapted model's own likelihoods is
bounded by the top-two posterior mass under those beliefs, and the pool measurement shows why that
bites: on the outputs that leak, the correct grid is dominated on multiplicity, mean augmentation NLL
and mean beam score at once, so no monotone rule over the recorded statistics can lift it. The theorem
names exactly two escapes, and both are channels that read the task through something other than the
adapted model: executing a program against the demonstrations, or scoring under a different model.

Two scorers are supported and the distinction matters for what the result licenses. The default is the
un-adapted base, which shares weights with the deployed model but not beliefs: it has never seen this
task's sixteen augmentations, so its errors are not the adapter's errors, but it is not an independent
model. Setting ARC_SCORER_MODEL to the separately trained 2B sibling gives the channel the theorem
actually names. Run the cheap one first; a null there says the dominators dominate under a different
belief state too, which is evidence for the ceiling rather than against it, and it lowers the prior on
the expensive one without settling it.

The cost of being wrong is bounded and known. The pool contains 12.333 task-credits against 9.833
attained on this slice, so a perfect re-ranker is worth about 2.5 credits and no more. This measures
how much of that a second model can actually reach.

Usage: python rig_crossscore.py [POOL_JSON]
Env:   ARC_SCORER_MODEL  path to the scoring model, default the un-adapted base
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P                                          # noqa: E402
from arc_loader import ArcDataset, QwenFormatter                       # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer           # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                 # noqa: E402

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(P.BASE, "data", "pool_dump.json")
# Which model provides the second opinion. The default is the un-adapted base, which shares weights
# with the deployed model but not beliefs. Pass the 2B sibling's path for the stronger test: it is
# separately trained, so it is the channel the theorem actually names.
SCORER = os.environ.get("ARC_SCORER_MODEL", P.MODEL_DIR)
SCORER_NAME = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(SCORER)))) or "base"


def kgmon(c):
    """The deployed selector's score: vote count minus mean augmentation NLL."""
    return len(c["beam"]) - float(np.mean([np.mean(a) for a in c["aug"]]))


def grid_str(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


@torch.no_grad()
def answer_nll(model, enc, prefix_text, answer_text):
    """Mean negative log-likelihood per token of answer_text given prefix_text."""
    pre = enc(prefix_text)
    ans = enc(answer_text)
    ids = torch.tensor([pre + ans], device="cuda")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        logits = model(input_ids=ids).logits[0].float()
    # token t is predicted from position t-1
    lp = torch.log_softmax(logits[len(pre) - 1:-1], dim=-1)
    tgt = torch.tensor(ans, device="cuda")
    return float(-lp[torch.arange(len(ans)), tgt].mean())


def main():
    pool = json.load(open(POOL))
    dd = P.data_dir()
    tasks = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    enc = P.encoder_for(tok)

    same = os.path.normpath(SCORER) == os.path.normpath(P.MODEL_DIR)
    print("scorer: %s (%s)" % (SCORER_NAME,
          "un-adapted base, shares weights with the deployed model"
          if same else "SEPARATELY TRAINED model, the channel the theorem names"), flush=True)
    stok = tok if same else AutoTokenizer.from_pretrained(SCORER)
    senc = enc if same else P.encoder_for(stok)
    model = AutoModelForCausalLM.from_pretrained(SCORER, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()

    rows, scored = [], 0
    for bk, cands in sorted(pool.items()):
        task_id, idx = bk.rsplit("_", 1)
        t = tasks.get(task_id)
        if t is None or int(idx) >= len(t["test"]):
            continue
        prefix = fmt.fmt_train(t["train"]) + "<|im_start|>user\n" \
            + grid_str(t["test"][int(idx)]["input"]) + "<|im_end|><|im_start|>assistant\n"
        for c in cands:
            c["base_nll"] = answer_nll(model, senc, prefix, grid_str(c["grid"]) + "<|im_end|>")
            scored += 1
        rows.append((bk, cands))
        if len(rows) % 10 == 0:
            print("  scored %d outputs, %d candidates" % (len(rows), scored), flush=True)

    # Three rankers over the same pool. base_nll is negated so that higher is better everywhere.
    rankers = {
        "kgmon (deployed)": kgmon,
        "scorer-NLL alone": lambda c: -c["base_nll"],
        "kgmon + scorer-NLL": lambda c: kgmon(c) - c["base_nll"],
        "multiplicity + scorer-NLL": lambda c: len(c["beam"]) - c["base_nll"],
    }
    print("\n%-26s %8s %8s" % ("ranker", "top1", "top2"))
    print("-" * 46)
    res = {}
    for name, f in rankers.items():
        t1 = t2 = 0
        for bk, cands in rows:
            order = sorted(cands, key=f, reverse=True)
            t1 += int(order[0]["correct"])
            t2 += int(any(c["correct"] for c in order[:2]))
        res[name] = (t1, t2)
        print("%-26s %8d %8d" % (name, t1, t2))
    print("\n(%d outputs, %d candidates scored; the pool contains a correct grid for %d of them)"
          % (len(rows), scored, sum(1 for _, c in rows if any(x["correct"] for x in c))))

    base_t2 = res["kgmon (deployed)"][1]
    best = max((v[1], k) for k, v in res.items())
    print("\nVERDICT")
    if best[0] > base_t2:
        print("  A second belief state re-ranks the pool: top-2 rises from %d to %d outputs under %r."
              % (base_t2, best[0], best[1]))
        print("  This is the escape route the ceiling theorem allows, and it is worth fetching the")
        print("  separately trained 2B to test the stronger version.")
    else:
        print("  No gain. Top-2 stays at %d of %d outputs under every ranker tried, so the candidates"
              % (base_t2, len(rows)))
        print("  that outrank the truth do so under a belief state that never saw this task's")
        print("  augmentations as well. The dominance is a property of the candidates rather than of")
        print("  the adapter, which is evidence for the ceiling and not against it. A separately")
        print("  trained model could still differ, but this removes the cheap reason to expect it to.")

    out = os.path.join(P.EXP, "crossscore_%s.json" % SCORER_NAME)
    json.dump({"scorer": SCORER_NAME, "results": {k: list(v) for k, v in res.items()}, "n_outputs": len(rows),
               "n_candidates": scored}, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
