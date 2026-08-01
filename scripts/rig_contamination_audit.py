"""Is the ARC-AGI-2 public evaluation set a valid measurement instrument for public checkpoints?

While testing the ceiling theorem's second escape route, scoring our candidate pool under a
separately trained model, the 2B sibling qwen3_2b_grids15_sft141 assigned the true public-eval
answers a median per-token NLL of 0.0002 nats and preferred the truth over every decoded alternative
on 40 of the 41 outputs where our 4B never produced the truth at all. Over a 16-token vocabulary
uniform is 2.77 nats, and a model that reasons still carries real uncertainty at ambiguous cells.
0.0002 across several hundred cells is recall, not understanding.

That observation is not yet a finding, and this script is built around the two ways it could be
wrong rather than around the conclusion it suggests. A likelihood contrast alone cannot separate
"memorised it" from "is very good at it", so the decisive instrument here is an ordering inversion:
establish a capability ordering between two models on data minted after they were published, measure
the same models on the suspect set, and read contamination off the interaction. A 2B beating a 4B
from the same author and recipe on the suspect set while losing to it on fresh data is not
capability. This needs no training-data access, no private split, and no cooperation from the
model's author.

Nothing here licenses a claim about intent. Many ARC recipes deliberately train on the public
evaluation set because the private set is what gets scored, which is standard practice. The
defensible claim is narrower: the public evaluation set is not a valid measurement instrument for
these checkpoints, and it is widely used as one.

Stages, each gating the next. See docs/contamination-audit.md.

  g1  token round-trip, CPU. The shipped tokenizer is WordLevel over 16 tokens and has already
      produced one silent zero-supervision bug, so prove the scored ids decode back to the truth
      before believing any number computed from them.
  g2  reproduce the NLL under the round-tripped encoder, both models.
  g3  greedy decode on public eval. A model at 0.0002 nats must reproduce the answers; if it does
      not, the likelihood and the decode disagree and one of them is measuring the wrong thing.
      This is the cheap test that can kill the whole line.
  g4  mint fresh RE-ARC instances under our own seed and verify each with RE-ARC's own verifier.
  g5  the 2x2 and its interaction term.

Usage: python rig_contamination_audit.py <g1|g2|g3|g4|g5> [N_TASKS]
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P                                          # noqa: E402
from arc_loader import QwenFormatter                                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer           # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                 # noqa: E402

MODELS = {
    "4B sft139": P.MODEL_DIR,
    "2B sft141": os.path.join(P.BASE, "models", "kagglehub", "models", "sorokin",
                              "qwen3_2b_grids15_sft141", "transformers", "bfloat16", "1"),
}
REARC = os.path.join(P.BASE, "re-arc")
FRESH = os.path.join(P.EXP, "fresh_rearc.json")
OUT = os.path.join(P.EXP, "contamination")

# Our own seed, chosen now and recorded, so the fresh instances provably postdate every published
# checkpoint. Difficulty is bounded away from the easy tail because a fresh column where both models
# sit at ceiling establishes no ordering and cannot support the interaction test.
FRESH_SEED = 20260801
FRESH_DIFF = (0.35, 1.0)
FRESH_TRAIN_PAIRS = 4

MAX_NEW = 1000          # 30x30 grid is 900 cells plus 30 newlines
PROMPT_CAP = 6000       # leave room for MAX_NEW inside the trained context
BATCH = int(os.environ.get("ARC_AUDIT_BATCH", "4"))


def grid_str(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def parse_grid(s):
    """Inverse of grid_str. Returns None for anything that is not a rectangular digit grid."""
    rows = [r for r in s.split("\n") if r != ""]
    if not rows or not all(r.isdigit() for r in rows):
        return None
    if len({len(r) for r in rows}) != 1:
        return None
    return [[int(c) for c in r] for r in rows]


def decoder_for(tokenizer):
    """Exact inverse of encoder_for over answer text: digits, newlines and the end marker.

    Ambiguity is impossible on this alphabet as long as the three id classes are disjoint, which is
    asserted rather than assumed. Prompt text is not invertible at the id level (the newline id is
    shared with the turn markers) and is deliberately out of scope: what gets scored is the answer.
    """
    v = tokenizer.get_vocab()
    nl = next((i for t, i in v.items() if tokenizer.convert_tokens_to_string([t]) == "\n"), None)
    assert nl is not None, "no newline token in the vocabulary"
    digits = {v[str(d)]: str(d) for d in range(10)}
    end = v["<|im_end|>"]
    assert len(digits) == 10, "digit ids collide: %s" % digits
    assert nl not in digits and end not in digits and nl != end, \
        "id classes overlap, the inverse would be ambiguous"

    def decode(ids):
        out = []
        for i in ids:
            if i in digits:
                out.append(digits[i])
            elif i == nl:
                out.append("\n")
            elif i == end:
                out.append("<|im_end|>")
            else:
                out.append("<?%d>" % i)
        return "".join(out)

    return decode


def load_eval():
    dd = P.data_dir()
    tasks = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    return tasks, sols


def prompt_for(fmt, task, idx):
    return fmt.fmt_train(task["train"]) + "<|im_start|>user\n" \
        + grid_str(task["test"][idx]["input"]) + "<|im_end|><|im_start|>assistant\n"


def items(tasks, sols, limit):
    """(task_id, test_index, prompt_task, truth_grid) over the evaluation set, in sorted order."""
    out = []
    for tid in sorted(tasks):
        t = tasks[tid]
        for i in range(len(t["test"])):
            if tid in sols and i < len(sols[tid]):
                out.append((tid, i, t, sols[tid][i]))
    return out[:limit] if limit else out


# ----------------------------------------------------------------------------- g1

def g1(limit):
    """Prove the scored ids decode back to the truth, for both tokenizers, before believing any NLL.

    encoder_for raises on an unencodable character rather than dropping it, and maps each digit and
    newline to exactly one id, so silent loss should be structurally impossible. That is an argument
    from reading the code, which is the mode that produced the last three defects, so it is checked.
    """
    tasks, sols = load_eval()
    rows = items(tasks, sols, limit)
    print("checking %d evaluation outputs against %d tokenizers\n" % (len(rows), len(MODELS)))
    ok = True
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        enc, dec = P.encoder_for(tok), decoder_for(tok)
        bad, n_ids = [], 0
        for tid, i, _, truth in rows:
            text = grid_str(truth) + "<|im_end|>"
            try:
                ids = enc(text)
            except ValueError as e:
                bad.append((tid, i, "encode: %s" % e))
                continue
            n_ids += len(ids)
            back = dec(ids)
            if back != text:
                bad.append((tid, i, "round-trip differs"))
            elif parse_grid(back.replace("<|im_end|>", "")) != [list(r) for r in truth]:
                bad.append((tid, i, "parses to a different grid"))
        v = tok.get_vocab()
        print("%-10s vocab %d  scored ids %d  mismatches %d"
              % (name, len(v), n_ids, len(bad)))
        for b in bad[:5]:
            print("     %s[%d]: %s" % b)
        ok &= not bad
    print("\nG1 %s" % ("PASS: the scored ids are the truth, so a near-zero NLL is a fact about the "
                       "model" if ok else
                       "FAIL: the encoding is lossy, every NLL computed from it is void"))
    return ok


# ----------------------------------------------------------------------------- g2

@torch.no_grad()
def answer_nll(model, enc, prefix_text, answer_text):
    """Mean NLL per token of answer_text given prefix_text.

    ids = pre + ans, and logits[len(pre)-1 : -1] has length len(ans) with position len(pre)-1
    predicting ans[0]. No off-by-one and the answer is never visible to its own prediction.
    """
    pre, ans = enc(prefix_text), enc(answer_text)
    ids = torch.tensor([pre + ans], device="cuda")
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        logits = model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(logits[len(pre) - 1:-1], dim=-1)
    tgt = torch.tensor(ans, device="cuda")
    return float(-lp[torch.arange(len(ans)), tgt].mean())


def g2(limit):
    tasks, sols = load_eval()
    rows = items(tasks, sols, limit)
    res = {}
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        enc = P.encoder_for(tok)
        fmt = QwenFormatter(tokenizer=tok)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to("cuda").eval()
        vals = []
        for tid, i, t, truth in rows:
            pre = prompt_for(fmt, t, i)
            if len(enc(pre)) > PROMPT_CAP:
                continue
            vals.append(answer_nll(model, enc, pre, grid_str(truth) + "<|im_end|>"))
        res[name] = vals
        print("%-10s n=%3d  median %.5f  mean %.5f  p90 %.5f  frac<0.01 %.2f"
              % (name, len(vals), float(np.median(vals)), float(np.mean(vals)),
                 float(np.percentile(vals, 90)), float(np.mean(np.array(vals) < 0.01))), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(OUT + "_g2.json", "w"), indent=1)
    a, b = res["4B sft139"], res["2B sft141"]
    print("\nG2: the smaller model is %s the larger on truth-likelihood (median %.5f vs %.5f)"
          % ("BELOW" if np.median(b) < np.median(a) else "above", np.median(b), np.median(a)))
    return True


# ----------------------------------------------------------------------------- g3 / g5

@torch.no_grad()
def greedy_batch(model, tok, enc, dec, prompts):
    """Left-padded batched greedy decode.

    Padding must be on the LEFT so every sequence's last real token sits at the same index and
    generation continues from it, and the mask has to be passed explicitly because pad and eos share
    an id here, which is the condition transformers warns it cannot infer.

    No forced attention backend. Decode ran at 193 s per output under sdpa_kernel(CUDNN_ATTENTION),
    and batching four prompts changed nothing, which ruled out launch overhead and pointed at the
    kernel. rig_decodebench.py measured cuDNN at 1.9 tok/s against 6.9 for the torch default on the
    same prompt, so the backend that the training path wants is 3.6x wrong here: cuDNN was picked
    for queries thousands of tokens wide, and a decode step has a query length of one. Flash and
    mem-efficient report no available kernel on sm_120.
    """
    end = tok.get_vocab()["<|im_end|>"]
    outs = []
    for s in range(0, len(prompts), BATCH):
        chunk = [enc(p) for p in prompts[s:s + BATCH]]
        width = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), width), end, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, c in enumerate(chunk):
            ids[j, width - len(c):] = torch.tensor(c, dtype=torch.long)
            mask[j, width - len(c):] = 1
        gen = model.generate(input_ids=ids.to("cuda"), attention_mask=mask.to("cuda"),
                             max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=end, pad_token_id=end)
        for j in range(len(chunk)):
            outs.append(dec(gen[j, width:].tolist()).replace("<|im_end|>", ""))
        del gen
        torch.cuda.empty_cache()
    return outs


def sweep(rows, label):
    """Greedy exact-match accuracy for every model over (prompt_task, truth) rows."""
    res = {}
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        enc, dec = P.encoder_for(tok), decoder_for(tok)
        fmt = QwenFormatter(tokenizer=tok)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to("cuda").eval()
        prompts, truths, skipped = [], [], 0
        for tid, i, t, truth in rows:
            pre = prompt_for(fmt, t, i)
            if len(enc(pre)) > PROMPT_CAP:
                skipped += 1
                continue
            prompts.append(pre)
            truths.append([list(r) for r in truth])
        # Group similar lengths so padding does not dominate: a short prompt batched with a long one
        # pays the long one's cost at every step.
        order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
        t0 = time.time()
        outs = greedy_batch(model, tok, enc, dec, [prompts[k] for k in order])
        got = [None] * len(prompts)
        for k, o in zip(order, outs):
            got[k] = parse_grid(o)
        hit = sum(int(g == tr) for g, tr in zip(got, truths))
        res[name] = {"hit": hit, "n": len(prompts), "skipped": skipped, "secs": time.time() - t0}
        print("%-10s %-14s %3d/%3d = %5.1f%%   (%d skipped for prompt length, %.0fs, batch %d)"
              % (name, label, hit, len(prompts), 100.0 * hit / max(len(prompts), 1), skipped,
                 time.time() - t0, BATCH), flush=True)
        del model
        torch.cuda.empty_cache()
    return res


def g3(limit):
    tasks, sols = load_eval()
    rows = items(tasks, sols, limit)
    print("greedy decode, no TTT, on the ARC-AGI-2 PUBLIC EVALUATION set (%d outputs)\n" % len(rows))
    res = sweep(rows, "public eval")
    json.dump(res, open(OUT + "_g3.json", "w"), indent=1)
    a = res["4B sft139"]["hit"] / max(res["4B sft139"]["n"], 1)
    b = res["2B sft141"]["hit"] / max(res["2B sft141"]["n"], 1)
    print("\nG3")
    if b < 0.5:
        print("  The 2B does NOT reproduce the answers it scores at 0.0002 nats (%.1f%%)." % (100 * b))
        print("  Teacher-forced likelihood and free-running decode disagree, so one of them is")
        print("  measuring something other than what we think. Resolve that before going further:")
        print("  the contamination reading is not supported by decode.")
    else:
        print("  The 2B reproduces %.1f%% of the public evaluation set by greedy decode with no" % (100 * b))
        print("  test-time training, against %.1f%% for the 4B built on the same recipe. Whether" % (100 * a))
        print("  that is memorisation or capability is exactly what g4/g5 decide.")
    return True


# ----------------------------------------------------------------------------- g4

def g4(limit):
    """Mint fresh instances under our seed and verify each with RE-ARC's own verifier."""
    sys.path.insert(0, REARC)
    from random import seed as set_seed
    import main as rearc                                                 # noqa: E402

    gens, vers = rearc.get_generators(), rearc.get_verifiers()
    keys = sorted(set(gens) & set(vers))
    print("RE-ARC exposes %d generators, %d verifiers, %d usable concepts"
          % (len(gens), len(vers), len(keys)))
    set_seed(FRESH_SEED)
    want = limit or 120
    out, rejected, failed = {}, 0, 0
    for k in keys:
        if len(out) >= want:
            break
        try:
            exs = [gens[k](*FRESH_DIFF) for _ in range(FRESH_TRAIN_PAIRS + 1)]
        except Exception:
            failed += 1
            continue
        good = True
        for e in exs:
            try:
                if vers[k](e["input"]) != e["output"]:
                    good = False
                    break
            except Exception:
                good = False
                break
        if not good:
            rejected += 1
            continue
        as_list = [{"input": [list(r) for r in e["input"]],
                    "output": [list(r) for r in e["output"]]} for e in exs]
        out["fresh_" + k] = {"train": as_list[:FRESH_TRAIN_PAIRS],
                             "test": [{"input": as_list[-1]["input"]}],
                             "solution": [as_list[-1]["output"]]}
    json.dump({"seed": FRESH_SEED, "difficulty": list(FRESH_DIFF), "tasks": out},
              open(FRESH, "w"))
    sizes = [len(t["test"][0]["input"]) * len(t["test"][0]["input"][0]) for t in out.values()]
    print("minted %d verified tasks (%d generators raised, %d failed their own verifier)"
          % (len(out), failed, rejected))
    print("test-input areas: median %d, max %d" % (int(np.median(sizes)), max(sizes)))
    print("wrote %s" % FRESH)
    print("\nG4 %s" % ("PASS" if len(out) >= 40 else "FAIL: too few verified tasks to measure on"))
    return len(out) >= 40


# ----------------------------------------------------------------------------- g5

def g5(limit):
    if not os.path.isfile(FRESH):
        raise SystemExit("run g4 first to mint the fresh column")
    fresh = json.load(open(FRESH))
    rows = [(tid, 0, t, t["solution"][0]) for tid, t in sorted(fresh["tasks"].items())][:limit or None]
    print("greedy decode on %d FRESH instances, seed %d, difficulty %s\n"
          % (len(rows), fresh["seed"], fresh["difficulty"]))
    res = sweep(rows, "fresh RE-ARC")
    json.dump(res, open(OUT + "_g5.json", "w"), indent=1)

    if not os.path.isfile(OUT + "_g3.json"):
        print("\nrun g3 to complete the 2x2")
        return True
    ev = json.load(open(OUT + "_g3.json"))
    print("\n%-12s %14s %16s %12s" % ("model", "public eval", "fresh RE-ARC", "gap"))
    print("-" * 58)
    acc = {}
    for name in MODELS:
        e = ev[name]["hit"] / max(ev[name]["n"], 1)
        f = res[name]["hit"] / max(res[name]["n"], 1)
        acc[name] = (e, f)
        print("%-12s %13.1f%% %15.1f%% %11.1f" % (name, 100 * e, 100 * f, 100 * (e - f)))

    (e4, f4), (e2, f2) = acc["4B sft139"], acc["2B sft141"]
    inter = (e2 - e4) - (f2 - f4)
    print("\ninteraction (2B-4B on eval) minus (2B-4B on fresh): %+.1f points" % (100 * inter))
    ceiling = min(f4, f2) > 0.95 or max(f4, f2) < 0.05
    print("\nG5")
    if ceiling:
        print("  NO RESULT. The fresh column is at %s, so it establishes no capability ordering and"
              % ("ceiling" if min(f4, f2) > 0.95 else "floor"))
        print("  the interaction is uninterpretable. Re-mint g4 with different difficulty bounds.")
    elif inter > 0.10 and f2 < f4:
        print("  ORDERING INVERSION. The 2B loses to the 4B on data minted after both were published")
        print("  and beats it on the public evaluation set. Capability does not do that. The public")
        print("  evaluation set is not a valid measurement instrument for this checkpoint.")
    else:
        print("  No inversion. The 2B's advantage on public eval, if any, is matched by an advantage")
        print("  on fresh data, which is what capability looks like. Report this as a negative and")
        print("  find another explanation for the 0.0002 nats.")
    return True


STAGES = {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5}

if __name__ == "__main__":
    os.makedirs(P.EXP, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "g1"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    assert stage in STAGES, "stage must be one of %s" % sorted(STAGES)
    print("=== %s ===" % stage, flush=True)
    sys.exit(0 if STAGES[stage](n) else 1)
