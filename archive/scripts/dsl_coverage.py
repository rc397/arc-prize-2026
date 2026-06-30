"""Disjoint coverage on ARC-2 eval. Does the program-search proposer solve tasks the 4B-TTT misses?

Builds the 120 ARC-2 public eval tasks from the Kaggle challenges plus solutions, takes the 4B-TTT solve
set as the union over every AIRV scale dump we have (the full 4B coverage envelope), runs the proposer
over the same 120 tasks (sample K programs, keep those that reproduce all train pairs, count solved when
a survivor reproduces every test output), then reports proposer solves, 4B solves, and the disjoint gain.

Run on the rig after the proposer adapter exists: python dsl_coverage.py
"""
import sys
import json
import concurrent.futures as cf

import dsl_exec

ARC_DSL = r"D:\arc\arc-dsl"
sys.path.insert(0, ARC_DSL)
import dsl as _dsl
import constants as _const

NS_BASE = {k: getattr(_dsl, k) for k in dir(_dsl) if not k.startswith("__")}
NS_BASE.update({k: getattr(_const, k) for k in dir(_const) if not k.startswith("__")})

BASE = r"D:\arc\models\qwen2.5-coder-1.5b"
ADAPTER = r"D:\arc\experiments\dsl_proposer"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
FBS = ["airv_ttt_val_s125_eval.json", "airv_ttt_val_s100_eval.json", "airv_ttt_val_s150_eval.json",
       "airv_ttt_val_s85_eval.json", "airv_ttt_val_s70_eval.json", "airv_nottt_eval.json"]
K = 16
TIMEOUT = 2.0


def grid(g):
    return tuple(tuple(r) for r in g)


def make_prompt(train):
    lines = ["# Map each INPUT grid to its OUTPUT grid by composing arc-dsl primitives.",
             "# Train examples:"]
    for i, e in enumerate(train, 1):
        gi = "\n".join(" ".join(str(c) for c in r) for r in e["input"])
        go = "\n".join(" ".join(str(c) for c in r) for r in e["output"])
        lines += ["# example %d" % i, "# INPUT", "# " + gi.replace("\n", "\n# "),
                  "# OUTPUT", "# " + go.replace("\n", "\n# ")]
    lines.append("")
    return "\n".join(lines) + "\n"


def extract_def(text):
    if "def solve" not in text:
        return None
    text = text[text.index("def solve"):]
    out = []
    for ln in text.splitlines():
        out.append(ln)
        if ln.strip().startswith("return "):
            break
    return "\n".join(out)


def run_prog(src, I):
    ns = dict(NS_BASE)
    exec(src, ns)
    return ns["solve"](I)


def safe_run(src, I):
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return ex.submit(run_prog, src, I).result(timeout=TIMEOUT)
        except Exception:
            return None


def fourb_solved(ch, sol):
    solved = set()
    for f in FBS:
        try:
            pred = json.load(open("D:/arc/experiments/" + f))
        except Exception:
            continue
        for tid in ch:
            if tid not in pred or tid not in sol:
                continue
            ok = True
            for i, truth in enumerate(sol[tid]):
                p = pred[tid][i] if i < len(pred[tid]) else {}
                if p.get("attempt_1") != truth and p.get("attempt_2") != truth:
                    ok = False
                    break
            if ok:
                solved.add(tid)
    return solved


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    ch = json.load(open(CH))
    sol = json.load(open(SOL))
    fourb = fourb_solved(ch, sol)
    print("4B-TTT union solves %d / %d" % (len(fourb), len(ch)), flush=True)

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    prop = set()
    missed = [tid for tid in ch if tid not in fourb]
    print("probing %d tasks the 4B misses (only these can be disjoint)" % len(missed), flush=True)
    for n, tid in enumerate(missed, 1):
        try:
            t = ch[tid]
            train = [(grid(e["input"]), grid(e["output"])) for e in t["train"]]
            tests = [(grid(t["test"][i]["input"]), grid(sol[tid][i])) for i in range(len(t["test"]))]
            ids = tok(make_prompt(t["train"]), return_tensors="pt").to("cuda")
            cands = []
            with torch.no_grad():
                for _ in range(max(1, K // 4)):
                    out = model.generate(**ids, do_sample=True, temperature=0.8, top_p=0.95,
                                         max_new_tokens=256, num_return_sequences=4,
                                         pad_token_id=tok.eos_token_id)
                    for o in out:
                        cands.append(tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True))
            defs = {d for d in (extract_def(c) for c in cands) if d}
            tr_in = [inp for inp, _ in train]
            tr_out = [o for _, o in train]
            te_in = [inp for inp, _ in tests]
            te_out = [o for _, o in tests]
            survivors = [s for s in defs if dsl_exec.run_candidate(s, tr_in, TIMEOUT) == tr_out]
            ok = any(dsl_exec.run_candidate(s, te_in, TIMEOUT) == te_out for s in survivors)
            if ok:
                prop.add(tid)
            print("%3d/%d %s survivors=%d SOLVED=%s" % (n, len(missed), tid, len(survivors), ok), flush=True)
        except Exception as e:
            print("%3d/%d %s SKIPPED %s" % (n, len(missed), tid, type(e).__name__), flush=True)
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    disjoint = prop - fourb
    print("=== COVERAGE proposer=%d 4B=%d proposer_disjoint=%d %s ===" %
          (len(prop), len(fourb), len(disjoint), sorted(disjoint)), flush=True)


if __name__ == "__main__":
    main()
