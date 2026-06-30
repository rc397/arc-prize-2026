"""Go/no-go for program search: does the fine-tuned proposer solve HELD-OUT tasks?

For each of the 40 held-out ARC-1 tasks: sample K candidate programs from the proposer, execute each in
the arc-dsl namespace against the task's TRAIN pairs (keep only those reproducing all of them, the
partial-credit verifier), then apply survivors to the TEST input. A task counts as solved if any
train-faithful survivor also reproduces the test output. Reports the held-out solve rate.

Run on the rig after dsl_finetune.py: python dsl_validate.py
"""
import sys
import json
import concurrent.futures as cf

import torch

ARC_DSL = r"D:\arc\arc-dsl"
sys.path.insert(0, ARC_DSL)
import dsl as _dsl
import constants as _const

NS_BASE = {k: getattr(_dsl, k) for k in dir(_dsl) if not k.startswith("__")}
NS_BASE.update({k: getattr(_const, k) for k in dir(_const) if not k.startswith("__")})

BASE = r"D:\arc\models\qwen2.5-coder-1.5b"
ADAPTER = r"D:\arc\experiments\dsl_proposer"
VAL = r"D:\arc\experiments\dsl_val_tasks.json"
K = 48
TIMEOUT = 1.0


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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    val = json.load(open(VAL))
    solved = train_faithful = 0
    for tid, t in val.items():
        train = [(grid(e["input"]), grid(e["output"])) for e in t["train"]]
        test = [(grid(e["input"]), grid(e["output"])) for e in t["test"]]
        ids = tok(make_prompt(t["train"]), return_tensors="pt").to("cuda")
        cands = []
        with torch.no_grad():
            for _ in range(K // 8):
                out = model.generate(**ids, do_sample=True, temperature=0.8, top_p=0.95,
                                     max_new_tokens=256, num_return_sequences=8,
                                     pad_token_id=tok.eos_token_id)
                for o in out:
                    cands.append(tok.decode(o[ids.input_ids.shape[1]:], skip_special_tokens=True))
        survivors = []
        for text in cands:
            src = extract_def(text)
            if not src:
                continue
            if all(safe_run(src, inp) == out for inp, out in train):
                survivors.append(src)
        tf = len(survivors) > 0
        train_faithful += tf
        ok = any(all(safe_run(src, inp) == out for inp, out in test) for src in survivors)
        solved += ok
        print("%s  survivors=%d  train_faithful=%s  SOLVED=%s" % (tid, len(survivors), tf, ok), flush=True)

    print("=== HELD-OUT (n=%d): train-faithful %d, SOLVED %d ===" % (len(val), train_faithful, solved), flush=True)


if __name__ == "__main__":
    main()
