"""Continued bf16 LoRA fine-tune of the NVARC base on the corpus from build_ft_corpus.py.

Reuses src/arc/model.py:load_model and src/arc/ttt.py:make_adapter (the same LoRA targets the
per-task TTT uses). A bf16 base + gradient checkpointing + a small LoRA optimizer fits 16 GB, the
footprint the per-task TTT already runs at, so no bitsandbytes / 4-bit is needed. Trains one sequence
at a time with gradient accumulation, early-stops on the held-out val slice, then merges the best
adapter into the base and writes the HF checkpoint `qwen3-4b-arc2ft` for GGUF conversion.

This produces a BETTER STATIC BASE. It is kept only if it clears the gate (no-TTT(FT) > 12.79% and
FT+TTT > 16.28% on the 120 public eval); otherwise we ship the stock base.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.arc.model import load_model        # noqa: E402
from src.arc.ttt import make_adapter        # noqa: E402


def load_jsonl(path, max_len):
    ex = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if len(d["ids"]) <= max_len:
                ex.append((d["ids"], d["labels"]))
    return ex


@torch.no_grad()
def val_loss(peft_model, val, device, limit=200):
    peft_model.eval()
    tot, n = 0.0, 0
    for ids, labels in val[:limit]:
        ii = torch.tensor([ids], device=device)
        ll = torch.tensor([labels], device=device)
        tot += float(peft_model(input_ids=ii, labels=ll).loss)
        n += 1
    peft_model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="NVARC base HF dir")
    ap.add_argument("--corpus", required=True, help="dir with train.jsonl / val.jsonl")
    ap.add_argument("--out", required=True, help="output dir for adapter + merged checkpoint")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048, help="drop sequences longer than this (VRAM)")
    ap.add_argument("--limit", type=int, default=0, help="cap train examples (0 = all); use for a pilot")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = "cuda"
    torch.manual_seed(a.seed)

    train = load_jsonl(os.path.join(a.corpus, "train.jsonl"), a.max_len)
    val = load_jsonl(os.path.join(a.corpus, "val.jsonl"), a.max_len)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(train)
    if a.limit:
        train = train[:a.limit]
    print("train %d / val %d (max_len %d)" % (len(train), len(val), a.max_len), flush=True)
    os.makedirs(a.out, exist_ok=True)
    best_dir = os.path.join(a.out, "adapter_best")

    model, tok = load_model(a.model, dtype=torch.bfloat16, device=device)
    peft_model, _ = make_adapter(model, rank=a.rank, alpha=a.alpha, dropout=a.dropout)
    base = peft_model.get_base_model()
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    steps = max(1, int(len(train) * a.epochs))
    best = float("inf")
    t0 = time.time()
    run = 0.0
    peft_model.train()
    for step in range(steps):
        ids, labels = train[step % len(train)]
        ii = torch.tensor([ids], device=device)
        ll = torch.tensor([labels], device=device)
        loss = peft_model(input_ids=ii, labels=ll).loss / a.accum
        loss.backward()
        run += float(loss) * a.accum
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
        if (step + 1) % a.eval_every == 0:
            vl = val_loss(peft_model, val, device)
            print("step %d/%d trainloss %.4f valloss %.4f %.0fs"
                  % (step + 1, steps, run / a.eval_every, vl, time.time() - t0), flush=True)
            run = 0.0
            if vl < best:
                best = vl
                peft_model.save_pretrained(best_dir)
    if not os.path.isdir(best_dir):       # fewer than eval_every steps: save the final adapter
        peft_model.save_pretrained(best_dir)

    # Merge the best adapter into a fresh base and write the HF checkpoint for GGUF conversion.
    from peft import PeftModel
    del peft_model, model
    torch.cuda.empty_cache()
    base2, tok2 = load_model(a.model, dtype=torch.bfloat16, device=device)
    merged = PeftModel.from_pretrained(base2, best_dir).merge_and_unload()
    ckpt = os.path.join(a.out, "qwen3-4b-arc2ft")
    merged.save_pretrained(ckpt)
    tok2.save_pretrained(ckpt)
    print("DONE best valloss %.4f -> %s" % (best, ckpt), flush=True)


if __name__ == "__main__":
    main()
