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


def padded(ids, labels, mult):
    """Round a sequence up to a multiple of `mult`, masking the padding out of both the loss and
    the attention.

    This is the whole fix, and it is about the ALLOCATOR rather than about the model. Every example
    in this corpus is a different length, so every step asks CUDA's caching allocator for a block
    size it has never seen; nothing is ever reused, reserved memory climbs to the card's limit, and
    from then on each step pays for synchronising frees. Measured: 1.48 s/step over the first 15
    steps, 2.97 over the first 19, and 11.5 averaged across the pilot's 8,000. Rounding to a multiple
    of 256 leaves at most eight distinct shapes, which the allocator reuses forever.

    On Linux the supported fix is PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. This rig runs
    native Windows, where that allocator prints "expandable_segments not supported on this platform"
    and is ignored, so the padding is not a preference.

    The loss is unchanged, not approximately: padded label positions are -100 and cross-entropy
    ignores them, and the attention mask keeps the padding out of every score.
    """
    n = len(ids)
    if not mult or n % mult == 0:
        return ids, labels, [1] * n
    m = ((n + mult - 1) // mult) * mult
    return ids + [0] * (m - n), labels + [-100] * (m - n), [1] * n + [0] * (m - n)


@torch.no_grad()
def val_loss(peft_model, val, device, limit=200, pad_mult=0):
    peft_model.eval()
    tot, n = 0.0, 0
    for ids, labels in val[:limit]:
        pi, pl, am = padded(ids, labels, pad_mult)
        ii = torch.tensor([pi], device=device)
        ll = torch.tensor([pl], device=device)
        aa = torch.tensor([am], device=device)
        tot += float(peft_model(input_ids=ii, labels=ll, attention_mask=aa).loss)
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
    ap.add_argument("--save-every", type=int, default=0,
                    help="write a step-indexed adapter every N steps (0 = off). Only needed for "
                         "runs that get gated at several durations rather than once at the end.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from out/resume.json if present; a 41-hour run on a box "
                         "Windows Update reboots without asking needs this")
    ap.add_argument("--pad-mult", type=int, default=256,
                    help="round every sequence up to a multiple of this so the CUDA "
                         "caching allocator sees repeating shapes; 0 disables")
    ap.add_argument("--trace", type=int, default=0,
                    help="print per-phase timing for the first N steps and for the val pass")
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
    start_step = 0
    # Resume is keyed on the SHUFFLED order, which is deterministic given --seed, so restarting
    # replays exactly the sequence the killed run would have seen. Without that the resumed run
    # trains on a different curriculum and the two halves are not one experiment.
    rpath = os.path.join(a.out, "resume.json")
    if a.resume and os.path.isfile(rpath):
        st = json.load(open(rpath))
        start_step, best = st["step"], st["best"]
        from peft import set_peft_model_state_dict
        import safetensors.torch as sft
        sd = sft.load_file(os.path.join(a.out, "resume_adapter.safetensors"))
        set_peft_model_state_dict(peft_model, sd)
        print("resumed at step %d/%d, best valloss %.4f" % (start_step, steps, best), flush=True)
    t0 = time.time()
    run = 0.0
    peft_model.train()
    # Per-phase timing for the first few steps. The pilot ran at 11.7 s/step while an isolated
    # harness measured the identical configuration at 1.37, and no amount of reading the code found
    # the difference, so the loop reports where its own time goes.
    for step in range(start_step, steps):
        ts = time.time()
        ids, labels = train[step % len(train)]
        pi, pl, am = padded(ids, labels, a.pad_mult)
        ii = torch.tensor([pi], device=device)
        ll = torch.tensor([pl], device=device)
        aa = torch.tensor([am], device=device)
        t_data = time.time()
        loss = peft_model(input_ids=ii, labels=ll, attention_mask=aa).loss / a.accum
        t_fwd = time.time()
        loss.backward()
        t_bwd = time.time()
        run += float(loss) * a.accum
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
        t_opt = time.time()
        if a.trace and step < a.trace:
            print("  [trace] step %2d len %4d  data %.3f  fwd %.3f  bwd %.3f  opt %.3f  total %.3f"
                  % (step, len(ids), t_data - ts, t_fwd - t_data, t_bwd - t_fwd,
                     t_opt - t_bwd, t_opt - ts), flush=True)
        if (step + 1) % a.eval_every == 0:
            tv = time.time()
            vl = val_loss(peft_model, val, device, pad_mult=a.pad_mult)
            if a.trace:
                print("  [trace] val pass %.1fs" % (time.time() - tv), flush=True)
            print("step %d/%d trainloss %.4f valloss %.4f %.0fs"
                  % (step + 1, steps, run / a.eval_every, vl, time.time() - t0), flush=True)
            run = 0.0
            if vl < best:
                best = vl
                peft_model.save_pretrained(best_dir)
            # checkpoint for resume, written every eval whether or not val improved
            try:
                import safetensors.torch as sft
                from peft import get_peft_model_state_dict
                sft.save_file({k: v.contiguous() for k, v in
                               get_peft_model_state_dict(peft_model).items()},
                              os.path.join(a.out, "resume_adapter.safetensors"))
                json.dump({"step": step + 1, "best": best},
                          open(os.path.join(a.out, "resume.json"), "w"))
            except Exception as e:
                print("  [resume checkpoint failed: %s]" % e, flush=True)
        # Step-indexed ADAPTER checkpoints, so a long run can be gated at several durations rather
        # than only at its end. Adapters and not merged models: merging needs a second base resident
        # (about 8 GB) and would not fit alongside training on a 16 GB card. scripts/merge_ft_adapter.py
        # turns each of these into a gate-able checkpoint afterwards, one at a time.
        if a.save_every and (step + 1) % a.save_every == 0:
            sdir = os.path.join(a.out, "adapter_step%d" % (step + 1))
            try:
                peft_model.save_pretrained(sdir)
                print("  saved step checkpoint %s" % os.path.basename(sdir), flush=True)
            except Exception as e:
                print("  [step checkpoint failed: %s]" % e, flush=True)
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
