"""LoRA fine-tune a small code model to propose arc-dsl programs from serialized ARC grids.

Trains on dsl_train.jsonl ({tid, prompt, completion}); loss only on the program (completion). Saves the
adapter to dsl_proposer/. Single-GPU, bf16, manual loop (batch 1 + grad accumulation) to avoid extra deps.

Run on the rig: python dsl_finetune.py
"""
import os
import sys
import json
import time
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = r"D:\arc\models\qwen2.5-coder-1.5b"
TRAIN = r"D:\arc\experiments\dsl_train.jsonl"
OUT = r"D:\arc\experiments\dsl_proposer"
MAXLEN = 4096
EPOCHS = 4
ACCUM = 8
LR = 1e-4


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples = []
    skipped = 0
    for line in open(TRAIN, encoding="utf-8"):
        r = json.loads(line)
        p_ids = tok(r["prompt"] + "\n", add_special_tokens=False).input_ids
        c_ids = tok(r["completion"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = p_ids + c_ids
        if len(ids) > MAXLEN:
            skipped += 1
            continue
        labels = [-100] * len(p_ids) + c_ids
        examples.append((ids, labels))
    print("examples %d  skipped(too long) %d" % (len(examples), skipped), flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    rng = random.Random(0)
    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        rng.shuffle(examples)
        run = 0.0
        opt.zero_grad()
        for i, (ids, labels) in enumerate(examples):
            x = torch.tensor([ids], device="cuda")
            y = torch.tensor([labels], device="cuda")
            out = model(input_ids=x, labels=y)
            (out.loss / ACCUM).backward()
            run += out.loss.item()
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
        print("epoch %d  mean_loss %.4f  t=%.0fs" % (ep, run / len(examples), time.time() - t0), flush=True)

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved adapter to %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
