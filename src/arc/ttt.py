"""Per-task test-time training: fit a fresh LoRA adapter on one task's train pairs.

Each step draws a random augmentation and a leave-one-out target (predict one train
pair's output from the others), with loss only on the output tokens. The adapter is
reset to its initial state before each task so tasks do not leak into each other.
"""
from __future__ import annotations

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from . import augment as A
from .model import IM_END, build_prompt_ids, grid_to_ids

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_adapter(model, rank=32, alpha=32, dropout=0.0):
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    init = {k: v.detach().clone() for k, v in peft_model.named_parameters() if "lora_" in k}
    return peft_model, init


def reset_adapter(peft_model, init):
    with torch.no_grad():
        for k, v in peft_model.named_parameters():
            if k in init:
                v.copy_(init[k])


def _example(task, rng):
    n = len(task["train"])
    aug = A.random_augmentation(rng, n_examples=n)
    pairs = A.apply_task(aug, task)["train"]
    t = int(rng.integers(0, n))
    context = [pairs[i] for i in range(n) if i != t]
    prompt = build_prompt_ids(context, pairs[t]["input"])
    target = grid_to_ids(pairs[t]["output"]) + [IM_END]
    return prompt + target, [-100] * len(prompt) + target


def test_time_train(peft_model, init, task, rng, steps=150, lr=1e-4,
                    grad_clip=1.0, device="cuda"):
    reset_adapter(peft_model, init)
    peft_model.config.use_cache = False
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    loss_val = 0.0
    for _ in range(steps):
        ids, labels = _example(task, rng)
        ii = torch.tensor([ids], device=device)
        ll = torch.tensor([labels], device=device)
        loss = peft_model(input_ids=ii, labels=ll).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        opt.zero_grad()
        loss_val = float(loss.detach())
    peft_model.gradient_checkpointing_disable()
    peft_model.eval()
    peft_model.config.use_cache = True
    return loss_val
