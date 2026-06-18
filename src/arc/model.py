"""NVARC Qwen3-4B grid model: loading and ID-level ChatML prompt assembly.

The 16-token vocab is ids 0-9 = colors, 10 = newline, 11/12 = user/assistant, 13 = pad,
14/15 = im_start/im_end. The role tokens cannot be produced by tokenizing text, so
prompts are built at the id level; a color value equals its token id. The prompt is a
ChatML few-shot sequence of input/output turns ending with the assistant opener.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

Grid = List[List[int]]

# Fixed vocab ids for this model (vocab_size == 16)
NL, USER, ASSISTANT, PAD, IM_START, IM_END = 10, 11, 12, 13, 14, 15


def load_model(model_dir: str, dtype=torch.bfloat16, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    model.to(device).eval()
    return model, tok


def grid_to_ids(grid: Grid) -> List[int]:
    ids: List[int] = []
    for i, row in enumerate(grid):
        if i:
            ids.append(NL)
        ids.extend(int(v) for v in row)   # color value == token id (0..9)
    return ids


def ids_to_grid(ids: Sequence[int]) -> Grid:
    grid: Grid = []
    row: List[int] = []
    for t in ids:
        t = int(t)
        if t == NL:
            grid.append(row)
            row = []
        elif 0 <= t <= 9:
            row.append(t)
        # ignore specials (roles / im_start / im_end / pad)
    if row:
        grid.append(row)
    return grid


def _turn(role_id: int, content_ids: List[int]) -> List[int]:
    return [IM_START, role_id, NL] + content_ids + [IM_END]


def build_prompt_ids(train_pairs, test_input: Grid) -> List[int]:
    """ChatML few-shot prompt ending at the assistant opener (model continues)."""
    seq: List[int] = []
    for p in train_pairs:
        seq += _turn(USER, grid_to_ids(p["input"])) + [NL]
        seq += _turn(ASSISTANT, grid_to_ids(p["output"])) + [NL]
    seq += _turn(USER, grid_to_ids(test_input)) + [NL]
    seq += [IM_START, ASSISTANT, NL]
    return seq


@torch.inference_mode()
def greedy_generate(model, prompt_ids: List[int], max_new_tokens: int = 1200,
                    device: str = "cuda") -> List[int]:
    inp = torch.tensor([prompt_ids], device=device)
    out = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=IM_END, pad_token_id=PAD)
    return out[0, inp.shape[1]:].tolist()
