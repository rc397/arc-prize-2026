"""Merge one saved LoRA adapter into a fresh base and write a gate-able HF checkpoint.

finetune_base.py merges only its BEST adapter, once, at the end of a run. A duration experiment needs
several points on the same run gated separately, so it saves step-indexed adapters instead
(--save-every) and this turns any one of them into the merged checkpoint rig_ftgate.py expects.

Kept as a separate script rather than a flag on the trainer because merging loads a second base, about
8 GB, which cannot be resident alongside training on a 16 GB card. Run these after the training
process has exited.

The merge path is deliberately identical to the tail of finetune_base.main so a checkpoint produced
here is indistinguishable from one the trainer would have written itself.

Usage: python merge_ft_adapter.py <adapter_dir> <out_ckpt_dir> [base_model_dir]
"""
import os
import sys

import torch

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

DEFAULT_BASE = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                            "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    adapter, out = sys.argv[1], sys.argv[2]
    base_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_BASE
    assert os.path.isdir(adapter), "no adapter at %s" % adapter
    if os.path.isdir(out) and os.path.isfile(os.path.join(out, "config.json")):
        print("already merged: %s" % out, flush=True)
        return 0

    from peft import PeftModel
    from src.arc.model import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("merging %s into a fresh base" % os.path.basename(adapter), flush=True)
    base, tok = load_model(base_dir, dtype=torch.bfloat16, device=device)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print("wrote %s" % out, flush=True)
    del merged, base
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
