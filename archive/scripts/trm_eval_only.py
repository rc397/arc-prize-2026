"""Eval-only driver for a released TRM ARC-AGI-2 checkpoint on the rig (single GPU, native Windows).

Loads the ARC-Prize-verified arc_v2_public checkpoint into the TRM model, runs the ARC evaluator over
the reproduced data/arc2concept-aug-1000 test split, and reports pass@k. The ARC evaluator writes a
submission.json (attempt_1/attempt_2 per test output) which a separate script compares against our 4B
solved set for the disjoint-coverage measurement.

Alignment check: TRM keeps a per-puzzle embedding, so the build must reproduce the same puzzle ids the
checkpoint trained on. The puzzle_emb shape print below is the first signal; reproducing ~6.2% pass@2 is
the real confirmation. Run from D:\\arc\\trm with the .venv python.
"""
import os
import sys
import yaml
import torch

os.environ["DISABLE_COMPILE"] = "1"          # skip torch.compile (no triton on native Windows)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")

TRM = r"D:\arc\trm"
sys.path.insert(0, TRM)
os.chdir(TRM)

import torch.distributed as dist

CKPT = r"D:\arc\trm\ckpt\arc_v2_public\step_723914"
CFG = r"D:\arc\trm\ckpt\arc_v2_public\all_config.yaml"


def main():
    # single process on native Windows: gloo init is unsupported, and the only dist call the ARC
    # evaluator makes is gather_object, which is trivial for world_size=1. Stub it.
    def _gather_object(obj, object_gather_list=None, dst=0, group=None):
        if object_gather_list is not None:
            object_gather_list[0] = obj
    dist.gather_object = _gather_object
    torch.cuda.set_device(0)

    from pretrain import (PretrainConfig, create_dataloader, create_evaluators,
                          init_train_state, evaluate)

    cfg = yaml.safe_load(open(CFG))
    cfg["data_paths"] = ["data/arc2concept-aug-1000"]
    cfg["load_checkpoint"] = None            # we load manually (DISABLE_COMPILE strips _orig_mod.)
    cfg["checkpoint_path"] = "eval_out_v2"
    cfg["eval_save_outputs"] = []
    cfg["global_batch_size"] = 256           # checkpoint trained at 768 on 80GB cards; shrink for 16GB
    config = PretrainConfig(**cfg)

    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1)
    evaluators = create_evaluators(config, eval_metadata)
    train_state = init_train_state(config, train_metadata, rank=0, world_size=1)

    # manual checkpoint load with prefix strip
    sd = torch.load(CKPT, map_location="cuda")
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    model = train_state.model
    pe = "model.inner.puzzle_emb.weights"
    ms = model.state_dict()
    print("ALIGN puzzle_emb  model", tuple(ms[pe].shape), " ckpt", tuple(sd[pe].shape),
          "MATCH" if ms[pe].shape == sd[pe].shape else "MISMATCH (misaligned build)", flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print("load_state_dict missing", len(missing), "unexpected", len(unexpected), flush=True)
    model.eval()

    metrics = evaluate(config, train_state, eval_loader, eval_metadata, evaluators,
                       rank=0, world_size=1, cpu_group=None)
    print("METRICS", metrics, flush=True)


if __name__ == "__main__":
    main()
