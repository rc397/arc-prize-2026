%%writefile run.py
#!/usr/bin/env python
"""Standalone in-process debug driver for the v7 (ARChitects) pipeline. No GitHub repo needed.

Unlike the notebook, which spawns the workers as subprocesses and hides their output, this runs one
worker in THIS process over a few eval tasks, so any error surfaces with a full traceback. Edit the
sibling modules (arc_solver.py and friends) and re-run to iterate.

Needs unsloth installed. The full rank-256 / 8192-context config wants a 24GB GPU; for a fast plumbing
check on less VRAM set ARC_RANK=16 ARC_MAXSEQ=2048 ARC_N_EVAL=1.

Env knobs: ARC_MODEL_PATH, ARC_DATA_DIR, ARC_N_EVAL, ARC_OUT_DIR, ARC_RANK, ARC_MAXSEQ.
"""
import os, sys, json, time, queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("ARC_MODEL_PATH", "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
os.environ.setdefault("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
os.environ.setdefault("ARC_N_EVAL", "1")
os.environ.setdefault("ARC_OUT_DIR", os.path.join(HERE, "inference_outputs"))
# for a fast low-VRAM plumbing check, uncomment the next line:
# os.environ["ARC_RANK"], os.environ["ARC_MAXSEQ"] = "16", "2048"

print("MODEL ", os.environ["ARC_MODEL_PATH"])
print("DATA  ", os.environ["ARC_DATA_DIR"])
print("N_EVAL", os.environ["ARC_N_EVAL"], "RANK", os.environ.get("ARC_RANK", "256"),
      "MAXSEQ", os.environ.get("ARC_MAXSEQ", "8192"))
os.makedirs(os.environ["ARC_OUT_DIR"], exist_ok=True)

ch = os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_challenges.json")
sol = os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_solutions.json")
keys = sorted(json.load(open(ch)).keys())[:int(os.environ["ARC_N_EVAL"])]
print("tasks:", keys)

from arc_solver import worker
q = queue.Queue()
for k in keys:
    q.put(k)
q.put(None)

worker(0, q, time.time() + 3 * 3600)

from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
data = ArcDataset.from_file(ch).load_replies(sol).change_keys(keys, keep_flags=True)
dec = ArcDecoder(data.split_multi_replies(), n_guesses=2)
dec.load_decoded_results(os.environ["ARC_OUT_DIR"])
sub = data.get_submission(dec.run_selection_algo())
print("SCORE over %d tasks:" % len(keys), data.validate_submission(sub))
