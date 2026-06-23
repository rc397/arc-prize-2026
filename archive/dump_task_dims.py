"""Dump task_dims.csv for fig_vram_wall (A2): per eval task, the test grid dims, exact context-token
count (the real id-level prompt), the largest train pair (maxpair, the VRAM driver and the actual
exclusion metric), and trainable/excluded from the val manifest."""
import sys, json, csv
sys.path.insert(0, r"D:\arc")
from src.arc.data import load_consolidated
from src.arc.model import build_prompt_ids, grid_to_ids

CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"
allt, _ = load_consolidated(CH, SOL, n=None)
man = json.load(open(r"D:\arc\experiments\ttt_manifest_val.json"))


def maxpair(t):
    return max(len(grid_to_ids(p["input"])) + len(grid_to_ids(p["output"])) for p in t["train"])


rows = []
for tid, t in allt.items():
    ti = t["test"][0]["input"]
    rows.append([tid, len(t["train"]), len(ti), len(ti[0]),
                 len(build_prompt_ids(t["train"], ti)), maxpair(t), int(tid in man)])
with open(r"D:\arc\experiments\task_dims.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task_id", "n_pairs", "test_h", "test_w", "context_tokens", "maxpair", "trainable"])
    w.writerows(rows)
print("task_dims.csv:", len(rows), "tasks, trainable", sum(r[6] for r in rows), flush=True)
