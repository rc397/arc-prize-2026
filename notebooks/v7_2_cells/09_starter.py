%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()
    NPROCS = torch.cuda.device_count() or 1

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = os.path.join(os.environ.get("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"), "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(os.environ.get("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"), "arc-agi_evaluation_challenges.json")

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    N_EVAL = int(os.environ.get("ARC_N_EVAL", "40"))
    keys = sorted(data.keys())
    if not rerun_mode and N_EVAL > 0:
        keys = keys[:N_EVAL]
    for key in keys:
        queue.put(key)
    for _ in range(NPROCS):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=NPROCS)