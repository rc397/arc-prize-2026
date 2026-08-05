"""Why a DFS batch went ragged: token lengths per view, per batch, for every fresh task.

CPU only, no model load. inference_turbo_dfs builds one tensor from the batch's prefixes, so a
batch whose views tokenize to different lengths raises before a single forward pass. The deployed
solver groups views by the same hard-coded offsets we copied, which means whatever breaks here is
capable of breaking a scoring run too, and that is worth knowing separately from fixing it.

Usage: python rig_batchlen.py [SOURCE=fresh|eval] [N_TASKS=24]
"""
import json
import os
import sys
from collections import defaultdict

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")


def main(source, n_tasks):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    if source == "eval":
        path = os.path.join(P.data_dir(), "arc-agi_evaluation_challenges.json")
        arc = ArcDataset.from_file(path)
        keys = sorted(json.load(open(path)))[:n_tasks]
    else:
        fresh = json.load(open(os.path.join(BASE, "experiments", "fresh_rearc.json")))
        tasks = fresh["tasks"]
        keys = sorted(tasks)[:n_tasks]
        challenges = {k: {"train": tasks[k]["train"], "test": tasks[k]["test"]} for k in keys}
        tmp = os.path.join(BASE, "_batchlen_challenges.json")
        json.dump(challenges, open(tmp, "w"))
        arc = ArcDataset.from_file(tmp)

    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    enc = P.encoder_for(tok)
    mnt = fmt.max_new_tokens()

    bad = 0
    for key in keys:
        pz = arc.change_keys([key])
        multi = pz.split_multi_replies()
        eval_ds = multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=fmt, name="input", max_len=P.MAXSEQ - mnt)

        by_test = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            by_test[subkey.split(".")[0].split("_")[1]].append(subkey)

        batches = []
        for pairs in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
            for _, subkeys in by_test.items():
                for p in pairs:
                    h = [subkeys[o:o + 2] for o in p]
                    batches.extend([h[0] + h[1]] if P.BATCH_VIEWS == 4 else h)

        # n_pairs counts the demonstrations that survived cut_to_len for this view
        rows = []
        for bi, subkeys in enumerate(batches):
            lens = [len(enc(eval_ds.get(sk, fmt)["input"])) for sk in subkeys]
            if len(set(lens)) > 1:
                npairs = [eval_ds.get(sk, fmt)["input"].count("<|im_start|>assistant")
                          for sk in subkeys]
                rows.append((bi, subkeys, lens, npairs))
        if rows:
            bad += 1
            print("\n%s RAGGED in %d/%d batches" % (key, len(rows), len(batches)), flush=True)
            for bi, subkeys, lens, npairs in rows[:3]:
                print("  batch %d lens=%s demos_kept=%s" % (bi, lens, npairs), flush=True)
                print("      %s" % [sk.split(".", 1)[1] if "." in sk else sk for sk in subkeys],
                      flush=True)
        else:
            print("%s ok" % key, flush=True)

    print("\n%d/%d %s tasks have at least one ragged batch" % (bad, len(keys), source), flush=True)
    print("VERDICT: %s" % ("the deployed batching assumption does not hold on these tasks"
                           if bad else "all batches length-homogeneous"), flush=True)
    return True


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else "fresh",
                       int(sys.argv[2]) if len(sys.argv) > 2 else 24) else 1)
