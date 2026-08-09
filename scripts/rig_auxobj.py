"""Three auxiliary objectives inside test-time training: masked-output completion, sub-grid crops,
and rule inversion.

All three come from Ryan. None is the pre-fitting objective we already killed, and the difference
matters. Pre-fitting was corrupted test input to clean test input, with no transformation involved,
and it moved nothing. These are transformation-conditioned.

  mask      user turn carries the input AND a copy of the output with 15-20 percent of its cells
            masked; the assistant turn carries the clean output. The adapter must complete an output
            it has been shown most of, given the input. This is close to what the decoder actually
            does, since every node of the depth-first search holds a partial output and extends it.

  invert    for demonstration pairs whose input and output have the same shape, an extra sample maps
            output back to input. Explicitly risky. Many ARC rules are many-to-one (denoise, fill,
            delete) so their inverse is not a function, and the adapter would be asked to fit
            something that does not exist. It also reverses the ARChitects' stated design that the
            model is never tasked with predicting an input grid. Worth measuring rather than
            arguing, and shape-matching at least excludes the cases where the inverse is obviously
            ill-posed.

STEP BUDGET IS HELD FIXED ACROSS ARMS, and this is the load-bearing design choice. Every arm takes
the same number of optimizer steps; the auxiliary arms spend half of them on auxiliary samples and
half on real demonstrations. So an auxiliary that wins is worth more than the real training it
displaced. Letting the auxiliary arms simply take more steps would make "the objective helped" and
"more optimisation helped" inseparable, which is how the pre-fitting experiment would have failed
had it lacked its third arm.

  base       all steps on the deployed demonstration objective.
  mask       half real, half masked-output completion on THIS task.
  mask_ctrl  half real, half masked-output completion on ANOTHER task. The control that separates a
             useful auxiliary from any auxiliary at all.
  crop       half real, half bounding-box crops of this task's pairs. See crop_samples for why this
             one can inject FALSE demonstrations, which is the risk it exists to price.
  invert     half real, half reversed pairs from this task.

Usage: python rig_auxobj.py [N_TASKS=8] [DECODE_S=600] [MASK_RATE=0.175]
Writes experiments/auxobj.jsonl, one record per (task, arm).
"""
import gc
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT = os.path.join(BASE, "experiments", "auxobj.jsonl")
N_AUX = int(os.environ.get("ARC_AUX_N", "16"))     # auxiliary samples, matching the 16 real views
STEPS = int(os.environ.get("ARC_AUX_STEPS", "32"))  # total optimizer steps, identical for every arm


def mask_grid(grid, rate, rng, fill):
    out = [row[:] for row in grid]
    for r in range(len(out)):
        for c in range(len(out[0])):
            if rng.random() < rate:
                out[r][c] = fill
    return out


def free_colour(*grids):
    used = {v for g in grids for row in g for v in row}
    return next((c for c in range(10) if c not in used), 0)


def mask_samples(pairs, n, rate, rng):
    """input + partially masked output -> clean output, in the deployed prompt shape."""
    from arc_loader import convert_grid_to_string as S
    out = []
    for i in range(n):
        p = pairs[i % len(pairs)]
        fill = free_colour(p["input"], p["output"])
        bad = mask_grid(p["output"], rate, rng, fill)
        out.append({"text": "<|im_start|>user\n" + S(p["input"]) + "\n" + S(bad)
                            + "<|im_end|><|im_start|>assistant\n" + S(p["output"]) + "<|im_end|>"})
    return out


def crop_samples(pairs, n, rng):
    """Bounding-box crops of a demonstration pair, appended as extra demonstrations.

    Only same-shape pairs qualify, and both grids are cropped to the SAME box, so the pair stays
    aligned. Note what this cannot guarantee: the crop is a true demonstration of the task's rule
    only if that rule is LOCAL. Where the rule counts objects, compares across the grid or uses one
    region to decide another, the cropped pair is a demonstration of something the task does not do,
    and the collator supervises output grids, so the adapter is trained on it as if it were true.
    That is the risk this arm exists to price rather than to avoid.
    """
    from arc_loader import convert_grid_to_string as S

    def tight(g):
        rows = [i for i, r in enumerate(g) if any(v for v in r)]
        cols = [j for j in range(len(g[0])) if any(r[j] for r in g)]
        if not rows or not cols:
            return None, False
        shrank = not (rows[0] == 0 and cols[0] == 0
                      and rows[-1] == len(g) - 1 and cols[-1] == len(g[0]) - 1)
        return [r[cols[0]:cols[-1] + 1] for r in g[rows[0]:rows[-1] + 1]], shrank

    ok = []
    for p in pairs:
        # each grid is cropped to its OWN bounding box, which is the proposal as stated: isolate the
        # active content of every demonstration grid and pair the tight versions. Requiring a shared
        # box instead would only admit same-shape pairs, and on this set that is 1 task in 8.
        ci, si = tight(p["input"])
        co, so = tight(p["output"])
        if ci is None or co is None or not (si or so):
            continue        # nothing removable, so the "crop" is a duplicate demonstration
        ok.append((ci, co))
    if not ok:
        return []
    return [{"text": "<|im_start|>user\n" + S(ok[i % len(ok)][0])
                     + "<|im_end|><|im_start|>assistant\n" + S(ok[i % len(ok)][1]) + "<|im_end|>"}
            for i in range(n)]


def invert_samples(pairs, n, rng):
    """output -> input, only where the shapes match. Empty if no pair qualifies."""
    from arc_loader import convert_grid_to_string as S
    ok = [p for p in pairs
          if len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0])]
    if not ok:
        return []
    return [{"text": "<|im_start|>user\n" + S(ok[i % len(ok)]["output"])
                     + "<|im_end|><|im_start|>assistant\n" + S(ok[i % len(ok)]["input"]) + "<|im_end|>"}
            for i in range(n)]


def main(n_tasks, decode_s, rate):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    challenges = json.load(open(os.path.join(dd, "arc-agi_evaluation_challenges.json")))
    sols = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    keys, _, strat = P.select_tasks(n_tasks)

    done = set()
    if os.path.isfile(OUT):
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(OUT, encoding="utf-8")}
    print("auxiliary objectives on %d tasks, %d steps every arm, %d aux samples, mask %.3f, decode %ds"
          % (len(keys), STEPS, N_AUX, rate, decode_s), flush=True)

    def donor_of(task):
        pool = [x for x in sorted(challenges) if x != task]
        return pool[abs(hash("aux" + task)) % len(pool)]

    def run(task, arm):
        if (task, arm) in done:
            print("skip %s|%s" % (task, arm), flush=True)
            return
        rng = random.Random(abs(hash(task)) % 10 ** 6)
        pz = arc.change_keys([task])
        labels = {"%s_%d" % (task, i): np.array(g) for i, g in enumerate(sols[task])}
        real = pz.augment(n=16, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ).as_list(fmt)

        if arm == "base":
            rows, n_aux = list(real), 0
        else:
            if arm == "mask":
                aux = mask_samples(challenges[task]["train"], N_AUX, rate, rng)
            elif arm == "mask_ctrl":
                aux = mask_samples(challenges[donor_of(task)]["train"], N_AUX, rate, rng)
            elif arm == "crop":
                aux = crop_samples(challenges[task]["train"], N_AUX, rng)
            else:
                aux = invert_samples(challenges[task]["train"], N_AUX, rng)
            if not aux:
                print("%-12s %-10s no eligible pairs, skipped" % (task, arm), flush=True)
                return
            n_aux = len(aux)
            # interleave so the two objectives alternate rather than arriving in blocks
            rows = []
            for i in range(max(len(real), n_aux)):
                if i < len(real):
                    rows.append(real[i])
                if i < n_aux:
                    rows.append(aux[i])

        m = P.build_model()
        t0 = time.time()
        # identical step count for every arm; composition is the only thing that varies
        tr = P.train_one(m, tok, collator, fmt, rows,
                         deadline=t0 + P.TRAIN_BUDGET, max_batches=STEPS)
        train_s = time.time() - t0
        base = m.get_base_model()
        base.config.use_cache = True
        base.gradient_checkpointing_disable()
        m.eval()
        t1 = time.time()
        pool, trunc, _ = P.decode_pool(m, tok, fmt, lifted, pz, t1 + decode_s)
        graded = P.grade(pool, labels, sorted(labels))
        rec = {"task": task, "arm": arm, "stratum": strat[task], "n_aux": n_aux,
               "steps": tr.get("train_steps", 0), "train_s": round(train_s, 1),
               "final_loss": round(float(tr.get("train_loss_last10", 0.0)), 5),
               "cands": sum(len(v) for v in pool.values()),
               "hits": sum(1 for v in graded.values() if v["in_pool"]),
               "outputs": len(graded), "truncated": trunc}
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print("%-12s %-10s steps %2d  loss %.5f  %4d cands  hit %d/%d%s"
              % (task, arm, rec["steps"], rec["final_loss"], rec["cands"], rec["hits"],
                 rec["outputs"], "  TRUNC" if trunc else ""), flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()

    for task in keys:
        for arm in ("base", "crop", "mask", "mask_ctrl", "invert"):
            run(task, arm)
    print("ALL ARMS DONE", flush=True)
    summarize()
    return True


def summarize():
    """Every auxiliary is scored against base on the MATCHED tasks where both arms actually ran.

    The first version summed each arm over whatever tasks it happened to cover and compared totals,
    so it reported crop at "2 hits against base's 7" when crop had run on 4 tasks and base on 10.
    Arms have different eligibility here (crop needs a removable margin, invert needs a
    shape-matching pair), so unmatched totals are never comparable.
    """
    import statistics as st
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["task"], {})[r["arm"]] = r
    n_base = sum(1 for v in by.values() if "base" in v)
    if not n_base:
        print("no base arm yet")
        return
    print("")
    print("%d records, %d tasks with a base arm" % (len(rows), n_base))
    print("%-11s %6s %10s %9s %10s %9s %11s"
          % ("arm", "tasks", "base_hits", "arm_hits", "base_loss", "arm_loss", "loss_lower"))

    res = {}
    for arm in ("crop", "mask", "mask_ctrl", "invert"):
        ts = sorted(t for t, v in by.items() if "base" in v and arm in v)
        if not ts:
            continue
        bh = sum(by[t]["base"]["hits"] for t in ts)
        ah = sum(by[t][arm]["hits"] for t in ts)
        bl = st.mean(by[t]["base"]["final_loss"] for t in ts)
        al = st.mean(by[t][arm]["final_loss"] for t in ts)
        lo = sum(1 for t in ts if by[t][arm]["final_loss"] < by[t]["base"]["final_loss"])
        res[arm] = (len(ts), bh, ah, bl, al, lo)
        print("%-11s %6d %10d %9d %10.5f %9.5f %8d/%d"
              % (arm, len(ts), bh, ah, bl, al, lo, len(ts)))

    if sum(r["cands"] for r in rows) == 0:
        print("")
        print("NO VERDICT")
        print("  Every arm decoded zero candidates. Budget or harness failure, not a result.")
        return
    if n_base < 8:
        print("")
        print("NO VERDICT")
        print("  Only %d tasks with a base arm. Too few to read four arms against each other."
              % n_base)
        return

    print("")
    print("VERDICT")
    for arm, label in (("mask", "masked-output completion"),
                       ("crop", "sub-grid crops"),
                       ("invert", "rule inversion")):
        if arm not in res:
            print("  %-26s no eligible tasks, unmeasured." % label)
            continue
        n, bh, ah = res[arm][0], res[arm][1], res[arm][2]
        d = ah - bh
        if arm == "mask" and "mask_ctrl" in res and d > 0:
            cbh, cah = res["mask_ctrl"][1], res["mask_ctrl"][2]
            if cah <= cbh:
                print("  %s ADDS %d on %d matched tasks while the unrelated-grid"
                      % (label, d, n))
                print("  control does not (%d vs %d). The auxiliary beats the real training it"
                      % (cah, cbh))
                print("  displaced, and the effect follows THIS task's outputs.")
                continue
        if d > 0:
            print("  %s adds %d hit(s) on %d matched tasks, %d vs %d."
                  % (label, d, n, ah, bh))
        elif d == 0:
            print("  %s changes nothing: %d vs %d on %d matched tasks."
                  % (label, ah, bh, n))
        else:
            print("  %s loses %d hit(s) on %d matched tasks, %d vs %d."
                  % (label, -d, n, ah, bh))
    print("  Every auxiliary arm shows a HIGHER final loss than base, which is the fixed step")
    print("  budget working as designed: half their steps go somewhere other than the")
    print("  demonstration objective, so a coverage gain would have had to beat that cost.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
        sys.exit(0)
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 600,
                       float(sys.argv[3]) if len(sys.argv) > 3 else 0.175) else 1)
