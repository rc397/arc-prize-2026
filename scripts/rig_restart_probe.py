"""Ask whether an independent test-time training run misses the same tasks as the first one.

The 40-task measurement run says the deployed system spends 69 percent of its budget on tasks it never
solves, that final training loss does not separate a solve from a miss (solved 0.0006 to 0.0253, missed
0.0027 to 0.0910, overlapping), and that the slowest solve of the run landed at 1118 seconds while the
four tasks that ran past 1200 returned nothing. Decide-early, heavy-tailed, no usable progress signal:
that is the regime where restart theory says a hard cap plus a fresh attempt beats waiting, and
Appendix F already carries the mathematics. Capping at 1200 costs no measured solve and frees about 7800
seconds per worker, which is roughly fifteen restarts.

All of that is worthless if a second run fails the same tasks. This measures exactly that, and nothing
else. Note the deployed pipeline cannot answer it as shipped, because every seed in it is hardcoded
(random_state 42, trainer seed 42, train augment seed 1, eval augment seed 2), so rerunning a task
reproduces the same adapter bit for bit. The restart lever exists only if the training augmentation seed
is varied, which changes WHICH sixteen of the D4-and-colour-permutation space the adapter sees. That is
the axis varied here, and it is also the one most likely to decorrelate.

Fidelity. The training condition is the deployed one, rank 256 with alpha 32 and rslora over the same
nine target modules at 8192 context, one epoch, lr 5e-5 cosine with 0.1 warmup, completion-only loss
through the notebook's own collator, and the decoder is the deployed turbo_dfs lifted verbatim out of
arc_solver.py by AST extraction rather than reimplemented. Three things differ and none is semantic.
Unsloth is unavailable here so the model loads through plain transformers and peft. Gradient
checkpointing is forced on where deployment leaves it off, which costs speed and not gradients. And DFS
batches run two views at a time rather than four, which the builder's own self-test proves covers the
identical thirty-two views.

One caveat to carry into any writeup. The base sets tie_word_embeddings, so embed_tokens and lm_head
share a weight, and peft warns when both are adapted; Unsloth handles that pairing itself and plain peft
does not. Deployment adapts both, so this keeps both, and the comparison stays internally valid because
every seed runs the identical setup. It does mean the absolute solve rate here is not directly
comparable to a Kaggle number, only seed against seed. The same applies if the adapter degrades to the
seven-module tier under memory pressure, which the summary detects and refuses to compare across.

One measurement is omitted deliberately. The deployed selector's score_aug term needs a second forward
pass per candidate under fresh augmentations, which doubles the cost and only affects RANKING. Since the
pool analysis already closes selection by proof, the primary metric here is whether the truth is present
in the pool at all, which is exact. A multiplicity-and-beam proxy for the selector is reported alongside
and is labelled as a proxy wherever it appears.

Reads: is the union over K seeds meaningfully larger than one seed. If yes, restarts are the plateau
lever and the cap pays for them. If no, the misses are hard for the model and no scheduling buys them.

Usage: python rig_restart_probe.py [N_TASKS=20] [K_SEEDS=3]
Resumable, one JSON line per (task, seed); rerunning skips completed cells.
"""
import ast
import builtins
import gc
import json
import os
import sys
import time
import typing
from collections import defaultdict

BASE = r"D:\arc"
# runs on the rig (D:\arc) or from a laptop checkout, so resolve rather than assume
V7 = next((p for p in (os.path.join(BASE, "notebooks", "v7_debug"),
                       os.path.join(r"D:\ARC_AGI-2,3", "notebooks", "v7_debug"))
           if os.path.isdir(p)), None)
assert V7, "v7_debug modules not found, sync them to D:\\arc\\notebooks\\v7_debug"
sys.path.insert(0, V7)

N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
K_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

MODEL_DIR = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                         "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")
DATA_DIR = next((p for p in (os.path.join(BASE, "data", "kaggle"), os.path.join(BASE, "data"))
                 if os.path.isfile(os.path.join(p, "arc-agi_evaluation_challenges.json"))), None)
assert DATA_DIR, "evaluation challenges not found under D:\\arc\\data"
EXP = os.path.join(BASE, "experiments")
STATUS = os.path.join(BASE, "eval_status_restart.json")
RESULTS = os.path.join(EXP, "restart_probe.jsonl")
SUMMARY = os.path.join(EXP, "restart_probe_summary.json")

# the deployed configuration, from arc_solver.py's peft_params and train_args
RANK, ALPHA, DROPOUT = 256, 32, 0.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "embed_tokens", "lm_head"]
MAXSEQ = 8192
LR, EPOCHS, WARMUP_RATIO, GRAD_CLIP = 5e-5, 1, 0.1, 1.0
DFS_FLOOR = 0.2
BATCH_VIEWS = 2          # 2 or 4, provably the same thirty-two views either way
# Decode gets its own budget measured from the END of training, not from the start of the cell.
# Deployment caps the whole task at 1800s but trains in about 210s on an L4 under unsloth, so decode
# there gets roughly 1590s. Training here takes about 770s, so a cell-relative cap would have left
# decode barely 400s and shrunk every pool against the deployed condition. That would depress the
# solve rate for every seed and bias the union-against-single contrast this probe exists to measure.
DECODE_BUDGET = 1500

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                 # noqa: E402
from peft import LoraConfig, get_peft_model                            # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,         # noqa: E402
                          DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup)

from arc_loader import ArcDataset, QwenFormatter                       # noqa: E402
from arc_decoder import score_kgmon                                    # noqa: E402


def _restore_cache(cache, pos):
    """Undo the in-place growth a model call leaves on a Cache, restoring the branch point.

    turbo_dfs recurses with one cache object and calls the model repeatedly inside a while loop,
    passing the SAME cache each time. That is only correct if a model call leaves its input cache
    untouched, which was true of the legacy tuple API the deployed pipeline was written against:
    outputs.past_key_values came back as a new tuple. transformers 5 replaced that with Cache
    objects that append in place, so the second iteration branches from an already-extended cache,
    positions run past what was allocated, and CUDA raises an illegal memory access. On entry to
    turbo_dfs the cache holds exactly `pos` tokens, so cropping back to pos before each call
    reproduces the old semantics precisely rather than approximating them.
    """
    try:
        if hasattr(cache, "crop") and cache.get_seq_length() > pos:
            cache.crop(pos)
    except Exception:
        pass


class _CropBeforeEachBranch(ast.NodeTransformer):
    def __init__(self):
        self.hits = 0

    def visit_While(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.parse("_restore_cache(cache, pos)").body[0])
        self.hits += 1
        return node


def lift_from_solver():
    """Pull the deployed decoder and collator out of arc_solver.py without importing it.

    That module imports unsloth on line 1, which does not exist here. Taking the definitions through
    the AST keeps them byte-faithful to what actually ships, which a hand reimplementation would not.
    The single exception is the cache restore documented on _restore_cache, inserted into turbo_dfs's
    loop, which is required to run the algorithm at all on this transformers version.
    """
    src = open(os.path.join(V7, "arc_solver.py"), encoding="utf-8").read()
    body = ast.parse(src).body
    want_def = {"turbo_dfs", "inference_turbo_dfs", "QwenDataCollatorForCompletionOnlyLM"}
    keep = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want_def]
    missing = want_def - {n.name for n in keep}
    assert not missing, "could not lift %s from arc_solver.py" % sorted(missing)

    # Pull in the module-level constants the lifted code reads, transitively, rather than naming
    # them by hand. Hand-listing missed ARC_TOKENS and PAD_ID, which turbo_dfs reads on every
    # branch, and the resulting NameError only showed up deep inside a rig run.
    assigns = {}
    for n in body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigns.setdefault(t.id, n)

    def free_names(nodes):
        return {x.id for n in nodes for x in ast.walk(n)
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}

    def bound_names(nodes):
        out = {getattr(n, "name", None) for n in nodes}
        for n in nodes:
            for x in ast.walk(n):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    out.add(x.id)
                elif isinstance(x, ast.arg):
                    out.add(x.arg)
        return out - {None}

    supplied = {"torch", "np", "time", "defaultdict", "DataCollatorForLanguageModeling",
                "Any", "Union", "_restore_cache"} | set(dir(builtins))
    while True:
        need = free_names(keep) - bound_names(keep) - supplied
        pull = [assigns[n] for n in sorted(need) if n in assigns and assigns[n] not in keep]
        if not pull:
            break
        keep.extend(pull)
    keep.sort(key=body.index)
    unresolved = free_names(keep) - bound_names(keep) - supplied
    assert not unresolved, "lifted code reads names nothing provides: %s" % sorted(unresolved)

    patcher = _CropBeforeEachBranch()
    for node in keep:
        if getattr(node, "name", None) == "turbo_dfs":
            patcher.visit(node)
            ast.fix_missing_locations(node)
    # a patch that silently fails to apply is worse than no patch, and this one is the difference
    # between a correct search and a CUDA fault twenty minutes in
    assert patcher.hits == 1, "expected exactly one DFS loop to patch, found %d" % patcher.hits

    # the collator's method signature is annotated, so these must be the real typing objects
    ns = {"torch": torch, "np": np, "time": time, "defaultdict": defaultdict,
          "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
          "Any": typing.Any, "Union": typing.Union, "_restore_cache": _restore_cache}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<lifted>", "exec"), ns)
    return ns


def heartbeat(phase, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "ts": time.time(), "extra": extra}, f)
    os.replace(tmp, STATUS)


# The adapter itself fits at 8.87 GB with all nine targets, but its 530M trainable parameters carry
# AdamW moments and gradients that only allocate on the first training step, which is where 16GB runs
# out. So the tier is global and degrades on the first OOM ANYWHERE, not just at adapter creation, and
# every record carries the tier it ran under. The summary refuses to compare across tiers.
TIERS = [("full", TARGETS), ("no-embed", TARGETS[:7])]
TIER_IDX = [0]


def build_model(tokenizer):
    """Load the base and attach a fresh adapter at the current global tier."""
    tier, targets = TIERS[TIER_IDX[0]]
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.to("cuda")
    peft_model = get_peft_model(model, LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=targets,
        bias="none", use_rslora=True, task_type="CAUSAL_LM"))
    return peft_model, tier


def train_one(peft_model, tokenizer, collator, formatter, train_ds):
    base = peft_model.get_base_model()
    base.config.use_cache = False
    # Reentrant checkpointing forks and restores the CUDA RNG so recomputation replays it, and on
    # this torch build that restore faults with an illegal memory access before the first step
    # completes. The non-reentrant implementation uses saved-tensor hooks instead and does not
    # touch the device generator, which is also the variant torch now recommends.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    peft_model.train()

    rows = train_ds.as_list(formatter)
    batches = []
    for r in rows:
        ids = tokenizer(r["text"], return_tensors=None)["input_ids"][:MAXSEQ]
        batches.append({"input_ids": ids})
    steps = len(batches) * EPOCHS
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP_RATIO * steps), steps)

    losses = []
    for _ in range(EPOCHS):
        for b in batches:
            batch = collator([b])
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # forward AND backward must sit inside the same backend context. Gradient checkpointing
            # replays the forward during backward, and if that replay picks a different kernel it
            # saves a different number of tensors and torch raises CheckpointError. rig_attnprobe.py
            # had this right by keeping the whole step in one context.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                loss = peft_model(**batch).loss
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
    del opt, sched
    gc.collect()
    torch.cuda.empty_cache()
    # the deployed banner reports the trainer's mean loss over the run, so match that
    return float(np.mean(losses)) if losses else 0.0


def decode_pool(peft_model, tokenizer, formatter, lifted, puzzle_ds, deadline):
    """Run the deployed DFS over the sixteen eval views and return the candidate pool per base key."""
    base = peft_model.get_base_model()
    base.config.use_cache = True
    base.gradient_checkpointing_disable()
    peft_model.eval()

    max_new_tokens = formatter.max_new_tokens()
    multi = puzzle_ds.split_multi_replies()
    eval_ds = multi.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=MAXSEQ - max_new_tokens)

    by_test = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        by_test[subkey.split(".")[0].split("_")[1]].append(subkey)

    batches = []
    for pairs in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
        for _, subkeys in by_test.items():
            for p in pairs:
                h = [subkeys[o:o + 2] for o in p]
                batches.extend([h[0] + h[1]] if BATCH_VIEWS == 4 else h)

    max_score = -np.log(DFS_FLOOR)
    pool = defaultdict(list)
    truncated = False
    with torch.inference_mode():
        for subkeys in batches:
            if not subkeys:
                continue
            if time.time() > deadline:
                truncated = True
                break
            tokens = [tokenizer.encode(eval_ds.get(sk, formatter)["input"]) for sk in subkeys]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                dfs_result = lifted["inference_turbo_dfs"](
                    peft_model, tokens, max_new_tokens, max_score, deadline)
            for subkey_id, scored_beams in dfs_result:
                subkey = subkeys[subkey_id]
                bk = subkey.split(".")[0]
                for beam_score, token_seq in scored_beams:
                    arr = formatter.convert_tokens_to_array(token_seq)
                    if arr is None:
                        continue
                    sol = multi.invert_mod(arr, subkey, inv_perm=True)
                    pool[bk].append({"beam_score": float(beam_score), "solution": sol})
    return pool, truncated


def grade(pool, labels):
    """Exact truth-in-pool per base key, plus a labelled proxy for the deployed selector."""
    out = {}
    for bk, cands in pool.items():
        lab = labels.get(bk)
        if lab is None:
            continue
        in_pool = any(np.array_equal(c["solution"], lab) for c in cands)
        # score_aug needs a second forward pass per candidate and only affects ranking, so the
        # deployed selector is approximated by its multiplicity term with beam score as tiebreak.
        guesses = {"g%d" % i: {"beam_score": c["beam_score"], "score_aug": [c["beam_score"]],
                               "solution": c["solution"]} for i, c in enumerate(cands)}
        top2 = score_kgmon(guesses)[:2] if guesses else []
        out[bk] = {"in_pool": bool(in_pool), "n_cands": len(cands),
                   "selected_proxy": bool(any(np.array_equal(g, lab) for g in top2))}
    return out


def main():
    os.makedirs(EXP, exist_ok=True)
    heartbeat("start")

    ch = os.path.join(DATA_DIR, "arc-agi_evaluation_challenges.json")
    sol = os.path.join(DATA_DIR, "arc-agi_evaluation_solutions.json")
    solutions = json.load(open(sol))
    arc = ArcDataset.from_file(ch)
    keys = sorted(json.load(open(ch)).keys())[:N_TASKS]

    labels = {}
    for k in keys:
        for i, g in enumerate(solutions[k]):
            labels["%s_%d" % (k, i)] = np.array(g)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    formatter = QwenFormatter(tokenizer=tokenizer)
    lifted = lift_from_solver()
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tokenizer, mlm=False)

    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                done.add((r["task"], r["seed"]))
            except Exception:
                pass
    print("resuming with %d cells already done" % len(done), flush=True)

    for key in keys:
        for seed in range(1, K_SEEDS + 1):     # seed 1 IS the deployed adapter
            if (key, seed) in done:
                continue
            heartbeat("cell", "%s seed%d" % (key, seed))
            rec = None
            for attempt in (0, 1):            # a second pass only happens after a tier degrade
                t0 = time.time()
                rec = {"task": key, "seed": seed}
                peft_model = None
                try:
                    torch.manual_seed(seed)    # LoRA init, deployment fixes this at 42
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    peft_model, tier = build_model(tokenizer)
                    rec["tier"] = tier

                    puzzle_ds = arc.change_keys([key])
                    # THE varied axis: which sixteen augmentations the adapter trains on
                    train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=seed)
                    train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=MAXSEQ)

                    rec["train_loss"] = train_one(peft_model, tokenizer, collator, formatter, train_ds)
                    rec["train_s"] = round(time.time() - t0, 1)

                    t1 = time.time()
                    pool, trunc = decode_pool(peft_model, tokenizer, formatter, lifted, puzzle_ds,
                                              t1 + DECODE_BUDGET)
                    rec["decode_s"] = round(time.time() - t1, 1)
                    rec["truncated"] = trunc
                    rec["outputs"] = grade(pool, labels)
                    rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                    rec["wall_s"] = round(time.time() - t0, 1)
                    rec["ok"] = True
                except torch.cuda.OutOfMemoryError as e:
                    rec["ok"] = False
                    rec["error"] = "OOM at tier %s" % TIERS[TIER_IDX[0]][0]
                    if attempt == 0 and TIER_IDX[0] + 1 < len(TIERS):
                        TIER_IDX[0] += 1
                        print("  OOM, degrading global tier to %s and retrying"
                              % TIERS[TIER_IDX[0]][0], flush=True)
                        del peft_model
                        peft_model = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                except Exception as e:
                    rec["ok"] = False
                    rec["error"] = "%s %s" % (type(e).__name__, str(e).replace("\n", " ")[:200])
                finally:
                    # An exception raised in finally escapes every except above it, which is how a
                    # single poisoned cell killed the whole run instead of being recorded. Cleanup
                    # is never allowed to propagate.
                    try:
                        del peft_model
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as ce:
                        rec.setdefault("error", "cleanup %s %s"
                                       % (type(ce).__name__, str(ce).replace("\n", " ")[:120]))
                        rec["cuda_poisoned"] = "illegal memory access" in str(ce).lower()
                break

            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            # a CUDA illegal access poisons the context for the life of the process, so every
            # later cell would fail too. Stop and let a relaunch resume from the record above.
            if rec.get("cuda_poisoned"):
                print("CUDA context poisoned, stopping so a relaunch can resume", flush=True)
                heartbeat("poisoned", "%s seed%d" % (key, seed))
                return
            n_in = sum(1 for v in rec.get("outputs", {}).values() if v["in_pool"])
            n_out = len(rec.get("outputs", {}))
            print("%-10s seed%d  %-5s loss %-9s in-pool %d/%d  train %4.0fs decode %4.0fs %s" % (
                key, seed, "OK" if rec.get("ok") else "FAIL",
                round(rec.get("train_loss", 0), 5), n_in, n_out,
                rec.get("train_s", 0), rec.get("decode_s", 0), rec.get("error", "")), flush=True)

    summarize()
    heartbeat("done")


def summarize():
    """The decision: does the union over seeds beat one seed, and by how much."""
    rows = [json.loads(l) for l in open(RESULTS)] if os.path.exists(RESULTS) else []
    rows = [r for r in rows if r.get("ok")]
    if not rows:
        print("no completed cells")
        return
    # a mid-run tier degrade changes the adapter, so comparing across tiers would attribute a
    # capacity difference to seed variance. Keep only the tier with the most complete coverage.
    tiers = defaultdict(list)
    for r in rows:
        tiers[r.get("tier", "?")].append(r)
    if len(tiers) > 1:
        best = max(tiers, key=lambda t: len(tiers[t]))
        print("mixed tiers present %s, restricting to %r (%d of %d cells)"
              % ({k: len(v) for k, v in tiers.items()}, best, len(tiers[best]), len(rows)))
        rows = tiers[best]
    # a task is only comparable if every seed of it completed
    per_task = defaultdict(set)
    for r in rows:
        per_task[r["task"]].add(r["seed"])
    full = {t for t, s in per_task.items() if len(s) == len({r["seed"] for r in rows})}
    dropped = len(per_task) - len(full)
    if dropped:
        print("dropping %d task(s) that did not complete every seed" % dropped)
    rows = [r for r in rows if r["task"] in full]
    if not rows:
        print("no task completed all seeds yet")
        return

    by_seed = defaultdict(dict)          # seed -> output key -> in_pool
    by_seed_sel = defaultdict(dict)
    for r in rows:
        for bk, v in r["outputs"].items():
            by_seed[r["seed"]][bk] = v["in_pool"]
            by_seed_sel[r["seed"]][bk] = v["selected_proxy"]
    seeds = sorted(by_seed)
    common = set.intersection(*[set(by_seed[s]) for s in seeds]) if seeds else set()
    n = len(common)

    single = {s: sum(by_seed[s][b] for b in common) for s in seeds}
    union = sum(any(by_seed[s][b] for s in seeds) for b in common)
    inter = sum(all(by_seed[s][b] for b in seeds) for b in common)
    sel1 = sum(by_seed_sel[seeds[0]][b] for b in common) if seeds else 0

    print("\n=== restart decision, %d test outputs covered by all %d seeds ===" % (n, len(seeds)))
    for s in seeds:
        print("  seed %d alone : truth in pool for %d/%d (%.1f%%)" % (s, single[s], n, 100 * single[s] / max(n, 1)))
    print("  union of %d   : %d/%d (%.1f%%)" % (len(seeds), union, n, 100 * union / max(n, 1)))
    print("  all seeds     : %d/%d   (agreement core)" % (inter, n))
    mean_single = sum(single.values()) / len(seeds)
    gain = union - mean_single
    print("  marginal gain from restarts: %+.1f outputs over the mean single seed (%.1f%% of %d)"
          % (gain, 100 * gain / max(n, 1), n))
    if mean_single:
        print("  scaled to the hidden 240: about %+.1f task credits, %+.2f points"
              % (gain / n * 240, gain / n * 240 * 100 / 240))
    print("  selector proxy on seed %d alone: %d/%d, so pool presence exceeds selection by %d"
          % (seeds[0], sel1, n, single[seeds[0]] - sel1))
    print("\n  READ: union materially above single means restarts are the lever and the 1200s cap")
    print("        funds them. Union at or near single means the misses are hard and no scheduling")
    print("        recovers them, in which case stop here rather than building the policy.")

    json.dump({"n_outputs": n, "seeds": seeds, "single": single, "union": union,
               "intersection": inter, "selector_proxy_seed1": sel1},
              open(SUMMARY, "w"), indent=1)
    print("\nSUMMARY", SUMMARY, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        main()
