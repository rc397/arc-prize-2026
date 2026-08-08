"""Does a task embedding retrieve tasks with the SAME RULE, or just tasks that look alike?

This gates the whole cross-task-transfer proposal. That design retrieves the top-K most similar
tasks by an embedding of their demonstrations and fuses their adapters as an initialisation. Every
step after retrieval is wasted if the embedding does not actually recover rule identity, and nobody
has checked, because on real ARC data there is no ground truth for "same rule".

RE-ARC supplies exactly that ground truth for free. Its generators emit unlimited pairs from one
underlying transformation, so slicing a single generator's pool into several disjoint instances gives
tasks that are different in every surface respect and identical in rule. Instances from different
generators are the negatives.

The embedding under test is the one proposed: format the demonstrations as the deployed prompt does,
run the frozen base model, mean-pool the final hidden states.

The control is the part that makes this a measurement rather than a demo. Same-generator instances
also share palette, grid sizes and density, because the same generator produced them. So a positive
result on its own cannot distinguish "recovers the rule" from "recovers the surface statistics", and
only the second is easy. We therefore score a deliberately dumb surface baseline on identical pairs:
colour histogram, grid dimensions, density. If the model embedding does not clearly beat counting
colours, it is not carrying rule information and the retrieval premise fails even though the
same-versus-different separation looks good.

Reported: mean cosine within and across generators, the separation in pooled standard deviations,
precision@1 (is a task's nearest neighbour from its own generator) and ROC AUC over all pairs, for
both the model embedding and the surface baseline.

Usage: python rig_taskembed.py [N_GENERATORS=24] [INSTANCES_PER=6] [DEMOS=4]
Writes experiments/taskembed.json.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
REARC = os.path.join(BASE, "re-arc")
POOLS = os.path.join(BASE, "experiments", "rearc_pools")
OUT = os.path.join(BASE, "experiments", "taskembed.json")


def ensure_pools(n_examples, seed):
    """RE-ARC writes one json per generator, each a list of {input, output} pairs from one rule."""
    tasks_dir = os.path.join(POOLS, "tasks")
    if os.path.isdir(tasks_dir) and len(os.listdir(tasks_dir)) > 50:
        return tasks_dir
    cwd = os.getcwd()
    sys.path.insert(0, REARC)                 # re-arc uses flat `from dsl import *`
    os.chdir(REARC)
    try:
        import main as rearc
        rearc.generate_dataset(path=POOLS, seed=seed, n_examples=n_examples)
    finally:
        os.chdir(cwd)
    return tasks_dir


def slice_instances(pairs, n_inst, n_demo, rng):
    """Disjoint groups of n_demo pairs from one generator's pool: same rule, different surface."""
    need = n_inst * n_demo
    if len(pairs) < need:
        return []
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    out = []
    for i in range(n_inst):
        grp = idx[i * n_demo:(i + 1) * n_demo]
        out.append([pairs[j] for j in grp])
    return out


def surface_vector(demos):
    """A deliberately dumb descriptor: colour histogram plus size and density. The thing the model
    embedding has to beat before any claim about rules is credible."""
    import numpy as np
    hist = np.zeros(10, dtype=np.float64)
    hs, ws, dens = [], [], []
    for d in demos:
        for g in (d["input"], d["output"]):
            a = np.asarray(g)
            for v in range(10):
                hist[v] += float((a == v).sum())
            hs.append(a.shape[0])
            ws.append(a.shape[1])
            dens.append(float((a != 0).mean()))
    hist = hist / max(hist.sum(), 1.0)
    v = np.concatenate([hist, [np.mean(hs) / 30.0, np.mean(ws) / 30.0,
                               np.std(hs) / 30.0, np.std(ws) / 30.0, np.mean(dens)]])
    return v / (np.linalg.norm(v) + 1e-9)


def main(n_gen, n_inst, n_demo):
    import numpy as np
    import torch
    import rig_restart_probe as P
    from arc_loader import QwenFormatter
    from transformers import AutoTokenizer

    rng = random.Random(0)
    tasks_dir = ensure_pools(n_examples=max(200, n_inst * n_demo * 3), seed=7)
    files = sorted(f for f in os.listdir(tasks_dir) if f.endswith(".json"))
    rng.shuffle(files)

    groups = []                                # (generator_id, [demos])
    for fn in files:
        pairs = json.load(open(os.path.join(tasks_dir, fn), encoding="utf-8"))
        insts = slice_instances(pairs, n_inst, n_demo, rng)
        if len(insts) == n_inst:
            for d in insts:
                groups.append((fn[:-5], d))
        if len({g for g, _ in groups}) >= n_gen:
            break
    gens = sorted({g for g, _ in groups})
    print("%d generators x %d instances = %d tasks, %d demos each"
          % (len(gens), n_inst, len(groups), n_demo), flush=True)
    assert len(gens) >= 4, "not enough generators with a deep enough pool"

    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    enc = P.encoder_for(tok)
    m = P.build_model()
    base = m.get_base_model()
    base.config.use_cache = False
    base.gradient_checkpointing_disable()
    m.eval()

    from arc_loader import convert_grid_to_string
    embs, surf, labels = [], [], []
    with torch.inference_mode():
        for i, (gid, demos) in enumerate(groups):
            # the deployed prompt shape: demonstration pairs as user/assistant turns
            txt = "".join("<|im_start|>user\n" + convert_grid_to_string(d["input"])
                          + "<|im_end|><|im_start|>assistant\n" + convert_grid_to_string(d["output"])
                          + "<|im_end|>" for d in demos)
            ids = torch.tensor([enc(txt)[:P.MAXSEQ]], device="cuda", dtype=torch.long)
            h = m(input_ids=ids, output_hidden_states=True,
                  return_dict=True).hidden_states[-1][0].float()
            v = h.mean(0)
            v = (v / (v.norm() + 1e-9)).cpu().numpy()
            embs.append(v)
            surf.append(surface_vector(demos))
            labels.append(gid)
            if (i + 1) % 20 == 0:
                print("  embedded %d/%d" % (i + 1, len(groups)), flush=True)

    E = np.stack(embs)
    S = np.stack(surf)
    y = np.array([gens.index(g) for g in labels])
    res = {"n_generators": len(gens), "instances_per": n_inst, "n_demos": n_demo,
           "n_tasks": len(groups)}
    # Mean-pooled LLM hidden states are anisotropic: every pair sits near cosine 1 because the
    # vectors occupy a thin cone, which makes absolute similarity nearly uninformative and retrieval
    # fragile. Centring removes the shared component. It is the standard remedy and costs nothing, so
    # the question is whether it is needed here, not whether to prefer it in advance.
    Ec = E - E.mean(0, keepdims=True)
    Ec = Ec / (np.linalg.norm(Ec, axis=1, keepdims=True) + 1e-9)

    for name, M in (("model_embedding", E), ("model_embedding_centred", Ec),
                    ("surface_baseline", S)):
        C = M @ M.T
        n = len(M)
        same, diff = [], []
        for a in range(n):
            for b in range(a + 1, n):
                (same if y[a] == y[b] else diff).append(float(C[a, b]))
        same_a, diff_a = np.array(same), np.array(diff)
        pooled = np.sqrt((same_a.var(ddof=1) + diff_a.var(ddof=1)) / 2) + 1e-12
        d_eff = (same_a.mean() - diff_a.mean()) / pooled
        np.fill_diagonal(C, -2.0)
        nn = C.argmax(1)
        p_at_1 = float((y[nn] == y).mean())
        # AUC over all pairs: probability a same-rule pair scores above a different-rule pair
        allv = np.concatenate([same_a, diff_a])
        order = allv.argsort()
        ranks = np.empty(len(allv), dtype=np.float64)
        ranks[order] = np.arange(1, len(allv) + 1)
        r_same = ranks[:len(same_a)].sum()
        auc = float((r_same - len(same_a) * (len(same_a) + 1) / 2) / (len(same_a) * len(diff_a)))
        res[name] = {"cos_same": round(float(same_a.mean()), 4),
                     "cos_diff": round(float(diff_a.mean()), 4),
                     "cohens_d": round(float(d_eff), 3),
                     "precision_at_1": round(p_at_1, 3),
                     "auc": round(auc, 4),
                     "chance_p_at_1": round(1.0 * (n_inst - 1) / (len(groups) - 1), 3)}
        print("\n%s" % name, flush=True)
        for k, v in res[name].items():
            print("  %-16s %s" % (k, v), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    sb = res["surface_baseline"]
    # judge on whichever form of the embedding is better, and say which
    me = max((res["model_embedding"], res["model_embedding_centred"]), key=lambda r: r["auc"])
    which = "centred" if me is res["model_embedding_centred"] else "raw"
    print("\nbest embedding form: %s (auc %.4f)" % (which, me["auc"]), flush=True)
    print("\nVERDICT", flush=True)
    if me["auc"] < 0.6:
        print("  The embedding barely separates same-rule from different-rule pairs (AUC %.3f)."
              % me["auc"], flush=True)
        print("  Retrieval cannot work, so the adapter-bank design fails at step 5 regardless of how")
        print("  the fusion in step 6 is implemented. Kill it here.")
    elif me["auc"] <= sb["auc"] + 0.02:
        print("  The embedding separates (AUC %.3f) but no better than counting colours and grid"
              % me["auc"], flush=True)
        print("  sizes (AUC %.3f). It is recovering surface statistics, not rules, and on real ARC"
              % sb["auc"], flush=True)
        print("  tasks those come apart. The retrieval premise is not supported.")
    else:
        print("  The embedding separates same-rule pairs at AUC %.3f against %.3f for the surface"
              % (me["auc"], sb["auc"]), flush=True)
        print("  baseline, so it carries rule information beyond appearance. Retrieval is viable and")
        print("  the next question is step 6, where averaging LoRA factors is ill-defined and the")
        print("  fusion must happen in delta-W space instead.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 24,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 6,
                       int(sys.argv[3]) if len(sys.argv) > 3 else 4) else 1)
