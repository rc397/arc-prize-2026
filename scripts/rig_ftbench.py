"""Why does the compositional fine-tune cost 11.7 seconds a step, and what actually fixes it?

The pilot took 26 hours for 8,000 examples, which prices the full 78,792-example corpus at eleven
days on this card and puts the only model-changing lever we have out of reach. Before renting
anything, find out where the time goes.

The suspicion, and it is only a suspicion until this runs. `src/arc/model.py:load_model` never passes
`attn_implementation`, while `rig_restart_probe.build_model` explicitly asks for sdpa, and this
repository's own TTT docstring records "~13s/step under eager training" against "~2s/step" otherwise.
We measured 11.7. Eager attention also materialises an n-by-n score matrix per head, which at a
1,561-token median and 32 heads is hundreds of megabytes per layer, and the run sat at 97.6% of 16 GB.
One cause would explain the speed and the memory together.

Four configurations, each timed on the same sequences, reported in TOKENS per second so that batch
sizes are comparable:

  eager_b1        what the pilot ran
  sdpa_b1         one line changed
  sdpa_bucket     plus length-bucketed batching under a token cap. The corpus median is 1,561
                  tokens, so this is worth less here than it would be on short sequences: about 2.4
                  sequences per 4,096-token batch. Measured anyway rather than assumed.
  sdpa_bucket_nc  plus gradient checkpointing off, which trades memory for about a third of the
                  compute and is only affordable if sdpa frees enough

Every configuration trains the same LoRA on the same data for the same number of SEQUENCES, so the
comparison is throughput at equal work. An out-of-memory failure is reported as a result, not raised.

Usage: python rig_ftbench.py [N_SEQ=48] [MAX_LEN=2048] [TOKEN_CAP=4096]
"""
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
ROOT = os.path.dirname(HERE)
for _p in (HERE, BASE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(BASE, "_corpus_comp", "train.jsonl")
MODEL = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                     "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")


def load_some(path, n, max_len):
    """First n examples at or under max_len, in file order so every config sees the same work."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if len(d["ids"]) <= max_len:
                out.append((d["ids"], d["labels"]))
                if len(out) == n:
                    break
    return out


def bucket(examples, token_cap):
    """Sort by length, then fill batches until padded size would exceed the cap.

    Padded size is max_len_in_batch times batch_size, which is what the GPU actually allocates, not
    the sum of the true lengths. Sorting first is what keeps that product close to the sum.
    """
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches, cur, mx = [], [], 0
    for i in order:
        L = len(examples[i][0])
        if cur and max(mx, L) * (len(cur) + 1) > token_cap:
            batches.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, L)
    if cur:
        batches.append(cur)
    return batches


def collate(examples, idxs, device):
    import torch
    rows = [examples[i] for i in idxs]
    m = max(len(r[0]) for r in rows)
    ids = torch.zeros(len(rows), m, dtype=torch.long)
    lab = torch.full((len(rows), m), -100, dtype=torch.long)
    att = torch.zeros(len(rows), m, dtype=torch.long)
    for k, (i_, l_) in enumerate(rows):
        n = len(i_)
        ids[k, :n] = torch.tensor(i_, dtype=torch.long)
        lab[k, :n] = torch.tensor(l_, dtype=torch.long)
        att[k, :n] = 1
    return (ids.to(device), lab.to(device), att.to(device))


def run(name, attn, token_cap, checkpointing, examples, device="cuda", reentrant=False):
    import torch
    from transformers import AutoModelForCausalLM
    from src.arc.ttt import make_adapter

    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        kw = {"dtype": torch.bfloat16}
        if attn:
            kw["attn_implementation"] = attn
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).to(device)
        peft_model, _ = make_adapter(model, rank=32, alpha=32, dropout=0.05)
        base = peft_model.get_base_model()
        base.config.use_cache = False
        if checkpointing:
            # finetune_base.py calls this with no arguments, which selects the REENTRANT variant.
            # rig_restart_probe deliberately does not: reentrant checkpointing forks and restores the
            # CUDA RNG so it can replay the forward, and on this torch build that path is the
            # suspect. This flag is what makes the two comparable.
            if reentrant:
                base.gradient_checkpointing_enable()
            else:
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        params = [p for p in peft_model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4)
        peft_model.train()

        batches = bucket(examples, token_cap) if token_cap else [[i] for i in range(len(examples))]
        # one warmup batch, excluded from timing, so cuda init and autotune do not land in the number
        ids, lab, att = collate(examples, batches[0], device)
        peft_model(input_ids=ids, labels=lab, attention_mask=att).loss.backward()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        t0 = time.time()
        n_seq = n_tok = 0
        for bidx in batches:
            ids, lab, att = collate(examples, bidx, device)
            loss = peft_model(input_ids=ids, labels=lab, attention_mask=att).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_seq += len(bidx)
            n_tok += sum(len(examples[i][0]) for i in bidx)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        out = {"name": name, "ok": True, "seconds": round(dt, 1), "batches": len(batches),
               "seqs": n_seq, "tokens": n_tok, "s_per_seq": round(dt / n_seq, 3),
               "tok_per_s": round(n_tok / dt), "peak_gb": round(peak, 2),
               "mean_batch": round(n_seq / len(batches), 2)}
    except Exception as e:
        out = {"name": name, "ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:120])}
    finally:
        for v in ("opt", "peft_model", "model", "base"):
            if v in dir():
                pass
        gc.collect()
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass
    return out


def main(n_seq, max_len, token_cap):
    examples = load_some(CORPUS, n_seq, max_len)
    lens = sorted(len(e[0]) for e in examples)
    print("%d sequences, median %d tokens, max %d, %d tokens total"
          % (len(examples), lens[len(lens) // 2], lens[-1], sum(lens)), flush=True)
    print("")

    # attn None means "pass nothing", which is what src/arc/model.py:load_model does.
    configs = [
        ("pilot_exact", None, 0, True, True),
        ("attn_unset", None, 0, True, False),
        ("eager_b1", "eager", 0, True, False),
        ("sdpa_b1", "sdpa", 0, True, False),
        ("sdpa_bucket", "sdpa", token_cap, True, False),
        ("sdpa_bucket_nc", "sdpa", token_cap, False, False),
    ]
    only = os.environ.get("ARC_BENCH_ONLY")
    if only:
        want = set(only.split(","))
        configs = [c for c in configs if c[0] in want]
    rows = []
    for name, attn, cap, ckpt, reent in configs:
        r = run(name, attn, cap, ckpt, examples, reentrant=reent)
        rows.append(r)
        if r["ok"]:
            print("%-16s %6.1fs  %5d tok/s  %6.3f s/seq  batch %4.2f  peak %5.2f GB"
                  % (name, r["seconds"], r["tok_per_s"], r["s_per_seq"], r["mean_batch"],
                     r["peak_gb"]), flush=True)
        else:
            print("%-16s FAILED  %s" % (name, r["error"]), flush=True)

    ok = [r for r in rows if r["ok"]]
    print("")
    print("VERDICT")
    if len(ok) < 2:
        print("  Not enough configurations completed to compare.")
        return True
    base_row = next((r for r in ok if r["name"] == "eager_b1"), ok[0])
    best = max(ok, key=lambda r: r["tok_per_s"])
    speed = best["tok_per_s"] / base_row["tok_per_s"]
    print("  Best is '%s' at %.2fx the pilot's throughput (%d against %d tokens/s)."
          % (best["name"], speed, best["tok_per_s"], base_row["tok_per_s"]))
    sd = next((r for r in ok if r["name"] == "sdpa_b1"), None)
    if sd:
        print("  Of that, changing ONE argument to sdpa accounts for %.2fx, and it drops peak memory"
              % (sd["tok_per_s"] / base_row["tok_per_s"]))
        print("  from %.2f GB to %.2f GB, which is what makes the rest affordable."
              % (base_row["peak_gb"], sd["peak_gb"]))
    days = 78792 * (sum(len(e[0]) for e in examples) / len(examples)) / best["tok_per_s"] / 86400
    old = 78792 * (sum(len(e[0]) for e in examples) / len(examples)) / base_row["tok_per_s"] / 86400
    print("  Full corpus, one epoch: %.1f days becomes %.1f days." % (old, days))
    with open(os.path.join(BASE, "experiments", "ftbench.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return True


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 48,
                       int(sys.argv[2]) if len(sys.argv) > 2 else 2048,
                       int(sys.argv[3]) if len(sys.argv) > 3 else 4096) else 1)
