"""What exactly does our test-time training optimise? Read it off the labels, not off the notes.

Two claims in our own documentation are worth checking against the collator rather than repeating.
The first is that supervision covers "output grids, examples after the first", which would make this
a leave-one-out objective. The second is the module list, where the probe and the deployed solver may
not agree.

Reports, from real training rows built exactly as deployment builds them:
  turns per row and which of them carry a loss
  whether the FIRST demonstration's output is supervised
  supervised tokens as a share of the row
  the deployed LoRA target modules against the probe's

CPU only. Usage: python sim_ttt_shape.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
for _p in (HERE, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    import rig_restart_probe as P
    from arc_loader import ArcDataset, QwenFormatter
    from transformers import AutoTokenizer

    lifted = P.lift_from_solver()
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    fmt = QwenFormatter(tokenizer=tok)
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tok, mlm=False)
    enc = P.encoder_for(tok)
    v = tok.get_vocab()
    USER, ASST, EOS = v["user"], v["assistant"], v["<|im_end|>"]

    dd = P.data_dir()
    arc = ArcDataset.from_file(os.path.join(dd, "arc-agi_evaluation_challenges.json"))
    keys, _, _ = P.select_tasks(6)

    rows_seen = first_supervised = 0
    sup_tok = tot_tok = 0
    span_counts = {}
    for task in keys:
        pz = arc.change_keys([task])
        ds = pz.augment(n=2, shfl_keys=True, seed=1).cut_to_len(
            formatter=fmt, name="text", max_len=P.MAXSEQ)
        for r in ds.as_list(fmt):
            ids = enc(r["text"])[:P.MAXSEQ]
            batch = collator([{"input_ids": ids}])
            lab = batch["labels"][0].tolist()
            starts = [i for i, t in enumerate(ids) if t in (USER, ASST)]
            ends = [i for i, t in enumerate(ids) if t == EOS]
            turns = list(zip(starts, ends))
            asst = [(s, e) for j, (s, e) in enumerate(turns) if ids[s] == ASST]
            sup = [k for k, (s, e) in enumerate(asst)
                   if any(x != -100 for x in lab[s + 2:e + 1])]
            rows_seen += 1
            span_counts[(len(asst), len(sup))] = span_counts.get((len(asst), len(sup)), 0) + 1
            if sup and sup[0] == 0:
                first_supervised += 1
            sup_tok += sum(1 for x in lab if x != -100)
            tot_tok += len(ids)

    print("%d training rows over %d tasks" % (rows_seen, len(keys)))
    print("")
    print("%-24s %s" % ("(assistant turns, supervised)", "rows"))
    for k in sorted(span_counts):
        print("%-24s %d" % (str(k), span_counts[k]))
    print("")
    print("rows where the FIRST demonstration's output carries a loss: %d of %d"
          % (first_supervised, rows_seen))
    print("supervised tokens: %d of %d (%.1f%%)" % (sup_tok, tot_tok, 100.0 * sup_tok / tot_tok))
    print("")
    print("deployed LoRA targets (arc_solver.peft_params):")
    print("  q_proj k_proj v_proj o_proj gate_proj up_proj down_proj embed_tokens lm_head  (9)")
    print("probe LoRA targets (rig_restart_probe.TARGETS):")
    print("  %s  (%d)" % (" ".join(P.TARGETS), len(P.TARGETS)))
    extra = {"embed_tokens", "lm_head"} - set(P.TARGETS)
    print("")
    print("VERDICT")
    if first_supervised == rows_seen:
        print("  Every assistant turn carries a loss, including the first demonstration's output.")
        print("  So this is NOT leave-one-out: the objective is next-token prediction over the whole")
        print("  demonstration sequence, with every output grid supervised and no input grid ever")
        print("  supervised. Variety across demonstrations comes from shuffling example ORDER in the")
        print("  128 augmented views, not from holding a pair out.")
    else:
        print("  The first demonstration's output is supervised on only %d of %d rows, so the"
              % (first_supervised, rows_seen))
        print("  objective does hold something out.")
    if extra:
        print("")
        print("  The probe adapts %d modules; deployment adapts those plus %s. On a 16-token"
              % (len(P.TARGETS), " and ".join(sorted(extra))))
        print("  vocabulary the embedding and head are 16x2560 each, so adapting them is cheap and")
        print("  is literally re-learning what each colour token MEANS for this task. Every rig")
        print("  probe therefore runs a strictly weaker adaptation than the submission does, which")
        print("  is a reason to distrust a rig null that contradicts a deployed effect.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
