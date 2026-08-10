"""Does the prompt budget actually cost us demonstrations? The number the whole token-saving family needs.

Run-length encoding, coordinate interleaving, digit-position prefixes, uniform row deletion and the
rest of that cluster all argue the same way: the serialization is wasteful, so compress it. Every one
of them was priced by how many TOKENS it saves, and none by what those tokens are worth.

They are worth exactly one thing. `cut_to_len` drops demonstration pairs from the FRONT of the prompt
until the text fits, so an over-long view trains or decodes on fewer examples than the task provides.
That is the only channel by which a token saving can change an answer. If no view ever loses a
demonstration, the family is dead whatever its compression ratio, and if many do, it has a real
motivation for the first time.

The count is exact and needs no tokenizer. The encoder is one token per digit, one per newline, and
a fixed three for a turn opener plus one for its close (rig_restart_probe.encoder_for), so a grid of
h by w costs h*w + h - 1 plus its wrapper. Three details decide whether the number is right:

  the two paths differ. Training rows are cut on the full text at MAXSEQ. Eval rows are cut on the
  INPUT only, against MAXSEQ minus max_new_tokens, which is 931 for a 30x30 reply. The eval limit is
  the tighter one and it is the one that governs which candidates can exist at all.

  the training text never contains the test pair. With no known reply, ArcDataset.get falls back to
  fmt_train(..., last_is_challenge=True), which reformats the last TRAINING pair as the query. On
  Kaggle the answer is unknown, so this is always the path taken.

  the D4 group contains the transpose, so a grid pays w-1 row separators in half the views and h-1
  in the other half. cut_to_len runs per view, so the longer orientation is the one that drops
  demonstrations, and both are counted here.

Usage: python sim_token_budget.py
"""
import json
import os
import sys

DD = os.path.join("experiments", "_data")
KAGGLE, RIG = 8192, 6144
MAX_NEW = 931            # len(encode(fmt_reply(30x30))) + 1, the reserve decode_pool subtracts
SAVINGS = 0.139          # measured run-length saving, sim_rle_saving.py


def grid_tokens(g, transposed):
    h, w = len(g), len(g[0])
    return h * w + (w - 1 if transposed else h - 1)


def pair_tokens(p, t):
    # <|im_start|> user \n GRID <|im_end|>   then the same shape for the assistant turn
    return (3 + grid_tokens(p["input"], t) + 1) + (3 + grid_tokens(p["output"], t) + 1)


def kept(body, tail, limit):
    """cut_to_len drops from the front until it fits. Returns demonstrations kept."""
    n = len(body)
    while n > 0 and sum(body[len(body) - n:]) + tail > limit:
        n -= 1
    return n


def survey(ch, limit, path, scale=1.0):
    """path 'train' cuts the full text; path 'eval' cuts the input against the reduced limit."""
    views = lose = lost = total = 0
    for tid, task in ch.items():
        for t in (False, True):
            body = [int(pair_tokens(p, t) * scale) for p in task["train"]]
            if path == "train":
                targets = [0]
            else:
                targets = [int((3 + grid_tokens(q["input"], t) + 1 + 3) * scale)
                           for q in task["test"]]
            for tail in targets:
                views += 1
                total += len(body)
                k = kept(body, tail, limit)
                lost += len(body) - k
                lose += int(k < len(body))
    return views, lose, lost, total


def main():
    ch = json.load(open(os.path.join(DD, "arc-agi_evaluation_challenges.json")))

    print("%-22s %8s %10s %12s %12s" % ("path and limit", "views", "lose >=1", "demos lost",
                                        "of demos"))
    print("-" * 68)
    cases = [("train, Kaggle 8192", KAGGLE, "train"),
             ("eval,  Kaggle 7261", KAGGLE - MAX_NEW, "eval"),
             ("train, rig 6144", RIG, "train"),
             ("eval,  rig 5213", RIG - MAX_NEW, "eval")]
    rows = {}
    for name, limit, path in cases:
        v, lose, lost, total = survey(ch, limit, path)
        rows[name] = (v, lose, lost, total, limit, path)
        print("%-22s %8d %10d %12d %12d" % (name, v, lose, lost, total))

    print("")
    print("with a %.0f%% serialization saving applied to every grid:" % (100 * SAVINGS))
    for name, limit, path in cases:
        v, lose, lost, total = survey(ch, limit, path, scale=1 - SAVINGS)
        was = rows[name][2]
        print("  %-22s demos lost %d, against %d now (%+d)" % (name, lost, was, lost - was))

    print("")
    print("VERDICT")
    v, lose, lost, total, limit, path = rows["eval,  Kaggle 7261"]
    ve, lose_e, lost_e, _, _, _ = rows["train, Kaggle 8192"]
    print("  On the deployed path, %d of %d eval views lose a demonstration and %d of %d"
          % (lose, v, lost, total))
    print("  demonstrations are dropped; on the training path, %d views and %d demonstrations."
          % (lose_e, lost_e))
    if lost + lost_e <= 0.02 * total:
        print("  That is the entire prize available to compression. The serialization is not the")
        print("  binding constraint at 8192, so run-length encoding, coordinate interleaving and")
        print("  digit-position prefixes are saving tokens nothing is competing for. The family is")
        print("  dead on this number, independent of how well any of them compresses.")
    else:
        print("  Compression has real room here: it buys back demonstrations, which is the only")
        print("  currency that can change an answer.")
    rv, rlose, rlost, rtotal, _, _ = rows["eval,  rig 5213"]
    print("")
    print("  Over the full 120 tasks, the rig's 6144 loses %d demonstrations across %d eval views"
          % (rlost, rlose))
    print("  against %d at Kaggle's 8192. That does NOT automatically apply to our probes, which"
          % lost)
    print("  run a stratified slice, and rig_restart_probe's smoke stage asserts that 6144 cuts")
    print("  nothing on the twenty tasks it selects. The two statements are compatible: the cutting")
    print("  is concentrated in the long tail. Check the slice before quoting either number.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
