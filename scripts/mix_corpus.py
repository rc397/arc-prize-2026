"""Interleave the composed corpus with single-rule data, to test whether mixing prevents forgetting.

The full compositional fine-tune learned what it was given and lost what it was not: composed-rule
coverage went 3 of 8 to 6 of 8 with candidate generation up 200 percent, while ARC-AGI-2 coverage
went 5 of 20 to 3 and candidates fell 47 percent, and per-task test-time training did not rebuild it.
That is catastrophic forgetting on a narrow distribution, and the textbook remedy is to keep some of
the original distribution in the training stream. We never tried it.

The design is EQUAL COMPUTE, not equal composed exposure. The pilot spent 8,000 steps entirely on
composed data; this spends 8,000 steps on half composed and half single-rule. So the comparison
answers the question a compute-poor project actually faces --- given a fixed training budget, is a
mixture better than the pure thing --- rather than the question of whether more total data helps,
which we already know costs 23 hours to ask.

Interleaving rather than concatenating matters: the trainer walks the file in order with gradient
accumulation every 8 steps, so a concatenated corpus would spend its first half in one distribution
and its second half in the other, which is a curriculum and not a mixture.

Usage: python mix_corpus.py <COMPOSED_DIR> <SINGLE_DIR> <OUT_DIR> [N_EACH=4000] [N_VAL_EACH=250]
"""
import io
import json
import os
import sys


def head(path, n, max_len=2048):
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if len(d["ids"]) <= max_len:
                out.append(line.rstrip("\n"))
                if len(out) == n:
                    break
    return out


def write(path, rows):
    with io.open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")


def main(comp_dir, single_dir, out_dir, n_each, n_val_each):
    os.makedirs(out_dir, exist_ok=True)
    for name, n in (("train.jsonl", n_each), ("val.jsonl", n_val_each)):
        a = head(os.path.join(comp_dir, name), n)
        b = head(os.path.join(single_dir, name), n)
        if len(a) < n or len(b) < n:
            print("WARNING %s: composed %d, single %d, wanted %d each" % (name, len(a), len(b), n))
        mixed = []
        for i in range(max(len(a), len(b))):
            if i < len(a):
                mixed.append(a[i])
            if i < len(b):
                mixed.append(b[i])
        write(os.path.join(out_dir, name), mixed)
        print("%s: %d composed + %d single = %d rows, strictly alternating"
              % (name, len(a), len(b), len(mixed)))
    return True


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2], sys.argv[3],
                       int(sys.argv[4]) if len(sys.argv) > 4 else 4000,
                       int(sys.argv[5]) if len(sys.argv) > 5 else 250) else 1)
