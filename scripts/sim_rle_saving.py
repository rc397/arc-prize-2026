"""What would run-length compressing background actually save? Idea 4, priced properly.

The bounding-box measurement answered a different question: how much EDGE margin is removable. This
one asks how much of a grid is background at all, how many whole rows are solid background, and what
a run-length encoding of background runs would save in tokens. Those can be large even when the
bounding box is tight, so the earlier 5.8 percent figure does not bound this.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P

ch = json.load(open(os.path.join(P.data_dir(), "arc-agi_evaluation_challenges.json")))
keys = sorted(ch)[:120]

cells = bg_cells = 0
rows_tot = rows_solid = 0
tok_plain = tok_rle = 0
MINRUN = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # runs shorter than this are not worth a marker

for k in keys:
    grids = []
    for p in ch[k]["train"]:
        grids += [p["input"], p["output"]]
    for t in ch[k]["test"]:
        grids.append(t["input"])
    for g in grids:
        for row in g:
            rows_tot += 1
            cells += len(row)
            bg_cells += sum(1 for v in row if v == 0)
            if all(v == 0 for v in row):
                rows_solid += 1
            # plain: one token per cell. rle: a background run of length L >= MINRUN becomes
            # 2 tokens (marker + length); everything else stays one token per cell.
            tok_plain += len(row)
            i = 0
            while i < len(row):
                if row[i] == 0:
                    j = i
                    while j < len(row) and row[j] == 0:
                        j += 1
                    L = j - i
                    tok_rle += 2 if L >= MINRUN else L
                    i = j
                else:
                    tok_rle += 1
                    i += 1

print("120 tasks, %d rows, %d cells" % (rows_tot, cells))
print("background (colour 0) share of cells : %.1f%%" % (100.0 * bg_cells / cells))
print("rows that are ENTIRELY background    : %d of %d (%.1f%%)" % (rows_solid, rows_tot, 100.0 * rows_solid / rows_tot))
print("cell tokens, plain                   : %d" % tok_plain)
print("cell tokens, run-length (min run %d)  : %d" % (MINRUN, tok_rle))
print("saving                               : %.1f%%" % (100.0 * (1 - tok_rle / tok_plain)))
print()
print("median prompt is 2198 tokens; after this saving it would be about %d" % int(2198 * tok_rle / tok_plain))
