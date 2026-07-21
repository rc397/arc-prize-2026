"""RETIRED. This built no submission and should not build one without a new rank sweep behind it.

It forks the v6 deep-TTT submission (18.19 hidden) into v7-B by pushing the LoRA rank to 128, everything
else identical. The premise was an early partial reading that higher rank helped the greedy vote. The full
sweep afterwards refuted it. At 60 steps the scores are 24, 27, 28 and 24 of 125 for ranks 16, 32, 64 and
128, an inverted U peaking at the rank v6.1 already ships. Rank 128 is joint worst, so this build would be
expected to score below the 18.19 it forked from, and it was retired without spending a submission slot.
Rank 64 leads rank 32 by one output, inside seed noise at double the adapter cost, so no rank on the curve
is worth resubmitting for either.

Kept for the record and for the rank knob, since a future sweep at a different training context could move
the peak. The notebook it produced is archived under notebooks/archive/.

Usage: python build_v7b_from_v6.py [RANK=128]
"""
import json
import os
import re
import sys

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "notebooks", "arc_submission_v6.ipynb")
DST = os.path.join(ROOT, "notebooks", "arc_submission_v7b.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))
n_rank = n_amb = n_scr = n_title = 0
for c in nb["cells"]:
    s = "".join(c["source"])
    s, k = re.subn(r"TTT_RANK(\s*)=(\s*)32",
                   lambda m: "TTT_RANK%s=%s%d" % (m.group(1), m.group(2), RANK), s)
    n_rank += k
    s, k = re.subn(r"ADAPTER_MB\s*=\s*126 if TTT_RANK >= 32 else 63",
                   "ADAPTER_MB = max(63, int(126 * TTT_RANK / 32))", s)
    n_amb += k
    s2 = s.replace("loras_v6", "loras_v7b")
    n_scr += (s2 != s)
    s = s2
    s2 = s.replace("v6 (deep-regularized per-task TTT", "v7-B (rank-pushed deep-regularized per-task TTT")
    n_title += (s2 != s)
    s = s2
    c["source"] = s.splitlines(keepends=True)

assert n_rank >= 1 and n_amb >= 1 and n_scr >= 1 and n_title >= 1, \
    "a patch did not apply: rank=%d amb=%d scratch=%d title=%d" % (n_rank, n_amb, n_scr, n_title)
json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1)
print("wrote", DST, "at rank", RANK, "| hits rank=%d amb=%d scratch=%d title=%d"
      % (n_rank, n_amb, n_scr, n_title))
