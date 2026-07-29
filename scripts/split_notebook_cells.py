"""Split a built notebook into one file per cell, for pasting into Kaggle by hand.

Kaggle's notebook import has failed on these builds more than once, and the fallback is to create a
blank notebook and paste cell by cell. Doing that from the .ipynb JSON is error-prone, so this writes
numbered files in execution order with names that say what each cell is.

Usage: python split_notebook_cells.py v7.5a
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v7.5a"
NB_DIR = r"D:\ARC_AGI-2,3\notebooks"
FILES = {"v7.1": "arc_submission_v7.ipynb", "v7.2": "arc_submission_v7_2.ipynb",
         "v7.3": "arc_submission_v7_3.ipynb", "v7.4": "arc_submission_v7_4.ipynb",
         "v7.5a": "arc_submission_v7_5a.ipynb", "v7.5b": "arc_submission_v7_5b.ipynb",
         "v7.5c": "arc_submission_v7_5c.ipynb"}
assert VARIANT in FILES, "unknown variant %r, expected one of %s" % (VARIANT, sorted(FILES))

SRC = os.path.join(NB_DIR, FILES[VARIANT])
OUT = os.path.join(NB_DIR, VARIANT.replace(".", "_") + "_cells")


def inert(text):
    """True when a cell cannot execute anything, so pasting it is optional.

    The source notebook carries fully commented-out duplicates of two modules, several hundred lines
    each. Verified rather than assumed: every non-blank line must be a comment.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    return not lines or all(l.lstrip().startswith("#") for l in lines)


def name_for(text, is_md):
    if is_md:
        return "intro", "md"
    m = re.match(r"^%%writefile\s+(\S+)", text)
    if m:
        return os.path.splitext(os.path.basename(m.group(1)))[0], "py"
    m = re.match(r"^#\s*%%writefile\s+(\S+)", text)
    if m and inert(text):
        return "SKIP_commented_" + os.path.splitext(os.path.basename(m.group(1)))[0], "py"
    if "ARC_MODEL_PATH" in text:
        return "config", "py"
    if "pip" in text and "unsloth" in text:
        return "install_unsloth", "py"
    if "uninstall" in text:
        return "uninstall_tf", "py"
    if "end_time" in text and len(text) < 400:
        return "endtime", "py"
    if "mp.spawn" in text or "local_worker" in text:
        return "starter", "py"
    if "validate_submission" in text or "get_submission" in text:
        return "scoring", "py"
    if text.lstrip().startswith("#") and text.count("#") > text.count("\n") * 0.5:
        return "commented_ref", "py"
    return "cell", "py"


nb = json.load(open(SRC, encoding="utf-8"))
os.makedirs(OUT, exist_ok=True)
for stale in os.listdir(OUT):
    os.remove(os.path.join(OUT, stale))

manifest = []
for i, c in enumerate(nb["cells"], start=1):
    text = "".join(c["source"])
    is_md = c["cell_type"] == "markdown"
    base, ext = name_for(text, is_md)
    skip = (not is_md) and inert(text)
    if skip and not base.startswith("SKIP"):
        base = "SKIP_empty" if not text.strip() else "SKIP_" + base
        ext = "py"
    fname = "%02d_%s.%s" % (i, base, ext)
    with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    manifest.append((fname, c["cell_type"], len(text.splitlines()), skip))

need = [m for m in manifest if not m[3]]
skipped = [m for m in manifest if m[3]]
with open(os.path.join(OUT, "00_ORDER.txt"), "w", encoding="utf-8") as f:
    f.write("Paste into a FRESH Kaggle notebook in this order, one cell each.\n")
    f.write("The first is a Markdown cell, every other is Code.\n")
    f.write("Files marked SKIP execute nothing (every non-blank line is a comment, or the cell is\n")
    f.write("empty) and can be left out. They are kept here only so the numbering matches the\n")
    f.write("notebook.\n\n")
    for fname, ctype, n, sk in manifest:
        f.write("%-34s %-9s %4d lines%s\n" % (fname, ctype, n, "   <- skip" if sk else ""))
    f.write("\n%d cells to paste, %d skippable (%d lines saved).\n"
            % (len(need), len(skipped), sum(m[2] for m in skipped)))

print("wrote", len(manifest), "cells to", OUT)
for fname, ctype, n, sk in manifest:
    print("  %-34s %-9s %4d lines%s" % (fname, ctype, n, "   <- skip" if sk else ""))
print("\n%d to paste, %d skippable, %d lines of dead comments avoided"
      % (len(need), len(skipped), sum(m[2] for m in skipped)))
