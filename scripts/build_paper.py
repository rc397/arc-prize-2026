"""Build the paper and FAIL LOUDLY on anything LaTeX complains about, not just undefined references.

Written after shipping a malformed table. A script inserted a section whose tabular rows ended in one
backslash instead of two, LaTeX reported "Extra alignment tab has been changed to \\cr" twenty-two
times, the rows ran together in the PDF, and the check in use at the time grepped only for the word
"undefined" and reported everything fine. A build check that passes a broken document is worse than
no check, because it converts a visible failure into an invisible one.

Fails on: any line starting "!", any "Extra alignment", any undefined reference or citation, and any
multiply-defined label. Warns on overfull boxes without failing, since a few are unavoidable.

Usage: python build_paper.py [--quick]     (--quick runs one pass instead of two)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(HERE, "paper")
TEX = "the-16gb-ceiling.tex"
LOG = os.path.join(PAPER, "the-16gb-ceiling.log")

FATAL = [
    (re.compile(r"^! "), "LaTeX error"),
    (re.compile(r"Extra alignment tab"), "malformed table row (a row needs two backslashes)"),
    (re.compile(r"Misplaced alignment tab"), "stray & outside a tabular"),
    (re.compile(r"Reference .* undefined"), "undefined reference"),
    (re.compile(r"Citation .* undefined"), "undefined citation"),
    (re.compile(r"multiply.defined", re.I), "duplicate label"),
    (re.compile(r"Runaway argument"), "unbalanced braces"),
]


def main(quick):
    passes = 1 if quick else 2
    for i in range(passes):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", TEX],
                       cwd=PAPER, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(LOG, encoding="utf-8", errors="replace").read().split("\n")

    hits = {}
    for line in log:
        for pat, label in FATAL:
            if pat.search(line):
                hits.setdefault(label, []).append(line.strip()[:110])
    overfull = sum(1 for l in log if l.startswith("Overfull"))
    pages = re.findall(r"\((\d+) pages", "\n".join(log))

    print("pages: %s   overfull boxes: %d" % (pages[-1] if pages else "?", overfull))
    if not hits:
        print("BUILD CLEAN")
        return True
    print("BUILD HAS PROBLEMS")
    for label, lines in hits.items():
        print("  %-46s %d" % (label, len(lines)))
        for l in lines[:3]:
            print("      %s" % l)
        if len(lines) > 3:
            print("      ... and %d more" % (len(lines) - 3))
    return False


if __name__ == "__main__":
    sys.exit(0 if main("--quick" in sys.argv) else 1)
