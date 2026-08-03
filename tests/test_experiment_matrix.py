"""Keep the experiment record honest about the builds it describes.

Five faults in this project have been the same species, a claim about what the code does surviving
because nothing checked it against the code. The builder already defends the notebooks: every knob
must reach the solver, no cap may stay hardcoded, no off-switch may be a truthy string. Nothing
defended the documentation, and notebooks/EXPERIMENTS.md drifted three claims out of date. It listed
a cap for v7.1 that never ran, credited v7.2's gain to a cap change that never happened, and omitted
v7.4 and the whole v7.5 bisection.

These are cheap structural checks, not a substitute for reading. They catch drift, not wrongness.

Run: python tests/test_experiment_matrix.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO, "scripts", "build_v7_from_architects.py")
MATRIX = os.path.join(REPO, "notebooks", "EXPERIMENTS.md")

failures = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        failures.append(msg)


def builder_variants():
    """Variant names and env dicts, read without importing (the builder runs a build on import)."""
    src = open(BUILDER, encoding="utf-8").read()
    block = src[src.index("VARIANTS = {"):src.index("\n}", src.index("VARIANTS = {")) + 2]
    names = re.findall(r'^    "(v[\d.]+[a-c]?)":', block, re.M)
    envs = {}
    for name in names:
        seg = block[block.index('"%s":' % name):]
        m = re.search(r'"env":\s*\{(.*?)\}', seg, re.S)
        envs[name] = dict(re.findall(r'"([A-Z_]+)":\s*"([^"]*)"', m.group(1))) if m else {}
    return names, envs


def main():
    names, envs = builder_variants()
    doc = open(MATRIX, encoding="utf-8").read()
    print("builder defines %d variants: %s\n" % (len(names), ", ".join(names)))

    print("every built variant appears in the matrix")
    for n in names:
        check(n in doc, "%s documented" % n)

    print("\nno variant is claimed that the builder cannot build")
    # The family-tree section only. Prose legitimately names things that do not exist, such as the
    # line recording that there is no v7.3a, and later sections carry tables of RUNS rather than
    # variants, where "v7.2 replicate" is a correct row and not a buildable name.
    section = doc.split("## The family tree", 1)[-1].split("\n## ", 1)[0]
    rows = [r for r in section.splitlines() if r.startswith("| v7.")]
    check(bool(rows), "the family-tree section still has a variant table")
    for n in sorted({r.split("|")[1].strip() for r in rows}):
        check(n in names, "table row %s exists in the builder" % n)

    print("\ncommit cost stays cheap")
    for n in names:
        v = envs[n].get("ARC_N_EVAL", "4")
        check(v.isdigit() and 0 < int(v) <= 8,
              "%s commits at ARC_N_EVAL=%s (must be 1..8, and 0 means all 120)" % (n, v))
    # 4 rather than 1 because NPROCS=4: one task exercises only rank 0, so a broken worker passes.
    check(all(envs[n].get("ARC_N_EVAL", "4") == "4" for n in names),
          "every variant commits at 4, one task per mp.spawn rank")

    print("\nthe builder still defends what it learned to defend")
    src = open(BUILDER, encoding="utf-8").read()
    for marker, why in (
            ('"per-task cap still hardcoded"', "cap cannot be hardcoded again"),
            ('"truthy-string gate on the fill"', "an off-switch cannot be a truthy string"),
            ("commit-subset gate missing", "commits cannot silently return to the full eval set"),
            ("is echoed but never read by the solver", "knobs must reach the code they claim to set")):
        check(marker in src, why)

    print("\nfacts the matrix must not lose")
    check("1800" in doc and "never wired" in doc.lower() or "cap ACTUALLY run" in doc,
          "the matrix distinguishes the cap set from the cap run")
    check("mikelou1" in doc, "the matrix names the notebook that actually carries 33.89")

    print("\n%s (%d checks failed)" % ("PASS" if not failures else "FAIL", len(failures)))
    return not failures


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
