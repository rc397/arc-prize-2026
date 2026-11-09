"""Push v7.7b, truncation-fair view scheduling, so it is submittable in the next UTC window.

The 16 evaluation views decode in FOUR batches and the per-task cap is checked only between them,
while each DFS call may burn a flat 540 seconds. Simulated against the budget the scoring rerun
actually has, 240 tasks over 12 hours on four L4s giving about 720 seconds a task and TTT taking
roughly 272 of them, the flat constant decodes ONE batch of four. Fair scheduling, dividing the
remaining task time over the batches still to run, decodes all four. Four of sixteen views against
sixteen of sixteen, and multiplicity across views is what kgmon ranks on.

This is C1 in the ranked levers and was never built. Its kill condition asked for the skip rate to be
logged first, which we could not do: the commit run has no time pressure and Kaggle does not expose
the scoring run's log. The arithmetic is the evidence and the score is the test.

Usage: python push_v77b.py
"""
import json
import os
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
WORK = os.path.join(BASE, "_kpush77b")
NB = "arc-agi-2-v7-7-greedy-fallback.ipynb"
SRC = os.path.join(BASE, "notebooks", "arc_submission_v7_7b.ipynb")
SLUG = "rc7530/arc-agi-2-v7-7b-fair-view-scheduling"   # must match what the TITLE slugs to
UTIL = "rc7530/pip-install-unsloth-flash-patch"


def main():
    assert os.path.isfile(SRC), "build it first: python build_v7_from_architects.py v7.7b (%s)" % SRC
    os.makedirs(WORK, exist_ok=True)
    shutil.copy(SRC, os.path.join(WORK, NB))

    meta = {
        "id": SLUG,
        "title": "ARC AGI 2 v7 7b fair view scheduling",
        "code_file": NB,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # REQUIRED, and its absence is silent. Without it the kernel gets machine_shape "Gpu" and
        # Kaggle may allocate a Tesla P100 (sm_60), on which this stack dies with "no kernel image is
        # available for execution on the device", but torch 2.8+cu128 supports sm_70 and up. The
        # SCORING rerun uses L4s and would have been fine; it is the commit run that fails, and a
        # kernel whose commit fails has no output and therefore cannot be submitted at all.
        # v7-2-replicate carries NvidiaL4, which is where this string came from.
        "machine_shape": "NvidiaL4",
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "kernel_sources": [UTIL],
        # Exactly the string the working kernels carry, pulled from v7-2-replicate's own metadata.
        # The mirza mirror path and a lowercase "transformers" are both REJECTED as an invalid
        # model source, and the push still succeeds when they are, leaving a kernel with no model
        # attached and a run that fails at the assert. Check invalidModelSources in the response.
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
    }
    json.dump(meta, open(os.path.join(WORK, "kernel-metadata.json"), "w"), indent=1)

    api = KaggleApi()
    api.authenticate()
    print("pushing %s" % SLUG, flush=True)
    r = api.kernels_push(WORK)
    print(r, flush=True)
    bad = getattr(r, "invalidModelSources", None) or (r.get("invalidModelSources")
                                                      if isinstance(r, dict) else None)
    if not bad:
        try:
            bad = json.loads(str(r)).get("invalidModelSources")
        except Exception:
            bad = None
    assert not bad, "model source rejected: %s, the kernel would run without a model" % bad

    # The kernel is not submittable until this commit run finishes and leaves an output behind,
    # which is exactly the precondition v7.3 now fails.
    status = None
    for i in range(90):
        time.sleep(60)
        try:
            st = api.kernels_status(SLUG)
            status = st.get("status") if isinstance(st, dict) else getattr(st, "status", str(st))
            print("  [%d min] %s" % (i + 1, status), flush=True)
            s = str(status).upper()
            if "COMPLETE" in s or "ERROR" in s or "CANCEL" in s:
                break
        except Exception as e:
            print("  status err %s" % str(e)[:120], flush=True)
    print("final status: %s" % status, flush=True)
    ok = status and "COMPLETE" in str(status).upper()
    print("SUBMITTABLE TOMORROW" if ok else "NOT SUBMITTABLE, the commit run did not complete")
    return bool(ok)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
