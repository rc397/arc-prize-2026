"""Push v7.7, the DFS cache-crop build, so it is submittable in the next UTC window.

turbo_dfs threads one KV cache through the recursion and reuses it for every sibling with no
truncation. On transformers 4.55, which the Kaggle image runs, DynamicCache appends rather than
overwrites, so each sibling attends over every branch explored before it and the cache grows for the
whole 540-second search. Our own local harness clones the cache per branch and its docstring says
why, so the decoder we validated and the decoder we ship have differed all along.

Measured on ten tasks, one adapter decoded three ways: cropping recovers 27 percent more candidates
(51 against 65, ahead on 8 of 10) and is identical to a full deepcopy on every task. The solve
difference was one task in ten, far under the roughly three points this instrument resolves, so this
goes out as a CORRECTNESS fix rather than a scored lever.

Usage: python push_v77.py
"""
import json
import os
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
WORK = os.path.join(BASE, "_kpush77")
NB = "arc-agi-2-v7-7-greedy-fallback.ipynb"
SRC = os.path.join(BASE, "notebooks", "arc_submission_v7_7.ipynb")
SLUG = "rc7530/arc-agi-2-v7-7-dfs-cache-crop"   # must match what the TITLE slugs to
UTIL = "rc7530/pip-install-unsloth-flash-patch"


def main():
    assert os.path.isfile(SRC), "build it first: python build_v7_from_architects.py v7.7 (%s)" % SRC
    os.makedirs(WORK, exist_ok=True)
    shutil.copy(SRC, os.path.join(WORK, NB))

    meta = {
        "id": SLUG,
        "title": "ARC AGI 2 v7 7 dfs cache crop",
        "code_file": NB,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        # REQUIRED, and its absence is silent. Without it the kernel gets machine_shape "Gpu" and
        # Kaggle may allocate a Tesla P100 (sm_60), on which this stack dies with "no kernel image is
        # available for execution on the device" -- torch 2.8+cu128 supports sm_70 and up. The
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
    assert not bad, "model source rejected: %s -- the kernel would run without a model" % bad

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
