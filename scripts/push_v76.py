"""Push v7.6, the greedy-fallback-only build, so it is submittable tomorrow.

v7.3 bundled the greedy fallback with a loss cut and a 1200-second cap and scored 28.61, so the one
mechanism worth isolating never was. It is also no longer reachable: Kaggle holds no output for that
kernel and both of its versions return 400 FAILED_PRECONDITION, so re-testing it means rebuilding.

This is v7.2's pipeline with ARC_GREEDY_FALLBACK=1 and nothing else changed. Verified in the built
notebook before pushing: the fallback and its 180-second budget are set, ARC_LOSS_CUT is absent so it
defaults to 0, and ARC_TASK_TIMEOUT is absent so it defaults to 1800.

The push triggers a commit run at ARC_N_EVAL=4, which is what produces the submission.json output a
code submission needs; the scoring rerun always does all 240. That commit run is the only part of
this that draws on the weekly interactive quota, and four tasks is a small draw.

The model is attached as the mirza mirror. There is no "sorokin" model on Kaggle; the rig's local
kagglehub path and that mirror are the same NVARC checkpoint.

Usage: python push_v76.py
"""
import json
import os
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
WORK = os.path.join(BASE, "_kpush76")
NB = "arc-agi-2-v7-6-greedy-fallback.ipynb"
SRC = os.path.join(BASE, "notebooks", "arc_submission_v7_6.ipynb")
SLUG = "rc7530/arc-agi-2-v7-6-greedy-fallback-only"   # must match what the TITLE slugs to
UTIL = "rc7530/pip-install-unsloth-flash-patch"


def main():
    assert os.path.isfile(SRC), "build it first: python build_v7_from_architects.py v7.6 (%s)" % SRC
    os.makedirs(WORK, exist_ok=True)
    shutil.copy(SRC, os.path.join(WORK, NB))

    meta = {
        "id": SLUG,
        "title": "ARC AGI 2 v7 6 greedy fallback only",
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
