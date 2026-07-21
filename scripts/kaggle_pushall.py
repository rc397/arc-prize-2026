"""Rebuild Ryan's unsloth utility fork slim, push it, wait for COMPLETE, then push v7 fully wired.

Runs on the rig (Kaggle token lives there). The slim utility installs unsloth pinned against the
image's own torch 2.6.0, deletes the tree's duplicates of image-shipped heavy packages (torch, nvidia,
triton, numpy), and swaps the flash-attention wheel to the cu124torch2.6 cp312 build, cutting the
mounted tree by roughly a factor of ten so Kaggle session provisioning stops timing out. v7 is pushed
as a new private kernel with competition, model, and utility-script sources declared in metadata, so
it arrives with every attachment already wired.
"""
import json
import os
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

UTIL = "rc7530/pip-install-unsloth-flash-patch"
WORK = r"D:\arc\_kpush"
UTIL_DIR = os.path.join(WORK, "util")
V7_DIR = os.path.join(WORK, "v7")
V7_SRC = r"D:\arc\notebooks_arc_submission_v7.ipynb"

SLIM_PIP = "!pip install --target=/kaggle/working unsloth==2025.9.7 unsloth_zoo==2025.9.9 torch==2.6.0\n"
CLEANUP = (
    "# the image already ships identical versions of the heavy packages, drop the tree's copies so the\n"
    "# utility mount stays small enough for session provisioning\n"
    "import shutil, glob\n"
    "for pat in [\"torch\", \"torchgen\", \"functorch\", \"triton\", \"numpy\", \"numpy.libs\",\n"
    "            \"nvidia*\", \"cuda*\", \"sympy\", \"networkx\", \"torchvision\", \"torchao\", \"mpmath\"]:\n"
    "    for d in glob.glob(\"/kaggle/working/\" + pat) + glob.glob(\"/kaggle/working/\" + pat + \"-*.dist-info\"):\n"
    "        shutil.rmtree(d, ignore_errors=True)\n"
    "import subprocess\n"
    "print(subprocess.run(['du', '-sh', '/kaggle/working'], capture_output=True, text=True).stdout)\n"
)
SLIM_FLASH = ("!pip install --no-deps --no-build-isolation --target=/kaggle/working "
              "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/"
              "flash_attn-2.7.4+cu124torch2.6-cp312-cp312-linux_x86_64.whl\n")


def lines(s):
    return s.splitlines(keepends=True)


def main():
    api = KaggleApi()
    api.authenticate()
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(UTIL_DIR)
    os.makedirs(V7_DIR)

    print("pulling fork with metadata", flush=True)
    api.kernels_pull(UTIL, path=UTIL_DIR, metadata=True)
    nb_path = None
    for f in os.listdir(UTIL_DIR):
        if f.endswith(".ipynb"):
            nb_path = os.path.join(UTIL_DIR, f)
    assert nb_path, "no notebook in pulled fork"
    nb = json.load(open(nb_path, encoding="utf-8"))
    cells = nb["cells"]
    assert "unsloth==2025.9.7" in "".join(cells[0]["source"]), "cell 0 unexpected"
    assert "flash_attn" in "".join(cells[1]["source"]), "cell 1 unexpected"
    cells[0]["source"] = lines(SLIM_PIP)
    cells[1]["source"] = lines(SLIM_FLASH)
    cells.insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": lines(CLEANUP)})
    for c in cells:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1)

    meta_path = os.path.join(UTIL_DIR, "kernel-metadata.json")
    meta = json.load(open(meta_path))
    meta["enable_gpu"] = True
    meta["enable_internet"] = True
    json.dump(meta, open(meta_path, "w"), indent=1)
    print("pushing slim utility", flush=True)
    r = api.kernels_push(UTIL_DIR)
    print("push:", r, flush=True)

    print("polling utility status", flush=True)
    status = None
    for i in range(60):
        time.sleep(60)
        try:
            st = api.kernels_status(UTIL)
            status = st.get("status") if isinstance(st, dict) else getattr(st, "status", str(st))
            print("  [%d min] %s" % (i + 1, status), flush=True)
            s = str(status).upper()
            if "COMPLETE" in s or "ERROR" in s or "CANCEL" in s:
                break
        except Exception as e:
            print("  status err", str(e)[:120], flush=True)
    assert status and "COMPLETE" in str(status).upper(), "utility run did not complete: %s" % status

    print("preparing v7 push", flush=True)
    shutil.copy(V7_SRC, os.path.join(V7_DIR, "arc-agi-2-v7-architects.ipynb"))
    v7meta = {
        "id": "rc7530/arc-agi-2-v7-architects",
        "title": "ARC-AGI-2 v7 ARChitects reproduction",
        "code_file": "arc-agi-2-v7-architects.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "kernel_sources": [UTIL],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"],
    }
    json.dump(v7meta, open(os.path.join(V7_DIR, "kernel-metadata.json"), "w"), indent=1)
    try:
        r = api.kernels_push(V7_DIR)
        print("v7 push:", r, flush=True)
    except Exception as e:
        print("v7 push with sorokin model failed (%s), retrying with mirza mirror" % str(e)[:150], flush=True)
        v7meta["model_sources"] = ["mirzamilanfarabi/qwen3-4b-grids15-sft139/transformers/bfloat16/1"]
        json.dump(v7meta, open(os.path.join(V7_DIR, "kernel-metadata.json"), "w"), indent=1)
        r = api.kernels_push(V7_DIR)
        print("v7 push (mirror):", r, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
