"""Fetch a llama.cpp build new enough for Qwen3.8 and a quant that fits 16 GB.

Two things make this less trivial than a download. The rig's llama.cpp is build 9724 from June and
Qwen3.8 quants need b10419 or newer, so the binary is replaced rather than reused. And the CUDA
build matters: the card is Blackwell sm_120, which the 12.4 build predates, so it gets the 13.3 one.
A 12.4 binary would load and then fail for lack of a kernel image, which is the same failure that
cost a Kaggle submission earlier in this project.

Q3_K_M is chosen over the larger IQ4_XS because only 14.97 GiB of VRAM is actually free. IQ4_XS at
14.5 GiB would leave under half a gigabyte for the KV cache and compute buffers, and a partial CPU
offload would make the throughput measurement meaningless, since measuring throughput is the point.

Downloads resume, because 13.6 GB over a home connection should not have to start again.

Usage: python fetch_qwen38.py
"""
import os
import sys
import time
import urllib.request
import zipfile

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
DEST = os.path.join(BASE, "models", "qwen38")
LLAMA_TAG = "b10488"
LLAMA_ZIP = "llama-%s-bin-win-cuda-13.3-x64.zip" % LLAMA_TAG
LLAMA_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/%s/%s" % (LLAMA_TAG, LLAMA_ZIP))
LLAMA_DIR = os.path.join(BASE, "llamacpp_%s" % LLAMA_TAG)
GGUF = "Qwen3.8-27B-Q3_K_M.gguf"
GGUF_URL = "https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/" + GGUF


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def fetch(url, path, expect=0):
    """Resumable GET. Returns the final size on disk."""
    have = os.path.getsize(path) if os.path.isfile(path) else 0
    if expect and have >= expect:
        say("already complete: %s (%.2f GB)" % (os.path.basename(path), have / 1e9))
        return have
    req = urllib.request.Request(url, headers={"User-Agent": "arc-rig/1.0"})
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        say("resuming %s at %.2f GB" % (os.path.basename(path), have / 1e9))
    else:
        say("downloading %s" % os.path.basename(path))
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0) + have
        mode = "ab" if (have and r.status == 206) else "wb"
        if mode == "wb":
            have = 0
        done = have
        nxt = time.time() + 20
        with open(path, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() >= nxt:
                    rate = (done - have) / max(1e-9, time.time() - t0) / 1e6
                    pct = (100.0 * done / total) if total else 0
                    eta = ((total - done) / 1e6 / rate / 60) if rate > 0 and total else 0
                    say("  %.2f / %.2f GB (%.0f%%) at %.1f MB/s, eta %.0f min"
                        % (done / 1e9, total / 1e9, pct, rate, eta))
                    nxt = time.time() + 20
    say("done: %s (%.2f GB in %.1f min)"
        % (os.path.basename(path), done / 1e9, (time.time() - t0) / 60))
    return done


def main():
    os.makedirs(DEST, exist_ok=True)
    zpath = os.path.join(DEST, LLAMA_ZIP)
    fetch(LLAMA_URL, zpath)
    say("extracting to %s" % LLAMA_DIR)
    os.makedirs(LLAMA_DIR, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(LLAMA_DIR)
    exe = None
    for root, _dirs, files in os.walk(LLAMA_DIR):
        for f in files:
            if f.lower() == "llama-server.exe":
                exe = os.path.join(root, f)
    say("llama-server: %s" % (exe or "NOT FOUND"))

    gpath = os.path.join(DEST, GGUF)
    fetch(GGUF_URL, gpath, expect=14605735520)
    say("model at %s (%.2f GB)" % (gpath, os.path.getsize(gpath) / 1e9))
    say("READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
