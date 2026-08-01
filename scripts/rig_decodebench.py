"""Why is greedy decode running at 5 tokens per second on a 4B model?

Measured 193 s per output in the audit's g3, and batching four prompts changed nothing, which rules
out per-token launch overhead and points at the per-step cost itself. A bandwidth estimate for this
model says roughly 15 ms per step, so the observed rate is about an order of magnitude off. The
forced cuDNN attention backend is the first suspect: it was chosen for the training path, where
queries are thousands of tokens wide, and a decode step has a query length of one.

Compares attention backends at a fixed token budget on a real prompt. Same model, same prompt, same
number of new tokens, so the only variable is kernel selection.

Usage: python rig_decodebench.py [N_TOKENS=120]
"""
import contextlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig_restart_probe as P                                          # noqa: E402
import rig_contamination_audit as A                                    # noqa: E402
from arc_loader import QwenFormatter                                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer           # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                 # noqa: E402

N_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def main():
    tok = AutoTokenizer.from_pretrained(P.MODEL_DIR)
    enc = P.encoder_for(tok)
    fmt = QwenFormatter(tokenizer=tok)
    tasks, sols = A.load_eval()
    rows = A.items(tasks, sols, 6)
    tid, idx, t, _ = rows[0]
    prompt = A.prompt_for(fmt, t, idx)
    ids = torch.tensor([enc(prompt)], device="cuda")
    print("task %s, prompt %d tokens, generating %d new tokens per arm\n"
          % (tid, ids.shape[1], N_TOK), flush=True)

    model = AutoModelForCausalLM.from_pretrained(P.MODEL_DIR, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    end = tok.get_vocab()["<|im_end|>"]

    arms = [("cuDNN (what g3 forces)", lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION)),
            ("torch default", contextlib.nullcontext),
            ("flash", lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION)),
            ("mem-efficient", lambda: sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION))]

    best = None
    for name, ctx in arms:
        try:
            with torch.no_grad(), ctx():                       # warm up kernel selection
                model.generate(input_ids=ids, max_new_tokens=4, do_sample=False,
                               eos_token_id=end, pad_token_id=end)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), ctx():
                out = model.generate(input_ids=ids, max_new_tokens=N_TOK, do_sample=False,
                                     min_new_tokens=N_TOK, eos_token_id=end, pad_token_id=end)
            torch.cuda.synchronize()
            dt = time.time() - t0
            n = out.shape[1] - ids.shape[1]
            rate = n / dt
            print("%-24s %6.1f s for %d tokens  = %5.1f tok/s" % (name, dt, n, rate), flush=True)
            if best is None or rate > best[1]:
                best = (name, rate)
        except Exception as e:
            print("%-24s unavailable: %s" % (name, str(e)[:80]), flush=True)

    print("\nfastest: %s at %.1f tok/s" % best)
    print("A 900-token grid costs %.0f s per output at that rate, against the 193 s g3 measured."
          % (900.0 / best[1]))


if __name__ == "__main__":
    main()
