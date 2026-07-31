"""Ask whether an independent test-time training run misses the same tasks as the first one.

The 40-task measurement run says the deployed system spends 69 percent of its budget on tasks it never
solves, that final training loss does not separate a solve from a miss (solved 0.0006 to 0.0253, missed
0.0027 to 0.0910, overlapping), and that the slowest solve of the run landed at 1118 seconds while the
four tasks that ran past 1200 returned nothing. Decide-early, heavy-tailed, no usable progress signal:
that is the regime where restart theory says a hard cap plus a fresh attempt beats waiting, and
Appendix F already carries the mathematics. Capping at 1200 costs no measured solve and frees about 7800
seconds per worker, which is roughly fifteen restarts.

All of that is worthless if a second run fails the same tasks. This measures exactly that.

WHAT THE ANSWER LICENSES, precisely. The deployed pipeline hardcodes every seed (adapter init 42,
trainer 42, train augment 1, eval augment 2), so rerunning a task reproduces the same adapter bit for
bit and a naive restart returns the identical pool. The lever exists only if the training augmentation
seed varies. But augment(n=16, seed=s) ALWAYS applies all eight dihedral views; the seed selects which
sixteen of the colour-permutation space, plus example order. So a null result licenses "colour
permutation restarts do not help", NOT "restarts do not help". Say so wherever the number appears.

THE NEGATIVE CONTROL IS NOT OPTIONAL. The DFS admits a continuation on a hard threshold,
score < -log(0.2), evaluated on bf16 logits. Kernel nondeterminism can flip a borderline token and
prune or admit a whole subtree, so two runs of the SAME seed need not agree. Without measuring that,
"union above single" is guaranteed to look positive whether or not seeds decorrelate. The claim is
cross-seed disagreement MINUS replicate disagreement, and the replicate arm measures the subtrahend.

Fidelity, stated honestly. Training matches the deployed rank 256, alpha 32, rslora, 8192 context, one
epoch at lr 5e-5 cosine with 0.1 warmup, completion-only loss through the notebook's own collator, and
the decoder is the deployed turbo_dfs lifted out of arc_solver.py by AST rather than reimplemented.
Four things differ. Unsloth is unavailable so the model loads through plain transformers and peft.
Gradient checkpointing is forced on to fit 16GB. DFS batches run two views rather than four, which the
builder's self-test proves covers the identical thirty-two views. And LoRA covers seven modules, not
nine: vocab_size is 16, so embed_tokens and lm_head adapters are 1.3M of 501.5M parameters, and our own
best single-card TTT (src/arc/ttt.py) trains the same seven. Absolute solve rate here is therefore not
comparable to a Kaggle number. Seed against seed is, because every cell shares one configuration.

One measurement is omitted deliberately. The deployed selector's score_aug term needs a second forward
pass per candidate and only affects RANKING, and the pool analysis already closes selection by proof, so
the primary metric is truth-in-pool, which is exact. The selector figure is a labelled proxy.

Usage: python rig_restart_probe.py [N_TASKS=20] [K_SEEDS=3]
       python rig_restart_probe.py summary
Env:   ARC_PROBE_OPTIM   adamw | adamw_offload (default) | adafactor
       ARC_PROBE_RANK    default 256      ARC_PROBE_MAXSEQ  default 8192
       ARC_PROBE_BASE    override D:\\arc (lets the offline stages run anywhere)
Resumable: one JSON line per (task, seed, replicate). Completed cells are skipped, failed cells are
retried up to MAX_ATTEMPTS, and a cell that faults hard still leaves a marker so it cannot loop.
"""
import ast
import builtins
import gc
import json
import math
import os
import sys
import threading
import time
import typing
from collections import Counter, defaultdict

# turbo_dfs recurses once per generated token and max_new_tokens is about 931 for a 30x30 grid,
# against CPython's default limit of 1000. Raising the limit alone segfaults on Windows, whose
# default thread stack is 1 MB, so decode also runs on a thread created with a 64 MB stack.
sys.setrecursionlimit(20000)
STACK_BYTES = 64 * 1024 * 1024

BASE = os.environ.get("ARC_PROBE_BASE", r"D:\arc")
V7 = next((p for p in (os.path.join(BASE, "notebooks", "v7_debug"),
                       os.path.join(r"D:\ARC_AGI-2,3", "notebooks", "v7_debug"))
           if os.path.isdir(p)), None)
assert V7, "v7_debug modules not found, sync them to %s" % os.path.join(BASE, "notebooks", "v7_debug")
sys.path.insert(0, V7)

REPO = r"D:\ARC_AGI-2,3"
EXP = os.path.join(BASE, "experiments")
STATUS = os.path.join(BASE, "eval_status_restart.json")
RESULTS = os.path.join(EXP, "restart_probe.jsonl")
SUMMARY = os.path.join(EXP, "restart_probe_summary.json")
MODEL_DIR = os.path.join(BASE, "models", "kagglehub", "models", "sorokin",
                         "qwen3_4b_grids15_sft139", "transformers", "bfloat16", "1")

# the deployed configuration, from arc_solver.py's peft_params and train_args
RANK = int(os.environ.get("ARC_PROBE_RANK", "256"))
ALPHA, DROPOUT = 32, 0.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Deployment sets 8192, but that is a CAP and not a length. Measured across the twenty selected
# tasks (smoke stage s0b): the longest training text is 3849 tokens and the longest eval input is
# 4783, so 5714 covers both including generation. 6144 truncates nothing, which s0b asserts, and
# reclaims about a quarter of the activation memory for free. The usual objection to this lever,
# that a tighter cap makes cut_to_len drop pairs seed-dependently and so inflates the measured
# decorrelation, does not apply when nothing is being cut at all.
MAXSEQ = int(os.environ.get("ARC_PROBE_MAXSEQ", "6144"))
LR, EPOCHS, WARMUP_RATIO, GRAD_CLIP = 5e-5, 1, 0.1, 1.0
LORA_INIT_SEED = 42       # deployment's random_state. Held FIXED so only augmentation varies.
DFS_FLOOR = 0.2
BATCH_VIEWS = 2           # 2 or 4, provably the same thirty-two views either way
OPTIM = os.environ.get("ARC_PROBE_OPTIM", "adamw_offload")
# Decode gets its own budget measured from the END of training. Deployment caps the whole task at
# 1800s but trains in about 210s on an L4 under unsloth, so decode there gets roughly 1590s. Training
# here takes far longer, so a cell-relative cap would shrink every pool against the deployed condition,
# depressing every seed and compressing the very contrast this probe exists to measure.
DECODE_BUDGET = int(os.environ.get("ARC_PROBE_DECODE_BUDGET", "1500"))
TRAIN_BUDGET = int(os.environ.get("ARC_PROBE_TRAIN_BUDGET", "2400"))
POOL_CAP = 200_000        # a flat-distribution adapter can explode the tree; abort rather than host-OOM
MAX_ATTEMPTS = 2          # per (task, seed, replicate)

# The cuBLAS failure that killed run 4 was a workspace allocation at 15836 MiB of 16384: the
# allocator had the bytes but not a contiguous block. Expandable segments target exactly that, cost
# nothing, and must be set before torch initialises its allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
from torch.nn.attention import sdpa_kernel, SDPBackend                 # noqa: E402
from peft import LoraConfig, get_peft_model                            # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,         # noqa: E402
                          DataCollatorForLanguageModeling)

from arc_loader import ArcDataset, QwenFormatter                       # noqa: E402
from arc_decoder import score_kgmon                                    # noqa: E402


def timing_csv():
    """Locate the Kaggle measurement outcomes. The rig has no repo checkout, so resolve rather than
    assume a laptop path, and resolve it in ONE place: a second hardcoded read in the smoke stage
    survived the first fix and crashed s3 after it had done all of its expensive work."""
    p = next((c for c in (os.path.join(BASE, "data", "task_timing.csv"),
                          os.path.join(EXP, "task_timing.csv"),
                          os.path.join(REPO, "paper", "data", "task_timing.csv"))
              if os.path.isfile(c)), None)
    assert p, "task_timing.csv not found; copy it to %s" % os.path.join(BASE, "data")
    return p


def encoder_for(tokenizer):
    """Return an exact encoder for this prompt format, because the shipped tokenizer is not one.

    The model's tokenizer is WordLevel with no merges over a 16-token vocabulary, and its
    pre-tokenizer cannot reach the entries 'user' (11) and 'assistant' (12): tokenizer.encode("user")
    returns the empty list, and "<|im_start|>user\\n" and "<|im_start|>assistant\\n" both encode to
    [14, 10], indistinguishable. The completion-only collator keys on exactly 11 and 12 to find the
    assistant spans, so with this tokenizer every label stays -100, the loss is NaN over zero
    targets, and the gradient norm is 0. Measured: four training rows, zero supervised labels,
    NaN loss at every rank and both optimizers.

    Kaggle does not hit this. paper/data/task_timing.csv records finite converged losses from the
    deployed run, so there the collator does find spans, which fixes the intended tokenization:
    "<|im_start|>user\\n" is [im_start, user, newline]. This builds exactly that, taking every id
    from the model's own vocabulary rather than hardcoding, and the round-trip assertion below
    checks the result decodes back to the original text.
    """
    v = tokenizer.get_vocab()
    need = ["<|im_start|>", "<|im_end|>", "user", "assistant"]
    missing = [t for t in need if t not in v]
    assert not missing, "vocabulary lacks %s, this encoder does not apply" % missing
    nl = next((i for t, i in v.items() if tokenizer.convert_tokens_to_string([t]) == "\n"), None)
    assert nl is not None, "no newline token in the vocabulary"
    digits = {str(d): v[str(d)] for d in range(10)}
    parts = [("<|im_start|>user\n", [v["<|im_start|>"], v["user"], nl]),
             ("<|im_start|>assistant\n", [v["<|im_start|>"], v["assistant"], nl]),
             ("<|im_end|>", [v["<|im_end|>"]])]

    def encode(text):
        ids, i = [], 0
        while i < len(text):
            for lit, seq in parts:
                if text.startswith(lit, i):
                    ids += seq
                    i += len(lit)
                    break
            else:
                c = text[i]
                if c in digits:
                    ids.append(digits[c])
                elif c == "\n":
                    ids.append(nl)
                else:
                    raise ValueError("unencodable character %r at %d" % (c, i))
                i += 1
        return ids

    return encode


def data_dir():
    """Resolved lazily so the offline stages can import this module without the rig's data tree."""
    d = next((p for p in (os.path.join(BASE, "data", "kaggle"), os.path.join(BASE, "data"))
              if os.path.isfile(os.path.join(p, "arc-agi_evaluation_challenges.json"))), None)
    assert d, "evaluation challenges not found under %s" % os.path.join(BASE, "data")
    return d


# ---------------------------------------------------------------- lifting the deployed decoder

CACHE_STATS = {"calls": 0, "shrinks": 0}


def _branch_cache(cache, pos):
    """Undo the in-place growth a model call leaves on a Cache, restoring the branch point.

    turbo_dfs recurses with one cache object and calls the model repeatedly inside a while loop,
    passing the SAME cache each time. That is only correct if a model call leaves its input cache
    untouched, which was true of the legacy tuple API the deployed pipeline was written against.
    transformers 5 replaced that with Cache objects that append in place, so the second branch
    continues from an already-extended cache while position_ids still says pos, RoPE and the causal
    mask disagree, and CUDA raises an illegal memory access. On entry the cache holds exactly pos
    tokens, so cropping back to pos reproduces the old semantics rather than approximating them.

    The counters exist because the previous version swallowed every exception here, which made a
    silent no-op indistinguishable from success. The smoke stage asserts both are non-zero.

    Returns the cache so the injected statement is an assignment. That costs nothing here and lets
    the smoke stage substitute an implementation that rebuilds the cache from manually sliced
    tensors, which is independent of Cache.crop and so can check it rather than assume it.
    """
    CACHE_STATS["calls"] += 1
    if cache.get_seq_length() > pos:
        cache.crop(pos)
        CACHE_STATS["shrinks"] += 1
        assert cache.get_seq_length() == pos, "crop left %d tokens, expected %d" % (
            cache.get_seq_length(), pos)
    return cache


class _CropBeforeEachBranch(ast.NodeTransformer):
    def __init__(self):
        self.hits = 0

    def visit_While(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.parse("cache = _branch_cache(cache, pos)").body[0])
        self.hits += 1
        return node


def lift_from_solver(patch_cache=True):
    """Pull the deployed decoder and collator out of arc_solver.py without importing it.

    That module imports unsloth on line 1, which does not exist here. Taking the definitions through
    the AST keeps them byte-faithful to what ships, which a reimplementation would not. The single
    exception is the cache restore documented on _branch_cache. patch_cache=False yields the
    unpatched decoder, which the smoke stage needs for its clone-arm comparison.
    """
    src = open(os.path.join(V7, "arc_solver.py"), encoding="utf-8").read()
    body = ast.parse(src).body
    want_def = {"turbo_dfs", "inference_turbo_dfs", "QwenDataCollatorForCompletionOnlyLM"}
    keep = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want_def]
    missing = want_def - {n.name for n in keep}
    assert not missing, "could not lift %s from arc_solver.py" % sorted(missing)

    # Pull in the module-level constants the lifted code reads, transitively, rather than naming them
    # by hand. Hand-listing missed ARC_TOKENS and PAD_ID, which turbo_dfs reads on every branch, and
    # the NameError only surfaced deep inside a rig run.
    assigns = {}
    for n in body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigns.setdefault(t.id, n)

    def free_names(nodes):
        return {x.id for n in nodes for x in ast.walk(n)
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}

    def bound_names(nodes):
        out = {getattr(n, "name", None) for n in nodes}
        for n in nodes:
            for x in ast.walk(n):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    out.add(x.id)
                elif isinstance(x, ast.arg):
                    out.add(x.arg)
        return out - {None}

    supplied = {"torch", "np", "time", "defaultdict", "DataCollatorForLanguageModeling",
                "Any", "Union", "_branch_cache"} | set(dir(builtins))
    while True:
        need = free_names(keep) - bound_names(keep) - supplied
        pull = [assigns[n] for n in sorted(need) if n in assigns and assigns[n] not in keep]
        if not pull:
            break
        keep.extend(pull)
    keep.sort(key=body.index)
    unresolved = free_names(keep) - bound_names(keep) - supplied
    assert not unresolved, "lifted code reads names nothing provides: %s" % sorted(unresolved)

    if patch_cache:
        patcher = _CropBeforeEachBranch()
        for node in keep:
            if getattr(node, "name", None) == "turbo_dfs":
                patcher.visit(node)
                ast.fix_missing_locations(node)
        # a patch that silently fails to apply is worse than no patch
        assert patcher.hits == 1, "expected exactly one DFS loop to patch, found %d" % patcher.hits

    # the collator's method signature is annotated, so these must be the real typing objects
    ns = {"torch": torch, "np": np, "time": time, "defaultdict": defaultdict,
          "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
          "Any": typing.Any, "Union": typing.Union, "_branch_cache": _branch_cache}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<lifted>", "exec"), ns)
    return ns


def run_deep(fn, *args, **kwargs):
    """Run fn on a thread with a large stack, so deep DFS recursion raises rather than segfaults."""
    out = {}

    def target():
        try:
            out["r"] = fn(*args, **kwargs)
        except BaseException as e:                                     # noqa: BLE001
            out["e"] = e

    prev = threading.stack_size(STACK_BYTES)
    try:
        t = threading.Thread(target=target)
        t.start()
        t.join()
    finally:
        threading.stack_size(prev)
    if "e" in out:
        raise out["e"]
    return out["r"]


def heartbeat(phase, extra=""):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "ts": time.time(), "extra": extra}, f)
    os.replace(tmp, STATUS)


def is_memory_error(e):
    """Memory pressure does not always arrive as OutOfMemoryError.

    Near the ceiling cuBLAS fails to obtain a contiguous workspace and surfaces
    CUBLAS_STATUS_INTERNAL_ERROR from an ordinary matmul, as a plain RuntimeError.
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    m = str(e).lower()
    return ("out of memory" in m or "cublas_status_internal_error" in m
            or "cublas_status_alloc_failed" in m)


# ---------------------------------------------------------------- optimizer

class CPUOffloadAdamW:
    """AdamW whose fp32 master weights and moments live on the host.

    Identical update rule to torch.optim.AdamW; only residency changes. The moments dominate GPU
    footprint (about 4 GB for 500M parameters against an 8 GB base and 16 GB card) and are touched
    once per step, so hosting them trades PCIe bandwidth for headroom at zero semantic cost. That
    matters here because the alternatives, a lower rank or a shorter context, both change the
    quantity being measured rather than the memory used to measure it.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.master = [p.detach().to("cpu", torch.float32).clone() for p in self.params]
        self.m = [torch.zeros_like(x) for x in self.master]
        self.v = [torch.zeros_like(x) for x in self.master]
        self.t = 0

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach().to("cpu", torch.float32)
            w = self.master[i]
            if self.wd:
                w.mul_(1 - self.lr * self.wd)
            self.m[i].mul_(b1).add_(g, alpha=1 - b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (self.v[i] / bc2).sqrt_().add_(self.eps)
            w.addcdiv_(self.m[i] / bc1, denom, value=-self.lr)
            p.copy_(w.to(p.dtype))


def cosine_lr(step, total, warmup_ratio, base_lr):
    """transformers' get_cosine_schedule_with_warmup at num_cycles=0.5, inlined.

    Inlined so the offloaded and stock optimizers share one schedule; the smoke stage asserts it
    matches the library's curve to 1e-12 rather than taking the equivalence on trust.
    """
    w = max(1, int(warmup_ratio * total))
    if step < w:
        return base_lr * float(step) / float(w)
    prog = float(step - w) / float(max(1, total - w))
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))


def make_optimizer(params):
    if OPTIM == "adamw":
        return torch.optim.AdamW(params, lr=LR, weight_decay=0.0), "adamw"
    if OPTIM == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=LR, scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=0.0), "adafactor"
    return CPUOffloadAdamW(params, lr=LR, weight_decay=0.0), "adamw_offload"


def set_lr(opt, lr):
    if isinstance(opt, CPUOffloadAdamW):
        opt.lr = lr
    else:
        for g in opt.param_groups:
            g["lr"] = lr


# ---------------------------------------------------------------- model, train, decode

ADAPTER_BF16 = os.environ.get("ARC_PROBE_ADAPTER_BF16", "1") not in ("", "0", "false", "False")


def build_model():
    """Load the base and attach a fresh adapter. LoRA init is fixed so only augmentation varies.

    get_peft_model defaults to autocast_adapter_dtype=True, which upcasts lora_A and lora_B to fp32.
    At rank 256 over seven modules that is 500M parameters, so the upcast costs a gigabyte of
    parameters and another of gradients on top of an 8 GB base, and s2 measured every optimizer
    reserving past the 16 GB card and thrashing into host memory at 62 to 150 seconds per step.
    Keeping the adapter in bf16 is what unsloth does and costs nothing semantically here.
    """
    torch.manual_seed(LORA_INIT_SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.to("cuda")
    peft_model = get_peft_model(model, LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=TARGETS,
        bias="none", use_rslora=True, task_type="CAUSAL_LM"),
        autocast_adapter_dtype=not ADAPTER_BF16)
    if ADAPTER_BF16:
        for n, p in peft_model.named_parameters():
            if p.requires_grad and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
    return peft_model


def adapter_stats(peft_model):
    tp = [p for p in peft_model.parameters() if p.requires_grad]
    return {"trainable_m": round(sum(p.numel() for p in tp) / 1e6, 1),
            "param_dtype": str(tp[0].dtype) if tp else "none"}


def train_one(peft_model, tokenizer, collator, formatter, train_ds, deadline=None,
              max_batches=None):
    base = peft_model.get_base_model()
    base.config.use_cache = False
    # Reentrant checkpointing forks and restores the CUDA RNG so recomputation can replay it, and on
    # this torch build that restore faults with an illegal memory access. The non-reentrant variant
    # uses saved-tensor hooks and never touches the device generator.
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    peft_model.train()

    rows = train_ds.as_list(formatter)
    # cut_to_len drops demonstration pairs from the front until the text fits, and if MAXSEQ is
    # small enough it drops them all, after which fmt_train indexes an empty list and the failure
    # surfaces as an IndexError deep in the loader. Fail here with a legible reason instead.
    assert rows, ("cut_to_len left no training examples at MAXSEQ=%d; the cap is below this task's "
                  "shortest single demonstration pair" % MAXSEQ)
    enc = encoder_for(tokenizer)
    batches = [{"input_ids": enc(r["text"])[:MAXSEQ]} for r in rows]
    # zero supervision produces a NaN loss over no targets rather than an error, which is how the
    # broken tokenization stayed invisible. Refuse to train on it.
    probe = collator([batches[0]])
    assert int((probe["labels"] != -100).sum()) > 0, (
        "collator produced no supervised labels; the encoder is not emitting the user/assistant "
        "control ids the completion-only mask keys on")
    # max_batches is for the smoke stages only. The schedule length follows it so a reduced run
    # still sees a complete warmup-and-decay curve rather than only its opening.
    if max_batches:
        batches = batches[:max_batches]
    steps = max(1, len(batches) * EPOCHS)
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt, opt_name = make_optimizer(params)

    losses, step, truncated = [], 0, False
    for _ in range(EPOCHS):
        for b in batches:
            if deadline is not None and time.time() > deadline:
                truncated = True
                break
            set_lr(opt, cosine_lr(step, steps, WARMUP_RATIO, LR))
            batch = {k: v.to("cuda") for k, v in collator([b]).items()}
            # forward AND backward inside one backend context. Gradient checkpointing replays the
            # forward during backward, and a replay under a different kernel saves a different number
            # of tensors, which raises CheckpointError. rig_attnprobe.py had this right.
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                loss = peft_model(**batch).loss
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            step += 1
        if truncated:
            break
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    # task_timing.csv holds a FINAL loss, so record all three rather than one and compare like
    # with like later.
    return {"train_loss_mean": float(np.mean(losses)) if losses else 0.0,
            "train_loss_final": float(losses[-1]) if losses else 0.0,
            "train_loss_last10": float(np.mean(losses[-10:])) if losses else 0.0,
            "train_steps": len(losses), "train_truncated": truncated, "optim": opt_name}


def decode_pool(peft_model, tokenizer, formatter, lifted, puzzle_ds, deadline):
    """Run the deployed DFS over the sixteen eval views, returning the pool keyed by base key.

    Every expected base key is created up front. A key with no parseable candidate is an
    in_pool=False observation, not a missing one; letting it vanish would drop that output from the
    seed comparison entirely and bias the headline number in an uncontrolled direction.
    """
    base = peft_model.get_base_model()
    base.config.use_cache = True
    base.gradient_checkpointing_disable()
    peft_model.eval()

    enc = encoder_for(tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    multi = puzzle_ds.split_multi_replies()
    eval_ds = multi.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=MAXSEQ - max_new_tokens)

    by_test = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        by_test[subkey.split(".")[0].split("_")[1]].append(subkey)

    batches = []
    for pairs in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
        for _, subkeys in by_test.items():
            for p in pairs:
                h = [subkeys[o:o + 2] for o in p]
                batches.extend([h[0] + h[1]] if BATCH_VIEWS == 4 else h)

    pool = {sk.split(".")[0]: [] for sk in eval_ds.keys}
    max_score = -np.log(DFS_FLOOR)
    truncated = overflow = False
    with torch.inference_mode():
        for subkeys in batches:
            if not subkeys:
                continue
            if time.time() > deadline:
                truncated = True
                break
            tokens = [enc(eval_ds.get(sk, formatter)["input"]) for sk in subkeys]
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                dfs_result = run_deep(lifted["inference_turbo_dfs"],
                                      peft_model, tokens, max_new_tokens, max_score, deadline)
            for subkey_id, scored_beams in dfs_result:
                subkey = subkeys[subkey_id]
                bk = subkey.split(".")[0]
                for beam_score, token_seq in scored_beams:
                    arr = formatter.convert_tokens_to_array(token_seq)
                    if arr is None:
                        continue
                    sol = multi.invert_mod(arr, subkey, inv_perm=True)
                    pool.setdefault(bk, []).append(
                        {"beam_score": float(beam_score), "solution": sol})
            if sum(len(v) for v in pool.values()) > POOL_CAP:
                overflow = True
                break
    return pool, truncated, overflow


def grade(pool, labels, expected_keys):
    """Exact truth-in-pool per expected base key, plus a labelled proxy for the deployed selector."""
    out = {}
    for bk in expected_keys:
        lab = labels.get(bk)
        if lab is None:
            continue
        cands = pool.get(bk, [])
        in_pool = any(np.array_equal(c["solution"], lab) for c in cands)
        # score_aug needs a second forward pass per candidate and only affects ranking, so the
        # deployed selector is approximated by its multiplicity term with beam score as tiebreak.
        guesses = {"g%d" % i: {"beam_score": c["beam_score"], "score_aug": [c["beam_score"]],
                               "solution": c["solution"]} for i, c in enumerate(cands)}
        top2 = score_kgmon(guesses)[:2] if guesses else []
        out[bk] = {"in_pool": bool(in_pool), "n_cands": len(cands),
                   "selected_proxy": bool(any(np.array_equal(g, lab) for g in top2))}
    return out


# ---------------------------------------------------------------- task selection

def select_tasks(n_tasks):
    """Stratified sample from the measurement slice, with population weights for reweighting.

    Sampling the first N tasks wastes the run: of the first twenty, fifteen are outright misses that
    cannot discriminate between seeds and only four ever had the truth in the pool, so most cells
    would land where no seed can differ while still inflating the denominator. Strata come from the
    Kaggle outcomes in paper/data/task_timing.csv:
      A  reachable      solved or in-pool, the truth entered the pool at least once
      B  near miss      missed, but trained to a below-median final loss, so it trained fine and
                        decoded wrong, which is where a restart could plausibly recover it
      C  hard           missed at above-median loss, or decoded nothing at all: the floor
    Returns (keys, weight_by_key, stratum_by_key). Weight is N_stratum / n_sampled_from_stratum, so a
    weighted mean estimates the population rate over the 40-task slice.
    """
    import csv
    rows = list(csv.DictReader(open(timing_csv())))
    for r in rows:
        r["loss"] = float(r["train_loss"])
    misses = [r for r in rows if r["outcome"] not in ("solved", "in-pool")]
    med = float(np.median([r["loss"] for r in misses])) if misses else 0.0
    strat = {}
    for r in rows:
        if r["outcome"] in ("solved", "in-pool"):
            strat[r["task"]] = "A"
        elif r["outcome"] != "empty" and r["loss"] <= med:
            strat[r["task"]] = "B"
        else:
            strat[r["task"]] = "C"
    pop = Counter(strat.values())

    # Take all of A (it is the only stratum where seeds can differ on a reachable truth), then fill
    # with B, then C as a floor. Deterministic: sorted order, no RNG, so the sample is reproducible.
    order = {"A": 0, "B": 1, "C": 2}
    want = {"A": pop["A"], "B": max(0, min(pop["B"], n_tasks - pop["A"] - 2)), "C": 2}
    while sum(want.values()) > n_tasks:
        for s in ("C", "B", "A"):
            if sum(want.values()) > n_tasks and want[s] > 0:
                want[s] -= 1
    keys, taken = [], Counter()
    for t in sorted(strat, key=lambda k: (order[strat[k]], k)):
        s = strat[t]
        if taken[s] < want[s]:
            keys.append(t)
            taken[s] += 1
    weight = {t: pop[strat[t]] / float(taken[strat[t]]) for t in keys}
    return sorted(keys), weight, {t: strat[t] for t in keys}


def cell_plan(keys, k_seeds, n_replicates, strat):
    """(task, seed, replicate) cells. replicate=1 repeats seed 1 as the negative control."""
    cells = [(t, s, 0) for t in keys for s in range(1, k_seeds + 1)]
    reach = [t for t in keys if strat.get(t) == "A"][:n_replicates]
    cells += [(t, 1, 1) for t in reach]
    return cells


def plan_todo(cells, records):
    """Split cells into still-to-run, already-done, and attempt-exhausted.

    A completed cell is skipped. A FAILED cell is retried, which the previous version did not do:
    it added every seen cell to the done set regardless of outcome, so one fault removed that cell
    permanently and the supervisor's relaunching accomplished nothing. Attempts are counted from
    the started markers written before each cell, so a hard crash that never writes a result still
    counts against the cap and cannot loop forever.
    """
    done_ok, attempts = set(), Counter()
    for r in records:
        cid = (r["task"], r["seed"], r.get("replicate", 0))
        if r.get("marker") == "started":
            attempts[cid] += 1
        elif r.get("ok"):
            done_ok.add(cid)
    todo = [c for c in cells if c not in done_ok and attempts[c] < MAX_ATTEMPTS]
    exhausted = [c for c in cells if c not in done_ok and attempts[c] >= MAX_ATTEMPTS]
    return todo, done_ok, exhausted


# ---------------------------------------------------------------- main

def main():
    os.makedirs(EXP, exist_ok=True)
    heartbeat("start")
    n_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    k_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    dd = data_dir()
    ch = os.path.join(dd, "arc-agi_evaluation_challenges.json")
    solutions = json.load(open(os.path.join(dd, "arc-agi_evaluation_solutions.json")))
    arc = ArcDataset.from_file(ch)

    keys, weight, strat = select_tasks(n_tasks)
    cells = cell_plan(keys, k_seeds, n_reps, strat)
    print("CONFIG rank=%d maxseq=%d optim=%s targets=%d tasks=%d seeds=%d reps=%d cells=%d"
          % (RANK, MAXSEQ, OPTIM, len(TARGETS), len(keys), k_seeds, n_reps, len(cells)), flush=True)
    print("STRATA " + " ".join("%s=%d" % (s, sum(1 for t in keys if strat[t] == s))
                               for s in ("A", "B", "C")), flush=True)

    labels = {}
    for k in keys:
        for i, g in enumerate(solutions[k]):
            labels["%s_%d" % (k, i)] = np.array(g)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    formatter = QwenFormatter(tokenizer=tokenizer)
    lifted = lift_from_solver()
    collator = lifted["QwenDataCollatorForCompletionOnlyLM"](tokenizer=tokenizer, mlm=False)

    records = []
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    todo, done_ok, exhausted = plan_todo(cells, records)
    print("resume: %d done, %d exhausted, %d to run"
          % (len(done_ok), len(exhausted), len(todo)), flush=True)

    for key, seed, rep in todo:
        cid = "%s seed%d rep%d" % (key, seed, rep)
        heartbeat("cell", cid)
        with open(RESULTS, "a") as f:
            f.write(json.dumps({"task": key, "seed": seed, "replicate": rep,
                                "marker": "started", "ts": time.time()}) + "\n")
        t0 = time.time()
        rec = {"task": key, "seed": seed, "replicate": rep, "stratum": strat[key],
               "weight": weight[key]}
        peft_model = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peft_model = build_model()
            rec.update(adapter_stats(peft_model))

            puzzle_ds = arc.change_keys([key])
            # THE varied axis: which sixteen augmentations the adapter trains on
            train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=seed)
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=MAXSEQ)

            rec.update(train_one(peft_model, tokenizer, collator, formatter, train_ds,
                                 deadline=t0 + TRAIN_BUDGET))
            rec["train_s"] = round(time.time() - t0, 1)

            t1 = time.time()
            CACHE_STATS.update(calls=0, shrinks=0)
            pool, trunc, overflow = decode_pool(peft_model, tokenizer, formatter, lifted,
                                                puzzle_ds, t1 + DECODE_BUDGET)
            rec["decode_s"] = round(time.time() - t1, 1)
            rec["truncated"] = trunc
            rec["pool_overflow"] = overflow
            rec["cache_calls"] = CACHE_STATS["calls"]
            rec["cache_shrinks"] = CACHE_STATS["shrinks"]
            rec["outputs"] = grade(pool, labels, [k for k in labels if k.startswith(key + "_")])
            rec["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["ok"] = True
        except Exception as e:                                          # noqa: BLE001
            rec["ok"] = False
            rec["error"] = "%s %s" % (type(e).__name__, str(e).replace("\n", " ")[:200])
            if is_memory_error(e):
                rec["memory_pressure"] = True
            # An allocator or cuBLAS fault leaves the context unreliable; continuing in this process
            # only manufactures more misleading failures.
            rec["cuda_poisoned"] = is_memory_error(e) or "illegal memory access" in str(e).lower()
        finally:
            # An exception raised in finally escapes every except above it, which is how one bad
            # cell killed a whole run instead of being recorded. Cleanup never propagates, and it
            # never clears a poison flag the handler already set.
            try:
                del peft_model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as ce:                                     # noqa: BLE001
                rec.setdefault("error", "cleanup %s" % type(ce).__name__)
                if "illegal memory access" in str(ce).lower() or is_memory_error(ce):
                    rec["cuda_poisoned"] = True

        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        n_in = sum(1 for v in rec.get("outputs", {}).values() if v["in_pool"])
        n_out = len(rec.get("outputs", {}))
        print("%-10s s%d r%d %-4s %s loss %-8s in-pool %d/%d  train %4.0fs decode %4.0fs %s" % (
            key, seed, rep, strat[key], "OK" if rec.get("ok") else "FAIL",
            round(rec.get("train_loss_final", 0), 5), n_in, n_out,
            rec.get("train_s", 0), rec.get("decode_s", 0), rec.get("error", "")), flush=True)
        if rec.get("cuda_poisoned"):
            print("CUDA context unreliable, stopping so a relaunch resumes", flush=True)
            heartbeat("poisoned", cid)
            # Returning normally exits 0, which the supervisor reads as "completed" and stops.
            # Stopping early and reporting success are opposite things and must not share an exit
            # code: one poisoned cell ended a 65-cell run at 37 because of exactly that.
            return False

    summarize()
    heartbeat("done")
    return True


def summarize(path=None):
    """The decision: union over seeds against one seed, net of the replicate floor."""
    src = path or RESULTS
    rows = []
    if os.path.exists(src):
        for line in open(src):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("marker") != "started" and r.get("ok"):
                rows.append(r)
    if not rows:
        print("no completed cells")
        return None

    # Truncation is only a confound when it is ASYMMETRIC. Every cell gets the same decode budget,
    # so a task whose seeds were all cut at the deadline is still a fair seed-against-seed
    # comparison on a uniformly smaller pool; a task where one seed finished and another was cut is
    # not, because the difference could be time rather than capability. Excluding all truncated
    # cells would have dropped 10 of the first 16, including both tasks that showed any seed
    # variation at all, which is discarding the measurement to keep the bookkeeping tidy.
    trunc_by_task = defaultdict(set)
    for r in rows:
        trunc_by_task[r["task"]].add(bool(r.get("truncated")))
    mixed = {t for t, v in trunc_by_task.items() if len(v) > 1}
    hard = [r for r in rows if r.get("train_truncated") or r.get("pool_overflow")
            or r["task"] in mixed]
    n_trunc = sum(1 for r in rows if r.get("truncated"))
    print("decode truncated in %d of %d cells, uniform within task except %s"
          % (n_trunc, len(rows), sorted(mixed) or "none"))
    if hard:
        print("EXCLUDING %d cell(s) for mixed truncation, train truncation or pool overflow: %s"
              % (len(hard), sorted({"%s s%d" % (r["task"], r["seed"]) for r in hard})))
        rows = [r for r in rows if r not in hard]
    if n_trunc:
        print("NOTE: pools are smaller than the deployed condition wherever decode was cut. That")
        print("      lowers every seed's rate together, so the union-against-single contrast stays")
        print("      valid, but the absolute rates are floors and not estimates of deployment.")

    main_rows = [r for r in rows if r.get("replicate", 0) == 0]
    rep_rows = [r for r in rows if r.get("replicate", 0) == 1]

    per_task = defaultdict(set)
    for r in main_rows:
        per_task[r["task"]].add(r["seed"])
    seeds = sorted({r["seed"] for r in main_rows})
    full = {t for t, s in per_task.items() if s == set(seeds)}
    dropped = sorted(set(per_task) - full)
    if dropped:
        print("DROPPED %d task(s) missing a seed: %s" % (len(dropped), dropped))
    if not full:
        print("no task completed all seeds yet")
        return None

    by_seed, weight, strat = defaultdict(dict), {}, {}
    for r in main_rows:
        if r["task"] not in full:
            continue
        weight[r["task"]] = r.get("weight", 1.0)
        strat[r["task"]] = r.get("stratum", "?")
        for bk, v in r["outputs"].items():
            by_seed[r["seed"]][bk] = v["in_pool"]

    common = sorted(set.intersection(*[set(by_seed[s]) for s in seeds]))
    n = len(common)
    task_of = {b: b.rsplit("_", 1)[0] for b in common}
    w = {b: weight.get(task_of[b], 1.0) for b in common}
    tw = sum(w.values()) or 1.0

    def wrate(pred):
        return sum(w[b] for b in common if pred(b)) / tw

    single = {s: wrate(lambda b, s=s: by_seed[s][b]) for s in seeds}
    union = wrate(lambda b: any(by_seed[s][b] for s in seeds))
    inter = wrate(lambda b: all(by_seed[s][b] for s in seeds))
    mean_single = sum(single.values()) / len(seeds)

    print("\n=== restart decision: %d test outputs across %d tasks, %d seeds ==="
          % (n, len(full), len(seeds)))
    print("    rates are reweighted to the 40-task slice using stratum weights")
    for s in seeds:
        print("  seed %d alone : %.1f%% of outputs have the truth in pool" % (s, 100 * single[s]))
    print("  union of %d   : %.1f%%" % (len(seeds), 100 * union))
    print("  all seeds     : %.1f%%  (agreement core)" % (100 * inter))
    print("  raw counts    : " + " ".join(
        "s%d=%d" % (s, sum(by_seed[s][b] for b in common)) for s in seeds)
        + " union=%d of %d" % (sum(any(by_seed[s][b] for s in seeds) for b in common), n))

    # The negative control. Cross-seed disagreement is only evidence of decorrelation to the extent
    # it exceeds the disagreement two runs of the SAME seed produce through kernel nondeterminism.
    rep_line, net = "replicate arm not run, gain is NOT interpretable", None
    if rep_rows:
        pairs = 0
        dis_rep = 0
        for r in rep_rows:
            base = next((x for x in main_rows
                         if x["task"] == r["task"] and x["seed"] == r["seed"]), None)
            if not base:
                continue
            for bk, v in r["outputs"].items():
                if bk in base["outputs"]:
                    pairs += 1
                    dis_rep += int(v["in_pool"] != base["outputs"][bk]["in_pool"])
        dis_cross = cross = 0
        for b in common:
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    cross += 1
                    dis_cross += int(by_seed[seeds[i]][b] != by_seed[seeds[j]][b])
        r_rate = dis_rep / pairs if pairs else float("nan")
        c_rate = dis_cross / cross if cross else float("nan")
        net = c_rate - r_rate
        rep_line = ("replicate disagreement %.3f (%d pairs), cross-seed %.3f (%d pairs), net %+.3f"
                    % (r_rate, pairs, c_rate, cross, net))
    print("  NEGATIVE CONTROL: " + rep_line)

    by_strat = defaultdict(lambda: [0, 0])
    for b in common:
        s = strat.get(task_of[b], "?")
        by_strat[s][1] += 1
        by_strat[s][0] += int(any(by_seed[k][b] for k in seeds))
    print("  raw union by stratum: " + " ".join(
        "%s=%d/%d" % (s, v[0], v[1]) for s, v in sorted(by_strat.items())))

    # Fidelity against the deployed run, on the tasks they share. Compare LIKE WITH LIKE: the CSV
    # holds the trainer's mean loss over the run, not a final loss, so the probe's mean is the
    # comparable statistic and its final (about 25x smaller) is not. This is reported before the
    # headline because a probe training at a different operating point would make the headline mean
    # something different, and because agreement here is the only evidence that a stack running
    # plain peft rather than unsloth reproduces the deployed condition at all.
    try:
        import csv
        kag = {r["task"]: r for r in csv.DictReader(open(timing_csv()))}
        pairs = [(r["task"], r.get("train_loss_mean"), float(kag[r["task"]]["train_loss"]),
                  r.get("wall_s"), float(kag[r["task"]]["wall_seconds"]),
                  any(v["in_pool"] for v in r["outputs"].values()),
                  kag[r["task"]]["outcome"] in ("solved", "in-pool"))
                 for r in main_rows if r["seed"] == seeds[0] and r["task"] in kag]
        if pairs:
            ratios = sorted(p[1] / p[2] for p in pairs if p[1] and p[2])
            agree = sum(1 for p in pairs if p[5] == p[6])
            print("\n  FIDELITY vs the deployed run, %d shared tasks at seed %d" % (len(pairs), seeds[0]))
            print("    loss mean ratio probe/kaggle: median %.2f, range %.2f to %.2f"
                  % (ratios[len(ratios) // 2], ratios[0], ratios[-1]))
            print("    wall ratio probe/kaggle     : median %.2f"
                  % sorted(p[3] / p[4] for p in pairs if p[3] and p[4])[len(pairs) // 2])
            print("    outcome agreement           : %d/%d tasks" % (agree, len(pairs)))
            if agree < len(pairs) * 0.7:
                print("    WARNING: outcomes disagree on more than a third of shared tasks, which")
                print("             is a porting problem rather than a result. Do not trust the")
                print("             headline until it is explained.")
    except Exception as e:                                              # noqa: BLE001
        print("\n  FIDELITY check unavailable: %s" % type(e).__name__)

    gain = union - mean_single
    print("\n  reweighted gain from restarts: %+.1f percentage points" % (100 * gain))
    if net is not None and net <= 0.02:
        print("  WARNING: cross-seed disagreement does not exceed the replicate floor. This run")
        print("           measured nondeterminism, not decorrelation. The answer is NO RESULT.")
    print("  READ: union materially above single, AND net disagreement clearly positive, means")
    print("        restarts are the lever. Otherwise the misses are hard and scheduling cannot")
    print("        recover them. The claim is scoped to COLOUR-PERMUTATION restarts either way.")

    out = {"n_outputs": n, "n_tasks": len(full), "seeds": seeds,
           "single_weighted": single, "union_weighted": union, "intersection_weighted": inter,
           "gain_weighted": gain, "net_disagreement": net, "dropped_tasks": dropped,
           "excluded_cells": len(hard), "decode_truncated_cells": n_trunc,
           "mixed_truncation_tasks": sorted(mixed)}
    json.dump(out, open(SUMMARY, "w"), indent=1, default=str)
    print("\nSUMMARY", SUMMARY, flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        sys.exit(0 if main() else 3)
