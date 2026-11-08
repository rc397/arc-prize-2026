import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model, build_prompt_ids, grid_to_ids, IM_END
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, reset_adapter
from src.arc import augment as A

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"


def example_capped(task, rng, max_tokens):
    n = len(task["train"])
    aug = A.random_augmentation(rng, n_examples=n)
    pairs = A.apply_task(aug, task)["train"]
    t = int(rng.integers(0, n))
    ctx_idx = [i for i in range(n) if i != t]
    rng.shuffle(ctx_idx)
    chosen = []
    for i in ctx_idx:
        plen = len(build_prompt_ids([pairs[j] for j in chosen + [i]], pairs[t]["input"]))
        if plen > max_tokens and chosen:
            break
        chosen = chosen + [i]
    prompt = build_prompt_ids([pairs[i] for i in chosen], pairs[t]["input"])
    target = grid_to_ids(pairs[t]["output"]) + [IM_END]
    return prompt + target, [-100] * len(prompt) + target


model, tok = load_model(MODEL)
tasks, sols = load_consolidated(CH, SOL, n=3)
peft, init = make_adapter(model, rank=32)
base = peft.get_base_model()
reset_adapter(peft, init)
base.config.use_cache = False
base.gradient_checkpointing_enable()
base.enable_input_require_grads()
peft.train()
params = [p for p in peft.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-4)
rng = np.random.default_rng(0)
task = tasks[list(tasks.keys())[0]]
for cap in [1500, 800]:
    for s in range(5):
        t = time.time()
        ids, labels = example_capped(task, rng, cap)
        ii = torch.tensor([ids], device="cuda")
        ll = torch.tensor([labels], device="cuda")
        loss = peft(input_ids=ii, labels=ll).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        print("cap=%d step=%d seqlen=%4d %.2fs" % (cap, s, len(ids), time.time() - t), flush=True)
