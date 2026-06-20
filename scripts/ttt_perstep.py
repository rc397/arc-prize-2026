import sys, time
sys.path.insert(0, r"D:\arc")
import numpy as np, torch
from src.arc.model import load_model
from src.arc.data import load_consolidated
from src.arc.ttt import make_adapter, reset_adapter, _example

MODEL = r"D:\arc\models\kagglehub\models\sorokin\qwen3_4b_grids15_sft139\transformers\bfloat16\1"
CH = r"D:\arc\data\kaggle\arc-agi_evaluation_challenges.json"
SOL = r"D:\arc\data\kaggle\arc-agi_evaluation_solutions.json"

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
for s in range(12):
    t = time.time()
    ids, labels = _example(task, rng)
    ii = torch.tensor([ids], device="cuda")
    ll = torch.tensor([labels], device="cuda")
    loss = peft(input_ids=ii, labels=ll).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    print("step %2d seqlen=%4d loss=%.3f  %.2fs" % (s, len(ids), float(loss), time.time() - t), flush=True)
