"""Build notebooks/arc_submission_v7.ipynb from the public ARChitects notebook, made Kaggle-runnable.

Source is learned-from-aimo.ipynb, the public port of the ARChitects' 2024-winning pipeline (heavy
rank-256 per-task TTT, a DFS decoder, augmentation-product selection, plus a small symbolic arm) onto
our grids15_sft139 base, claimed ~31 on ARC-AGI-2. We change only what would break a fresh run:
  - a config cell exposing the model path, data dir, and eval-task count as env vars,
  - an Unsloth install cell for interactive (internet-on) runs,
  - the hardcoded model/data paths rewired to those env vars,
  - the four-task dev filter opened to a configurable eval slice.
Their method and symbolic arm are otherwise untouched, so the first interactive run reproduces their
claim on our model. The method is theirs and the paper attributes it; this only verifies the number.
"""
import json
import os
import re
import sys

# Named variants. The pipeline is identical in every one, only the config cell differs, so a variant is
# a documented set of knob values rather than a fork. v7.1 is the 30.56 submission and its knobs are the
# upstream defaults. Anything else exists to answer one question and says so in its own header.
VARIANTS = {
    "v7.1": {
        "file": "arc_submission_v7.ipynb",
        "env": {},
        "note": "The submitted configuration, scored 30.56 on the hidden set. Upstream defaults.",
    },
    "v7.2": {
        "file": "arc_submission_v7_2.ipynb",
        "env": {"ARC_DFS_FLOOR": "0.1"},
        "note": ("Search-width arm. Halves the DFS probability floor from 0.2 to 0.1 so more branches "
                 "survive, which buys candidates and costs time. Run it at ARC_N_EVAL=6 first and read "
                 "seconds per task, since the hidden run has 240 tasks against a twelve-hour wall and "
                 "v7.1 cleared that by roughly an hour. If the projection exceeds the wall this arm "
                 "cannot be submitted whatever it scores."),
    },
    "v7.3": {
        "file": "arc_submission_v7_3.ipynb",
        "env": {"ARC_DFS_FLOOR": "0.35"},
        "note": ("Narrow-search control for v7.2. Raising the floor prunes earlier and runs faster. Its "
                 "purpose is the gradient rather than the score, since v7.2 and v7.3 either bracket the "
                 "0.2 default or point consistently one way, and one batch then closes the lever."),
    },
}

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v7.1"
assert VARIANT in VARIANTS, "unknown variant %r, expected one of %s" % (VARIANT, sorted(VARIANTS))
SPEC = VARIANTS[VARIANT]

SRC = r"C:\Users\rchen\Downloads\learned-from-aimo.ipynb"
DST = os.path.join(r"D:\ARC_AGI-2,3\notebooks", SPEC["file"])

DATA_DEFAULT = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
MODEL_DEFAULT = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def rewire(text):
    for fname in ("arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json",
                  "arc-agi_evaluation_solutions.json"):
        text = text.replace(
            '"%s/%s"' % (DATA_DEFAULT, fname),
            'os.path.join(os.environ.get("ARC_DATA_DIR", "%s"), "%s")' % (DATA_DEFAULT, fname))
    text = text.replace(
        'model_name="%s",' % MODEL_DEFAULT,
        'model_name=os.environ.get("ARC_MODEL_PATH", "%s"),' % MODEL_DEFAULT)
    # disable the AIF arm: on the 40-task eval slice it fast-pathed 12 tasks with zero correct,
    # sacrificing the LLM pipeline on each. One neutered dict kills the early-exit, the beam
    # pruning, and keeps the confidence print alive. Reverts behavior to sorokin's original.
    text = text.replace(
        "        executor.shutdown(wait=False)\n",
        "        executor.shutdown(wait=False)\n"
        "        # AIF arm disabled: measured 12/39 fast-paths with zero correct on the eval slice\n"
        "        aif_result = {'confidence': 0.0, 'grid': None, 'expected_count': -1, 'is_solved': False}\n")
    # restore sorokin's DFS batching. The aimo edit split his 4-view batches into pairs, halving
    # decode throughput at zero semantic benefit. Same views, same order, grouped 4 per DFS call.
    text = text.replace(
        "        batches = []\n"
        "        for test_id, subkeys in test_id_to_subkeys.items():\n"
        "            for offset in [0, 4]:\n"
        "                batches.append(subkeys[offset:offset+2])\n"
        "            for offset in [2, 6]:\n"
        "                batches.append(subkeys[offset:offset+2])\n"
        "        for test_id, subkeys in test_id_to_subkeys.items():\n"
        "            for offset in [8, 12]:\n"
        "                batches.append(subkeys[offset:offset+2])\n"
        "            for offset in [10, 14]:\n"
        "                batches.append(subkeys[offset:offset+2])\n",
        "        batches = []\n"
        "        for test_id, subkeys in test_id_to_subkeys.items():\n"
        "            batch = []\n"
        "            for offset in [0, 4]:\n"
        "                batch.extend(subkeys[offset:offset+2])\n"
        "            batches.append(batch)\n"
        "            batch = []\n"
        "            for offset in [2, 6]:\n"
        "                batch.extend(subkeys[offset:offset+2])\n"
        "            batches.append(batch)\n"
        "        for test_id, subkeys in test_id_to_subkeys.items():\n"
        "            batch = []\n"
        "            for offset in [8, 12]:\n"
        "                batch.extend(subkeys[offset:offset+2])\n"
        "            batches.append(batch)\n"
        "            batch = []\n"
        "            for offset in [10, 14]:\n"
        "                batch.extend(subkeys[offset:offset+2])\n"
        "            batches.append(batch)\n")
    # Deployed selector as an env knob. The default stays kgmon, which is upstream's choice.
    # A 40-task read had kgmon's solved set as a strict subset of probmul's, three discordant tasks all
    # favouring probmul, and probmul weights each vote by confidence where kgmon only counts votes, so
    # the mechanism looked sound. The full 120-task run refutes it. kgmon scores 33.8 against probmul's
    # 33.3, each wins one task the other loses, and on the eighty tasks outside the original slice kgmon
    # is ahead by a full credit. Three discordant cases were an anecdote and the mechanism story made a
    # tuned result feel principled. Left as ARC_SELECTOR=probmul for anyone wanting to re-test it.
    text = text.replace(
        "    def run_selection_algo(self, selection_algorithm=score_kgmon):",
        "    def run_selection_algo(self, selection_algorithm=None):\n"
        "        if selection_algorithm is None:\n"
        '            selection_algorithm = (score_full_probmul_3 if os.environ.get("ARC_SELECTOR", "kgmon")\n'
        "                                   == \"probmul\" else score_kgmon)")
    # dump the full candidate pool for offline selection analysis. Piggybacks on the benchmark method,
    # where every candidate is already loaded with its beam score, per-augmentation NLLs and correctness.
    # Writes /kaggle/working/pool_dump.json, which survives the output pull. Eval mode only, harmless in
    # the hidden rerun where there are no labels.
    text = text.replace(
        "            print(f\" acc: {score:5.1f}/{num_puzzles:3} ('{name}')\")",
        "            print(f\" acc: {score:5.1f}/{num_puzzles:3} ('{name}')\")\n"
        "\n"
        "        import json as _json\n"
        "        _pool = {}\n"
        "        for _bk, _subs in self.decoded_results.items():\n"
        "            _lab = labels.get(_bk)\n"
        "            _cands = {}\n"
        "            for _sk, _s in _subs.items():\n"
        "                _grid = [list(map(int, _r)) for _r in _s[\"solution\"]]\n"
        "                _key = _json.dumps(_grid)\n"
        "                _c = _cands.setdefault(_key, {\"grid\": _grid, \"count\": 0, \"beam\": [], \"aug\": [],\n"
        "                    \"correct\": bool(np.array_equal(_s[\"solution\"], _lab)) if _lab is not None else False})\n"
        "                _c[\"count\"] += 1\n"
        "                _c[\"beam\"].append(float(_s[\"beam_score\"]))\n"
        "                _c[\"aug\"].append([float(_x) for _x in _s[\"score_aug\"]])\n"
        "            _pool[_bk] = list(_cands.values())\n"
        "        try:\n"
        "            _json.dump(_pool, open(\"/kaggle/working/pool_dump.json\", \"w\"))\n"
        "            print(\"POOL DUMP: %d outputs -> /kaggle/working/pool_dump.json\" % len(_pool))\n"
        "        except Exception as _e:\n"
        "            print(\"pool dump failed:\", _e)\n")
    # DFS probability floor as an env knob. This is the single biggest search-budget dial, it sets how
    # improbable a continuation may get before the tree search abandons it. Lower keeps more branches
    # alive and costs time, so it trades directly against the per-task cap above.
    text = text.replace(
        "    max_score = -np.log(0.2)\n",
        '    max_score = -np.log(float(os.environ.get("ARC_DFS_FLOOR", "0.2")))\n')
    # per-task time cap as an env knob. The aimo edit raised sorokin's 1200 to 1800; the stripped
    # pipeline needs the tighter cap to fit 240 hidden tasks inside the 12h rerun.
    text = text.replace(
        "if spend_time > 1800 or time.time() > end_time:",
        'if spend_time > int(os.environ.get("ARC_TASK_TIMEOUT", "1800")) or time.time() > end_time:')
    # adapt the hardcoded 4-worker/4-GPU launch to whatever GPUs the session actually has
    text = text.replace(
        "    args = parser.parse_args()",
        "    args = parser.parse_args()\n    NPROCS = torch.cuda.device_count() or 1")
    text = text.replace("    for _ in range(4):", "    for _ in range(NPROCS):")
    text = text.replace(
        "mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)",
        "mp.spawn(local_worker, args=(queue, args.end_time), nprocs=NPROCS)")
    # make the heavy knobs and the output dir env-overridable (defaults preserve their config), so the
    # debug run.py can shrink them for a fast low-VRAM plumbing check without touching the real notebook
    text = text.replace("        r=256,", '        r=int(os.environ.get("ARC_RANK", "256")),')
    text = text.replace("    max_seq_length = 8192",
                        '    max_seq_length = int(os.environ.get("ARC_MAXSEQ", "8192"))')
    text = text.replace('    dir_outputs = "/kaggle/inference_outputs"',
                        '    dir_outputs = os.environ.get("ARC_OUT_DIR", "/kaggle/inference_outputs")')
    return text


FILTER_OLD = (
    '    for key in sorted(data.keys()):\n'
    '        if not rerun_mode:\n'
    '            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:\n'
    '                continue\n'
    '        queue.put(key)')
FILTER_NEW = (
    '    N_EVAL = int(os.environ.get("ARC_N_EVAL", "40"))\n'
    '    keys = sorted(data.keys())\n'
    '    if not rerun_mode and N_EVAL > 0:\n'
    '        keys = keys[:N_EVAL]\n'
    '    for key in keys:\n'
    '        queue.put(key)')

nb = json.load(open(SRC, encoding="utf-8"))

n_filter = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    text = rewire("".join(c["source"]))
    if FILTER_OLD in text:
        text = text.replace(FILTER_OLD, FILTER_NEW)
        n_filter += 1
    c["source"] = text.splitlines(keepends=True)

assert n_filter == 1, "task filter not patched exactly once (got %d)" % n_filter
alltext = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
assert "ARC_MODEL_PATH" in alltext, "model-path rewire missed"
assert "ARC_DATA_DIR" in alltext, "data-dir rewire missed"
assert "0934a4d8" not in alltext, "old dev filter still present"

intro = md_cell(
    "# v7 - ARChitects method reproduction (verify the ~31 claim on our model)\n\n"
    "This is the public ARChitects 2024-winning pipeline ported onto our grids15_sft139 base. It is "
    "**their** method (heavy rank-256 per-task TTT, a DFS decoder, augmentation-product selection). It "
    "is reproduced here only to measure its real number on our model. Credit is theirs and the paper "
    "attributes it.\n\n"
    "**Before running**\n\n"
    "1. Attach the grids15_sft139 model and the arc-prize-2026-arc-agi-2 competition data.\n"
    "2. Set `ARC_MODEL_PATH` and `ARC_DATA_DIR` in the config cell to match the Input panel on the right.\n"
    "3. Internet ON for this interactive run so Unsloth installs.\n"
    "4. `ARC_N_EVAL` picks how many eval tasks to score. The final cell prints the score against the eval "
    "solutions. Budget roughly 15-30 minutes per task, sharded over the 4 GPUs.\n\n"
    "The offline contest submission is a separate step, Unsloth rides in as an attached wheel and the "
    "rerun scores the full hidden set.\n")

config = code_cell(
    "import os, sys, glob\n"
    "# ============ unsloth tree wiring (works for utility-script AND notebook-output mounts) ============\n"
    "_hits = glob.glob('/kaggle/input/**/unsloth/__init__.py', recursive=True) \\\n"
    "      + glob.glob('/kaggle/usr/lib/**/unsloth/__init__.py', recursive=True)\n"
    "# prefer our own fork's tree over any stale utility mount that may still be attached\n"
    "_hits.sort(key=lambda p: (('rc7530' not in p), ('/kaggle/input/' not in p)))\n"
    "if _hits:\n"
    "    _U = os.path.dirname(os.path.dirname(_hits[0]))\n"
    "    sys.path.insert(0, _U)\n"
    "    os.environ['PYTHONPATH'] = _U + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
    "    print('unsloth tree wired from', _U)\n"
    "else:\n"
    "    print('unsloth tree not found in inputs, the install cell will need internet')\n"
    "# ============ CONFIG (paths self-discover, override only if the print is wrong) ============\n"
    'os.environ["ARC_MODEL_PATH"] = "%s"\n' % MODEL_DEFAULT +
    "# auto-locate the attached grids15 model if the default path is not the actual mount\n"
    'if not os.path.isdir(os.environ["ARC_MODEL_PATH"]):\n'
    '    hits = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "pip-install" not in p]\n'
    '    pref = [p for p in hits if "grids15" in p.lower().replace("_", "").replace("-", "")]\n'
    '    pick = (pref or hits)\n'
    '    if pick:\n'
    '        os.environ["ARC_MODEL_PATH"] = os.path.dirname(sorted(pick, key=len)[0])\n'
    'os.environ["ARC_DATA_DIR"]   = "%s"\n' % DATA_DEFAULT +
    'if not os.path.isdir(os.environ["ARC_DATA_DIR"]):\n'
    '    cand = glob.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True)\n'
    '    if cand:\n'
    '        os.environ["ARC_DATA_DIR"] = os.path.dirname(cand[0])\n'
    "# Interactive eval tasks to score (non-rerun). 0 = all 120. Use 6 for a smoke run.\n"
    'os.environ["ARC_N_EVAL"]     = "40"\n'
    "# Experiment knobs. Defaults reproduce the 30.56 submission, change one at a time.\n"
    "#   ARC_DFS_FLOOR    search width, lower keeps more branches and costs time (default 0.2)\n"
    "#   ARC_TASK_TIMEOUT per-task seconds (default 1800, the 30.56 submission used 1200)\n"
    "#   ARC_RANK         LoRA rank (default 256)      ARC_MAXSEQ  training context (default 8192)\n"
    + ("".join('os.environ["%s"] = "%s"\n' % (k, v) for k, v in sorted(SPEC["env"].items()))
       or '# os.environ["ARC_DFS_FLOOR"]    = "0.1"\n'
          '# os.environ["ARC_TASK_TIMEOUT"] = "1200"\n')
    + "# ====================================================\n"
    'print("MODEL ", os.environ["ARC_MODEL_PATH"], "exists:", os.path.isdir(os.environ["ARC_MODEL_PATH"]))\n'
    'print("DATA  ", os.environ["ARC_DATA_DIR"], "exists:", os.path.isdir(os.environ["ARC_DATA_DIR"]))\n'
    "# echo every knob so a smoke run proves the config landed instead of silently defaulting\n"
    'print("CONFIG N_EVAL=%s RANK=%s MAXSEQ=%s TIMEOUT=%s DFS_FLOOR=%s" % (\n'
    '    os.environ.get("ARC_N_EVAL", "40"), os.environ.get("ARC_RANK", "256"),\n'
    '    os.environ.get("ARC_MAXSEQ", "8192"), os.environ.get("ARC_TASK_TIMEOUT", "1800"),\n'
    '    os.environ.get("ARC_DFS_FLOOR", "0.2")))\n'
    'assert os.path.isdir(os.environ["ARC_MODEL_PATH"]), "model not found under /kaggle/input, attach it"\n'
    'assert os.path.isdir(os.environ["ARC_DATA_DIR"]), "competition data not found, attach it"\n')

install = code_cell(
    "# Unsloth arrives one of two ways.\n"
    "# PREFERRED (offline-safe, the route the 32.22 exemplar used): attach a utility-script notebook that\n"
    "# pip-installed unsloth==2025.9.7 + the flash-attn wheel with --target on the CURRENT Kaggle image\n"
    "# (fork sorokin/pip-install-unsloth-flash-patch, swap the cp311 flash-attn wheel URL for the cp312\n"
    "# build matching this image's torch, run it once with internet, attach YOUR fork). Its tree lands on\n"
    "# sys.path automatically and the import below just works, no pip here at all.\n"
    "# FALLBACK (interactive with internet only): plain pip install.\n"
    "try:\n"
    "    import unsloth  # noqa: F401\n"
    "    print('unsloth already importable (utility-script route), version', unsloth.__version__)\n"
    "except ImportError:\n"
    "    print('unsloth not on path, falling back to pip (needs internet)')\n"
    "    import subprocess, sys\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth', 'unsloth_zoo'], check=True)\n")

# order: intro, config, time-cell(orig 0), uninstall-tf(orig 1), install-unsloth, then the rest
nb["cells"] = [intro, config, nb["cells"][0], nb["cells"][1], install] + nb["cells"][2:]

assert "src.arc" not in alltext and "from src" not in alltext, "v7 must not import our repo"

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1)
print("wrote", DST)
print("cells:", len(nb["cells"]))

# ---- also emit a self-contained debug folder: the standalone modules + an in-process run.py ----
DEBUG_DIR = r"D:\ARC_AGI-2,3\notebooks\v7_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)
modules = {}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    m = re.match(r"^%%writefile\s+(\S+)\s*\n(.*)$", "".join(c["source"]), re.S)
    if m:
        modules[m.group(1)] = m.group(2)
for fname in ["arc_loader.py", "arc_decoder.py", "arc_aif.py", "arc_solver.py"]:
    assert fname in modules, "missing module %s" % fname
    with open(os.path.join(DEBUG_DIR, fname), "w", encoding="utf-8") as f:
        f.write(modules[fname])

RUN_PY = '''#!/usr/bin/env python
"""Standalone in-process debug driver for the v7 (ARChitects) pipeline. No GitHub repo needed.

Unlike the notebook, which spawns the workers as subprocesses and hides their output, this runs one
worker in THIS process over a few eval tasks, so any error surfaces with a full traceback. Edit the
sibling modules (arc_solver.py and friends) and re-run to iterate.

Needs unsloth installed. The full rank-256 / 8192-context config wants a 24GB GPU; for a fast plumbing
check on less VRAM set ARC_RANK=16 ARC_MAXSEQ=2048 ARC_N_EVAL=1.

Env knobs: ARC_MODEL_PATH, ARC_DATA_DIR, ARC_N_EVAL, ARC_OUT_DIR, ARC_RANK, ARC_MAXSEQ.
"""
import os, sys, json, time, queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("ARC_MODEL_PATH", "%s")
os.environ.setdefault("ARC_DATA_DIR", "%s")
os.environ.setdefault("ARC_N_EVAL", "1")
os.environ.setdefault("ARC_OUT_DIR", os.path.join(HERE, "inference_outputs"))
# for a fast low-VRAM plumbing check, uncomment the next line:
# os.environ["ARC_RANK"], os.environ["ARC_MAXSEQ"] = "16", "2048"

print("MODEL ", os.environ["ARC_MODEL_PATH"])
print("DATA  ", os.environ["ARC_DATA_DIR"])
print("N_EVAL", os.environ["ARC_N_EVAL"], "RANK", os.environ.get("ARC_RANK", "256"),
      "MAXSEQ", os.environ.get("ARC_MAXSEQ", "8192"))
os.makedirs(os.environ["ARC_OUT_DIR"], exist_ok=True)

ch = os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_challenges.json")
sol = os.path.join(os.environ["ARC_DATA_DIR"], "arc-agi_evaluation_solutions.json")
keys = sorted(json.load(open(ch)).keys())[:int(os.environ["ARC_N_EVAL"])]
print("tasks:", keys)

from arc_solver import worker
q = queue.Queue()
for k in keys:
    q.put(k)
q.put(None)

worker(0, q, time.time() + 3 * 3600)

from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
data = ArcDataset.from_file(ch).load_replies(sol).change_keys(keys, keep_flags=True)
dec = ArcDecoder(data.split_multi_replies(), n_guesses=2)
dec.load_decoded_results(os.environ["ARC_OUT_DIR"])
sub = data.get_submission(dec.run_selection_algo())
print("SCORE over %%d tasks:" %% len(keys), data.validate_submission(sub))
''' % (MODEL_DEFAULT, DATA_DEFAULT)

with open(os.path.join(DEBUG_DIR, "run.py"), "w", encoding="utf-8") as f:
    f.write(RUN_PY)
print("wrote debug folder", DEBUG_DIR, "->", sorted(os.listdir(DEBUG_DIR)))

# also embed run.py INTO the notebook, written next to the modules on Kaggle, so `!python run.py`
# is available for in-process debugging right where the notebook runs and fails
run_cell = code_cell("%%writefile run.py\n" + RUN_PY)
for i, c in enumerate(nb["cells"]):
    if "".join(c["source"]).startswith("%%writefile arc_solver.py"):
        nb["cells"].insert(i + 1, run_cell)
        break
json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1)
print("re-wrote", DST, "with embedded run.py, cells now", len(nb["cells"]))
