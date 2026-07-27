import os, sys, glob
# ============ unsloth tree wiring (works for utility-script AND notebook-output mounts) ============
_hits = glob.glob('/kaggle/input/**/unsloth/__init__.py', recursive=True) \
      + glob.glob('/kaggle/usr/lib/**/unsloth/__init__.py', recursive=True)
# prefer our own fork's tree over any stale utility mount that may still be attached
_hits.sort(key=lambda p: (('rc7530' not in p), ('/kaggle/input/' not in p)))
if _hits:
    _U = os.path.dirname(os.path.dirname(_hits[0]))
    sys.path.insert(0, _U)
    os.environ['PYTHONPATH'] = _U + os.pathsep + os.environ.get('PYTHONPATH', '')
    print('unsloth tree wired from', _U)
else:
    print('unsloth tree not found in inputs, the install cell will need internet')
# ============ CONFIG (paths self-discover, override only if the print is wrong) ============
os.environ["ARC_MODEL_PATH"] = "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
# auto-locate the attached grids15 model if the default path is not the actual mount
if not os.path.isdir(os.environ["ARC_MODEL_PATH"]):
    hits = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "pip-install" not in p]
    pref = [p for p in hits if "grids15" in p.lower().replace("_", "").replace("-", "")]
    pick = (pref or hits)
    if pick:
        os.environ["ARC_MODEL_PATH"] = os.path.dirname(sorted(pick, key=len)[0])
os.environ["ARC_DATA_DIR"]   = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
if not os.path.isdir(os.environ["ARC_DATA_DIR"]):
    cand = glob.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True)
    if cand:
        os.environ["ARC_DATA_DIR"] = os.path.dirname(cand[0])
# Interactive eval tasks to score (non-rerun). 0 = all 120. Use 6 for a smoke run.
# Experiment knobs. Defaults reproduce the 30.56 submission, change one at a time.
#   ARC_DFS_FLOOR    search width, lower keeps more branches and costs time (default 0.2)
#   ARC_TASK_TIMEOUT per-task seconds (default 1800, the 30.56 submission used 1200)
#   ARC_RANK         LoRA rank (default 256)      ARC_MAXSEQ  training context (default 8192)
#   ARC_LOSS_CUT     skip the search when training loss ends above this, 0 disables (v7.3: 0.05)
#   ARC_GREEDY_FALLBACK  fill search-empty tasks with a greedy decode (v7.3: 1)
os.environ["ARC_GREEDY_FALLBACK"] = "0"
os.environ["ARC_LOSS_CUT"] = "0"
os.environ["ARC_N_EVAL"] = "0"
os.environ["ARC_TASK_TIMEOUT"] = "1200"
# ====================================================
print("MODEL ", os.environ["ARC_MODEL_PATH"], "exists:", os.path.isdir(os.environ["ARC_MODEL_PATH"]))
print("DATA  ", os.environ["ARC_DATA_DIR"], "exists:", os.path.isdir(os.environ["ARC_DATA_DIR"]))
# echo every knob so a smoke run proves the config landed instead of silently defaulting
print("CONFIG N_EVAL=%s RANK=%s MAXSEQ=%s TIMEOUT=%s DFS_FLOOR=%s LOSS_CUT=%s GFB=%s" % (
    os.environ.get("ARC_N_EVAL", "40"), os.environ.get("ARC_RANK", "256"),
    os.environ.get("ARC_MAXSEQ", "8192"), os.environ.get("ARC_TASK_TIMEOUT", "1800"),
    os.environ.get("ARC_DFS_FLOOR", "0.2"), os.environ.get("ARC_LOSS_CUT", "0"),
    os.environ.get("ARC_GREEDY_FALLBACK", "0")))
assert os.path.isdir(os.environ["ARC_MODEL_PATH"]), "model not found under /kaggle/input, attach it"
assert os.path.isdir(os.environ["ARC_DATA_DIR"]), "competition data not found, attach it"
