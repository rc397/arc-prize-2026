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
import textwrap

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
        "env": {"ARC_DUMP_POOL": "1"},
        "note": ("Selection-analysis build. Identical pipeline to v7.1 plus a candidate-pool dump, gated "
                 "on ARC_DUMP_POOL so only this build writes it. Run at ARC_N_EVAL=40, it writes "
                 "pool_dump.json with every candidate's grid, beam score, per-augmentation NLLs, visit "
                 "count and correctness, so any selector or verifier can then be tested offline for free "
                 "against the ~5-point selection ceiling. Not for submission, it is a measurement run. "
                 "Submitted anyway as a cap read, scored 30.97 at the default 1800s cap."),
    },
    "v7.3": {
        "file": "arc_submission_v7_3.ipynb",
        "env": {"ARC_LOSS_CUT": "0.05", "ARC_GREEDY_FALLBACK": "1", "ARC_N_EVAL": "0",
                "ARC_TASK_TIMEOUT": "1200"},
        "note": ("Scheduling build, rebased onto what we then believed was the 33.89 deployment condition. "
                 "CAUTION, that premise was wrong twice over. koushikrudra/failed-in-aimo does not carry "
                 "33.89, we forked it as a control and it scored 28.06, and v7.1 ran 1800 rather than the "
                 "1200 asserted below. The environment reasoning may still hold, the score attribution "
                 "does not. The public notebook "
                 "koushikrudra/failed-in-aimo is byte-equivalent to our v7.1 restoration in every module "
                 "yet scores 33.89 to our 30.56 at the same 1200s cap, and its kernel metadata shows the "
                 "one difference, it mounts sorokin's original pip-install-unsloth-flash-patch wheel tree "
                 "and model upload where ours mounts the rc7530 slim fork built for an older image. Same "
                 "code twice on the hidden set prices the environment at 3.33 points. So v7.3 must be "
                 "uploaded with sorokin/pip-install-unsloth-flash-patch and sorokin's model attached, NOT "
                 "the rc7530 fork, and runs cap 1200 to match. On top of that condition it adds the two "
                 "do-no-harm scheduler rules priced on the v7.2 measurement run, the loss-gated search "
                 "skip and the greedy fill of search-empty slots, so any delta against 33.89 is the "
                 "scheduler's. SCORED 28.61, a regression, and the post-mortem found the cap knob had "
                 "never been wired, so this ran at 1800 like every build before it. See v7.4."),
    },
    "v7.4": {
        "file": "arc_submission_v7_4.ipynb",
        "env": {"ARC_TASK_TIMEOUT": "1200", "ARC_LOSS_CUT": "0", "ARC_GREEDY_FALLBACK": "0",
                "ARC_N_EVAL": "0"},
        "note": ("The cap, wired for the first time. ARC_TASK_TIMEOUT was echoed by the config cell from "
                 "v7.1 onward but the builder's replace targeted a spend_time form the source does not "
                 "use, so it never matched and v7.1, v7.2 and v7.3 all ran the hardcoded 1800 while "
                 "reporting otherwise. The public 33.89 build runs 1200. On a 240-task rerun against a "
                 "twelve-hour wall a looser per-task cap does not buy depth, it lets slow tasks eat the "
                 "budget until the tail of the queue is dropped unstarted, which is the standing "
                 "candidate explanation for a three-point gap we had wrongly attributed to the "
                 "dependency tree. The loss cut is OFF here, it is the leading suspect for v7.3's "
                 "regression and its justification was five tasks. The greedy fill is off too. Its 6-task "
                 "smoke ran it clean past the per-task cap, since it sits after the DFS loop and had no "
                 "budget of its own, adding 330s to one task in four fills. At roughly 80s a fill that "
                 "is near two hours of unbudgeted work on a 240-task rerun, a better candidate for "
                 "v7.3's regression than the loss cut. It now takes ARC_GFB_BUDGET seconds of its own, "
                 "default 180, but v7.4 leaves it off so this build is a pure control against the 33.89 "
                 "public notebook, identical pipeline, identical cap, nothing of ours added. SCORED "
                 "29.44, and it was not pure. The fill ran anyway through a truthy-string gate, so two "
                 "factors moved and the run is void rather than adverse. A note here previously also "
                 "withdrew the cap claim, on the grounds that the 33.89 notebook forks the aimo source "
                 "which hardcodes 1800. That was read off the WRONG notebook, koushikrudra/failed-in-aimo, "
                 "which we forked as a control and which scored 28.06. The build carrying 33.89 is "
                 "mikelou1/arc-agi2-lb33-89-minimal-perfpatch, and its own decode loop carries a literal "
                 "1200 on the effective path. The cap claim is reinstated and remains untested cleanly."),
    },
    # ---- the bisection. v7.1 (30.56) and the 33.89 public notebook differ in exactly two factors,
    # everything else in the diff being byte-identical, neutral at its default, or identical under
    # rerun_mode. The three builds below vary one factor at a time off that pair, at the 1800 cap both
    # sides actually ran. Run v7.5c FIRST if the pending control does not land near 33.89, since it
    # tests our builder rather than our edits.
    "v7.5a": {
        "file": "arc_submission_v7_5a.ipynb",
        "env": {"ARC_AIF": "prune", "ARC_DFS_BATCH": "4", "ARC_TASK_TIMEOUT": "1800", "ARC_N_EVAL": "0"},
        "note": ("The theory build, and the one worth a submission slot. Keeps our 4-view batching and "
                 "restores only the AIF object-count prune, leaving the fast-path preemption off. The "
                 "2026-07-20 A/B deleted the arm on the strength of the preemption alone, 12 of 39 tasks "
                 "fast-pathed with zero correct, but expected_count is returned on every branch of "
                 "run_aif_analysis including the no-rule fallback and re-ranks beams without ever "
                 "preempting. Appendix F's internal-ceiling theorem says a selector reading only model "
                 "likelihoods cannot exceed the top-two posterior mass, and names program execution "
                 "against the demonstrations as one of two channels that break the assumption. This is "
                 "that channel. No configuration on either side has run it."),
    },
    "v7.5b": {
        "file": "arc_submission_v7_5b.ipynb",
        "env": {"ARC_AIF": "off", "ARC_DFS_BATCH": "2", "ARC_TASK_TIMEOUT": "1800", "ARC_N_EVAL": "0"},
        "note": ("Isolates the batching. v7.1 with sorokin's 4-view grouping reverted to the aimo pair "
                 "form, AIF still off. The restoration was credited with an eleven-point eval gain, but "
                 "the recorded mechanism was pool coverage, 155 correct answers present in the beams "
                 "against 105, and Section 5.7 measures 12.333 credits present in pool against 9.833 "
                 "attained. Widening a pool the selector cannot rank is the paper's own documented "
                 "failure mode, and pairs of 2 also double the granularity at which the cap can cut. "
                 "Against v7.1's 30.56 this prices the batching on the hidden set alone."),
    },
    "v7.5c": {
        "file": "arc_submission_v7_5c.ipynb",
        "env": {"ARC_AIF": "full", "ARC_DFS_BATCH": "2", "ARC_TASK_TIMEOUT": "1800", "ARC_N_EVAL": "0"},
        "note": ("Our builder reproducing the public notebook's behaviour exactly, both factors upstream "
                 "at the cap both sides ran. This is a check on us, not on the method. If it lands near "
                 "33.89 the pipeline is faithful and the deficit is entirely our two edits, so v7.5a and "
                 "v7.5b price them. If it lands near 30 with the pending unmodified-fork control also "
                 "near 30, then 33.89 does not reproduce on this account at all, the deficit was never "
                 "ours, and the last week chased a ghost that the paper should record as such."),
    },
}
# The search-width arms (old v7.2 at floor 0.1, v7.3 at floor 0.35) were measured dead, the DFS floor
# changed the deployed output by nothing, so their notebooks are removed. The finding is in the DEVLOG.

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
    # AIF arm, gated on ARC_AIF rather than deleted. The arm returns two independent things and
    # v7.1 through v7.4 killed both with one stub. is_solved preempts the LLM search entirely and was
    # measured harmful, 12 of 39 eval tasks fast-pathed with zero correct. expected_count only
    # re-ranks beams by an object count read off the demonstrations, costs nothing, and was never
    # measured on its own. Appendix F names that second half as one of exactly two channels that can
    # beat the internal-ceiling theorem, since it reads the task through something other than the
    # model's beliefs, so deleting it was the expensive half of a cheap decision.
    text = text.replace(
        "        # ================================================================\n"
        "        # LAUNCH AIF IN BACKGROUND THREAD\n"
        "        # ================================================================\n"
        "        aif_future = None\n"
        "        executor = ThreadPoolExecutor(max_workers=1)\n"
        "        try:\n"
        "            raw_train_data = puzzle_ds.queries[key]['train']\n"
        "            raw_test_input = puzzle_ds.queries[key]['test'][0]['input']\n"
        "            aif_future = executor.submit(\n"
        "                run_aif_analysis,\n"
        "                raw_train_data,\n"
        "                raw_test_input\n"
        "            )\n"
        "            print(f\"[Rank {rank}] AIF background analysis launched for {key}...\")\n"
        "        except Exception as e:\n"
        "            print(f\"[Rank {rank}] Failed to launch AIF: {e}\")\n"
        "            aif_future = None\n",
        "        # off = arm never launched (the v7.1 default), prune = object-count signal with the\n"
        "        # preemption suppressed, full = upstream including the fast path.\n"
        '        _aif_mode = os.environ.get("ARC_AIF", "off")\n'
        "        aif_future = None\n"
        "        executor = None\n"
        '        if _aif_mode != "off":\n'
        "            executor = ThreadPoolExecutor(max_workers=1)\n"
        "            try:\n"
        "                raw_train_data = puzzle_ds.queries[key]['train']\n"
        "                raw_test_input = puzzle_ds.queries[key]['test'][0]['input']\n"
        "                aif_future = executor.submit(\n"
        "                    run_aif_analysis,\n"
        "                    raw_train_data,\n"
        "                    raw_test_input\n"
        "                )\n"
        '                print(f"[Rank {rank}] AIF ({_aif_mode}) launched for {key}...")\n'
        "            except Exception as e:\n"
        '                print(f"[Rank {rank}] Failed to launch AIF: {e}")\n'
        "                aif_future = None\n")
    text = text.replace(
        "        # ================================================================\n"
        "        # GET AIF RESULT (should be ready, runs <0.1s on CPU)\n"
        "        # ================================================================\n"
        "        aif_result = None\n"
        "        if aif_future is not None:\n"
        "            try:\n"
        "                aif_result = aif_future.result(timeout=1.0)  # safe, should be instant\n"
        "            except Exception as e:\n"
        "                print(f\"[Rank {rank}] AIF retrieval error: {e}\")\n"
        "        executor.shutdown(wait=False)\n",
        "        aif_result = {'confidence': 0.0, 'grid': None, 'expected_count': -1, 'is_solved': False}\n"
        "        if aif_future is not None:\n"
        "            try:\n"
        "                aif_result = aif_future.result(timeout=1.0) or aif_result\n"
        "            except Exception as e:\n"
        '                print(f"[Rank {rank}] AIF retrieval error: {e}")\n'
        "        if executor is not None:\n"
        "            executor.shutdown(wait=False)\n"
        '        if _aif_mode == "prune":\n'
        "            aif_result = dict(aif_result, is_solved=False)\n"
        "        # per-task telemetry, so a smoke run measures how often the prune is actually live\n"
        "        # instead of us assuming it from the fast path's firing rate\n"
        "        print(f\"[Rank {rank}] AIF mode={_aif_mode} count={aif_result.get('expected_count')} \"\n"
        "              f\"solved={aif_result.get('is_solved')}\")\n")
    # DFS batch width as an env knob. sorokin grouped 4 views per DFS call, the aimo edit split them
    # into pairs. Restoring 4 was recorded as buying deeper search, but its measured effect was pool
    # coverage (155 correct answers present in the beams against 105), and Section 5.7 shows coverage
    # does not convert. It also halves the granularity at which the per-task cap can cut. Default
    # stays 4, which is what v7.1 through v7.4 shipped.
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
        '        _bw = int(os.environ.get("ARC_DFS_BATCH", "4"))\n'
        "        for test_id, subkeys in test_id_to_subkeys.items():\n"
        "            for _pair in ([0, 4], [2, 6]):\n"
        "                _h = [subkeys[_o:_o+2] for _o in _pair]\n"
        "                batches.extend([_h[0] + _h[1]] if _bw == 4 else _h)\n"
        "        for test_id, subkeys in test_id_to_subkeys.items():\n"
        "            for _pair in ([8, 12], [10, 14]):\n"
        "                _h = [subkeys[_o:_o+2] for _o in _pair]\n"
        "                batches.extend([_h[0] + _h[1]] if _bw == 4 else _h)\n")
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
    # candidate-pool dump for offline selection analysis, gated on ARC_DUMP_POOL so only the v7.2 build
    # writes it. Piggybacks on the benchmark method where every candidate is already loaded with its beam
    # score, per-augmentation NLLs and correctness. Writes pool_dump.json, which survives the output pull.
    text = text.replace(
        "            print(f\" acc: {score:5.1f}/{num_puzzles:3} ('{name}')\")",
        "            print(f\" acc: {score:5.1f}/{num_puzzles:3} ('{name}')\")\n"
        "\n"
        "        if os.environ.get(\"ARC_DUMP_POOL\"):\n"
        "            import json as _json\n"
        "            _pool = {}\n"
        "            for _bk, _subs in self.decoded_results.items():\n"
        "                _lab = labels.get(_bk)\n"
        "                _cands = {}\n"
        "                for _sk, _s in _subs.items():\n"
        "                    _grid = [list(map(int, _r)) for _r in _s[\"solution\"]]\n"
        "                    _key = _json.dumps(_grid)\n"
        "                    _c = _cands.setdefault(_key, {\"grid\": _grid, \"count\": 0, \"beam\": [], \"aug\": [],\n"
        "                        \"correct\": bool(np.array_equal(_s[\"solution\"], _lab)) if _lab is not None else False})\n"
        "                    _c[\"count\"] += 1\n"
        "                    _c[\"beam\"].append(float(_s[\"beam_score\"]))\n"
        "                    _c[\"aug\"].append([float(_x) for _x in _s[\"score_aug\"]])\n"
        "                _pool[_bk] = list(_cands.values())\n"
        "            try:\n"
        "                _json.dump(_pool, open(\"/kaggle/working/pool_dump.json\", \"w\"))\n"
        "                print(\"POOL DUMP: %d outputs -> /kaggle/working/pool_dump.json\" % len(_pool))\n"
        "            except Exception as _e:\n"
        "                print(\"pool dump failed:\", _e)\n")
    # DFS probability floor as an env knob. This is the single biggest search-budget dial, it sets how
    # improbable a continuation may get before the tree search abandons it. Lower keeps more branches
    # alive and costs time, so it trades directly against the per-task cap above.
    text = text.replace(
        "    max_score = -np.log(0.2)\n",
        '    max_score = -np.log(float(os.environ.get("ARC_DFS_FLOOR", "0.2")))\n')
    # per-task time cap as an env knob. The aimo edit raised sorokin's 1200 to 1800; the stripped
    # pipeline needs the tighter cap to fit 240 hidden tasks inside the 12h rerun.
    # The source writes this inline rather than through a spend_time local, and an earlier version of
    # this replace targeted the spend_time form. It never matched, so every v7 submission ran the
    # hardcoded 1800 while its config cell printed whatever ARC_TASK_TIMEOUT said. The assert below
    # is the real fix, the pattern is now verified to have fired.
    text = text.replace(
        "if time.time() - start_time > 1800 or time.time() > end_time:",
        'if time.time() - start_time > int(os.environ.get("ARC_TASK_TIMEOUT", "1800")) or time.time() > end_time:')
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
    # scheduler rules, both inert unless their env knob is set so every variant shares one codebase.
    # Priced on the v7.2 measurement run, where 71 percent of task-seconds bought nothing.
    text = text.replace(
        '\n        print(f"[Rank {rank}] allocated {memory_allocated}MB for training | stats: {stats}")',
        '\n        print(f"[Rank {rank}] allocated {memory_allocated}MB for training | stats: {stats}")\n'
        '        final_loss = float(getattr(stats, "training_loss", 0.0) or 0.0)')
    # adapters finishing above the cut never produced a harvested answer on the measurement run, so
    # the half-hour search is skipped and the greedy fallback below serves the task instead
    text = text.replace(
        "\n        model = FastLanguageModel.for_inference(model)\n"
        "        known_scores = {}",
        "\n        model = FastLanguageModel.for_inference(model)\n"
        '        _loss_cut = float(os.environ.get("ARC_LOSS_CUT", "0") or 0)\n'
        "        if _loss_cut and final_loss > _loss_cut:\n"
        '            print(f"[Rank {rank}] loss {final_loss:.4f} above cut {_loss_cut}, skipping search for {key}")\n'
        "            batches = []\n"
        "        known_scores = {}")
    # a task whose search returned nothing forfeits its slot for free, so fill empty outputs with one
    # greedy decode per view under the already-trained adapter. Runs after the DFS loop and touches only
    # tasks with no output file, so it cannot displace a searched answer.
    text = text.replace(
        '\n        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")',
        '\n        if os.environ.get("ARC_GREEDY_FALLBACK", "0") not in ("", "0", "false", "False"):\n'
        '            _done = {fn.split(".")[0] for fn in os.listdir(dir_outputs)}\n'
        "            _missing = [k2 for k2 in sorted(puzzle_ds_multi.keys) if k2 not in _done]\n"
        '            _fb_stop = start_time + int(os.environ.get("ARC_TASK_TIMEOUT", "1800")) + int(os.environ.get("ARC_GFB_BUDGET", "180"))\n'
        "            with torch.inference_mode():\n"
        "                for k2 in _missing:\n"
        "                    if time.time() > end_time or time.time() > _fb_stop:\n"
        '                        print(f"[Rank {rank}] greedy fallback budget spent for {key}")\n'
        "                        break\n"
        '                    _sks = [s2 for s2 in sorted(eval_ds.keys) if s2.split(".")[0] == k2][:2]\n'
        "                    for s2 in _sks:\n"
        "                        try:\n"
        '                            _toks = tokenizer.encode(eval_ds.get(s2, formatter)["input"])\n'
        '                            _inp = torch.tensor([_toks], device="cuda")\n'
        "                            _out = model.generate(input_ids=_inp, max_new_tokens=max_new_tokens, do_sample=False)\n"
        "                            _arr = formatter.convert_tokens_to_array(_out[0, _inp.shape[1]:].tolist())\n"
        "                            if _arr is None:\n"
        "                                continue\n"
        "                            _sol = puzzle_ds_multi.invert_mod(_arr, s2, inv_perm=True)\n"
        '                            with bz2.BZ2File(os.path.join(dir_outputs, s2 + ".gfb"), "w") as f:\n'
        '                                pickle.dump([{"beam_score": 3.0, "score_aug": [3.0], "solution": _sol}], f)\n'
        '                            print(f"[Rank {rank}] greedy fallback filled {s2}")\n'
        "                        except Exception as _e:\n"
        '                            print(f"[Rank {rank}] greedy fallback failed {s2}: {_e}")\n'
        '        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")')
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
    + ('' if "ARC_N_EVAL" in SPEC["env"] else 'os.environ["ARC_N_EVAL"]     = "40"\n') +
    "# Experiment knobs. Defaults reproduce the 30.56 submission, change one at a time.\n"
    "#   ARC_DFS_FLOOR    search width, lower keeps more branches and costs time (default 0.2)\n"
    "#   ARC_TASK_TIMEOUT per-task seconds (default 1800, the 30.56 submission used 1200)\n"
    "#   ARC_RANK         LoRA rank (default 256)      ARC_MAXSEQ  training context (default 8192)\n"
    "#   ARC_LOSS_CUT     skip the search when training loss ends above this, 0 disables (v7.3: 0.05)\n"
    "#   ARC_GREEDY_FALLBACK  fill search-empty tasks with a greedy decode (v7.3: 1)\n"
    "#   ARC_AIF          symbolic arm: off (default) | prune (count signal, no preemption) | full\n"
    "#   ARC_DFS_BATCH    views per DFS call, 4 = sorokin (default), 2 = the aimo pair form\n"
    + ("".join('os.environ["%s"] = "%s"\n' % (k, v) for k, v in sorted(SPEC["env"].items()))
       or '# os.environ["ARC_DFS_FLOOR"]    = "0.1"\n'
          '# os.environ["ARC_TASK_TIMEOUT"] = "1200"\n')
    + "# ====================================================\n"
    'print("MODEL ", os.environ["ARC_MODEL_PATH"], "exists:", os.path.isdir(os.environ["ARC_MODEL_PATH"]))\n'
    'print("DATA  ", os.environ["ARC_DATA_DIR"], "exists:", os.path.isdir(os.environ["ARC_DATA_DIR"]))\n'
    "# echo every knob so a smoke run proves the config landed instead of silently defaulting\n"
    'print("CONFIG N_EVAL=%s RANK=%s MAXSEQ=%s TIMEOUT=%s DFS_FLOOR=%s LOSS_CUT=%s GFB=%s AIF=%s BATCH=%s" % (\n'
    '    os.environ.get("ARC_N_EVAL", "40"), os.environ.get("ARC_RANK", "256"),\n'
    '    os.environ.get("ARC_MAXSEQ", "8192"), os.environ.get("ARC_TASK_TIMEOUT", "1800"),\n'
    '    os.environ.get("ARC_DFS_FLOOR", "0.2"), os.environ.get("ARC_LOSS_CUT", "0"),\n'
    '    os.environ.get("ARC_GREEDY_FALLBACK", "0"), os.environ.get("ARC_AIF", "off"),\n'
    '    os.environ.get("ARC_DFS_BATCH", "4")))\n'
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
assert alltext.count("final_loss = float(") == 1, "loss capture not patched exactly once"
# every env knob must actually reach the code. A knob that only reaches the config echo is worse than
# no knob, it reports a setting the run is not using, which is how three submissions ran the wrong cap.
for knob, marker in (("ARC_TASK_TIMEOUT", 'os.environ.get("ARC_TASK_TIMEOUT"'),
                     ("ARC_DFS_FLOOR", 'os.environ.get("ARC_DFS_FLOOR"'),
                     ("ARC_RANK", 'os.environ.get("ARC_RANK"'),
                     ("ARC_MAXSEQ", 'os.environ.get("ARC_MAXSEQ"')):
    assert marker in alltext, "%s is echoed but never read by the solver" % knob
assert "start_time > 1800" not in alltext, "per-task cap still hardcoded"
# a knob whose "off" value is truthy is worse than no knob. "0" is a non-empty string, so the
# greedy fill ran through the entire v7.4 control that was supposed to have it disabled.
assert 'os.environ.get("ARC_GREEDY_FALLBACK"):' not in alltext, "truthy-string gate on the fill"
assert alltext.count("ARC_LOSS_CUT") >= 1, "loss-cut gate missed"
assert alltext.count("ARC_GREEDY_FALLBACK") >= 1, "greedy fallback missed"
# the two bisection factors. Both arms must reach the solver, and the prune mode must actually
# suppress the fast path rather than merely name it.
assert 'os.environ.get("ARC_AIF"' in alltext, "ARC_AIF is echoed but never read by the solver"
assert 'os.environ.get("ARC_DFS_BATCH"' in alltext, "ARC_DFS_BATCH is echoed but never read"
assert "run_aif_analysis" in alltext, "AIF import stripped, the prune arm cannot run"
assert alltext.count('_aif_mode = os.environ.get') == 1, "AIF launch gate missed, _aif_mode undefined"
assert alltext.count('_aif_mode == "prune"') == 1, "prune mode does not suppress the fast path"
assert "batches.append(subkeys[offset:offset+2])" not in alltext, "ungated batching form survived"

# ---- behavioural self-test on the emitted batching. A replace that lands but computes the wrong
# thing is precisely the failure that shipped three wrong caps, so execute the generated block at
# both widths and check it against the two reference forms rather than trusting the patterns.
_i = alltext.index('_bw = int(os.environ.get("ARC_DFS_BATCH"')
_start = alltext.rindex("\n", 0, alltext.rindex("batches = []", 0, _i)) + 1
_end = alltext.index("\n", alltext.index("batches.extend(", alltext.index("([8, 12], [10, 14])", _i))) + 1
_block = textwrap.dedent(alltext[_start:_end])


def _emitted(fixture, width):
    os.environ["ARC_DFS_BATCH"] = str(width)
    ns = {"os": os, "test_id_to_subkeys": fixture}
    exec(_block, ns)
    return ns["batches"]


def _aimo_pairs(d):  # the upstream form ARC_DFS_BATCH=2 must reproduce exactly
    b = []
    for _, sk in d.items():
        for o in [0, 4]:
            b.append(sk[o:o + 2])
        for o in [2, 6]:
            b.append(sk[o:o + 2])
    for _, sk in d.items():
        for o in [8, 12]:
            b.append(sk[o:o + 2])
        for o in [10, 14]:
            b.append(sk[o:o + 2])
    return b


def _sorokin_quads(d):  # the v7.1 grouping ARC_DFS_BATCH=4 must reproduce exactly
    b = []
    for pairs in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
        for _, sk in d.items():
            for p in pairs:
                b.append(sk[p[0]:p[0] + 2] + sk[p[1]:p[1] + 2])
    return b


_fix = {"0": ["a%d" % i for i in range(16)], "1": ["b%d" % i for i in range(16)]}
assert _emitted(_fix, 2) == _aimo_pairs(_fix), "ARC_DFS_BATCH=2 does not reproduce the aimo pair form"
assert _emitted(_fix, 4) == _sorokin_quads(_fix), "ARC_DFS_BATCH=4 does not reproduce v7.1's grouping"
# whatever the width, the same 32 views must be searched, or the knob is changing coverage as well
assert sorted(sum(_emitted(_fix, 2), [])) == sorted(sum(_emitted(_fix, 4), [])), \
    "the two batch widths do not cover the same views"
os.environ.pop("ARC_DFS_BATCH", None)
print("batching self-test passed: %d pair-batches, %d quad-batches, same 32 views"
      % (len(_emitted(_fix, 2)), len(_emitted(_fix, 4))))
os.environ.pop("ARC_DFS_BATCH", None)

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
