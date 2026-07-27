%%writefile arc_solver.py
from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict
from typing import Any, Union
from transformers import DataCollatorForLanguageModeling
import logging
from contextlib import redirect_stdout, redirect_stderr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import bz2
import pickle
from scipy.ndimage import label  # <-- NEW for pruning

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "Ċ": 10, "<|im_end|>": 15,
}
ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:
    n = logits.size(0)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)
    suffixes = defaultdict(list)
    candidates = dict()
    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))
    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0])
    
    while time.time() - start_time < 540 and time.time() < end_time:
        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0
        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1
        if num_alive_beams == 0:
            break
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )
        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))
    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


def worker(rank, queue, end_time):
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=int(os.environ.get("ARC_RANK", "256")),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = int(os.environ.get("ARC_MAXSEQ", "8192"))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.environ.get("ARC_MODEL_PATH", "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"),
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()
    max_score = -np.log(float(os.environ.get("ARC_DFS_FLOOR", "0.2")))

    if rerun_mode:
        test_path = os.path.join(os.environ.get("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"), "arc-agi_test_challenges.json")
    else:
        test_path = os.path.join(os.environ.get("ARC_DATA_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"), "arc-agi_evaluation_challenges.json")

    arc_test_set = ArcDataset.from_file(test_path)
    dir_outputs = os.environ.get("ARC_OUT_DIR", "/kaggle/inference_outputs")
    os.makedirs(dir_outputs, exist_ok=True)

    while not queue.empty():
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        load_result = set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])
        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        # AIF arm removed. It fast-pathed 12 of 39 tasks with zero correct on the eval slice,
        # so the arm is not launched at all and this stub stands in for its result.

        # ================================================================
        # TRAINING PHASE (GPU busy for ~3.5 minutes)
        # ================================================================
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args),
            )
            stats = trainer.train()
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            del trainer

        gc.collect()
        torch.cuda.empty_cache()

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training | stats: {stats}")
        final_loss = float(getattr(stats, "training_loss", 0.0) or 0.0)

        aif_result = {'confidence': 0.0, 'grid': None, 'expected_count': -1, 'is_solved': False}

        # ================================================================
        # EARLY EXIT: IF AIF SOLVED PERFECTLY AND ONLY ONE TEST CASE
        # ================================================================
        test_cases = puzzle_ds.queries[key]['test']
        if len(test_cases) == 1 and aif_result and aif_result.get('is_solved', False) and aif_result.get('grid') is not None:
            solution_grid = np.array(aif_result['grid'])
            print(f"[Rank {rank}] ⭐ AIF PERFECTLY SOLVED {key}! Skipping 15-min inference.")
            dummy_result = [{
                "beam_score": 0.0,
                "score_aug": [0.0, 0.0],
                "solution": solution_grid
            }]
            # Correct base key: {key}_0 (matches split_multi_replies)
            with bz2.BZ2File(os.path.join(dir_outputs, f"{key}_0.out0"), "w") as f:
                pickle.dump(dummy_result, f)
            print(f"[Rank {rank}] finished {key} in {time.time() - start_time:.1f}s (AIF fast-path)")
            continue

        # ================================================================
        # FALLBACK: LLM INFERENCE WITH AIF STRUCTURAL PRUNING
        # ================================================================
        print(f"[Rank {rank}] AIF confidence {aif_result.get('confidence',0):.2f}. Running LLM inference...")

        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
        for test_id, subkeys in test_id_to_subkeys.items():
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)

        model = FastLanguageModel.for_inference(model)
        _loss_cut = float(os.environ.get("ARC_LOSS_CUT", "0") or 0)
        if _loss_cut and final_loss > _loss_cut:
            print(f"[Rank {rank}] loss {final_loss:.4f} above cut {_loss_cut}, skipping search for {key}")
            batches = []
        known_scores = {}
        expected_count = aif_result.get('expected_count', -1) if aif_result else -1

        with torch.inference_mode():
            for subkeys in batches:
                if time.time() - start_time > int(os.environ.get("ARC_TASK_TIMEOUT", "1800")) or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {time.time()-start_time:.1f}s for {key}")
                    break

                print(f"[Rank {rank}] decoding {subkeys}")
                tokens = [tokenizer.encode(eval_ds.get(sk, formatter)["input"]) for sk in subkeys]
                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)

                for subkey_id, scored_beams in dfs_result:
                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []

                    for beam_score, token_seq in scored_beams:
                        array = formatter.convert_tokens_to_array(token_seq)
                        if array is None:
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                        # ============================================================
                        # AIF PRUNING: Penalize if generated object count mismatches
                        # ============================================================
                        if expected_count > 0:
                            bin_sol = (solution > 0).astype(np.int32)
                            _, gen_count = label(bin_sol)
                            if abs(gen_count - expected_count) > 0:
                                beam_score += abs(gen_count - expected_count) * 0.5

                        # ============================================================
                        # ORIGINAL SCORING LOGIC (unchanged)
                        # ============================================================
                        grid_id = (bk, tuple(map(tuple, solution)))
                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
                            aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length - max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores

                        decoded_result.append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "solution": solution,
                        })

                    if len(decoded_result):
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        if os.environ.get("ARC_GREEDY_FALLBACK"):
            _done = {fn.split(".")[0] for fn in os.listdir(dir_outputs)}
            _missing = [k2 for k2 in sorted(puzzle_ds_multi.keys) if k2 not in _done]
            with torch.inference_mode():
                for k2 in _missing:
                    if time.time() > end_time:
                        break
                    _sks = [s2 for s2 in sorted(eval_ds.keys) if s2.split(".")[0] == k2][:2]
                    for s2 in _sks:
                        try:
                            _toks = tokenizer.encode(eval_ds.get(s2, formatter)["input"])
                            _inp = torch.tensor([_toks], device="cuda")
                            _out = model.generate(input_ids=_inp, max_new_tokens=max_new_tokens, do_sample=False)
                            _arr = formatter.convert_tokens_to_array(_out[0, _inp.shape[1]:].tolist())
                            if _arr is None:
                                continue
                            _sol = puzzle_ds_multi.invert_mod(_arr, s2, inv_perm=True)
                            with bz2.BZ2File(os.path.join(dir_outputs, s2 + ".gfb"), "w") as f:
                                pickle.dump([{"beam_score": 3.0, "score_aug": [3.0], "solution": _sol}], f)
                            print(f"[Rank {rank}] greedy fallback filled {s2}")
                        except Exception as _e:
                            print(f"[Rank {rank}] greedy fallback failed {s2}: {_e}")
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        print(f"[Rank {rank}] finished {key} in {time.time() - start_time:.1f}s")