# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import warnings
import yaml
from datetime import timedelta

# The model's HF processor calls librosa.load() on video containers. soundfile/libsndfile
# can't open those, so librosa falls back to audioread (ffmpeg). That fallback works, but
# emits two benign warnings per sample; silence them here (the HF module can't be edited).
warnings.filterwarnings("ignore", message="PySoundFile failed", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"librosa.*")

import torch
import torch.distributed as dist

from avlm.inference.common.models import get_model
from avlm.inference.common.utils.json_io import load_jsonl, write_json, write_jsonl

def to_record(row, pred, idx, ground_truth, conversations, reasoning_trace, video_rel=None):
    return {
        "index": idx,
        "id": row.get("id"),
        "class": row.get("class"),
        "super_category": row.get("super_category"),
        "fine_category": row.get("fine_category"),
        "evaluation_type": row.get("evaluation_type"),
        "video-sound": video_rel or row.get("video-sound") or row.get("video"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "ground_truth": ground_truth,
        "prediction": pred,
        "inference_status": "success",
        # 'conversations': row.get("conversations"),
        'conversations': conversations,
        "reasoning_trace": reasoning_trace,
        "metadata": row.get("metadata"),
    }

def split_thinking(response):
    reasoning_trace = response
    last_think_end = response.rfind("</think>")
    if last_think_end != -1:
        prediction = response[last_think_end + len("</think>"):].lstrip("\n").strip()
    else:
        prediction = response
    return prediction, reasoning_trace


def _hf_content_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    )


def _init_dist():
    if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=os.environ.get('DIST_TIMOUT_HOURS', 4)))

def _rank_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

def _parse_qa(row):
    if "conversation" in row:
        question, answer = "", ""
        for turn in row.get("conversation", []):
            role = turn.get("role")
            if role == "user":
                question = _hf_content_text(turn.get("content", ""))
            elif role == "assistant":
                answer = _hf_content_text(turn.get("content", ""))
        return question.strip(), answer.strip()

    question, answer = "", ""
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            question = turn.get("value", "")
        elif turn.get("from") == "gpt":
            answer = turn.get("value", "")
    question = question.replace("<video-sound>", "").strip()
    return question, answer

def _video_rel(row):
    rel = row.get("video-sound") or row.get("video")
    if rel:
        return rel

    for turn in row.get("conversation", []):
        content = turn.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"video", "video-sound", "video_sound"}:
                return part.get("path") or part.get("video") or part.get("video-sound")

    return None

def _resolve_video_path(row, video_root):
    rel = _video_rel(row)
    if not rel:
        raise ValueError(f"no video path in row id={row.get('id')}")
    if os.path.isabs(rel):
        return rel
    return os.path.join(video_root, rel)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_bool(value):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inference-name", default=None)
    parser.add_argument("--base-output-dir", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--is-fsdp", type=parse_bool, default=None)
    parser.add_argument(
        "--fsdp-sharding-strategy",
        choices=("optim_grads_params", "no_shard"),
        default=None,
    )
    parser.add_argument("--max-inference-samples", type=int, default=None)
    parser.add_argument("--max-inference-samples-seed", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    cfg = load_config(args.config)
    max_inference_samples = args.max_inference_samples
    max_inference_samples_seed = args.max_inference_samples_seed

    if args.inference_name is not None:
        cfg["inference_name"] = args.inference_name

    if args.base_output_dir is not None:
        cfg["base_output_dir"] = args.base_output_dir

    if args.data_path is not None:
        cfg["data_path"] = args.data_path

    if args.model_path is not None:
        cfg["model_path"] = args.model_path

    if args.adapter_path is not None:
        cfg["adapter_path"] = args.adapter_path

    if args.is_fsdp is not None:
        cfg["is_fsdp"] = args.is_fsdp

    if args.fsdp_sharding_strategy is not None:
        cfg["fsdp_sharding_strategy"] = args.fsdp_sharding_strategy

    _init_dist()
    rank, world_size = _rank_world_size()

    base_output_dir = cfg['base_output_dir']
    inference_name = cfg['inference_name']
    output_dir = os.path.join(base_output_dir, inference_name)

    os.makedirs(output_dir, exist_ok=True)
    rank_results_dir = os.path.join(output_dir, "rank_results")
    os.makedirs(rank_results_dir, exist_ok=True)

    rows = load_jsonl(cfg['data_path'])
    if (max_inference_samples is not None and
        max_inference_samples > 0 and 
        max_inference_samples < len(rows)):
        if max_inference_samples_seed >= 0:
            rng = random.Random(max_inference_samples_seed)
            indices = sorted(rng.sample(range(len(rows)), max_inference_samples))
            rows = [rows[i] for i in indices]
        else:
            rows = rows[:max_inference_samples]

    model = get_model(**cfg)
    distribution = model.inference_distribution(rank, world_size)
    data_rank = distribution.data_rank
    data_world_size = distribution.data_world_size
    writes_results = distribution.writes_results
    requires_lockstep = distribution.requires_lockstep

    state_path = os.path.join(rank_results_dir, "inference_state.json")
    state = {
        "world_size": world_size,
        "data_path": cfg["data_path"],
    }
    if data_world_size != world_size:
        state["data_parallel_world_size"] = data_world_size

    if rank == 0:
        if args.resume and os.path.exists(state_path):
            with open(state_path) as f:
                old_state = json.load(f)
            if old_state != state:
                raise ValueError(f"Resume state mismatch: {old_state} != {state}")
        else:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

    if world_size > 1:
        dist.barrier()

    shard = [row for i, row in enumerate(rows) if i % data_world_size == data_rank]

    rank_path = os.path.join(rank_results_dir, f"rank_{data_rank}.json")
    rank_records = {}

    if args.resume and os.path.exists(rank_path):
        with open(rank_path) as f:
            rank_records = {int(k): v for k, v in json.load(f).items()}

    if writes_results and (not args.resume or not os.path.exists(rank_path)):
        with open(rank_path, "w") as f:
            json.dump({str(k): v for k, v in rank_records.items()}, f)

    work_items = []
    if requires_lockstep and rows:
        # EP/ETP collectives require every lane to execute the same number of
        # generation rounds. Missing or resumed items use rows[0] as a filler.
        num_rounds = (len(rows) + data_world_size - 1) // data_world_size
        for i in range(num_rounds):
            global_idx = data_rank + i * data_world_size
            is_real = global_idx < len(rows) and not (args.resume and global_idx in rank_records)
            row = rows[global_idx] if is_real else rows[0]
            work_items.append((i, row, global_idx, is_real))
    else:
        for i, row in enumerate(shard):
            global_idx = data_rank + i * data_world_size
            if args.resume and global_idx in rank_records:
                continue
            work_items.append((i, row, global_idx, True))

    for i, row, global_idx, is_real in work_items:
        if requires_lockstep:
            round_has_work = torch.tensor(int(is_real), dtype=torch.int32, device=model.device)
            dist.all_reduce(round_has_work, op=dist.ReduceOp.MAX)
            if int(round_has_work.item()) == 0:
                continue

        video_rel = _video_rel(row)
        video_path = _resolve_video_path(row, cfg['video_root'])
        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)

        question, ground_truth = _parse_qa(row)

        pred, conversations = model.predict(
            question,
            video_path,
            **cfg,
        )

        pred, reasoning_trace = split_thinking(pred)
        
        if is_real and writes_results:
            rec = to_record(row, pred, global_idx, ground_truth, conversations, reasoning_trace, video_rel=video_rel)
            rank_records[global_idx] = rec

            tmp_path = rank_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump({str(k): v for k, v in rank_records.items()}, f)
            os.replace(tmp_path, rank_path)

            if rank == 0 or world_size == 1:
                print(f"[{global_idx + 1}/{len(rows)}] {rec['id']}: {pred[:80]}...")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged = {}
        for r in range(data_world_size):
            p = os.path.join(rank_results_dir, f"rank_{r}.json")
            with open(p) as f:
                merged.update({int(k): v for k, v in json.load(f).items()})

        missing = [i for i in range(len(rows)) if i not in merged]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} predictions. First missing: {missing[:10]}")

        records = [merged[i] for i in range(len(rows))]
        write_jsonl(os.path.join(output_dir, "predictions.jsonl"), records)
        write_json(os.path.join(output_dir, "predictions.json"), records)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
