# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional training-time dump of decoded processed prompt text (collate path).

Enable with ``AVLM_LOG_PROCESSED_PROMPT=1`` and optional ``AVLM_LOG_PROCESSED_PROMPT_TAG``
(default ``run``) for the output filename. Logs land under ``${OUTPUT}/processed_prompt_logs/``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("avlm.processed_prompt_log")

_TRUTHY = frozenset({"1", "true", "True", "yes", "on"})
_rank_logged_count = 0
_seen_dedup_keys: set[tuple[str, int, int]] = set()
_active_train_iteration: int | None = None
_in_eval = False
_training_hooks_installed = False


def processed_prompt_log_enabled() -> bool:
    return os.environ.get("AVLM_LOG_PROCESSED_PROMPT", "").strip() in _TRUTHY


def _log_tag() -> str:
    return os.environ.get("AVLM_LOG_PROCESSED_PROMPT_TAG", "run").strip() or "run"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _rank_info() -> tuple[int, int, int]:
    """Return (rank, local_rank, world_size); fall back to single-process defaults."""

    def _int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    rank = _int("RANK", _int("LOCAL_RANK", 0))
    local_rank = _int("LOCAL_RANK", rank)
    world_size = _int("WORLD_SIZE", 1)
    return rank, local_rank, world_size


def _parallel_layout_from_env() -> dict[str, Any]:
    """Megatron rank layout: global_rank = tp_rank + dp_rank * tp_size."""
    rank, _, world_size = _rank_info()
    tp_size = _int_env("TP", 1)
    ep_size = _int_env("EP", 1)
    model_parallel = tp_size
    dp_size = max(1, world_size // model_parallel) if world_size >= model_parallel else 1
    dp_rank = rank // model_parallel if model_parallel else 0
    tp_rank = rank % tp_size if tp_size else 0
    processing_global_ranks = [dp_rank * model_parallel + t for t in range(tp_size)]
    return {
        "data_parallel_rank": dp_rank,
        "data_parallel_size": dp_size,
        "tensor_model_parallel_rank": tp_rank,
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_rank": 0,
        "pipeline_model_parallel_size": 1,
        "context_parallel_rank": 0,
        "context_parallel_size": 1,
        "expert_parallel_size": ep_size,
        "processing_global_ranks": processing_global_ranks,
        "collate_global_rank": rank,
    }


def _parallel_layout() -> dict[str, Any]:
    rank, _, world_size = _rank_info()
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized() and dist.is_initialized():
            tp_size = mpu.get_tensor_model_parallel_world_size()
            dp_size = mpu.get_data_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            dp_rank = mpu.get_data_parallel_rank()
            ep_size = mpu.get_expert_model_parallel_world_size()
            processing_global_ranks = sorted(
                dist.get_process_group_ranks(mpu.get_tensor_model_parallel_group())
            )
            return {
                "data_parallel_rank": dp_rank,
                "data_parallel_size": dp_size,
                "tensor_model_parallel_rank": tp_rank,
                "tensor_model_parallel_size": tp_size,
                "pipeline_model_parallel_rank": 0,
                "pipeline_model_parallel_size": 1,
                "context_parallel_rank": 0,
                "context_parallel_size": 1,
                "expert_parallel_size": ep_size,
                "processing_global_ranks": processing_global_ranks,
                "collate_global_rank": rank,
            }
    except Exception:
        pass
    return _parallel_layout_from_env()


def _log_dir() -> Path:
    output = os.environ.get("OUTPUT", "").strip()
    if output:
        return Path(output) / "processed_prompt_logs"
    output_base = os.environ.get("OUTPUT_BASE", "").strip()
    model_name = os.environ.get("MODEL_NAME", "megatron_bridge_sft").strip() or "megatron_bridge_sft"
    if output_base:
        return Path(output_base) / model_name / "processed_prompt_logs"
    workspace = os.environ.get("WORKSPACE", os.environ.get("CACHE_DIR", ".")).strip() or "."
    return (
        Path(workspace)
        / "avlm/training/megatron-bridge/sft/slurm/outputs"
        / model_name
        / "processed_prompt_logs"
    )


def _jsonl_path() -> Path:
    return _log_dir() / f"processed_prompts_{_log_tag()}.jsonl"


def _meta_path() -> Path:
    return _log_dir() / f"processed_prompts_{_log_tag()}.meta.json"


def _should_log_on_this_rank() -> bool:
    layout = _parallel_layout()
    return layout["tensor_model_parallel_rank"] == 0


def _input_ids_fingerprint(input_ids: list[int]) -> str:
    digest = hashlib.sha256(",".join(map(str, input_ids)).encode()).hexdigest()
    return digest[:16]


def _should_skip_dedup(input_ids: list[int], dp_rank: int, train_iteration: int) -> bool:
    key = (_input_ids_fingerprint(input_ids), int(dp_rank), int(train_iteration))
    if key in _seen_dedup_keys:
        return True
    _seen_dedup_keys.add(key)
    return False


def _append_jsonl_locked(record: dict[str, Any]) -> None:
    """Append one JSONL row under an exclusive lock."""
    path = _jsonl_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            fh.write(line)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _write_run_meta_once() -> None:
    rank, _, _ = _rank_info()
    if rank != 0:
        return
    meta_path = _meta_path()
    jsonl_path = _jsonl_path()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    _, _, world_size = _rank_info()
    layout = _parallel_layout()
    with meta_path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            if fh.read(1):
                return
            fh.seek(0)
            fh.truncate()
            if jsonl_path.is_file():
                jsonl_path.write_text("", encoding="utf-8")
            payload = {
                "tag": _log_tag(),
                "jsonl_path": str(jsonl_path),
                "jsonl_fields": [
                    "train_iteration",
                    "data_parallel_rank",
                    "processing_global_ranks",
                    "sample_key",
                    "encoder",
                    "seq_len",
                    "supervised_tokens",
                    "image_token_count",
                    "sound_token_count",
                    "processed_prompt_text",
                ],
                "world_size": world_size,
                "data_parallel_size": layout["data_parallel_size"],
                "tensor_model_parallel_size": layout["tensor_model_parallel_size"],
                "hf_model_id": os.environ.get("HF_MODEL_ID"),
                "model_name": os.environ.get("MODEL_NAME"),
                "started_at_unix": time.time(),
            }
            fh.write(json.dumps(payload, indent=2) + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def install_processed_prompt_training_hooks() -> None:
    """Patch Bridge train/eval so collate logs only during real training steps."""
    global _training_hooks_installed
    if _training_hooks_installed or not processed_prompt_log_enabled():
        return

    import megatron.bridge.training.eval as eval_mod
    import megatron.bridge.training.train as train_mod

    _orig_train_step = train_mod.train_step
    _orig_evaluate = eval_mod.evaluate

    def train_step(*args, **kwargs):
        global _active_train_iteration
        global_state = kwargs.get("global_state")
        if global_state is None and len(args) > 5:
            global_state = args[5]
        iteration = int(global_state.train_state.step) + 1 if global_state is not None else None
        _active_train_iteration = iteration
        try:
            return _orig_train_step(*args, **kwargs)
        finally:
            _active_train_iteration = None

    def evaluate(*args, **kwargs):
        global _in_eval
        _in_eval = True
        try:
            return _orig_evaluate(*args, **kwargs)
        finally:
            _in_eval = False

    train_mod.train_step = train_step
    eval_mod.evaluate = evaluate
    _training_hooks_installed = True
    logger.info("processed prompt training hooks installed (train-step gated logging)")


def log_encoded_sample(
    encoded_sample: Any,
    tokenizer: Any,
    *,
    sample_key: str,
    encoder_name: str,
    split: str = "train",
) -> None:
    """Decode ``input_ids`` and append one JSONL row when logging is enabled."""
    global _rank_logged_count

    if not processed_prompt_log_enabled():
        return
    if split != "train" or _in_eval or _active_train_iteration is None:
        return

    layout = _parallel_layout()
    if not _should_log_on_this_rank():
        return

    train_iteration = _active_train_iteration
    if train_iteration is None:
        return

    input_ids = encoded_sample.input_ids.reshape(-1).tolist()
    if _should_skip_dedup(input_ids, layout["data_parallel_rank"], train_iteration):
        return

    img_tok = tokenizer.convert_tokens_to_ids("<image>")
    sound_tok = tokenizer.convert_tokens_to_ids("<so_embedding>")
    loss_mask = getattr(encoded_sample, "loss_mask", None)
    supervised = int(loss_mask.sum().item()) if loss_mask is not None else None

    processed_prompt_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    _write_run_meta_once()
    record: dict[str, Any] = {
        "train_iteration": train_iteration,
        "data_parallel_rank": layout["data_parallel_rank"],
        "processing_global_ranks": layout["processing_global_ranks"],
        "sample_key": sample_key,
        "encoder": encoder_name,
        "seq_len": len(input_ids),
        "supervised_tokens": supervised,
        "image_token_count": input_ids.count(img_tok),
        "sound_token_count": input_ids.count(sound_tok),
    }
    record["processed_prompt_text"] = processed_prompt_text
    _append_jsonl_locked(record)
    _rank_logged_count += 1

    if _rank_logged_count == 1 and layout["tensor_model_parallel_rank"] == 0:
        logger.info(
            "processed prompt logging tag=%s dir=%s",
            _log_tag(),
            _log_dir(),
        )
    if _rank_logged_count <= 8:
        logger.info(
            "logged processed prompt iter=%d dp=%d processing_ranks=%s key=%s seq_len=%d",
            train_iteration,
            layout["data_parallel_rank"],
            layout["processing_global_ranks"],
            sample_key,
            record["seq_len"],
        )
