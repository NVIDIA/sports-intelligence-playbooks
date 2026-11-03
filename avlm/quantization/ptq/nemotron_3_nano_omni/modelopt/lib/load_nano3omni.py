# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Nemotron-3-Nano-Omni HF checkpoint for PTQ (single GPU)."""

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


def _local_checkpoint_repo_slug(model_path: Path) -> str:
    """Transformers cache folder name for a local checkpoint directory."""
    return model_path.name.lower().replace("-", "_hyphen_")


def _hf_modules_cache_root() -> Path:
    import os

    return Path(
        os.environ.get(
            "HF_MODULES_CACHE",
            os.path.join(os.environ.get("HF_HOME", ""), "modules"),
        )
    )


def _copy_checkpoint_py_files(target_dir: Path, py_files: list[Path]) -> int:
    import shutil

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in py_files:
        shutil.copy2(src, target_dir / src.name)
        copied += 1
    return copied


def _sync_trust_remote_code_from_checkpoint(model_path: Path) -> int:
    """Copy all top-level remote-code .py files into every HF modules hash dir."""
    from transformers.utils import TRANSFORMERS_DYNAMIC_MODULE_NAME

    model_path = model_path.resolve()
    if not model_path.is_dir():
        return 0

    slug_root = (
        _hf_modules_cache_root()
        / TRANSFORMERS_DYNAMIC_MODULE_NAME
        / _local_checkpoint_repo_slug(model_path)
    )
    if not slug_root.is_dir():
        return 0

    py_files = sorted(model_path.glob("*.py"))
    if not py_files:
        return 0

    copied = 0
    for hash_dir in slug_root.iterdir():
        if not hash_dir.is_dir() or hash_dir.name in {"__pycache__"}:
            continue
        copied += _copy_checkpoint_py_files(hash_dir, py_files)
    return copied


def _prime_model_remote_code_cache(model_path: Path) -> None:
    """Ensure the modeling.py revision dir exists and contains the full code tree.

    Processor/config auto_map use different content hashes than AutoModelForCausalLM.
    Syncing only after processor load misses the modeling hash (e.g. c52f8891…).
    """
    import os
    import shutil

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_cached_module_file
    from transformers.utils import TRANSFORMERS_DYNAMIC_MODULE_NAME

    model_path = model_path.resolve()
    py_files = sorted(model_path.glob("*.py"))
    if not py_files:
        return

    AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    cache_dir = str(_hf_modules_cache_root() / TRANSFORMERS_DYNAMIC_MODULE_NAME)
    model_str = str(model_path)
    copied = 0

    for module_file in ("configuration.py", "modeling.py"):
        try:
            cached = get_cached_module_file(
                model_str,
                module_file,
                cache_dir=cache_dir,
                local_files_only=os.path.isdir(model_str),
            )
        except Exception as exc:
            logger.warning("Could not cache %s for remote code: %s", module_file, exc)
            continue
        copied += _copy_checkpoint_py_files(Path(cached).parent, py_files)

    copied += _sync_trust_remote_code_from_checkpoint(model_path)
    if copied:
        logger.info(
            "Synced %d remote-code file(s) from %s into HF modules cache",
            copied,
            model_path,
        )


def load_nano3omni(model_path: Path | str, *, video_pruning_rate: float = 0.0):
    model_path = Path(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    _prime_model_remote_code_cache(model_path)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    prev_pruning = float(getattr(config, "video_pruning_rate", 0.0) or 0.0)
    config.video_pruning_rate = float(video_pruning_rate)
    if prev_pruning != config.video_pruning_rate:
        logger.info(
            "video_pruning_rate=%s in checkpoint → %s (EVS in generate when > 0)",
            prev_pruning,
            config.video_pruning_rate,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        config=config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            indexed_keys = set(json.load(f)["weight_map"])
        missing_state_dict = {}
        for shard in sorted(model_path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in indexed_keys:
                        missing_state_dict[key] = f.get_tensor(key)
        if missing_state_dict:
            logger.info("Loading %d weights missing from index (e.g. sound_encoder)", len(missing_state_dict))
            model.load_state_dict(missing_state_dict, strict=False)

    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<image>")
    model.sound_context_token_id = tokenizer.convert_tokens_to_ids("<so_embedding>")
    model.video_pruning_rate = float(video_pruning_rate)

    return model, processor, tokenizer
