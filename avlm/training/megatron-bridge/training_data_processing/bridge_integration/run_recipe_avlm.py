#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""AVLM training entrypoint: extends Bridge run_recipe without editing the cache checkout."""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

_BRIDGE_ROOT = Path(os.environ["MEGATRON_BRIDGE_ROOT"]).resolve()
_overlay = os.environ.get("MB_BRIDGE_OVERLAY", "").strip()
_OVERLAY_ROOT = Path(_overlay).resolve() if _overlay else None


def _quiet_noisy_runtime_logs() -> None:
    if os.environ.get("AVLM_QUIET_RUNTIME_LOGS", "1") == "0":
        return

    logging.getLogger("httpx").setLevel(os.environ.get("HTTPX_LOG_LEVEL", "WARNING"))
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("vllm").setLevel(os.environ.get("VLLM_LOGGING_LEVEL", "CRITICAL"))

    warning_filters = (
        r"The given NumPy array is not writable.*",
        r"The AccumulateGrad node.*stream does not match.*",
        r"barrier.*using the device under current context.*",
        r"MimoModelConfig is experimental.*",
        r"Field .* duplicates an ancestor field.*",
    )
    for message in warning_filters:
        warnings.filterwarnings("ignore", message=message, category=UserWarning)


def _bootstrap_import_paths() -> None:
    # Bridge src first so megatron.bridge keeps AutoBridge; overlay dirs are injected below.
    _integration_dir = Path(__file__).resolve().parent
    for path in (_integration_dir, _BRIDGE_ROOT / "src", _BRIDGE_ROOT / "scripts" / "training"):
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def _inject_overlay_paths() -> None:
    """Prepend staged AVLM modules onto existing Bridge packages (no overlay __init__.py stubs)."""
    if _OVERLAY_ROOT is None:
        return
    overlay_entries = (
        ("megatron.core.models.multimodal", "megatron/core/models/multimodal"),
        ("megatron.bridge.recipes.nemotron_omni", "megatron/bridge/recipes/nemotron_omni"),
    )
    for mod_name, rel in overlay_entries:
        directory = _OVERLAY_ROOT / rel
        if not directory.is_dir():
            continue
        pkg = importlib.import_module(mod_name)
        extra = str(directory)
        if extra not in list(pkg.__path__):
            pkg.__path__.insert(0, extra)


def _apply_megatron_core_patches() -> None:
    from hdo_resume_fix import apply_hdo_checkpoint_resume_patch
    from megatron_fsdp_buffer_index import apply_megatron_fsdp_buffer_index_patch
    from megatron_fsdp_empty_grad_clip import apply_megatron_fsdp_empty_grad_clip_patch
    from megatron_fsdp_mamba_checkpoint import apply_megatron_fsdp_mamba_checkpoint_patch
    from peft_fsdp_pretrained_load import apply_peft_fsdp_pretrained_load_patch
    from peft_recompute_inputs import apply_peft_recompute_inputs_patch
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.nemotron_omni_audio_mask import (
        apply_nemotron_omni_audio_mask_override,
    )
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.nemotron_omni_model_config import (
        apply_nemotron_omni_model_config_override,
    )
    from megatron.bridge.recipes.nemotron_omni.bridge_integration.nemotron_omni_squared_relu import (
        apply_nemotron_omni_squared_relu_override,
    )

    apply_nemotron_omni_audio_mask_override()
    apply_nemotron_omni_model_config_override()
    apply_nemotron_omni_squared_relu_override()
    apply_hdo_checkpoint_resume_patch()
    apply_megatron_fsdp_buffer_index_patch()
    apply_megatron_fsdp_empty_grad_clip_patch()
    apply_megatron_fsdp_mamba_checkpoint_patch()
    apply_peft_fsdp_pretrained_load_patch()
    apply_peft_recompute_inputs_patch()


_quiet_noisy_runtime_logs()
_bootstrap_import_paths()
_inject_overlay_paths()
_apply_megatron_core_patches()

from megatron.bridge.recipes.nemotron_omni.bridge_integration.task_encoder_collate import (  # noqa: E402
    install_packed_training_hooks,
)
from megatron.bridge.recipes.nemotron_omni.bridge_integration.processed_prompt_log import (  # noqa: E402
    install_processed_prompt_training_hooks,
    processed_prompt_log_enabled,
)

install_packed_training_hooks()
if processed_prompt_log_enabled():
    install_processed_prompt_training_hooks()

import megatron.bridge.recipes as recipes
from megatron.bridge.training.config import ConfigContainer

import run_recipe as bridge_run_recipe  # noqa: E402


def _load_avlm_recipe_builder(recipe_name: str) -> Callable[..., ConfigContainer]:
    try:
        mod = importlib.import_module("megatron.bridge.recipes.nemotron_omni.recipe")
    except ImportError as exc:
        raise AttributeError(
            f"Failed to import megatron.bridge.recipes.nemotron_omni.recipe "
            f"(overlay={_OVERLAY_ROOT}): {exc}",
        ) from exc
    if not hasattr(mod, recipe_name):
        raise AttributeError(
            f"Recipe {recipe_name!r} missing from megatron.bridge.recipes.nemotron_omni.recipe",
        )
    return getattr(mod, recipe_name)


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    hf_path: str | None = None,
) -> ConfigContainer:
    if hasattr(recipes, recipe_name):
        return bridge_run_recipe.load_recipe(
            recipe_name,
            peft_scheme,
            packed_sequence,
            seq_length,
            hf_path,
        )

    config_builder = _load_avlm_recipe_builder(recipe_name)

    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {}
        if "peft" in params or has_var_keyword:
            kwargs["peft"] = peft_scheme
        if ("packed_sequence" in params or has_var_keyword) and packed_sequence:
            kwargs["packed_sequence"] = packed_sequence
        if ("seq_length" in params or has_var_keyword) and seq_length is not None:
            kwargs["seq_length"] = seq_length
        if ("hf_path" in params or has_var_keyword) and hf_path is not None:
            kwargs["hf_path"] = hf_path
        return config_builder(**kwargs)
    except TypeError:
        return config_builder()


def _apply_peft_env_overrides(config: ConfigContainer) -> None:
    """Apply LORA_* env (from AVLM recipe YAML via _train_lib.sh) directly on config.peft.

    Bridge OmegaConf conversion excludes the PEFT object, so peft.* Hydra overrides fail.
    """
    peft = getattr(config, "peft", None)
    if peft is None:
        return
    dim = os.environ.get("LORA_DIM", "").strip()
    alpha = os.environ.get("LORA_ALPHA", "").strip()
    targets = os.environ.get("LORA_TARGET_MODULES", "").strip()
    if dim:
        peft.dim = int(dim)
    if alpha:
        peft.alpha = int(alpha)
    if targets:
        if targets.startswith("["):
            inner = targets[1:-1].strip()
            peft.target_modules = [p.strip() for p in inner.split(",") if p.strip()]
        else:
            peft.target_modules = [p for p in targets.split() if p]


def _strip_peft_hydra_overrides(cli_overrides: list[str] | None) -> list[str]:
    if not cli_overrides:
        return []
    return [item for item in cli_overrides if not item.startswith("peft.")]


def main() -> None:
    args, cli_overrides = bridge_run_recipe.parse_args()

    config = load_recipe(
        args.recipe,
        args.peft_scheme,
        args.packed_sequence,
        args.seq_length,
        args.hf_path,
    )

    if args.dataset is not None:
        mode = bridge_run_recipe.infer_mode_from_dataset(args.dataset)
        config = bridge_run_recipe.apply_dataset_override(
            config,
            dataset_type=args.dataset,
            packed_sequence=args.packed_sequence,
            seq_length=args.seq_length,
            cli_overrides=cli_overrides,
        )
    else:
        mode = bridge_run_recipe.infer_train_mode(args.recipe)

    _apply_peft_env_overrides(config)
    config = bridge_run_recipe.process_config_with_overrides(
        config,
        cli_overrides=_strip_peft_hydra_overrides(cli_overrides) or None,
    )

    if (
        hasattr(config, "model")
        and config.model is not None
        and hasattr(config, "dataset")
        and config.dataset is not None
        and hasattr(config.dataset, "seq_length")
        and config.model.seq_length != config.dataset.seq_length
    ):
        config.model.seq_length = config.dataset.seq_length

    if config.validation.eval_iters == -1:
        from megatron.bridge.recipes.nemotron_omni.dataset import load_jsonl_examples

        n = len(
            load_jsonl_examples(
                config.dataset.val_jsonl,
                config.dataset.val_video_root or config.dataset.video_root,
                jsonl_format=config.dataset.val_jsonl_format or config.dataset.train_jsonl_format,
                sample_ratio=config.dataset.val_sample_ratio,
                rng_seed=config.dataset.rng_seed,
            )
        )
        config.validation.eval_iters = math.ceil(n / config.train.global_batch_size)

    forward_step = bridge_run_recipe.load_forward_step(args.step_func, mode=mode)
    train_func = bridge_run_recipe.TRAIN_FUNCTIONS[mode]
    train_func(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
