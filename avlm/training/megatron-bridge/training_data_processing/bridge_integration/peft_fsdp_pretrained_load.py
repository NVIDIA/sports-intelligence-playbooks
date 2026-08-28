# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an FSDP-DTensor base checkpoint before applying PEFT."""

from __future__ import annotations

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)

_PATCHED = "_avlm_peft_fsdp_pretrained_load_patch"


def _needs_module_prefix(cfg, state_dict) -> bool:  # noqa: ANN001
    checkpoint = getattr(cfg, "checkpoint", None)
    if (
        getattr(cfg, "peft", None) is None
        or getattr(checkpoint, "ckpt_format", None) != "fsdp_dtensor"
        or not getattr(checkpoint, "finetune", False)
    ):
        return False

    model_state = state_dict.get("model")
    if not isinstance(model_state, dict) or not model_state:
        return False

    keys = tuple(str(key) for key in model_state)
    has_prefixed = [key == "module" or key.startswith("module.") for key in keys]
    if any(has_prefixed) and not all(has_prefixed):
        raise RuntimeError("FSDP-DTensor model state mixes module-prefixed and unprefixed keys")
    return not any(has_prefixed)


def apply_peft_fsdp_pretrained_load_patch() -> None:
    """Match wrapped checkpoint keys while loading the pre-wrap PEFT base model."""
    from megatron.bridge.training import checkpointing

    original_load = checkpointing.load_fsdp_dtensor_checkpoint
    if getattr(original_load, _PATCHED, False):
        return
    original_preprocess = checkpointing.preprocess_fsdp_dtensor_state_dict

    @wraps(original_preprocess)
    def preprocess(cfg, raw_state_dict, model):  # noqa: ANN001
        state_dict = original_preprocess(cfg, raw_state_dict, model)
        if not _needs_module_prefix(cfg, state_dict):
            return state_dict

        state_dict = state_dict.copy()
        state_dict["model"] = {
            f"module.{key}": value for key, value in state_dict["model"].items()
        }
        if state_dict.get("rng_state") is None:
            state_dict.pop("rng_state", None)
        logger.info("Using module-prefixed state keys for the pre-PEFT FSDP-DTensor base load")
        return state_dict

    @wraps(original_load)
    def load(*args, **kwargs):  # noqa: ANN002, ANN003
        bound = inspect.signature(original_load).bind_partial(*args, **kwargs)
        ckpt_cfg = bound.arguments.get("ckpt_cfg")
        cfg = bound.arguments.get("cfg")
        state_dict = bound.arguments.get("sharded_state_dict")
        strict_pretrained_load = (
            ckpt_cfg is not None
            and cfg is not None
            and isinstance(state_dict, dict)
            and _needs_module_prefix(cfg, state_dict)
        )
        if not strict_pretrained_load:
            return original_load(*args, **kwargs)

        previous = getattr(ckpt_cfg, "strict_fsdp_dtensor_load", False)
        ckpt_cfg.strict_fsdp_dtensor_load = True
        try:
            return original_load(*args, **kwargs)
        finally:
            ckpt_cfg.strict_fsdp_dtensor_load = previous

    setattr(load, _PATCHED, True)
    checkpointing.preprocess_fsdp_dtensor_state_dict = preprocess
    checkpointing.load_fsdp_dtensor_checkpoint = load
