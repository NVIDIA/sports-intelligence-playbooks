# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release CPU/GPU tensor references held by VLM training batches."""

from __future__ import annotations

import gc
from typing import Any

import torch


def release_tensor_tree(obj: Any) -> None:
    """Drop references inside nested batch structures (dict / list / tuple)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return
    if isinstance(obj, torch.Tensor):
        return
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            val = obj.pop(key, None)
            release_tensor_tree(val)
            del val
        return
    if isinstance(obj, list):
        while obj:
            val = obj.pop()
            release_tensor_tree(val)
            del val
        return
    if isinstance(obj, tuple):
        for val in obj:
            release_tensor_tree(val)


def release_vlm_batch(batch: dict[str, Any] | None) -> None:
    """Clear a collated / packed VLM batch dict so large CPU tensors can be freed."""
    if not batch:
        return
    for key in list(batch.keys()):
        val = batch.pop(key, None)
        release_tensor_tree(val)
        del val


def release_grad_accum_batches(batches: list[dict[str, Any]] | None) -> None:
    """Clear all microbatches from one optimizer step."""
    if not batches:
        return
    for batch in batches:
        release_vlm_batch(batch)
    batches.clear()


def post_optimizer_cpu_gc() -> None:
    """Run a full Python GC after an optimizer step (all generations)."""
    gc.collect()


def release_decode_intermediates(*objs: Any) -> None:
    """Drop references to decode / pretokenize temporaries (best-effort)."""
    for obj in objs:
        if isinstance(obj, dict):
            release_vlm_batch(obj)
        elif isinstance(obj, list):
            obj.clear()
        del obj


def maybe_gc_after_pack() -> None:
    """Optional GC after materializing one pack (``GC_AFTER_PACK=1``)."""
    import os

    if os.environ.get("GC_AFTER_PACK", "0") in ("1", "true", "yes"):
        gc.collect()
