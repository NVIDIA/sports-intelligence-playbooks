# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Avoid fused gradient-clipping calls with empty local FSDP shards."""

from __future__ import annotations

from functools import wraps

import torch


_PATCH_FLAG = "_avlm_empty_grad_clip_patch"


def apply_megatron_fsdp_empty_grad_clip_patch() -> None:
    """Skip clipping on ranks that have no local gradients to scale."""
    from megatron.core.optimizer import clip_grads as clip_grads_module
    from megatron.core.optimizer import optimizer as optimizer_module
    from megatron.core.utils import to_local_if_dtensor

    if getattr(clip_grads_module, _PATCH_FLAG, False):
        return

    original = clip_grads_module.clip_grad_by_total_norm_fp32

    @wraps(original)
    def clip_grad_by_total_norm_fp32(
        parameters,  # noqa: ANN001
        max_norm,  # noqa: ANN001
        total_norm,  # noqa: ANN001
        use_decoupled_grad: bool = False,
    ):
        parameter_list = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
        grad_name = "decoupled_grad" if use_decoupled_grad else "grad"
        local_parameters = []

        for parameter in parameter_list:
            grad = getattr(parameter, grad_name, None)
            if grad is not None and to_local_if_dtensor(grad).numel() > 0:
                local_parameters.append(parameter)

        if not local_parameters:
            return None

        return original(
            local_parameters,
            max_norm=max_norm,
            total_norm=total_norm,
            use_decoupled_grad=use_decoupled_grad,
        )

    clip_grads_module.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    optimizer_module.clip_grad_by_total_norm_fp32 = clip_grad_by_total_norm_fp32
    setattr(clip_grads_module, _PATCH_FLAG, True)


__all__ = ["apply_megatron_fsdp_empty_grad_clip_patch"]
