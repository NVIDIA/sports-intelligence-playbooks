# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""In-place patches for HybridDeviceOptimizer checkpoint resume with CPU offload."""

from __future__ import annotations

import functools
from typing import Any, Callable

_PATCHED = "_avlm_hdo_resume_patch_v2"


def _hdo_cpu_offload_active(optimizer: Any) -> bool:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    return isinstance(optimizer, HybridDeviceOptimizer) and optimizer.offload_fraction > 0


def _patch_distrib_optimizer_load_state_dict() -> None:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    if getattr(DistributedOptimizer.load_state_dict, _PATCHED, False):
        return

    original: Callable = DistributedOptimizer.load_state_dict

    @functools.wraps(original)
    def load_state_dict(self, state_dict):  # noqa: ANN001
        if not _hdo_cpu_offload_active(self.optimizer) or self.ddp_config.use_megatron_fsdp:
            return original(self, state_dict)

        inner_optimizer = self.optimizer
        inner_load = inner_optimizer.load_state_dict

        def _load_state_dict_keep_cpu_state(opt_state_dict: dict) -> None:
            """Apply param-group metadata only; keep CPU-resident Adam state from dummy_step."""
            for dst, src in zip(inner_optimizer.param_groups, opt_state_dict["param_groups"]):
                params = dst["params"]
                dst.clear()
                dst.update({key: value for key, value in src.items() if key != "params"})
                dst["params"] = params
            inner_optimizer._sync_hdo_param_groups_to_sub_optimizers()

        inner_optimizer.load_state_dict = _load_state_dict_keep_cpu_state
        try:
            return original(self, state_dict)
        finally:
            inner_optimizer.load_state_dict = inner_load

    setattr(load_state_dict, _PATCHED, True)
    DistributedOptimizer.load_state_dict = load_state_dict


def apply_hdo_checkpoint_resume_patch() -> None:
    """Patch HybridDeviceOptimizer and DistributedOptimizer for CPU-offload resume."""
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    if getattr(HybridDeviceOptimizer, _PATCHED, False):
        return

    def _update_fp32_params_by_new_state(self) -> None:
        if not self.param_update_in_fp32:
            return
        for param, v in self.state.items():
            fp32_param = self.param_to_fp32_param.get(param)
            if fp32_param is None or "master_param" not in v:
                continue
            fp32_param.data.copy_(v["master_param"])

    def _register_load_state_dict_hooks(self) -> None:
        def pre_load_state_dict_hook(self, state_dict):
            if not self.param_update_in_fp32:
                return state_dict

            new_state = {}
            for param, v in self.state.items():
                param = self.param_to_fp32_param.get(param, param)
                new_state[param] = v
            self.state = new_state

            for group in self.param_groups:
                for i, param in enumerate(group["params"]):
                    group["params"][i] = self.param_to_fp32_param.get(param, param)

            return state_dict

        self.register_load_state_dict_pre_hook(pre_load_state_dict_hook)

        def post_load_state_dict_hook(self) -> None:
            if self.param_update_in_fp32:
                new_state = {}
                for param, v in self.state.items():
                    orig_param = self.fp32_param_to_orig_param.get(param)
                    if orig_param is None:
                        orig_param = self.inner_param_to_orig_param.get(param, param)
                    new_state[orig_param] = v
                self.state = new_state

                for group in self.param_groups:
                    for i, param in enumerate(group["params"]):
                        group["params"][i] = self.fp32_param_to_orig_param.get(param, param)

            self._init_sub_optimizers()
            self._sync_hdo_param_groups_to_sub_optimizers()
            self._sync_hdo_state_to_sub_optimizers()

        self.register_load_state_dict_post_hook(post_load_state_dict_hook)

    HybridDeviceOptimizer._update_fp32_params_by_new_state = _update_fp32_params_by_new_state
    HybridDeviceOptimizer._register_load_state_dict_hooks = _register_load_state_dict_hooks
    setattr(HybridDeviceOptimizer, _PATCHED, True)

    _patch_distrib_optimizer_load_state_dict()
