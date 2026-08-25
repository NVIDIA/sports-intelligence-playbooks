"""Make Mamba parameters compatible with ``fsdp_dtensor`` checkpoints."""

from __future__ import annotations

import logging
from functools import wraps

import torch

logger = logging.getLogger(__name__)

_PATCHED = "_avlm_mamba_fsdp_checkpoint_patch"


def _strip_wrappers(path: str) -> str:
    parts = path.split(".")
    while parts and parts[0] in ("model", "module"):
        parts.pop(0)
    return ".".join(parts)


def _mamba_layouts(model):  # noqa: ANN001
    from megatron.core.ssm.mamba_mixer import MambaMixer

    layouts = {}
    for name, module in model.named_modules():
        if not isinstance(module, MambaMixer):
            continue

        prefix = _strip_wrappers(name)
        in_proj_sizes = [
            module.d_inner_local_tp,
            module.d_inner_local_tp,
            module.ngroups_local_tp * module.d_state,
            module.ngroups_local_tp * module.d_state,
            module.nheads_local_tp,
        ]
        conv_sizes = [
            module.d_inner_local_tp,
            module.ngroups_local_tp * module.d_state,
            module.ngroups_local_tp * module.d_state,
        ]
        in_proj = getattr(module.in_proj, "to_wrap", module.in_proj)
        layouts[f"{prefix}.in_proj.weight"] = (
            in_proj_sizes,
            ("z", "x", "B", "C", "dt"),
            in_proj.weight,
        )
        for parameter_name in ("conv1d.weight", "conv1d.bias"):
            parameter = getattr(module.conv1d, parameter_name.rsplit(".", 1)[-1], None)
            if parameter is not None:
                layouts[f"{prefix}.{parameter_name}"] = (
                    conv_sizes,
                    ("x", "B", "C"),
                    parameter,
                )
    return layouts


def _split_fused_tensor(data, dist_param, split_sizes):  # noqa: ANN001
    from torch.distributed._tensor import DTensor

    from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
        make_fsdp_dtensor,
    )
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import split_dtensor
    from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import (
        get_mcore_tensor_parallel_partition_dim,
        is_mcore_tensor_model_parallel,
    )
    from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes

    total_split = sum(split_sizes)
    if isinstance(data, DTensor) and data.shape[0] == total_split:
        return list(
            split_dtensor(
                data,
                split_sizes,
                dim=0,
                update_uneven_dtensor_chunk_meta=True,
            )
        )

    # PEFT loads base weights before the model is wrapped with Megatron-FSDP.
    if not hasattr(dist_param, "megatron_fsdp_slice"):
        if data.shape[0] != total_split:
            raise ValueError(f"Mamba fused tensor shape mismatch: {data.shape[0]} != {total_split}")
        return list(torch.split(data, split_sizes, dim=0))

    fsdp_slice = dist_param.megatron_fsdp_slice
    dist_index = dist_param.megatron_fsdp_dist_index
    tp_mesh = dist_index.get_submesh([dist_index.tp_dim], is_expert_parallel=False)
    global_shape = dist_param.shape
    data_size = dist_param.numel() // tp_mesh.mesh.numel()
    elements_per_unit = data_size // total_split

    if isinstance(data, DTensor):
        if data.shape != global_shape:
            raise ValueError(f"Mamba DTensor shape mismatch: {data.shape} != {global_shape}")
        local_tensor = data.to_local()
    else:
        expected_numel = fsdp_slice.stop - fsdp_slice.start
        if data.numel() != expected_numel:
            raise ValueError(
                "Mamba optimizer tensor is not an FSDP-local shard: "
                f"expected {expected_numel} elements, got {data.numel()}"
            )
        local_tensor = data

    per_tp_rank_shape = list(global_shape)
    if is_mcore_tensor_model_parallel(dist_param):
        tp_dim = get_mcore_tensor_parallel_partition_dim(dist_param)
        if tp_dim is None:
            raise ValueError("Mamba tensor-parallel partition dimension is missing")
        per_tp_rank_shape[tp_dim] //= tp_mesh.mesh.numel()

    results = []
    flat_offset = 0
    for size in split_sizes:
        component_numel = size * elements_per_unit
        component_start = flat_offset
        component_stop = flat_offset + component_numel
        shard_start = max(fsdp_slice.start, component_start)
        shard_stop = min(fsdp_slice.stop, component_stop)
        if shard_start >= shard_stop:
            shard_start = shard_stop = fsdp_slice.start

        component = local_tensor.reshape(-1)[
            shard_start - fsdp_slice.start : shard_stop - fsdp_slice.start
        ]
        component_shape = list(global_shape)
        component_shape[0] = -1
        component = component.reshape(component_shape)

        meta_shape = list(per_tp_rank_shape)
        meta_shape[0] = size
        meta = torch.empty(*meta_shape, device="meta")
        copy_tensor_model_parallel_attributes(meta, dist_param)
        results.append(
            make_fsdp_dtensor(
                component.data,
                meta,
                dist_index=dist_index,
                is_expert_param=False,
                run_check=True,
                update_uneven_dtensor_chunk_meta=True,
            )
        )
        flat_offset = component_stop

    return results


def _split_mamba_state_dict(model, model_state_dict, optimizer_state_dict):  # noqa: ANN001
    layouts = _mamba_layouts(model)
    if not layouts:
        return model_state_dict, optimizer_state_dict

    model_state_dict = model_state_dict.copy()
    split_count = 0
    for key in list(model_state_dict):
        layout = layouts.get(_strip_wrappers(key))
        if layout is None:
            continue
        sizes, component_names, parameter = layout
        components = _split_fused_tensor(model_state_dict.pop(key), parameter, sizes)
        for name, component in zip(component_names, components):
            model_state_dict[f"{key}.{name}"] = component
        split_count += 1

    if optimizer_state_dict is not None and optimizer_state_dict.get("state"):
        optimizer_state_dict = optimizer_state_dict.copy()
        new_optimizer_state = {}
        for key, state in optimizer_state_dict["state"].items():
            layout = layouts.get(_strip_wrappers(key))
            if layout is None:
                new_optimizer_state[key] = state
                continue
            sizes, component_names, parameter = layout
            component_states = {name: state.copy() for name in component_names}
            for state_name in ("exp_avg", "exp_avg_sq"):
                if state_name not in state:
                    continue
                components = _split_fused_tensor(state[state_name], parameter, sizes)
                for name, component in zip(component_names, components):
                    component_states[name][state_name] = component
            for name, component_state in component_states.items():
                new_optimizer_state[f"{key}.{name}"] = component_state
        optimizer_state_dict["state"] = new_optimizer_state

    if split_count:
        logger.info("Split %d fused Mamba tensors for fsdp_dtensor checkpointing", split_count)
    return model_state_dict, optimizer_state_dict


def apply_megatron_fsdp_mamba_checkpoint_patch() -> None:
    """Split fused Mamba tensors before FSDP checkpoint save and load."""
    from megatron.bridge.training import checkpointing

    original = checkpointing.preprocess_fsdp_dtensor_state_dict
    if getattr(original, _PATCHED, False):
        return

    @wraps(original)
    def preprocess(args, raw_state_dict, model):  # noqa: ANN001
        state_dict = raw_state_dict.copy()
        model_state, optimizer_state = _split_mamba_state_dict(
            model,
            state_dict["model"],
            state_dict.get("optimizer"),
        )
        state_dict["model"] = model_state
        if optimizer_state is not None:
            state_dict["optimizer"] = optimizer_state
        return original(args, state_dict, model)

    setattr(preprocess, _PATCHED, True)
    checkpointing.preprocess_fsdp_dtensor_state_dict = preprocess
