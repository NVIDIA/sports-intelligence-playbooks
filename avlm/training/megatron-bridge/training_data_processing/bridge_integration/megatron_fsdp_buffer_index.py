"""Backport Megatron-LM's corrected FSDP parameter-buffer indexing."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def apply_megatron_fsdp_buffer_index_patch() -> None:
    """Keep every FSDP parameter shard aligned to complete tensor rows."""
    from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as fsdp

    if getattr(fsdp, "_avlm_buffer_index_patch", False):
        return

    def build_data_parallel_buffer_index(
        param_shapes: List[torch.Size],
        data_parallel_rank: int,
        data_parallel_world_size: int,
        is_data_distributed: bool,
        ddp_config,
        bucket_id: int = 0,
        chunk_size_factor: int = 1,
    ) -> Tuple[Dict[int, object], object, object]:
        """Backport NVIDIA/Megatron-LM commit 07e17d347608e16085910e9e7b54079f4ab84a8c."""

        def pad_if_needed(index: int) -> int:
            if ddp_config.data_parallel_sharding_strategy != "no_shard":
                return fsdp._pad(index, data_parallel_world_size * chunk_size_factor)
            return index

        def add_item(param_id: int, shape: torch.Size, offset: int) -> None:
            item_index_map[param_id] = fsdp.TensorItemIndex(
                global_data_index=offset,
                size=shape.numel(),
                item_id=param_id,
                bucket_id=bucket_id,
                shape=shape,
            )

        fragments = []
        regular = []
        for param_id, shape in enumerate(param_shapes):
            target = fragments if shape.numel() < chunk_size_factor else regular
            target.append((param_id, shape))
        fragments.sort(key=lambda pair: -pair[1].numel())

        item_index_map = {}
        global_index = 0
        while regular:
            param_id, shape = regular.pop(0)
            numel = shape.numel()
            add_item(param_id, shape, global_index)
            if numel % chunk_size_factor == 0:
                global_index += numel
                continue

            gap_offset = global_index + numel
            global_index += (numel // chunk_size_factor + 1) * chunk_size_factor
            remainder = numel % chunk_size_factor
            remaining_space = chunk_size_factor - remainder

            partner = None
            for candidate in regular:
                candidate_remainder = candidate[1].numel() % chunk_size_factor
                if candidate_remainder and remainder + candidate_remainder <= chunk_size_factor:
                    partner = candidate
                    break

            if partner is not None:
                regular.remove(partner)
                partner_id, partner_shape = partner
                partner_numel = partner_shape.numel()
                partner_remainder = partner_numel % chunk_size_factor
                add_item(partner_id, partner_shape, global_index - partner_numel)
                remaining_space -= partner_remainder
                global_index += (partner_numel // chunk_size_factor) * chunk_size_factor

            for fragment in list(fragments):
                fragment_id, fragment_shape = fragment
                fragment_numel = fragment_shape.numel()
                if fragment_numel > remaining_space:
                    continue
                add_item(fragment_id, fragment_shape, gap_offset)
                remaining_space -= fragment_numel
                gap_offset += fragment_numel
                fragments.remove(fragment)

        slots = []
        for fragment in fragments:
            fragment_size = fragment[1].numel()
            for slot in slots:
                if sum(item[1].numel() for item in slot) + fragment_size <= chunk_size_factor:
                    slot.append(fragment)
                    break
            else:
                slots.append([fragment])

        for slot in slots:
            offset = 0
            for param_id, shape in slot:
                add_item(param_id, shape, global_index + offset)
                offset += shape.numel()
            global_index += chunk_size_factor

        bucket_index = fsdp.BucketIndex(
            bucket_id=bucket_id,
            global_data_index=0,
            size=pad_if_needed(global_index),
            items=list(item_index_map.values()),
        )
        shard_bucket_index = fsdp._get_dp_buffer_shard_bucket_index(
            bucket_index=bucket_index,
            is_data_distributed=is_data_distributed,
            data_parallel_world_size=data_parallel_world_size,
            data_parallel_rank=data_parallel_rank,
        )
        return item_index_map, bucket_index, shard_bucket_index

    fsdp.build_data_parallel_buffer_index = build_data_parallel_buffer_index
    fsdp._avlm_buffer_index_patch = True

