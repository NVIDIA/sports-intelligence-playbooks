# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Carry a merged LoRA into a ModelOpt FP8 checkpoint without re-running calibration.

A LoRA touches a small set of linear weights. Instead of quantizing the whole merged BF16
checkpoint again, copy the published FP8 checkpoint and re-quantize only those weights
from the merged BF16 checkpoint: per-tensor static FP8 (e4m3 weight + float32
`<weight>_scale`). Every other tensor, including the calibrated `input_scale`s, is kept
from the FP8 base.

The adapter directory is only read for its key names, to know which weights were touched.

usage:
  python avlm/training/automodel/lora/export/requantize_fp8_merged.py \
    --fp8-base /path/to/fp8_snapshot \
    --merged /path/to/merged_bf16 \
    --adapter /path/to/checkpoints_<layout>/<epoch_x_step_y>/model \
    --out /path/to/merged_fp8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from merge_lora_adapter import ADAPTER_WEIGHTS, INDEX, adapter_target_name, copy_non_weight_files

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def quantize_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (weight.float().abs().amax() / FP8_MAX).clamp(min=1e-12)
    quantized = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale.to(torch.float32)


def touched_weights(adapter_dir: str) -> list[str]:
    keys = load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS)).keys()
    return sorted({adapter_target_name(k)[0] for k in keys})


class ShardCache:
    """Loads one shard of the merged checkpoint at a time, in weight-map order."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.weight_map = json.load(open(os.path.join(root, INDEX)))["weight_map"]
        self.shard = ""
        self.tensors: dict[str, torch.Tensor] = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard != self.shard:
            self.shard, self.tensors = shard, load_file(os.path.join(self.root, shard))
        return self.tensors[key]


def requantize(fp8_base: str, merged: str, adapter: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    touched = touched_weights(adapter)
    fp8_map = json.load(open(os.path.join(fp8_base, INDEX)))["weight_map"]
    not_fp8 = [k for k in touched if k not in fp8_map or k + "_scale" not in fp8_map]
    if not_fp8:
        raise SystemExit(f"touched weights that are not FP8-quantized in the base: {not_fp8[:5]}")
    merged_shards = ShardCache(merged)

    done = 0
    for shard in sorted(set(fp8_map.values())):
        keys = [k for k in touched if fp8_map[k] == shard]
        if not keys:
            shutil.copy(os.path.join(fp8_base, shard), os.path.join(out, shard))
            print(f"{shard}: copied", flush=True)
            continue
        tensors = load_file(os.path.join(fp8_base, shard))
        for key in keys:
            quantized, scale = quantize_per_tensor(merged_shards.get(key))
            if tensors[key].shape != quantized.shape or tensors[key].dtype != quantized.dtype:
                raise SystemExit(f"{key}: base is {tensors[key].dtype} {tuple(tensors[key].shape)}, expected e4m3 {tuple(quantized.shape)}")
            scale_key = key + "_scale"
            tensors[key] = quantized
            tensors[scale_key] = scale.reshape(tensors[scale_key].shape).to(tensors[scale_key].dtype)
            done += 1
        save_file(tensors, os.path.join(out, shard), metadata={"format": "pt"})
        print(f"{shard}: {len(keys)} weights re-quantized", flush=True)

    if done != len(touched):
        raise SystemExit(f"re-quantized {done} weights but the adapter touches {len(touched)}")
    copy_non_weight_files(fp8_base, out)
    print(f"re-quantized {done} weights into {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp8-base", required=True, help="ModelOpt FP8 snapshot of the same base model")
    parser.add_argument("--merged", required=True, help="merged BF16 checkpoint from merge_lora_adapter.py")
    parser.add_argument("--adapter", required=True, help="adapter directory (read for its key names only)")
    parser.add_argument("--out", required=True, help="output directory for the FP8 checkpoint")
    args = parser.parse_args()
    requantize(args.fp8_base, args.merged, args.adapter, args.out)


if __name__ == "__main__":
    main()
