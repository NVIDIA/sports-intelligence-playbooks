# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge an AutoModel LoRA adapter into the BF16 Hugging Face base checkpoint.

For every adapted linear layer the merged weight is W + (alpha / r) * B @ A, written back
in the base weight's dtype. The output directory is a complete Hugging Face checkpoint
(shards, index, config, processor files) that vLLM and transformers load as is.

AutoModel saves the adapter with PEFT-style key names that differ from the Hub checkpoint
in two ways: a `base_model.model.` prefix and `language_model.model.` where the Hub
checkpoint says `language_model.backbone.`. Both are normalized here before lookup.

usage:
  python avlm/training/automodel/lora/export/merge_lora_adapter.py \
    --base /path/to/hf_snapshot \
    --adapter /path/to/checkpoints_<layout>/<epoch_x_step_y>/model \
    --out /path/to/merged_bf16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
INDEX = "model.safetensors.index.json"

# Adapter key fragments -> Hub checkpoint key fragments.
KEY_RENAMES = (
    ("base_model.model.", ""),
    ("language_model.model.", "language_model.backbone."),
)


def adapter_target_name(key: str) -> tuple[str, str]:
    """Map `<...>.lora_A.weight` / `<...>.lora_B.weight` to (base weight key, "A" | "B")."""
    for old, new in KEY_RENAMES:
        key = key.replace(old, new)
    module, kind = key.rsplit(".lora_", 1)
    return module + ".weight", kind.split(".")[0]


def load_lora_pairs(adapter_dir: str) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    cfg = json.load(open(os.path.join(adapter_dir, ADAPTER_CONFIG)))
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS)).items():
        name, kind = adapter_target_name(key)
        pairs.setdefault(name, {})[kind] = tensor
    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"adapter pairs without both A and B: {incomplete[:5]}")
    return pairs, scale


def copy_non_weight_files(src: str, dst: str) -> None:
    for name in os.listdir(src):
        if name.endswith(".safetensors"):
            continue
        path = os.path.join(src, name)
        target = os.path.join(dst, name)
        if os.path.isdir(path):
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy(path, target)


def merge(base: str, adapter: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    pairs, scale = load_lora_pairs(adapter)
    weight_map = json.load(open(os.path.join(base, INDEX)))["weight_map"]
    missing = [k for k in pairs if k not in weight_map]
    if missing:
        raise SystemExit(f"adapter keys not in base checkpoint: {missing[:5]}")

    merged = 0
    for shard in sorted(set(weight_map.values())):
        tensors = load_file(os.path.join(base, shard))
        for key in list(tensors):
            if key not in pairs:
                continue
            lora_a = pairs[key]["A"].float()
            lora_b = pairs[key]["B"].float()
            weight = tensors[key]
            if weight.shape != (lora_b.shape[0], lora_a.shape[1]):
                raise SystemExit(f"shape mismatch for {key}: {tuple(weight.shape)} vs B@A {lora_b.shape[0]}x{lora_a.shape[1]}")
            tensors[key] = (weight.float() + scale * (lora_b @ lora_a)).to(weight.dtype)
            merged += 1
        save_file(tensors, os.path.join(out, shard), metadata={"format": "pt"})
        print(f"{shard}: written", flush=True)

    if merged != len(pairs):
        raise SystemExit(f"merged {merged} weights but the adapter has {len(pairs)}")
    copy_non_weight_files(base, out)
    print(f"merged {merged} weights with scale {scale:g} into {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Hugging Face BF16 base snapshot directory")
    parser.add_argument("--adapter", required=True, help="directory holding adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--out", required=True, help="output directory for the merged checkpoint")
    args = parser.parse_args()
    merge(args.base, args.adapter, args.out)


if __name__ == "__main__":
    main()
