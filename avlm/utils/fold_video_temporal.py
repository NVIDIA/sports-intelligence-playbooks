# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fold a Nemotron-Omni checkpoint to one frame per temporal patch.

Usage:
    python avlm/utils/fold_video_temporal.py --src <checkpoint> --dst <folded_checkpoint>
"""

import argparse
import json
import os

from safetensors.torch import load_file, save_file

TENSOR = "vision_model.radio_model.model.patch_generator.video_embedder.weight"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="source model/consolidated dir")
    parser.add_argument("--dst", required=True, help="output model/consolidated dir")
    return parser.parse_args()


def main():
    args = parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)

    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    # Prefer the top-level setting when both configs specify a temporal patch size.
    temporal = config.get("video_temporal_patch_size")
    if temporal is None:
        temporal = config["vision_config"]["video_temporal_patch_size"]
    assert temporal >= 2, f"nothing to fold: video_temporal_patch_size={temporal}"

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shard_name = index["weight_map"][TENSOR]

    rewritten = {shard_name, "config.json", "processor_config.json", "model.safetensors.index.json"}
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if name in rewritten:
            continue
        link = os.path.join(dst, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join(src, name), link)

    tensors = load_file(os.path.join(src, shard_name))
    weight = tensors[TENSOR]
    out_features, in_features = weight.shape
    assert in_features % temporal == 0, (weight.shape, temporal)
    block = in_features // temporal
    # Summing frame blocks preserves the projection of repeated identical frames.
    folded = sum(
        weight[:, t * block : (t + 1) * block].float() for t in range(temporal)
    ).to(weight.dtype)
    tensors[TENSOR] = folded.contiguous()
    save_file(tensors, os.path.join(dst, shard_name), metadata={"format": "pt"})

    saved_bytes = (in_features - block) * out_features * weight.element_size()
    index["metadata"]["total_size"] -= saved_bytes
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    config["video_temporal_patch_size"] = 1
    config["vision_config"]["video_temporal_patch_size"] = 1
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(src, "processor_config.json")) as f:
        processor_config = json.load(f)
    processor_config["video_temporal_patch_dim"] = 1
    with open(os.path.join(dst, "processor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=2)


if __name__ == "__main__":
    main()
