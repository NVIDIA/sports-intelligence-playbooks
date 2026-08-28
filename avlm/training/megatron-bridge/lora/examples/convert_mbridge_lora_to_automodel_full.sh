#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

BRIDGE_ROOT="/opt/Megatron-Bridge"
HF_MODEL_PATH="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
LORA_CHECKPOINT="/path/to/mbridge_lora_checkpoint/iter_NNNNNNN"  # REPLACE_ME with the specific standard Megatron distributed-optimizer (torch_dist) LoRA checkpoint iter_* directory.
HF_OUTPUT="/path/to/automodel_full_model"  # REPLACE_ME with a new output directory.

NUM_GPUS=8
TP=1
PP=1
EP=8

cd "${BRIDGE_ROOT}"

MERGE_SCRIPT="$(mktemp /tmp/merge_lora.XXXXXX.py)"
trap 'rm -f "${MERGE_SCRIPT}"' EXIT

sed '/^    model = lora_peft(model, training=False)$/i\
    model = [model_chunk.module if hasattr(model_chunk, "module") else model_chunk for model_chunk in model]
' examples/peft/merge_lora.py > "${MERGE_SCRIPT}"

uv run torchrun --nproc-per-node="${NUM_GPUS}" \
  "${MERGE_SCRIPT}" \
  --lora-checkpoint "${LORA_CHECKPOINT}" \
  --hf-model-path "${HF_MODEL_PATH}" \
  --output "${HF_OUTPUT}" \
  --tp "${TP}" \
  --pp "${PP}" \
  --ep "${EP}"
