#!/usr/bin/env bash
set -euo pipefail

BRIDGE_ROOT="/opt/Megatron-Bridge"
HF_MODEL_PATH="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
LORA_CHECKPOINT="/path/to/mbridge_lora_checkpoint/iter_NNNNNNN"  # REPLACE_ME with the specific standard Megatron distributed-optimizer (torch_dist) LoRA checkpoint iter_* directory.
HF_OUTPUT="/path/to/automodel_lora_adapter"  # REPLACE_ME with a new output directory.

cd "${BRIDGE_ROOT}"

uv run python examples/conversion/adapter/export_adapter.py \
  --hf-model-path "${HF_MODEL_PATH}" \
  --lora-checkpoint "${LORA_CHECKPOINT}" \
  --output "${HF_OUTPUT}" \
  --trust-remote-code
