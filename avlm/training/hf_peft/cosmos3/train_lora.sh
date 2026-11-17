#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the Cosmos3 LoRA trainer inside the container built from this directory's Dockerfile.
# Required: TRAIN_JSONL, VAL_JSONL, FRAMES_ROOT, OUTPUT_DIR. Optional: MODEL_PATH, STEPS, LR,
# RANK, WARMUP, GLOBAL_BATCH_SIZE, plus EXTRA_ARGS passed through to train_lora.py.
set -euo pipefail

_COSMOS3_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PY=/opt/venv/bin/python3
[[ -x "${_PY}" ]] || _PY="$(command -v python3)"

: "${TRAIN_JSONL:?Set TRAIN_JSONL to the training conversation JSONL}"
: "${VAL_JSONL:?Set VAL_JSONL to the validation conversation JSONL}"
: "${FRAMES_ROOT:?Set FRAMES_ROOT to the frames root written by extract_frames.py}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the output directory}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck disable=SC2086
exec "${_PY}" "${_COSMOS3_DIR}/train_lora.py" \
  --model "${MODEL_PATH:-nvidia/Cosmos3-Nano}" \
  --train "${TRAIN_JSONL}" --val "${VAL_JSONL}" --frames "${FRAMES_ROOT}" --out "${OUTPUT_DIR}" \
  --steps "${STEPS:-150}" --lr "${LR:-2e-4}" --rank "${RANK:-64}" --warmup "${WARMUP:-10}" \
  --global-batch "${GLOBAL_BATCH_SIZE:-8}" ${EXTRA_ARGS:-}
