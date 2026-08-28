#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (($#)); then
  echo "error: eval configuration is environment-only; set INFERENCE_DIR before the command" >&2
  exit 2
fi
: "${INFERENCE_DIR:?Set INFERENCE_DIR}"

PYTHONPATH="." python3 avlm/evals/mcq/eval_mcq.py --inference-dir "${INFERENCE_DIR}"
