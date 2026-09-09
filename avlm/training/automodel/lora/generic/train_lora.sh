#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run AutoModel LoRA in an existing container without Slurm.
set -euo pipefail

_LORA_GENERIC_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x /opt/venv/bin/automodel ]]; then
  echo "error: /opt/venv/bin/automodel not found — run inside nemo-automodel:26.06" >&2
  exit 1
fi

# shellcheck source=../../../../utils/_source_params.sh
source "${_LORA_GENERIC_DIR}/../../../../utils/_source_params.sh"

# shellcheck source=_train_lib.sh
source "${_LORA_GENERIC_DIR}/_train_lib.sh"
lora_generic_init_launch
lora_generic_train
