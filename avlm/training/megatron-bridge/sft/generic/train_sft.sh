#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run Megatron-Bridge SFT in an existing container without Slurm.
set -euo pipefail

_GENERIC_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../../../utils/_source_params.sh
source "${_GENERIC_DIR}/../../../../utils/_source_params.sh"

# shellcheck source=_train_lib.sh
source "${_GENERIC_DIR}/_train_lib.sh"
mb_sft_generic_init_launch
mb_sft_generic_train
