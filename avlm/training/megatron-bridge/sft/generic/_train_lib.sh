# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generic Megatron-Bridge VLM SFT launch. Reuses the existing Slurm training implementation.

# shellcheck shell=bash
[[ -n "${_MB_SFT_GENERIC_TRAIN_LIB_LOADED:-}" && $(type -t mb_sft_generic_train 2>/dev/null) == function ]] && return 0
_MB_SFT_GENERIC_TRAIN_LIB_LOADED=1

_MB_SFT_GENERIC_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../slurm/_train_lib.sh
source "${_MB_SFT_GENERIC_LIB_DIR}/../slurm/_train_lib.sh"

mb_sft_generic_init_launch() {
  export MB_TRAIN_SLURM_DIR="${_MB_SFT_GENERIC_LIB_DIR}"
  mb_sft_init_launch local
}

mb_sft_generic_train() {
  mb_sft_train
}
