#!/usr/bin/env bash
# Run Megatron-Bridge LoRA in an existing container without Slurm.
set -euo pipefail

_GENERIC_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../../../utils/_source_params.sh
source "${_GENERIC_DIR}/../../../../utils/_source_params.sh"

# shellcheck source=_train_lib.sh
source "${_GENERIC_DIR}/_train_lib.sh"
mb_lora_generic_init_launch
mb_lora_generic_train
