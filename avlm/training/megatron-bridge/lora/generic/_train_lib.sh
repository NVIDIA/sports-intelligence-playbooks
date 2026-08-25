# Generic Megatron-Bridge VLM LoRA launch. Reuses the existing Slurm training implementation.

# shellcheck shell=bash
[[ -n "${_MB_LORA_GENERIC_TRAIN_LIB_LOADED:-}" && $(type -t mb_lora_generic_train 2>/dev/null) == function ]] && return 0
_MB_LORA_GENERIC_TRAIN_LIB_LOADED=1

_MB_LORA_GENERIC_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../slurm/_train_lib.sh
source "${_MB_LORA_GENERIC_LIB_DIR}/../slurm/_train_lib.sh"

mb_lora_generic_init_launch() {
  export MB_TRAIN_SLURM_DIR="${_MB_LORA_GENERIC_LIB_DIR}"
  mb_sft_init_launch local
}

mb_lora_generic_train() {
  mb_lora_train
}
