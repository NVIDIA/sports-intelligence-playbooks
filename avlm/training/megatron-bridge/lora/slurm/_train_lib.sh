# Megatron-Bridge VLM LoRA launch. Reuses sft/slurm/_train_lib.sh (PEFT recipe + checkpoint.load rules).

# shellcheck shell=bash
[[ -n "${_MB_LORA_TRAIN_LIB_LOADED:-}" && $(type -t mb_lora_train 2>/dev/null) == function ]] && return 0
_MB_LORA_TRAIN_LIB_LOADED=1

_MB_LORA_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../sft/slurm/_train_lib.sh
source "${_MB_LORA_LIB_DIR}/../../sft/slurm/_train_lib.sh"

mb_lora_train() {
  mb_sft_train
}

mb_lora_init_launch() {
  export MB_TRAIN_SLURM_DIR="${_MB_LORA_LIB_DIR}"
  mb_sft_init_launch "$@"
}
