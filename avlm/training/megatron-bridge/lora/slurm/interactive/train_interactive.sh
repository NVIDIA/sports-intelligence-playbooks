#!/usr/bin/env bash
# Run Megatron-Bridge LoRA (PEFT) inside an interactive Slurm container session.
#
#   bash avlm/training/megatron-bridge/lora/slurm/interactive/launch_interactive_session.sh
#   bash avlm/training/megatron-bridge/lora/slurm/interactive/train_interactive.sh
#
# Resume LoRA from checkpoints in outputs/<MODEL_NAME>/checkpoints_<layout>/:
#   RESUME_CHECKPOINT=1 bash .../train_interactive.sh
set -euo pipefail

if [[ "${INSIDE_INTERACTIVE_SESSION:-0}" != "1" ]]; then
  echo "error: start an interactive session first:" >&2
  echo "  bash avlm/training/megatron-bridge/lora/slurm/interactive/launch_interactive_session.sh" >&2
  exit 1
fi

_INTERACTIVE_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${_INTERACTIVE_DIR}/.." && pwd)"
_MB_LORA_CONFIG="${_SLURM_DIR}/../configs/nemotron_omni_mbridge_lora_singlenode.yaml"

# shellcheck source=../../../../utils/_source_params.sh
source "${_SLURM_DIR}/../../../../utils/_source_params.sh"
if [[ -z "${CONFIG_YAML:-}" && -z "${CONFIG_YAML_REL:-}" ]]; then
  export CONFIG_YAML="${_MB_LORA_CONFIG}"
fi

# shellcheck source=../_train_lib.sh
source "${_SLURM_DIR}/_train_lib.sh"
mb_lora_init_launch interactive

mb_lora_train
