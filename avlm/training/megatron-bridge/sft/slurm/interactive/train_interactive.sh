#!/usr/bin/env bash
# Run Megatron-Bridge SFT inside an interactive Slurm container session.
#
#   bash avlm/training/megatron-bridge/sft/slurm/interactive/launch_interactive_session.sh
#   bash avlm/training/megatron-bridge/sft/slurm/interactive/train_interactive.sh
#
# Optional processed-prompt logging for encoding diagnostics:
#   AVLM_LOG_PROCESSED_PROMPT=1 AVLM_LOG_PROCESSED_PROMPT_TAG=aligned MAX_STEPS=2 bash .../train_interactive.sh
# Writes ${OUTPUT}/processed_prompt_logs/processed_prompts_<tag>.jsonl (GBS rows per iteration).
#
# Override base Megatron checkpoint (wins over recipe YAML checkpoint.pretrained_checkpoint):
#   PRETRAINED_CHECKPOINT=/path/to/megatron-checkpoint \
#     bash avlm/training/megatron-bridge/sft/slurm/interactive/train_interactive.sh
set -euo pipefail

if [[ "${INSIDE_INTERACTIVE_SESSION:-0}" != "1" ]]; then
  echo "error: start an interactive session first:" >&2
  echo "  bash avlm/training/megatron-bridge/sft/slurm/interactive/launch_interactive_session.sh" >&2
  exit 1
fi

_INTERACTIVE_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${_INTERACTIVE_DIR}/.." && pwd)"
_MB_SFT_CONFIG="${_SLURM_DIR}/../configs/nemotron_omni_mbridge_sft_singlenode.yaml"

# shellcheck source=../../../../utils/_source_params.sh
source "${_SLURM_DIR}/../../../../utils/_source_params.sh"
if [[ -z "${CONFIG_YAML:-}" && -z "${CONFIG_YAML_REL:-}" ]]; then
  export CONFIG_YAML="${_MB_SFT_CONFIG}"
fi

# shellcheck source=../_train_lib.sh
source "${_SLURM_DIR}/_train_lib.sh"
mb_sft_init_launch interactive

mb_sft_train
