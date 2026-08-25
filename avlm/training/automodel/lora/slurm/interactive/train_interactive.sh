#!/usr/bin/env bash
# Run VLM LoRA training inside an interactive Slurm container session.
#
#   bash avlm/training/automodel/lora/slurm/interactive/launch_interactive_session.sh
#   bash avlm/training/automodel/lora/slurm/interactive/train_interactive.sh
set -euo pipefail

if [[ "${INSIDE_INTERACTIVE_SESSION:-0}" != "1" ]]; then
  echo "error: start an interactive session first:" >&2
  echo "  bash avlm/training/automodel/lora/slurm/interactive/launch_interactive_session.sh" >&2
  exit 1
fi

_INTERACTIVE_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${_INTERACTIVE_DIR}/.." && pwd)"

# shellcheck source=../../../../utils/_source_params.sh
source "${_SLURM_DIR}/../../../../utils/_source_params.sh"

# shellcheck source=../_train_lib.sh
source "${_SLURM_DIR}/_train_lib.sh"
lora_init_launch "${_SLURM_DIR}" interactive

lora_train
