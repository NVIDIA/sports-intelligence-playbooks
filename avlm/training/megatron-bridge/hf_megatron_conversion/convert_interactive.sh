#!/usr/bin/env bash
# HF ↔ Megatron conversion inside an interactive Slurm container session.
# Set CONVERSION_DIRECTION in conversion_local.yaml; optional RUN_PARITY_CHECK=1
#
#   bash avlm/training/megatron-bridge/hf_megatron_conversion/launch_interactive_session.sh
#   bash avlm/training/megatron-bridge/hf_megatron_conversion/convert_interactive.sh
set -euo pipefail

if [[ "${INSIDE_INTERACTIVE_SESSION:-0}" != "1" ]]; then
  echo "error: start an interactive session first:" >&2
  echo "  bash avlm/training/megatron-bridge/hf_megatron_conversion/launch_interactive_session.sh" >&2
  exit 1
fi

_MODE_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../../utils/_source_params.sh
source "${_MODE_DIR}/../../../utils/_source_params.sh"
training_cli_apply_arg_overrides "$@"
set -- "${_TRAINING_REMAINING_ARGS[@]}"

# shellcheck source=_convert_lib.sh
source "${_MODE_DIR}/_convert_lib.sh"
training_cli_commit_env_overrides "${_MODE_DIR}" "${MB_BRIDGE_CLI_OVERRIDE_KEYS[@]}"
mb_convert_init_launch interactive

mb_convert_run "$@"
