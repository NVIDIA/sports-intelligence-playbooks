#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# PTQ entrypoint: pick configs/quantization_*.yaml, run quant + export.
# Usage (repo root, inside GPU container):
# Default recipe: configs/quantization_hf_tennis.yaml
#   bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh [hf] [KEY=val ...]
#   quantize.sh hf QUANT_FORMAT=nvfp4
# Base reasoning recipe: QUANTIZATION_PARAMS=.../configs/quantization_hf.yaml quantize.sh
# Override file: QUANTIZATION_PARAMS=/path/to.yaml quantize.sh
set -euo pipefail

_MODE_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/_quantize_lib.sh
source "${_MODE_DIR}/lib/_quantize_lib.sh"
# shellcheck source=../../../../utils/_source_params.sh
source "${_AVLM_ROOT}/utils/_source_params.sh"

mb_quantize_peel_recipe_arg "$@"
training_cli_apply_arg_overrides "${_MB_QUANT_RECIPE_REMAINING_ARGS[@]}"
set -- "${_TRAINING_REMAINING_ARGS[@]}"
mb_quant_cli_commit_overrides "${_MODE_DIR}" "${_MB_QUANT_RECIPE_ARG:-hf}"
mb_quantize_resolve_params_file "${_MODE_DIR}" "${_MB_QUANT_RECIPE_ARG:-hf}"
mb_quantize_init_launch
mb_quantize_run "$@"
