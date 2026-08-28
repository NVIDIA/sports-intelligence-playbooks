#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run AutoModel inference inside an interactive Slurm container session.
#
#   bash avlm/inference/automodel/slurm/interactive/launch_interactive_session.sh
#   bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh
set -euo pipefail

if [[ "${INSIDE_AVLM_INFERENCE_SESSION:-0}" != "1" ]]; then
  echo "error: start an interactive inference session first:" >&2
  echo "  bash avlm/inference/automodel/slurm/interactive/launch_interactive_session.sh" >&2
  exit 1
fi

if [[ "${AVLM_INFERENCE_BACKEND:-}" != "automodel" ]]; then
  echo "error: expected AVLM_INFERENCE_BACKEND=automodel" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
_BACKEND_DIR="$(cd -- "${_SLURM_DIR}/.." && pwd)"
_INFERENCE_DIR="$(cd -- "${_BACKEND_DIR}/.." && pwd)"

source "${_INFERENCE_DIR}/common/utils/_inference_lib.sh"
inference_reject_config_args "$@"
inference_env_commit_overrides \
  "${_SLURM_DIR}" \
  "${INFERENCE_COMMON_ENV_OVERRIDE_KEYS[@]}" \
  "${INFERENCE_SBATCH_ENV_OVERRIDE_KEYS[@]}" \
  "${INFERENCE_SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${_SLURM_DIR}"
inference_export_config_overrides

: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
: "${INFERENCE_CONFIG:?Set INFERENCE_CONFIG}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-true}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

# shellcheck source=../../../../utils/_source_params.sh
source "${AVLM_UTILS_DIR}/_source_params.sh"
# Use the container's nemo_automodel source + /opt/venv env (same as training).
resolve_automodel_code_root || {
  echo "error: nemo_automodel resolve failed; use nemo-automodel_26_06+ or set AUTOMODEL_CODE_ROOT" >&2
  exit 1
}
export PYTHONPATH="${AUTOMODEL_CODE_ROOT}:${PYTHONPATH:-}"
ensure_inference_vlm_deps || echo "[WARN] VLM python deps install skipped or failed" >&2

cd "${REPO_ROOT}"

infer_args=(
  --config "${INFERENCE_CONFIG}"
)

[[ -n "${MAX_INFERENCE_SAMPLES:-}" ]] && infer_args+=(--max-inference-samples "${MAX_INFERENCE_SAMPLES}")
[[ -n "${MAX_INFERENCE_SAMPLES_SEED:-}" ]] && infer_args+=(--max-inference-samples-seed "${MAX_INFERENCE_SAMPLES_SEED}")
[[ -n "${INFERENCE_NAME_OVERRIDE:-}" ]] && infer_args+=(--inference-name "${INFERENCE_NAME_OVERRIDE}")
[[ -n "${BASE_OUTPUT_DIR_OVERRIDE:-}" ]] && infer_args+=(--base-output-dir "${BASE_OUTPUT_DIR_OVERRIDE}")
[[ -n "${DATA_PATH_OVERRIDE:-}" ]] && infer_args+=(--data-path "${DATA_PATH_OVERRIDE}")
[[ -n "${MODEL_PATH_OVERRIDE:-}" && "${MODEL_PATH_OVERRIDE}" != "null" ]] && infer_args+=(--model-path "${MODEL_PATH_OVERRIDE}")
[[ -n "${ADAPTER_PATH_OVERRIDE:-}" && "${ADAPTER_PATH_OVERRIDE}" != "null" ]] && infer_args+=(--adapter-path "${ADAPTER_PATH_OVERRIDE}")
[[ "${RESUME_OVERRIDE:-0}" == "1" || "${RESUME_OVERRIDE:-0}" == "true" || "${RESUME_OVERRIDE:-0}" == "True" || "${RESUME_OVERRIDE:-0}" == "TRUE" ]] && infer_args+=(--resume)

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  avlm/inference/common/run_inference.py \
  "${infer_args[@]}"
