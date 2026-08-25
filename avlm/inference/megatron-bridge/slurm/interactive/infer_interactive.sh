#!/usr/bin/env bash
# Run Megatron-Bridge inference inside an interactive Slurm container session.
#
#   bash avlm/inference/megatron-bridge/slurm/interactive/launch_interactive_session.sh
#   bash avlm/inference/megatron-bridge/slurm/interactive/infer_interactive.sh
set -euo pipefail

if [[ "${INSIDE_AVLM_INFERENCE_SESSION:-0}" != "1" ]]; then
  echo "error: start an interactive inference session first:" >&2
  echo "  bash avlm/inference/megatron-bridge/slurm/interactive/launch_interactive_session.sh" >&2
  exit 1
fi

if [[ "${AVLM_INFERENCE_BACKEND:-}" != "megatron_bridge" ]]; then
  echo "error: expected AVLM_INFERENCE_BACKEND=megatron_bridge" >&2
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
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

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
[[ -n "${IS_FSDP_OVERRIDE:-}" ]] && infer_args+=(--is-fsdp "${IS_FSDP_OVERRIDE}")
[[ -n "${FSDP_SHARDING_STRATEGY_OVERRIDE:-}" ]] && infer_args+=(--fsdp-sharding-strategy "${FSDP_SHARDING_STRATEGY_OVERRIDE}")

export MB_BRIDGE_OVERLAY="${MB_BRIDGE_OVERLAY:-${CACHE_DIR}/megatron_bridge_workspace/megatron_bridge_overlay}"
_MB_APPLY_PATCHES="${REPO_ROOT}/avlm/training/megatron-bridge/training_data_processing/bridge_integration/apply_patches.py"
python "${_MB_APPLY_PATCHES}" \
  --overlay-dir "${MB_BRIDGE_OVERLAY}" \
  --recipe-name conversation_jsonl_sft_config

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  avlm/inference/megatron-bridge/entrypoint.py \
  "${infer_args[@]}"
