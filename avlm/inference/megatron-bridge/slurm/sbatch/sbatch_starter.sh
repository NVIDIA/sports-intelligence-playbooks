#!/usr/bin/env bash
# Submit non-interactive Megatron-Bridge inference via Slurm.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
_BACKEND_DIR="$(cd -- "${_SLURM_DIR}/.." && pwd)"
_INFERENCE_DIR="$(cd -- "${_BACKEND_DIR}/.." && pwd)"
SRUN_SCRIPT="${SCRIPT_DIR}/srun.sh"

source "${_INFERENCE_DIR}/common/utils/_inference_lib.sh"
inference_reject_config_args "$@"
inference_env_commit_overrides \
  "${_SLURM_DIR}" \
  "${INFERENCE_COMMON_ENV_OVERRIDE_KEYS[@]}" \
  "${INFERENCE_SBATCH_ENV_OVERRIDE_KEYS[@]}" \
  "${INFERENCE_SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${_SLURM_DIR}"

: "${INFERENCE_CONFIG:?Set INFERENCE_CONFIG}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
: "${num_nodes:?Set num_nodes in launch_local.yaml}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"

inference_export_config_overrides

export CONTAINER_IMAGE CONTAINER_MOUNT_HOST
export INFERENCE_CONFIG MAX_INFERENCE_SAMPLES MAX_INFERENCE_SAMPLES_SEED \
  INFERENCE_NAME_OVERRIDE BASE_OUTPUT_DIR_OVERRIDE DATA_PATH_OVERRIDE \
  MODEL_PATH_OVERRIDE ADAPTER_PATH_OVERRIDE \
  IS_FSDP_OVERRIDE FSDP_SHARDING_STRATEGY_OVERRIDE \
  RESUME_OVERRIDE

if [[ "${SLURM_ACCOUNT_RACE}" == "1" || "${SLURM_ACCOUNT_RACE}" == "true" ]]; then
  slurm_accounts=( "${slurm_accounts_race[@]}" )
else
  slurm_accounts=( "${slurm_accounts_single[@]}" )
  if [[ -n "${SLURM_ACCOUNTS:-}" ]]; then
    # shellcheck disable=SC2206
    slurm_accounts=( ${SLURM_ACCOUNTS} )
  fi
fi

export use_exclusive="${use_exclusive:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"

resolve_batch_partition "${num_nodes}"

INFERENCE_JOB_NAME="$(resolve_inference_job_name)"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGS_DIR="${LOGS_DIR:-${AVLM_ROOT}/inference/logs/${INFERENCE_JOB_NAME}_${RUN_TIMESTAMP}}"
export LOGS_DIR

setup_cache_dir_env
export CONTAINER_MOUNTS="$(batch_container_mounts)"

mkdir -p "${LOGS_DIR}"

submit_one() {
  local account="$1"
  local out_file="$2"
  local job_name="avlm-infer-${INFERENCE_JOB_NAME}-${account}"

  local -a sbatch_cmd=(sbatch)
  if [[ "${use_exclusive}" == "1" ]]; then
    sbatch_cmd+=(--exclusive)
  fi

  sbatch_cmd+=(
    --job-name="${job_name}"
    --nodes="${num_nodes}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --mem="${SLURM_MEM:-0}"
    --time="${SLURM_TIME_LIMIT:-4:00:00}"
    -p "${partition}"
    -A "${account}"
    --dependency=singleton
    --output="${LOGS_DIR}/slurm_%j.out"
    --error="${LOGS_DIR}/slurm_%j.out"
    --export=ALL,TMPDIR=/tmp,TEMP=/tmp,TMP=/tmp
    "${SRUN_SCRIPT}"
  )

  if ! "${sbatch_cmd[@]}" >"${out_file}" 2>&1; then
    cat "${out_file}" >&2
    return 1
  fi

  if [[ "${RACE_QUIET_SUBMIT:-0}" != "1" ]]; then
    cat "${out_file}"
  fi
}

source "${AVLM_UTILS_DIR}/_slurm_account_race.sh"
run_slurm_account_race "${slurm_accounts[@]}"
