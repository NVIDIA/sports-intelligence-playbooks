#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit Megatron-Bridge LoRA through Slurm. See avlm/training/megatron-bridge/lora/LORA_GUIDE.MD.
set -euo pipefail

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slurm_dir="$(cd -- "${script_dir}/.." && pwd)"
srun_script="${script_dir}/srun.sh"

# shellcheck source=../../../../utils/_source_params.sh
source "${slurm_dir}/../../../../utils/_source_params.sh"

export CLUSTER_PARAMS="${CLUSTER_PARAMS:-${slurm_dir}/launch_local.yaml}"
[[ "${CLUSTER_PARAMS}" == /* ]] || CLUSTER_PARAMS="${slurm_dir}/${CLUSTER_PARAMS#"${slurm_dir}"/}"
export CLUSTER_PARAMS
training_cli_commit_env_overrides "${slurm_dir}" "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${slurm_dir}"
resolve_avlm_repo_roots_from_mode_dir "${slurm_dir}"

if [[ -z "${CONFIG_YAML:-}" ]]; then
  : "${CONFIG_YAML_REL:?Set CONFIG_YAML or CONFIG_YAML_REL in launch_local.yaml}"
  CONFIG_YAML="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}/${CONFIG_YAML_REL}"
fi
: "${MODEL_NAME:?Set MODEL_NAME in launch_local.yaml}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
export CONTAINER_IMAGE CONTAINER_MOUNT_HOST CACHE_DIR CONFIG_YAML CONFIG_YAML_REL MODEL_NAME

if [[ "${SLURM_ACCOUNT_RACE}" == "1" || "${SLURM_ACCOUNT_RACE}" == "true" ]]; then
  slurm_accounts=( "${slurm_accounts_race[@]}" )
else
  slurm_accounts=( "${slurm_accounts_single[@]}" )
  [[ -n "${SLURM_ACCOUNTS:-}" ]] && slurm_accounts=( ${SLURM_ACCOUNTS} )
fi

export use_exclusive="${use_exclusive:-1}"
: "${num_nodes:?Set num_nodes in launch_local.yaml}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"

resolve_batch_partition "${num_nodes}"

# shellcheck source=../../_prep_bridge_env.sh
source "${slurm_dir}/../../../../utils/_prep_bridge_env.sh"
# shellcheck source=../_train_lib.sh
source "${slurm_dir}/_train_lib.sh"
export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export WANDB_NAME="${WANDB_NAME:-${MODEL_NAME}_${RUN_TIMESTAMP}}"
mb_sft_commit_recipe_cli_overrides
mb_sft_load_recipe_env
mb_validate_parallelism_layout "${num_nodes}" "${GPUS_PER_NODE}"

export GC_EVERY_STEPS="${GC_EVERY_STEPS:-1}"
mb_sft_export_cpu_memory_env
export CKPT_LAYOUT_TAG="${CKPT_LAYOUT_TAG:-n${num_nodes}_pp1_tp${TP}_ep${EP}}"
export OUTPUT_BASE="${OUTPUT_BASE:-${slurm_dir}/outputs}"
export OUTPUT="${OUTPUT:-${OUTPUT_BASE}/${MODEL_NAME}}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT}/wandb}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT}/checkpoints_${CKPT_LAYOUT_TAG}}"
export LOGS_DIR="${LOGS_DIR:-${slurm_dir}/logs/${MODEL_NAME}_${RUN_TIMESTAMP}}"

inference_validation_validate
inference_validation_prepare_wandb_env

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: nodes=${num_nodes} GPUs=${GPUS_PER_NODE} MODEL=${MODEL_NAME} partition=${partition}"
  echo "  CHECKPOINT_DIR=${CHECKPOINT_DIR} LOGS_DIR=${LOGS_DIR}"
  exit 0
fi

setup_cache_dir_env
# Container checkout (default) or Lustre clone (MEGATRON_BRIDGE_GIT_BOOTSTRAP=1).
mb_resolve_bridge_root_login
export CONTAINER_MOUNTS="$(batch_container_mounts)"
mkdir -p "${LOGS_DIR}" "${OUTPUT}" "${CHECKPOINT_DIR}"

_mb_relpath() {
  [[ "$1" == "${REPO_ROOT}/"* ]] && echo "${1#"${REPO_ROOT}/"}" || echo "$1"
}

export SUBMIT_DETACH="${SUBMIT_DETACH:-1}"
if [[ "${SUBMIT_DETACHED:-0}" != "1" && "${SUBMIT_DETACH}" =~ ^(1|true)$ ]]; then
  if [[ "${SLURM_ACCOUNT_RACE}" =~ ^(1|true)$ && ${#slurm_accounts[@]} -gt 1 ]]; then
    submit_log="${LOGS_DIR}/submit.log"
    export SUBMIT_DETACHED=1
    export _AVLM_PIN_INHERITED_ENV=1
    if command -v stdbuf >/dev/null 2>&1; then
      nohup stdbuf -oL -eL bash "${script_dir}/sbatch_starter.sh" >>"${submit_log}" 2>&1 &
    else
      nohup bash "${script_dir}/sbatch_starter.sh" >>"${submit_log}" 2>&1 &
    fi
    echo "${!}" >"${LOGS_DIR}/race_watcher.pid"
    echo "Detached account race → $(_mb_relpath "${LOGS_DIR}")/submit.log  (pid ${!})"
    exit 0
  fi
fi

echo "Megatron-Bridge LoRA submit $(date '+%F %T')"
echo "  model=${MODEL_NAME}  nodes=${num_nodes}×${GPUS_PER_NODE}  GBS=${GLOBAL_BATCH_SIZE}  steps=${MAX_STEPS:-recipe}"
echo "  TP=${TP} EP=${EP}  packing=${USE_SEQUENCE_PACKING}  partition=${partition}"
echo "  logs=$(_mb_relpath "${LOGS_DIR}")/  ckpt=$(_mb_relpath "${CHECKPOINT_DIR}")/"

submit_one() {
  local account="$1" out_file="$2"
  local -a cmd=(sbatch)
  [[ "${use_exclusive}" == "1" ]] && cmd+=(--exclusive)
  cmd+=(
    --job-name="${MODEL_NAME}-${account}"
    --nodes="${num_nodes}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --time="${SLURM_TIME_LIMIT:-4:00:00}"
    -p "${partition}"
    -A "${account}"
    --dependency=singleton
    --output="${LOGS_DIR}/slurm_%j.out"
    --error="${LOGS_DIR}/slurm_%j.out"
    --export=ALL,TMPDIR=/tmp,TEMP=/tmp,TMP=/tmp
    "${srun_script}"
  )
  "${cmd[@]}" >"${out_file}" 2>&1
  [[ "${RACE_QUIET_SUBMIT:-0}" != "1" ]] && cat "${out_file}"
}

# shellcheck source=../../../../utils/_slurm_account_race.sh
source "${slurm_dir}/../../../../utils/_slurm_account_race.sh"
run_slurm_account_race "${slurm_accounts[@]}"
inference_validation_start_sbatch_watcher megatron_bridge mbridge_lora
