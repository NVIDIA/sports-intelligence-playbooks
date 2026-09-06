#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Submit AutoModel LoRA through Slurm. See avlm/training/automodel/lora/LORA_GUIDE.MD.
set -euo pipefail

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slurm_dir="$(cd -- "${script_dir}/.." && pwd)"
srun_script="${script_dir}/srun.sh"

# shellcheck source=../../../../utils/_source_params.sh
source "${slurm_dir}/../../../../utils/_source_params.sh"

# --- Launch config: default launch_local.yaml. Override: CLUSTER_PARAMS=... ---
export CLUSTER_PARAMS="${CLUSTER_PARAMS:-${slurm_dir}/launch_local.yaml}"
[[ "${CLUSTER_PARAMS}" == /* ]] || CLUSTER_PARAMS="${slurm_dir}/${CLUSTER_PARAMS#"${slurm_dir}"/}"
export CLUSTER_PARAMS
training_cli_commit_env_overrides "${slurm_dir}" "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${slurm_dir}"

# Repo root (standalone repo root) — mounted into the container.
resolve_avlm_repo_roots_from_mode_dir "${slurm_dir}"

training_require_config_path
if [[ -z "${CONFIG_YAML:-}" ]]; then
  CONFIG_YAML="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}/${CONFIG_YAML_REL}"
fi
: "${MODEL_NAME:?Set MODEL_NAME in launch_local.yaml}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
export CONTAINER_IMAGE CONTAINER_MOUNT_HOST CACHE_DIR CONFIG_YAML CONFIG_YAML_REL MODEL_NAME

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

: "${num_nodes:?Set num_nodes in launch_local.yaml}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"

# Partitions from launch_local.yaml by num_nodes. Override: partition=polar4 bash ...
resolve_batch_partition "${num_nodes}"

# Recipe YAML (nemotron config). Launch-level overrides: num_nodes, MODEL_NAME, MAX_STEPS, PACK_SIZE, ...
# shellcheck source=../_train_lib.sh
source "${slurm_dir}/_train_lib.sh"
training_cli_commit_recipe_overrides
lora_load_recipe_env "${CONFIG_YAML}"
lora_validate_parallelism_layout "${num_nodes}" "${GPUS_PER_NODE}"
lora_export_cpu_memory_env

export CKPT_LAYOUT_TAG="${CKPT_LAYOUT_TAG:-n${num_nodes}_tp${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}_pp${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}_ep${EP}}"

export OUTPUT_BASE="${OUTPUT_BASE:-${slurm_dir}/outputs}"
export OUTPUT="${OUTPUT:-${OUTPUT_BASE}/${MODEL_NAME}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT}/checkpoints_${CKPT_LAYOUT_TAG}}"
export RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export LOGS_DIR="${LOGS_DIR:-${slurm_dir}/logs/${MODEL_NAME}_${RUN_TIMESTAMP}}"

export WANDB_NAME="${WANDB_NAME:-${MODEL_NAME}_${RUN_TIMESTAMP}}"

inference_validation_validate
inference_validation_prepare_wandb_env

# W&B run name is passed via env (WANDB_NAME); srun.sh builds CLI overrides in _train_lib.sh.

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat <<EOF
DRY_RUN=1 — resolved sbatch config (no directories created, no submit)

  num_nodes         ${num_nodes}
  GPUS_PER_NODE     ${GPUS_PER_NODE}
  NPROC_PER_NODE    ${NPROC_PER_NODE}
  MODEL_NAME        ${MODEL_NAME}
  CONFIG_YAML       ${CONFIG_YAML}
  MAX_STEPS         ${MAX_STEPS:-<from recipe>}
  PACK_SIZE         ${PACK_SIZE:-<from recipe>}
  partition         ${partition}
  SLURM_TIME_LIMIT  ${SLURM_TIME_LIMIT:-4:00:00}
  use_exclusive     ${use_exclusive}
  SLURM_ACCOUNT_RACE ${SLURM_ACCOUNT_RACE}
  CKPT_LAYOUT_TAG   ${CKPT_LAYOUT_TAG}
  CHECKPOINT_DIR    ${CHECKPOINT_DIR}
  GLOBAL_BATCH_SIZE ${GLOBAL_BATCH_SIZE:-<from recipe>}
  DISPATCHER        ${DISPATCHER:-<from recipe>}
  VAL_SAMPLE_RATIO  ${VAL_SAMPLE_RATIO:-<from recipe>}
  GC_EVERY_STEPS    ${GC_EVERY_STEPS:-<from recipe>}

EOF
  exit 0
fi

setup_cache_dir_env
export CONTAINER_MOUNTS="$(batch_container_mounts)"

mkdir -p "${LOGS_DIR}" "${OUTPUT}" "${CHECKPOINT_DIR}"

_lora_relpath() {
  local p="$1"
  if [[ "${p}" == "${REPO_ROOT}/"* ]]; then
    echo "${p#"${REPO_ROOT}/"}"
  else
    echo "${p}"
  fi
}

_lora_print_detach_notice() {
  local pid="$1"
  local submit_log="$2"
  local rel_logs rel_submit steps_suffix=""
  local _tp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}"
  local _pp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}"
  local _cp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[cp]}"
  rel_logs="$(_lora_relpath "${LOGS_DIR}")"
  rel_submit="$(_lora_relpath "${submit_log}")"
  if [[ -n "${MAX_STEPS:-}" ]]; then
    steps_suffix="  |  steps ${MAX_STEPS}"
  fi

  if [[ "${USE_SEQUENCE_PACKING}" == "1" ]]; then
    _lora_data_mode="pack ${PACK_SIZE}"
  else
    _lora_data_mode="collate ${COLLATE_MAX_LENGTH} / ${MAX_VIDEO_FRAMES}fr"
  fi

  cat <<EOF

════════════════════════════════════════════════════════════
 LoRA batch submit (detached — account race in background)
════════════════════════════════════════════════════════════
 Model       ${MODEL_NAME}
 Cluster     ${num_nodes} nodes × ${GPUS_PER_NODE} GPUs  |  GBS ${GLOBAL_BATCH_SIZE}${steps_suffix}
 Parallel    TP=${_tp}  PP=${_pp}  CP=${_cp}  EP=${EP}  |  ${_lora_data_mode}
 Accounts    ${slurm_accounts[*]}

 Run dir     ${rel_logs}/
 Watcher     pid ${pid}
 Submit log  ${rel_submit}

 Follow      tail -f ${rel_submit}
 Cancel      bash avlm/utils/cancel_slurm_race.sh ${rel_logs}
 Queue       squeue -u \$USER
════════════════════════════════════════════════════════════
EOF
}

_lora_print_submit_log_header() {
  local rel_logs rel_output rel_ckpt
  local _tp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}"
  local _pp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}"
  local _cp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[cp]}"
  rel_logs="$(_lora_relpath "${LOGS_DIR}")"
  rel_output="$(_lora_relpath "${OUTPUT}")"
  rel_ckpt="$(_lora_relpath "${CHECKPOINT_DIR}")"

  echo "LoRA submit log — $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo ""
  echo "[config]"
  echo "  model              ${MODEL_NAME}"
  echo "  nodes              ${num_nodes} × ${GPUS_PER_NODE} GPUs"
  echo "  global_batch       ${GLOBAL_BATCH_SIZE}"
  if [[ -n "${MAX_STEPS:-}" ]]; then
    echo "  max_steps          ${MAX_STEPS} (override)"
  fi
  if [[ "${USE_SEQUENCE_PACKING}" == "1" ]]; then
    echo "  dataloader.use_sequence_packing true"
    echo "  pack_size          ${PACK_SIZE}"
  else
    echo "  dataloader.use_sequence_packing false"
    echo "  collate_max_length ${COLLATE_MAX_LENGTH}"
    echo "  max_video_frames   ${MAX_VIDEO_FRAMES}"
  fi
  echo "  gc_every_steps     ${GC_EVERY_STEPS}"
  echo "  wandb_mode         ${WANDB_MODE:-<from recipe>}"
  echo "  ckpt_every_steps   ${CKPT_EVERY_STEPS}"
  echo "  val_every_steps    ${VAL_EVERY_STEPS}"
  echo "  dispatcher         ${DISPATCHER}"
  echo "  val_sample_ratio   ${VAL_SAMPLE_RATIO}"
  echo "  parallelism        TP=${_tp} PP=${_pp} CP=${_cp} EP=${EP}"
  echo "  partition          ${partition}"
  echo "  exclusive          ${use_exclusive}"
  echo "  account_race       ${SLURM_ACCOUNT_RACE}"
  echo "  race_accounts      ${slurm_accounts[*]}"
  echo "  cluster_params     ${CLUSTER_PARAMS}"
  echo "  config_yaml        ${CONFIG_YAML}"
  echo "  wandb_name         ${WANDB_NAME} (entity/project from recipe YAML)"
  echo ""
  echo "[paths]"
  echo "  repo_root          $(_lora_relpath "${REPO_ROOT}")"
  echo "  logs               ${rel_logs}/"
  echo "  output             ${rel_output}/"
  echo "  checkpoints        ${rel_ckpt}/"
  echo "  container          ${CONTAINER_IMAGE}"
  echo "  automodel_code     ${AUTOMODEL_CODE_ROOT:-/opt/Automodel (container)}"
  echo ""
}

export SUBMIT_DETACH="${SUBMIT_DETACH:-1}"
if [[ "${SUBMIT_DETACHED:-0}" != "1" ]]; then
  if [[ "${SUBMIT_DETACH}" == "1" || "${SUBMIT_DETACH}" == "true" ]]; then
    if [[ "${SLURM_ACCOUNT_RACE}" == "1" || "${SLURM_ACCOUNT_RACE}" == "true" ]] && ((${#slurm_accounts[@]} > 1)); then
      submit_log="${LOGS_DIR}/submit.log"
      export SUBMIT_DETACHED=1
      export _AVLM_PIN_INHERITED_ENV=1
      if command -v stdbuf >/dev/null 2>&1; then
        _submit_bash=(stdbuf -oL -eL bash)
      else
        _submit_bash=(bash)
      fi
      nohup "${_submit_bash[@]}" "${script_dir}/sbatch_starter.sh" >>"${submit_log}" 2>&1 &
      _submit_pid=$!
      echo "${_submit_pid}" >"${LOGS_DIR}/race_watcher.pid"
      _lora_print_detach_notice "${_submit_pid}" "${submit_log}"
      exit 0
    fi
  fi
fi

_lora_print_submit_log_header

submit_one() {
  local account="$1"
  local out_file="$2"
  local job_name="${MODEL_NAME}-${account}"

  local -a sbatch_cmd=(sbatch)
  if [[ "${use_exclusive}" == "1" ]]; then
    sbatch_cmd+=(--exclusive)
  fi
  sbatch_cmd+=(
    --job-name="${job_name}"
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
  "${sbatch_cmd[@]}" >"${out_file}" 2>&1
  if [[ "${RACE_QUIET_SUBMIT:-0}" != "1" ]]; then
    cat "${out_file}"
  fi
}

# shellcheck source=../../../../utils/_slurm_account_race.sh
source "${slurm_dir}/../../../../utils/_slurm_account_race.sh"
run_slurm_account_race "${slurm_accounts[@]}"
inference_validation_start_sbatch_watcher automodel automodel_lora
