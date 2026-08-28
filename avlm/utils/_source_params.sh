# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source per-mode cluster parameters (SFT or LoRA).
# Usage: source_cluster_params "/path/to/training/automodel/sft/slurm"
#
# Loads CLUSTER_PARAMS if set (see sbatch/sbatch_starter.sh / interactive/train_interactive.sh), else
# launch_local.yaml in the mode directory. launch.yaml is a repo template only — copy it to
# launch_local.yaml (gitignored) and edit; it is never used at runtime.
#
# shellcheck shell=bash

_AVLM_UTILS_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Override precedence: (1) command line, (2) YAML, (3) inherited shell (~/.bashrc).
if [[ -z "${_AVLM_PARAMS_LOADED:-}" ]]; then
  declare -g -A _AVLM_CLI_OVERRIDES=()
  declare -g -A _AVLM_CLI_FROM_ARGS=()
  declare -g -A _AVLM_BASHRC_FALLBACK=()
  declare -g -a _AVLM_LAUNCH_CLI_KEYS=()
  declare -g -a _AVLM_REMAINING_ARGS=()
  declare -g -r -A AUTOMODEL_NEMOTRON_OMNI_PARALLELISM=(
    [tp]=1
    [pp]=1
    [cp]=1
  )
  _AVLM_PARAMS_LOADED=1
  # Backward-compatible names for training launch scripts. Inference shares
  # the implementation but deliberately exposes environment-only terminology.
  declare -gn _TRAINING_CLI_OVERRIDES=_AVLM_CLI_OVERRIDES
  declare -gn _TRAINING_CLI_FROM_ARGS=_AVLM_CLI_FROM_ARGS
  declare -gn _TRAINING_BASHRC_FALLBACK=_AVLM_BASHRC_FALLBACK
  declare -gn _TRAINING_REMAINING_ARGS=_AVLM_REMAINING_ARGS
  declare -gn _INFERENCE_ENV_OVERRIDES=_AVLM_CLI_OVERRIDES
  declare -gn _INFERENCE_ENV_FALLBACK=_AVLM_BASHRC_FALLBACK
fi

# Parse KEY=VAL tokens from script args (always command-line tier).
# Remaining args are written to _TRAINING_REMAINING_ARGS.
training_cli_apply_arg_overrides() {
  local arg key val
  _AVLM_REMAINING_ARGS=()
  for arg in "$@"; do
    if [[ "${arg}" == *=* && "${arg}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      export "${key}=${val}"
      _AVLM_CLI_OVERRIDES["${key}"]="${val}"
      _AVLM_CLI_FROM_ARGS["${key}"]=1
    else
      _AVLM_REMAINING_ARGS+=("${arg}")
    fi
  done
}

# Parent inherited exports only (/proc/PPID/environ). Do not parse ps args: a wrapper
# `bash -c 'VAR=val bash script.sh'` embeds VAR=val in the parent's cmdline even though
# the assignment applies to the child, which would misclassify prefix overrides as bashrc.
_parent_env_has_key() {
  local key="$1"
  [[ -r "/proc/${PPID}/environ" ]] || return 1
  tr '\0' '\n' < "/proc/${PPID}/environ" | grep -q "^${key}="
}

_parent_env_value() {
  local key="$1" line
  [[ -r "/proc/${PPID}/environ" ]] || return 1
  line="$(tr '\0' '\n' < "/proc/${PPID}/environ" | grep -a "^${key}=" | head -1 || true)"
  [[ -n "${line}" ]] || return 1
  printf '%s' "${line#"${key}="}"
}

# VAR=val bash script.sh — tier-1 only when the key was not inherited unchanged from
# the parent shell. Prefix assignments appear in the child but not in the parent's
# environ (or with a different value than the parent export).
training_cli_detect_prefix_overrides() {
  local key parent_val
  for key in "$@"; do
    [[ -n "${_AVLM_CLI_OVERRIDES[${key}]+x}" ]] && continue
    [[ -z "${!key+x}" ]] && continue
    if _parent_env_has_key "${key}"; then
      parent_val="$(_parent_env_value "${key}" 2>/dev/null || true)"
      if [[ "${!key}" == "${parent_val}" ]]; then
        continue
      fi
    fi
    _AVLM_CLI_OVERRIDES["${key}"]="${!key}"
  done
}

# Inherited shell exports unchanged from the parent (typical ~/.bashrc or export before bash).
training_cli_record_bashrc_fallback() {
  local key parent_val
  for key in "$@"; do
    [[ -n "${_AVLM_CLI_OVERRIDES[${key}]+x}" ]] && continue
    [[ -z "${!key+x}" ]] && continue
    if ! _parent_env_has_key "${key}"; then
      continue
    fi
    parent_val="$(_parent_env_value "${key}" 2>/dev/null || true)"
    if [[ "${!key}" == "${parent_val}" ]]; then
      _AVLM_BASHRC_FALLBACK["${key}"]="${!key}"
    fi
  done
}

training_cli_prepare_env_overrides() {
  local _mode_dir="$1"
  shift
  _AVLM_LAUNCH_CLI_KEYS=( "$@" )
  training_cli_detect_prefix_overrides "$@"
  training_cli_record_bashrc_fallback "$@"
}

training_cli_prepare_recipe_overrides() {
  training_cli_detect_prefix_overrides "${RECIPE_CLI_OVERRIDE_KEYS[@]}"
  training_cli_record_bashrc_fallback "${RECIPE_CLI_OVERRIDE_KEYS[@]}"
}

# Backward-compatible aliases used by entry scripts.
training_cli_commit_env_overrides() {
  training_cli_prepare_env_overrides "$@"
}
training_cli_commit_recipe_overrides() {
  training_cli_prepare_recipe_overrides
}

training_require_config_path() {
  if [[ -z "${CONFIG_YAML:-}" && -z "${CONFIG_YAML_REL:-}" ]]; then
    echo "error: set CONFIG_YAML or CONFIG_YAML_REL" >&2
    return 1
  fi
}

training_normalize_bool() {
  local key="${1:?boolean variable name required}"
  local value="${!key:-}"
  case "${value,,}" in
    1|true) value=1 ;;
    0|false) value=0 ;;
    *)
      echo "error: ${key} must be one of: 0, 1, true, false (got '${value}')" >&2
      return 1
      ;;
  esac
  printf -v "${key}" '%s' "${value}"
  export "${key}"
}

# Slurm jobs re-load recipe YAML on compute; keep submit-time recipe CLI overrides.
training_cli_pin_slurm_recipe_env() {
  local key
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    return 0
  fi
  for key in "${RECIPE_CLI_OVERRIDE_KEYS[@]}"; do
    if [[ -n "${!key+x}" && -z "${_AVLM_CLI_OVERRIDES[${key}]+x}" ]]; then
      _AVLM_CLI_OVERRIDES["${key}"]="${!key}"
    fi
  done
}

# Preserve launch overrides when a child launcher or Slurm job reloads YAML.
training_cli_pin_slurm_env() {
  local key
  # Interactive training can identify true command-prefix overrides from its
  # parent shell. Do not promote ordinary inherited values to CLI overrides.
  if [[ "${INSIDE_INTERACTIVE_SESSION:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -z "${SLURM_JOB_ID:-}" && "${_AVLM_PIN_INHERITED_ENV:-0}" != "1" ]]; then
    return 0
  fi
  for key in "$@"; do
    if [[ -n "${!key+x}" && -z "${_AVLM_CLI_OVERRIDES[${key}]+x}" ]]; then
      _AVLM_CLI_OVERRIDES["${key}"]="${!key}"
    fi
  done
}

# Apply YAML exports: CLI wins, then YAML, then bashrc fallback for unset keys.
apply_yaml_exports_preserving_cli() {
  local yaml_exports="$1" key
  for key in "${!_AVLM_CLI_OVERRIDES[@]}" "${!_AVLM_BASHRC_FALLBACK[@]}"; do
    unset "${key}" 2>/dev/null || true
  done
  # shellcheck disable=SC2086
  eval "${yaml_exports}"
  for key in "${!_AVLM_CLI_OVERRIDES[@]}"; do
    export "${key}=${_AVLM_CLI_OVERRIDES[${key}]}"
  done
  for key in "${!_AVLM_BASHRC_FALLBACK[@]}"; do
    [[ -n "${_AVLM_CLI_OVERRIDES[${key}]+x}" ]] && continue
    [[ -n "${!key+x}" ]] && continue
    export "${key}=${_AVLM_BASHRC_FALLBACK[${key}]}"
  done
}

# Keys read from launch_local.yaml that sbatch_starter accepts as CLI overrides.
# Keep in sync with test_cli_overrides.sh expectations.
SBATCH_CLI_OVERRIDE_KEYS=(
  num_nodes MODEL_NAME CONFIG_YAML_REL GPUS_PER_NODE NPROC_PER_NODE MAX_STEPS PACK_SIZE SEQ_LENGTH
  CACHE_DIR AUTOMODEL_CODE_ROOT
  partition use_exclusive SLURM_ACCOUNTS SLURM_ACCOUNT_RACE SLURM_TIME_LIMIT CHECKPOINT_DIR
  EXIT_MINS_BEFORE_LIMIT
  OUTPUT_BASE LOGS_DIR GLOBAL_BATCH_SIZE DISPATCHER VAL_SAMPLE_RATIO VAL_MAX_PACKS GC_EVERY_STEPS
  CKPT_EVERY_STEPS VAL_EVERY_STEPS WANDB_NAME WANDB_MODE
  AVLM_HF_RESIZE
  USE_SEQUENCE_PACKING COLLATE_MAX_LENGTH MAX_VIDEO_FRAMES CLUSTER_PARAMS
  RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS RECOMPUTE_MODULES
  FINE_GRAINED_ACTIVATION_OFFLOADING OFFLOAD_MODULES
  OPTIMIZER_CPU_OFFLOAD OPTIMIZER_OFFLOAD_FRACTION OVERLAP_CPU_OPTIMIZER_D2H_H2D USE_PRECISION_AWARE_OPTIMIZER
  BF16_OPTIMIZER_STATES USE_MEGATRON_FSDP
  PYTORCH_CUDA_ALLOC_CONF NCCL_NVLS_ENABLE TORCH_NCCL_AVOID_RECORD_STREAMS CUDA_DEVICE_MAX_CONNECTIONS
  INFERENCE_VALIDATION_ENABLED INFERENCE_VALIDATION_EVERY_STEPS INFERENCE_VALIDATION_INFERENCE_CONFIG
  INFERENCE_VALIDATION_DATA_PATH INFERENCE_VALIDATION_MAX_SAMPLES INFERENCE_VALIDATION_MAX_SAMPLES_SEED INFERENCE_VALIDATION_CLUSTER_PARAMS
  INFERENCE_VALIDATION_NUM_NODES INFERENCE_VALIDATION_GPUS_PER_NODE INFERENCE_VALIDATION_NPROC_PER_NODE
  VLM_SCORER_CONFIG
)

# Cluster settings supported through inline environment assignments and launch YAML,
# but not through trailing KEY=VALUE arguments.
SBATCH_ENV_ONLY_KEYS=(
  CONTAINER_IMAGE CONTAINER_MOUNT_HOST
  PARTITIONS FEW_NODES_PARTITIONS SINGLE_NODE_PARTITION SLURM_ACCOUNTS_RACE
)

GENERIC_CLI_OVERRIDE_KEYS=(
  MODEL_NAME CONFIG_YAML_REL NUM_GPUS NPROC_PER_NODE MAX_STEPS PACK_SIZE SEQ_LENGTH
  CACHE_DIR AUTOMODEL_CODE_ROOT
  SLURM_TIME_LIMIT GLOBAL_BATCH_SIZE DISPATCHER VAL_SAMPLE_RATIO VAL_MAX_PACKS
  GC_EVERY_STEPS CKPT_EVERY_STEPS VAL_EVERY_STEPS WANDB_NAME WANDB_MODE
  USE_SEQUENCE_PACKING COLLATE_MAX_LENGTH MAX_VIDEO_FRAMES CLUSTER_PARAMS
  BF16_OPTIMIZER_STATES USE_MEGATRON_FSDP
  PYTORCH_CUDA_ALLOC_CONF NCCL_NVLS_ENABLE TORCH_NCCL_AVOID_RECORD_STREAMS CUDA_DEVICE_MAX_CONNECTIONS
)

# Keys loaded from NeMo recipe YAML that may be overridden from the CLI.
RECIPE_CLI_OVERRIDE_KEYS=(
  GLOBAL_BATCH_SIZE DISPATCHER PACK_SIZE SEQ_LENGTH GC_EVERY_STEPS CKPT_EVERY_STEPS VAL_EVERY_STEPS
  VAL_SAMPLE_RATIO WANDB_MODE USE_SEQUENCE_PACKING COLLATE_MAX_LENGTH MAX_VIDEO_FRAMES
  INFERENCE_VALIDATION_ENABLED INFERENCE_VALIDATION_EVERY_STEPS INFERENCE_VALIDATION_INFERENCE_CONFIG
  INFERENCE_VALIDATION_DATA_PATH INFERENCE_VALIDATION_MAX_SAMPLES INFERENCE_VALIDATION_MAX_SAMPLES_SEED
  INFERENCE_VALIDATION_NUM_NODES
  VLM_SCORER_CONFIG
)

INFERENCE_VALIDATION_YAML_SELECTOR_ARGS=(
  INFERENCE_VALIDATION_ENABLED=inference_validation.enabled
  INFERENCE_VALIDATION_EVERY_STEPS=inference_validation.every_steps
  INFERENCE_VALIDATION_INFERENCE_CONFIG=inference_validation.inference_config
  INFERENCE_VALIDATION_DATA_PATH=inference_validation.data_path
  INFERENCE_VALIDATION_MAX_SAMPLES=inference_validation.max_samples
  INFERENCE_VALIDATION_MAX_SAMPLES_SEED=inference_validation.inference_max_samples_seed
  INFERENCE_VALIDATION_CLUSTER_PARAMS=inference_validation.cluster_params
  INFERENCE_VALIDATION_NUM_NODES=inference_validation.num_nodes
  INFERENCE_VALIDATION_GPUS_PER_NODE=inference_validation.gpus_per_node
  INFERENCE_VALIDATION_NPROC_PER_NODE=inference_validation.nproc_per_node
  VLM_SCORER_CONFIG=inference_validation.vlm_scorer_config
)

# Generic single-node training: torchrun world size from visible GPUs.
# Override: NUM_GPUS=4 bash generic/train_sft.sh  (alias: NPROC_PER_NODE=4)
generic_resolve_num_gpus() {
  local _count=0 _cvd _src="" _dev
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    _count="${NPROC_PER_NODE}"
    _src="NPROC_PER_NODE"
  elif [[ -n "${NUM_GPUS:-}" ]]; then
    _count="${NUM_GPUS}"
    _src="NUM_GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _cvd="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    if [[ -n "${_cvd}" ]]; then
      IFS=',' read -ra _dev <<< "${_cvd}"
      _count="${#_dev[@]}"
    fi
    _src="CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    _count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    _src="nvidia-smi"
  fi
  if [[ ! "${_count}" =~ ^[0-9]+$ ]] || (( _count < 1 )); then
    echo "error: need at least 1 visible GPU for generic training (set NUM_GPUS=... or NPROC_PER_NODE=...)" >&2
    [[ -n "${_src}" ]] && echo "hint: resolved ${_count} from ${_src}" >&2
    return 1
  fi
  export NPROC_PER_NODE="${_count}"
  echo "info: using ${NPROC_PER_NODE} GPU rank(s) (${_src:-auto-detect})" >&2
}

# Resolve REPO_ROOT and avlm from avlm package root (…/avlm).
# Standalone repo: REPO_ROOT is parent of avlm/.
# Nested in Automodel: REPO_ROOT is Automodel root when ../nemo_automodel exists.
resolve_avlm_repo_roots_from_avlm_root() {
  local avlm_root="$1"
  _AVLM_ROOT="$(cd -- "${avlm_root}" && pwd)"
  _SPORTS_INTEL_ROOT="$(cd -- "${_AVLM_ROOT}/.." && pwd)"
  if [[ -d "${_SPORTS_INTEL_ROOT}/../nemo_automodel" ]]; then
    _REPO_ROOT="$(cd -- "${_SPORTS_INTEL_ROOT}/.." && pwd)"
  else
    _REPO_ROOT="${_SPORTS_INTEL_ROOT}"
  fi
  export REPO_ROOT="${REPO_ROOT:-${_REPO_ROOT}}"
}

# Walk up from a mode directory until avlm/utils/_source_params.sh is found.
resolve_avlm_repo_roots_from_mode_dir() {
  local mode_dir="$1"
  local d="${mode_dir}"
  while [[ "${d}" != "/" ]]; do
    if [[ -f "${d}/utils/_source_params.sh" ]]; then
      resolve_avlm_repo_roots_from_avlm_root "${d}"
      return 0
    fi
    d="$(dirname -- "${d}")"
  done
  echo "error: avlm/utils/_source_params.sh not found above ${mode_dir}" >&2
  return 1
}

_pick_cluster_params_file() {
  local d="$1" f=""
  if [[ -n "${CLUSTER_PARAMS:-}" && -f "${CLUSTER_PARAMS}" ]]; then
    printf '%s\n' "${CLUSTER_PARAMS}"
    return 0
  fi
  if [[ -n "${CLUSTER_PARAMS:-}" ]]; then
    echo "warn: CLUSTER_PARAMS not found: ${CLUSTER_PARAMS}; trying launch_local.yaml" >&2
  fi
  if [[ -f "${d}/launch_local.yaml" ]]; then
    printf '%s\n' "${d}/launch_local.yaml"
    return 0
  fi
  return 1
}

_pick_conversion_params_file() {
  local d="$1"
  if [[ -n "${CONVERSION_PARAMS:-}" && -f "${CONVERSION_PARAMS}" ]]; then
    printf '%s\n' "${CONVERSION_PARAMS}"
    return 0
  fi
  if [[ -n "${CONVERSION_PARAMS:-}" ]]; then
    echo "warn: CONVERSION_PARAMS not found: ${CONVERSION_PARAMS}; trying conversion_local.yaml" >&2
  fi
  if [[ -f "${d}/conversion_local.yaml" ]]; then
    printf '%s\n' "${d}/conversion_local.yaml"
    return 0
  fi
  return 1
}

_export_yaml_params() {
  local f="$1" _py yaml_exports
  [[ "${f}" == *.yaml || "${f}" == *.yml ]] || {
    echo "error: params must be YAML: ${f}" >&2
    return 1
  }
  _py=/opt/venv/bin/python3
  [[ -x "${_py}" ]] || _py=$(command -v python3)
  yaml_exports="$("${_py}" "${_AVLM_UTILS_DIR}/_load_params_yaml.py" "${f}")"
  if ((${#_AVLM_LAUNCH_CLI_KEYS[@]} > 0)); then
    training_cli_pin_slurm_env "${_AVLM_LAUNCH_CLI_KEYS[@]}"
  else
    training_cli_pin_slurm_env "${SBATCH_CLI_OVERRIDE_KEYS[@]}"
  fi
  apply_yaml_exports_preserving_cli "${yaml_exports}"
}

source_conversion_params() {
  local mode_dir="$1" f
  f="$(_pick_conversion_params_file "${mode_dir}")" || {
    echo "error: missing ${mode_dir}/conversion_local.yaml (copy conversion.yaml → conversion_local.yaml and edit)" >&2
    return 1
  }
  echo "info: conversion params from ${f}" >&2
  _export_yaml_params "${f}"
}

source_cluster_params() {
  local mode_dir="$1" f _py yaml_exports key
  f="$(_pick_cluster_params_file "${mode_dir}")" || {
    echo "error: missing ${mode_dir}/launch_local.yaml (copy launch.yaml → launch_local.yaml and edit)" >&2
    return 1
  }
  echo "info: cluster params from ${f}" >&2
  _export_yaml_params "${f}"

  slurm_accounts_single=( "${slurm_accounts_single[@]+"${slurm_accounts_single[@]}"}" )
  slurm_accounts_race=( "${slurm_accounts_race[@]+"${slurm_accounts_race[@]}"}" )
  if [[ ${#slurm_accounts_single[@]} -eq 0 && -n "${SLURM_ACCOUNTS:-}" ]]; then
    # shellcheck disable=SC2206
    slurm_accounts_single=( ${SLURM_ACCOUNTS} )
  fi
  if [[ ${#slurm_accounts_race[@]} -eq 0 && -n "${SLURM_ACCOUNTS_RACE:-}" ]]; then
    # shellcheck disable=SC2206
    slurm_accounts_race=( ${SLURM_ACCOUNTS_RACE} )
  fi
}

# Resolve WANDB_MODE: CLI override > recipe wandb.mode > inherited shell export > online.
# If the resolved mode is online but WANDB_API_KEY is unset, fall back to offline.
finalize_wandb_mode_env() {
  local recipe_yaml="${1:-}"
  local mode="" yaml_mode="" _py

  if [[ -n "${_AVLM_CLI_OVERRIDES[WANDB_MODE]+x}" ]]; then
    mode="${_AVLM_CLI_OVERRIDES[WANDB_MODE]}"
  elif [[ -n "${recipe_yaml}" && -f "${recipe_yaml}" ]]; then
    _py=/opt/venv/bin/python3
    [[ -x "${_py}" ]] || _py=$(command -v python3)
    yaml_mode="$("${_py}" "${_AVLM_UTILS_DIR}/_load_params_yaml.py" "${recipe_yaml}" WANDB_MODE=wandb.mode \
      | sed -n 's/^export WANDB_MODE="\(.*\)"$/\1/p')"
    mode="${yaml_mode:-online}"
  elif [[ -n "${_AVLM_BASHRC_FALLBACK[WANDB_MODE]+x}" ]]; then
    mode="${_AVLM_BASHRC_FALLBACK[WANDB_MODE]}"
  else
    mode=online
  fi

  case "${mode}" in
    online|offline|disabled) ;;
    *)
      echo "error: invalid wandb.mode=${mode}; use online, offline, or disabled" >&2
      return 1
      ;;
  esac
  if [[ "${mode}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE=offline
    echo "info: WANDB_API_KEY unset; falling back to WANDB_MODE=offline" >&2
    return 0
  fi
  export WANDB_MODE="${mode}"
}

inference_validation_apply_defaults() {
  export INFERENCE_VALIDATION_ENABLED="${INFERENCE_VALIDATION_ENABLED:-0}"
}

inference_validation_enabled() {
  inference_validation_apply_defaults
  case "${INFERENCE_VALIDATION_ENABLED}" in
    1|true|True|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

inference_validation_validate() {
  inference_validation_enabled || return 0
  : "${INFERENCE_VALIDATION_INFERENCE_CONFIG:?inference_validation.inference_config is required when inference_validation.enabled=true}"
  : "${INFERENCE_VALIDATION_EVERY_STEPS:?inference_validation.every_steps is required when inference_validation.enabled=true}"
  if [[ ! "${INFERENCE_VALIDATION_EVERY_STEPS}" =~ ^[0-9]+$ ]] || ((INFERENCE_VALIDATION_EVERY_STEPS < 1)); then
    echo "error: inference_validation.every_steps must be a positive integer" >&2
    return 1
  fi
}

_inference_validation_sanitize_id() {
  local raw="$1"
  raw="${raw//[^A-Za-z0-9_-]/_}"
  printf '%s' "${raw:0:128}"
}

inference_validation_load_wandb_env() {
  inference_validation_enabled || return 0
  [[ -n "${CONFIG_YAML:-}" && -f "${CONFIG_YAML}" ]] || return 0
  [[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]] && return 0

  local _py _yaml_params _entity="${WANDB_ENTITY:-}" _project="${WANDB_PROJECT:-}"
  _py=/opt/venv/bin/python3
  [[ -x "${_py}" ]] || _py=$(command -v python3)
  _yaml_params="$("${_py}" "${_AVLM_UTILS_DIR}/_load_params_yaml.py" "${CONFIG_YAML}" \
    WANDB_ENTITY=wandb.entity \
    WANDB_PROJECT=wandb.project)"
  eval "${_yaml_params}"
  [[ -n "${_entity}" ]] && export WANDB_ENTITY="${_entity}"
  [[ -n "${_project}" ]] && export WANDB_PROJECT="${_project}"
  return 0
}

inference_validation_prepare_wandb_env() {
  inference_validation_enabled || return 0
  [[ "${WANDB_MODE:-}" == "disabled" ]] && return 0
  inference_validation_load_wandb_env
  if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    export WANDB_RUN_ID="$(_inference_validation_sanitize_id "${WANDB_NAME:-${MODEL_NAME:-avlm}}_${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}")"
  fi
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
}

inference_validation_start_watcher() {
  local inference_backend="${1:?inference backend required}"
  local checkpoint_kind="${2:?checkpoint kind required}"
  local train_job_id="${3:?training Slurm job id required}"
  local watcher_script log_dir output_dir pid_file pid

  inference_validation_enabled || return 0
  inference_validation_validate
  : "${REPO_ROOT:?REPO_ROOT must be set before starting inference validation watcher}"
  : "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set before starting inference validation watcher}"
  : "${OUTPUT:?OUTPUT must be set before starting inference validation watcher}"
  : "${LOGS_DIR:?LOGS_DIR must be set before starting inference validation watcher}"

  watcher_script="${REPO_ROOT}/avlm/utils/inference_validation_watcher.sh"
  [[ -f "${watcher_script}" ]] || {
    echo "error: missing ${watcher_script}" >&2
    return 1
  }

  log_dir="${LOGS_DIR}/inference_validation"
  output_dir="${OUTPUT}/inference_validation"
  mkdir -p "${log_dir}" "${output_dir}"
  pid_file="${log_dir}/watcher.pid"
  if [[ -s "${pid_file}" ]]; then
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "info: inference validation watcher already running (pid ${pid})" >&2
      return 0
    fi
  fi

  inference_validation_prepare_wandb_env
  export INFERENCE_VALIDATION_BACKEND="${inference_backend}"
  export INFERENCE_VALIDATION_CHECKPOINT_KIND="${checkpoint_kind}"
  export INFERENCE_VALIDATION_TRAIN_JOB_ID="${train_job_id}"
  export INFERENCE_VALIDATION_LOG_DIR="${log_dir}"
  export INFERENCE_VALIDATION_OUTPUT_DIR="${output_dir}"
  export AVLM_BASE_REPO_ROOT="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}"

  nohup bash "${watcher_script}" >>"${log_dir}/watcher.log" 2>&1 &
  pid=$!
  echo "${pid}" >"${pid_file}"
  export INFERENCE_VALIDATION_WATCHER_PID="${pid}"
  echo "info: inference validation watcher pid=${pid} log=${log_dir}/watcher.log" >&2
}

inference_validation_start_sbatch_watcher() {
  local inference_backend="${1:?inference backend required}"
  local checkpoint_kind="${2:?checkpoint kind required}"
  local job_id=""
  inference_validation_enabled || return 0
  if [[ -f "${LOGS_DIR}/race_jobs.tsv" ]]; then
    job_id="$(awk 'NR == 2 {print $2; exit}' "${LOGS_DIR}/race_jobs.tsv")"
  fi
  : "${job_id:?Could not resolve training Slurm job id for inference validation watcher}"
  inference_validation_start_watcher "${inference_backend}" "${checkpoint_kind}" "${job_id}"
}

# Derive cache env from CACHE_DIR (cluster_params sets CACHE_DIR only).
#   HF_HOME=${CACHE_DIR}/huggingface  XDG_CACHE_HOME=${CACHE_DIR}  TRITON_CACHE_DIR=${CACHE_DIR}/triton
setup_cache_dir_env() {
  local root="${CACHE_DIR:-${CORD_LUSTRE_CACHE:-}}"
  if [[ -z "${root}" ]]; then
    echo "error: CACHE_DIR unset. Set it in cluster_params or export CACHE_DIR=..." >&2
    return 1
  fi
  export CACHE_DIR="${root}"
  export HF_HOME="${HF_HOME:-${root}/huggingface}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${root}}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${root}/triton}"
  mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"
}

# Install lustre deep_ep wheel when recipe uses dispatcher: deepep (A100 needs pre-Hopper sm_80).
# Auto-selects wheel from GPU SM; skips if check_deepep_wheel.py already passes.
# Skip with SKIP_DEEPEP_INSTALL=1 (or SKIP_DEEPEP_INSTALL_ON_LAUNCH / ON_BATCH).
ensure_deepep() {
  if [[ "${SKIP_DEEPEP_INSTALL:-${SKIP_DEEPEP_INSTALL_ON_LAUNCH:-${SKIP_DEEPEP_INSTALL_ON_BATCH:-0}}}" == "1" ]]; then
    echo "[deepep] skip (SKIP_DEEPEP_INSTALL)" >&2
    return 0
  fi

  local _py _repo _check _install
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    _py="${PYTHON}"
  elif [[ -x /opt/venv/bin/python3 ]]; then
    _py=/opt/venv/bin/python3
  else
    _py="$(command -v python3)"
  fi

  _repo="${REPO_ROOT:-}"
  if [[ -z "${_repo}" ]]; then
    echo "[deepep] skip (REPO_ROOT unset)" >&2
    return 0
  fi

  _check="${_repo}/wheels/deepep/check_deepep_wheel.py"
  _install="${_repo}/wheels/deepep/install_deepep_wheel.sh"
  [[ -f "${_install}" ]] || {
    echo "[deepep] error: missing ${_install}" >&2
    return 1
  }

  if [[ -z "${CONFIG_YAML:-}" && -n "${CONFIG_YAML_REL:-}" ]]; then
    export CONFIG_YAML="${_repo}/${CONFIG_YAML_REL}"
  fi

  if [[ -n "${CONFIG_YAML:-}" && -f "${_check}" ]]; then
    if ! "${_py}" "${_check}" --needs-deepep; then
      echo "[deepep] skip (dispatcher is not deepep)" >&2
      return 0
    fi
  elif [[ "${DEEPEP_FORCE_INSTALL:-0}" != "1" ]]; then
    echo "[deepep] skip (CONFIG_YAML unset)" >&2
    return 0
  fi

  if [[ "${DEEPEP_FORCE_INSTALL:-0}" != "1" && -f "${_check}" ]]; then
    if "${_py}" "${_check}" 2>/dev/null; then
      echo "[deepep] ok (wheel matches GPU)" >&2
      return 0
    fi
  fi

  echo "[deepep] installing GPU-matched wheel" >&2
  bash "${_install}"
}

# nemo-automodel_26_06+ is missing some VLM runtime deps in /opt/venv.
# Skip with SKIP_VLM_TRAINING_DEPS=1 (or legacy SKIP_DECORD_INSTALL / SKIP_DECORD_INSTALL_ON_LAUNCH).
automodel_container_extra_pkgs_dir() {
  printf '%s\n' "${AUTOMODEL_EXTRA_PKGS_DIR:-${CACHE_DIR}/automodel_container_pkgs}"
}

automodel_prepend_extra_pkgs_pythonpath() {
  local _target
  _target="$(automodel_container_extra_pkgs_dir)"
  mkdir -p "${_target}"
  case ":${PYTHONPATH:-}:" in
    *":${_target}:"*) ;;
    *) export PYTHONPATH="${_target}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

# NeMo container: symlink imageio-ffmpeg as ffmpeg when system ffmpeg is absent.
automodel_ensure_ffmpeg_on_path() {
  command -v ffmpeg >/dev/null 2>&1 && return 0
  local _py _bin _dir _target
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    _py="${PYTHON}"
  elif [[ -x /opt/venv/bin/python3 ]]; then
    _py=/opt/venv/bin/python3
  else
    _py="$(command -v python3)"
  fi
  _target="$(automodel_container_extra_pkgs_dir)"
  _bin="$(PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)" || return 0
  for _dir in /usr/local/bin "${CACHE_DIR}/bin"; do
    mkdir -p "${_dir}" && ln -sf "${_bin}" "${_dir}/ffmpeg" && export PATH="${_dir}:${PATH}"
    command -v ffmpeg >/dev/null && { echo "[vlm-deps] ffmpeg → ${_dir}/ffmpeg" >&2; return 0; }
  done
}

ensure_vlm_training_deps() {
  if [[ "${SKIP_VLM_TRAINING_DEPS:-${SKIP_DECORD_INSTALL:-${SKIP_DECORD_INSTALL_ON_LAUNCH:-0}}}" == "1" ]]; then
    echo "[vlm-deps] skip (SKIP_VLM_TRAINING_DEPS)" >&2
    return 0
  fi

  if [[ -n "${CACHE_DIR:-}" ]]; then
    setup_cache_dir_env
  fi
  : "${CACHE_DIR:?CACHE_DIR must be set for VLM extra deps}"

  local _py _target _spec _pkg _mod _missing=()
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    _py="${PYTHON}"
  elif [[ -x /opt/venv/bin/python3 ]]; then
    _py=/opt/venv/bin/python3
  else
    _py="$(command -v python3)"
  fi
  [[ -x "${_py}" ]] || { echo "[vlm-deps] error: python not found" >&2; return 1; }

  automodel_prepend_extra_pkgs_pythonpath
  _target="$(automodel_container_extra_pkgs_dir)"

  for _spec in \
    "decord:decord" \
    "imageio-ffmpeg:imageio_ffmpeg" \
    "librosa:librosa" \
    "audioread:audioread" \
    "pooch:pooch" \
    "lazy-loader:lazy_loader"; do
    _pkg="${_spec%%:*}"
    _mod="${_spec##*:}"
    if PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}" 2>/dev/null; then
      echo "[vlm-deps] ${_pkg} ok ($(PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}; print(getattr(${_mod}, '__version__', 'unknown'))" 2>/dev/null || echo unknown))" >&2
      continue
    fi
    _missing+=("${_pkg}")
  done

  if ((${#_missing[@]} > 0)); then
    echo "[vlm-deps] installing container-missing deps: ${_missing[*]} → ${_target}" >&2
    # --no-deps: avoid shadowing the container's pinned numpy/scipy/torch stack.
    "${_py}" -m pip install --no-cache-dir --no-deps --target "${_target}" --upgrade "${_missing[@]}" || {
      echo "[vlm-deps] error: pip install of ${_missing[*]} into ${_target} failed" >&2
      return 1
    }
  fi

  local _fail=0
  for _mod in decord imageio_ffmpeg librosa; do
    PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}" 2>/dev/null \
      || { echo "[vlm-deps] error: ${_mod} still not importable after install (${_target})" >&2; _fail=1; }
  done
  (( _fail == 0 )) || return 1

  automodel_ensure_ffmpeg_on_path
  echo "[vlm-deps] ok: container python + extra deps (decord, imageio_ffmpeg, librosa) in ${_target}" >&2
}

# nemo_automodel ships in nemo-automodel_26_06+ at /opt/Automodel.
# Default: resolve_automodel_code_root → AUTOMODEL_CODE_ROOT=/opt/Automodel.
# Legacy git checkout (opt-in): AUTOMODEL_GIT_BOOTSTRAP=1 → ${CACHE_DIR}/Automodel.
# Pin matches nemo-automodel:26.06 (/opt/Automodel, nemo_automodel 0.5.0+d02f49cb).
_AUTOMODEL_GIT_URL="https://github.com/NVIDIA-NeMo/Automodel.git"
_AUTOMODEL_GIT_REF="d02f49cb314554715aabb97e8dba6599c9f6e9e0"

_automodel_nemo_tree_ok() {
  local root="$1"
  [[ -d "${root}/nemo_automodel" ]] || return 1
  [[ -f "${root}/nemo_automodel/components/datasets/vlm/collate_fns.py" ]] || return 1
  return 0
}

resolve_automodel_code_root() {
  if [[ "${AUTOMODEL_GIT_BOOTSTRAP:-0}" == "1" ]]; then
    bootstrap_automodel_git_checkout
    return $?
  fi

  local root="${AUTOMODEL_CODE_ROOT:-${AUTOMODEL_CONTAINER_ROOT:-/opt/Automodel}}"
  if ! _automodel_nemo_tree_ok "${root}"; then
    cat >&2 <<EOF
error: nemo_automodel not found at ${root}.

Use nemo-automodel_26_06+ (Nemotron Omni at /opt/Automodel).
Legacy git checkout: AUTOMODEL_GIT_BOOTSTRAP=1 (requires CACHE_DIR and network).
EOF
    return 1
  fi
  export AUTOMODEL_CODE_ROOT="$(cd -- "${root}" && pwd)"
  echo "info: nemo_automodel from ${AUTOMODEL_CODE_ROOT}" >&2
}

bootstrap_automodel_git_checkout() {
  local dir="${AUTOMODEL_GIT_DIR:-${CACHE_DIR}/Automodel}"

  _automodel_is_commit_sha() {
    [[ "${1:-}" =~ ^[0-9a-fA-F]{7,40}$ ]]
  }

  _automodel_verify_git_tree() {
    local root="${1:-${dir}}"
    _automodel_nemo_tree_ok "${root}" || return 1
    [[ -d "${root}/.git" ]] || return 1
    return 0
  }

  _automodel_clear_stale_git_locks() {
    local git_dir="$1/.git"
    [[ -d "${git_dir}" ]] || return 0
    local f
    for f in shallow.lock index.lock HEAD.lock config.lock; do
      [[ -f "${git_dir}/${f}" ]] && rm -f "${git_dir}/${f}"
    done
  }

  _automodel_head_matches_ref() {
    local root="$1" want="$2" head target=""
    head="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
    [[ -n "${head}" ]] || return 1
    target="$(git -C "${root}" rev-parse "${want}^{commit}" 2>/dev/null || true)"
    if [[ -n "${target}" ]]; then
      [[ "${head}" == "${target}" ]]
      return
    fi
    [[ "${head}" == "${want}" ]] || [[ "${head}" == ${want}* ]] || [[ "${want}" == ${head}* ]]
  }

  _automodel_fetch_ref() {
    local root="$1"
    if _automodel_is_commit_sha "${ref}" && [[ ${#ref} -lt 40 ]]; then
      echo "error: Automodel pin ${ref} is a short commit SHA; set _AUTOMODEL_GIT_REF to the full 40-character hash" >&2
      return 1
    fi
    if ! git -C "${root}" fetch --depth 1 origin "${ref}"; then
      if _automodel_is_commit_sha "${ref}"; then
        echo "error: Automodel git fetch failed in ${root} (commit=${ref})" >&2
      else
        echo "error: Automodel git fetch failed in ${root} (ref=${ref})" >&2
      fi
      return 1
    fi
    if ! git -C "${root}" reset --hard FETCH_HEAD; then
      echo "error: Automodel git reset failed in ${root}" >&2
      return 1
    fi
  }

  _automodel_clone_ref() {
    rm -rf "${dir}"
    if _automodel_is_commit_sha "${ref}"; then
      git init "${dir}" || return 1
      git -C "${dir}" remote add origin "${url}" || return 1
      _automodel_fetch_ref "${dir}" || return 1
    elif ! git clone --depth 1 --branch "${ref}" "${url}" "${dir}" 2>/dev/null; then
      rm -rf "${dir}"
      git init "${dir}" || return 1
      git -C "${dir}" remote add origin "${url}" || return 1
      _automodel_fetch_ref "${dir}" || return 1
    fi
  }

  _automodel_sync() {
    if [[ -d "${dir}/.git" ]] && _automodel_verify_git_tree "${dir}" && _automodel_head_matches_ref "${dir}" "${ref}"; then
      echo "info: Automodel checkout already at ${ref} ($(git -C "${dir}" rev-parse --short HEAD)); skipping fetch" >&2
      return 0
    fi
    _automodel_clear_stale_git_locks "${dir}"
    if [[ -d "${dir}/.git" ]]; then
      _automodel_fetch_ref "${dir}" || return 1
    else
      _automodel_clone_ref || return 1
    fi
    _automodel_verify_git_tree "${dir}"
  }

  : "${CACHE_DIR:?CACHE_DIR must be set before Automodel git bootstrap}"

  dir="${AUTOMODEL_GIT_DIR:-${CACHE_DIR}/Automodel}"
  local url="${_AUTOMODEL_GIT_URL}"
  local ref="${AUTOMODEL_GIT_REF:-${_AUTOMODEL_GIT_REF}}"
  local lockfile="${CACHE_DIR}/.automodel_bootstrap.lock"
  local _bootstrap_rc=0

  mkdir -p "${CACHE_DIR}"

  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 200 || exit 1
      if _automodel_verify_git_tree "${dir}" && _automodel_head_matches_ref "${dir}" "${ref}"; then
        echo "info: Automodel checkout present at ${dir} (${ref}); no sync needed" >&2
      else
        echo "info: Automodel git bootstrap (clone/sync) → ${dir} @ ${ref}" >&2
      fi
      _automodel_sync
    ) 200>"${lockfile}" || _bootstrap_rc=$?
  else
    echo "warn: flock not found; Automodel bootstrap may race on multi-node jobs" >&2
    _automodel_sync || _bootstrap_rc=$?
  fi

  if ((_bootstrap_rc != 0)) || ! _automodel_verify_git_tree "${dir}"; then
    cat >&2 <<EOF
error: Automodel git bootstrap failed (${dir}).

Re-run with AUTOMODEL_GIT_BOOTSTRAP=1 after fixing network/checkout, or use container /opt/Automodel (default).
EOF
    return 1
  fi

  export AUTOMODEL_CODE_ROOT="$(cd -- "${dir}" && pwd)"
  echo "info: Automodel from git @ $(git -C "${dir}" rev-parse --short HEAD 2>/dev/null || echo "${ref}") (${AUTOMODEL_CODE_ROOT})" >&2
}

# Slurm enroot mount list: host:container pairs (comma-separated).
# Ensures CACHE_DIR is visible in the container when not under CONTAINER_MOUNT_HOST.
batch_container_mounts() {
  local mount_host="${CONTAINER_MOUNT_HOST:-/lustre}"
  local base_repo_root="${AVLM_BASE_REPO_ROOT:-}"
  local -a mounts=( "${mount_host}:${mount_host}" )
  if [[ -n "${base_repo_root}" && "${base_repo_root}" != "${REPO_ROOT}" ]]; then
    mounts+=( "${base_repo_root}:${base_repo_root}" )
  fi
  mounts+=( "${REPO_ROOT}:${REPO_ROOT}" )
  if [[ -n "${CACHE_DIR:-}" ]]; then
    local m seen=0
    for m in "${mounts[@]}"; do
      if [[ "${m%%:*}" == "${CACHE_DIR}" ]]; then
        seen=1
        break
      fi
    done
    if [[ "${seen}" == "0" ]]; then
      mounts+=( "${CACHE_DIR}:${CACHE_DIR}" )
    fi
  fi
  local IFS=,
  echo "${mounts[*]}"
}

# First account for interactive srun (SLURM_ACCOUNTS, else SLURM_ACCOUNTS_RACE).
primary_slurm_account() {
  # shellcheck disable=SC2206
  local -a accts=( ${SLURM_ACCOUNTS:-} )
  if [[ ${#accts[@]} -eq 0 ]]; then
    accts=( ${SLURM_ACCOUNTS_RACE:-} )
  fi
  if [[ ${#accts[@]} -eq 0 ]]; then
    echo "error: set SLURM_ACCOUNTS (or SLURM_ACCOUNTS_RACE) in cluster_params" >&2
    return 1
  fi
  printf '%s\n' "${accts[0]}"
}

# Join partition names for sbatch -p (comma-separated).
_batch_partition_join() {
  local IFS=,
  echo "$*"
}

# Interactive srun/salloc: SINGLE_NODE_PARTITION from cluster_params.yaml.
interactive_slurm_partition() {
  if [[ -z "${SINGLE_NODE_PARTITION:-}" ]]; then
    echo "error: set SINGLE_NODE_PARTITION in cluster_params.yaml (e.g. interactive_singlenode)" >&2
    return 1
  fi
  printf '%s\n' "${SINGLE_NODE_PARTITION}"
}

# Pick sbatch partition from cluster_params when partition env is unset.
# >3: PARTITIONS. 2–3: FEW_NODES_PARTITIONS + PARTITIONS. 1: SINGLE_NODE + FEW_NODES + PARTITIONS.
# Override: partition=polar4 bash ...
resolve_batch_partition() {
  local nodes="${1:?}"
  [[ -n "${partition:-}" ]] && return 0

  # shellcheck disable=SC2206
  local -a few=( ${FEW_NODES_PARTITIONS:-} ) chosen=()
  if [[ -z "${PARTITIONS:-}" ]]; then
    echo "error: unset partition and missing PARTITIONS in cluster_params.yaml" >&2
    return 1
  fi
  if ((nodes > 3)); then
    # shellcheck disable=SC2206
    chosen=( ${PARTITIONS} )
  else
    [[ ${#few[@]} -eq 0 ]] && {
      echo "error: num_nodes=${nodes} requires FEW_NODES_PARTITIONS in cluster_params.yaml" >&2
      return 1
    }
    if ((nodes == 1)); then
      [[ -z "${SINGLE_NODE_PARTITION:-}" ]] && {
        echo "error: num_nodes=1 requires SINGLE_NODE_PARTITION in cluster_params.yaml" >&2
        return 1
      }
      chosen+=( "${SINGLE_NODE_PARTITION}" "${few[@]}" )
    else
      chosen+=( "${few[@]}" )
    fi
    # shellcheck disable=SC2206
    chosen+=( ${PARTITIONS} )
  fi
  partition="$(_batch_partition_join "${chosen[@]}")"
  export partition
}

# Inference launch helpers. AVLM_INFERENCE_EXPLICIT_ENV_KEYS contains variable
# names only; values continue to travel in their normal environment variables.
_inference_env_key_is_marked() {
  local want="$1" key
  for key in ${AVLM_INFERENCE_EXPLICIT_ENV_KEYS:-}; do
    [[ "${key}" == "${want}" ]] && return 0
  done
  return 1
}

inference_env_mark_explicit_keys() {
  local key marked="${AVLM_INFERENCE_EXPLICIT_ENV_KEYS:-}"
  for key in "$@"; do
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
      echo "error: invalid inference environment key: ${key}" >&2
      return 1
    }
    [[ -n "${!key+x}" ]] || continue
    if ! _inference_env_key_is_marked "${key}"; then
      marked="${marked:+${marked} }${key}"
      AVLM_INFERENCE_EXPLICIT_ENV_KEYS="${marked}"
    fi
  done
  export AVLM_INFERENCE_EXPLICIT_ENV_KEYS="${marked}"
}

inference_env_import_explicit_overrides() {
  local key
  for key in "$@"; do
    _inference_env_key_is_marked "${key}" || continue
    [[ -n "${!key+x}" ]] || continue
    _AVLM_CLI_OVERRIDES["${key}"]="${!key}"
  done
}

inference_env_prepare_overrides() {
  local _mode_dir="$1"
  shift
  _AVLM_LAUNCH_CLI_KEYS=( "$@" )
  if [[ -n "${AVLM_INFERENCE_EXPLICIT_ENV_KEYS+x}" ]]; then
    # A child-process provenance marker is authoritative: unmarked variables
    # are ordinary inherited fallbacks even if an exec boundary obscures the
    # original parent environment.
    inference_env_import_explicit_overrides "$@"
  else
    training_cli_detect_prefix_overrides "$@"
  fi
  training_cli_record_bashrc_fallback "$@"
}

inference_env_commit_overrides() {
  inference_env_prepare_overrides "$@"
}

inference_env_override_is_explicit() {
  local key="$1"
  [[ -n "${_INFERENCE_ENV_OVERRIDES[${key}]+x}" ]]
}

inference_env_forward_overrides() {
  local key
  for key in "${!_INFERENCE_ENV_OVERRIDES[@]}"; do
    inference_env_mark_explicit_keys "${key}"
  done
}

# Execute an inference shell entrypoint through a parent that retains ordinary
# inherited exports but does not contain explicit overrides. Re-applying only
# explicit values on the child command preserves precedence and remains
# compatible with target commits that predate the provenance marker.
inference_env_exec_sanitized() {
  local target="${1:?inference target required}"
  shift
  local key marker
  local -a explicit_keys=() explicit_assignments=() unset_args=()

  for key in "${_AVLM_LAUNCH_CLI_KEYS[@]}"; do
    if inference_env_override_is_explicit "${key}" && [[ -n "${!key+x}" ]]; then
      explicit_keys+=( "${key}" )
      explicit_assignments+=( "${key}=${!key}" )
    fi
  done

  for key in "${explicit_keys[@]}"; do
    unset_args+=( -u "${key}" )
  done
  marker="${explicit_keys[*]}"

  # Keep a sanitized shell alive as the target's parent. That parent boundary
  # matters for older commits whose prefix-env detection predates the marker.
  env "${unset_args[@]}" -u AVLM_INFERENCE_EXPLICIT_ENV_KEYS \
    bash -c '
      target="$1"
      marker="$2"
      assignment_count="$3"
      shift 3
      assignments=( "${@:1:assignment_count}" )
      shift "${assignment_count}"
      env "${assignments[@]}" \
        AVLM_INFERENCE_EXPLICIT_ENV_KEYS="${marker}" \
        bash "${target}" "$@" &
      child_pid=$!
      wait "${child_pid}"
      status=$?
      exit "${status}"
    ' avlm-inference-parent \
    "${target}" "${marker}" "${#explicit_assignments[@]}" \
    "${explicit_assignments[@]}" "$@"
}

inference_env_pin_slurm() {
  if ((${#_AVLM_LAUNCH_CLI_KEYS[@]} > 0)); then
    training_cli_pin_slurm_env "${_AVLM_LAUNCH_CLI_KEYS[@]}"
  else
    training_cli_pin_slurm_env "$@"
  fi
}

# --- VLM runtime deps (nemo-automodel_26_06 container gaps) ---

_inference_vlm_pkgs_dir() {
  printf '%s\n' "${AUTOMODEL_EXTRA_PKGS_DIR:-${CACHE_DIR}/automodel_container_pkgs}"
}

_inference_vlm_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
    printf '%s\n' "${PYTHON}"
  elif [[ -x /opt/venv/bin/python3 ]]; then
    printf '%s\n' /opt/venv/bin/python3
  else
    command -v python3
  fi
}

_inference_vlm_ffmpeg_on_path() {
  command -v ffmpeg >/dev/null 2>&1 && return 0
  local _py _bin _dir _target
  _py="$(_inference_vlm_python)"
  _target="$(_inference_vlm_pkgs_dir)"
  _bin="$(PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)" || return 0
  for _dir in /usr/local/bin "${CACHE_DIR}/bin"; do
    mkdir -p "${_dir}" && ln -sf "${_bin}" "${_dir}/ffmpeg" && export PATH="${_dir}:${PATH}"
    command -v ffmpeg >/dev/null && { echo "[vlm-deps] ffmpeg → ${_dir}/ffmpeg" >&2; return 0; }
  done
}

ensure_inference_vlm_deps() {
  if [[ "${SKIP_VLM_INFERENCE_DEPS:-0}" == "1" ]]; then
    echo "[vlm-deps] skip (SKIP_VLM_INFERENCE_DEPS)" >&2
    return 0
  fi
  : "${CACHE_DIR:?CACHE_DIR must be set for VLM extra deps}"

  local _py _target _spec _pkg _mod _missing=()
  _py="$(_inference_vlm_python)"
  [[ -x "${_py}" ]] || { echo "[vlm-deps] error: python not found" >&2; return 1; }

  _target="$(_inference_vlm_pkgs_dir)"
  mkdir -p "${_target}"
  case ":${PYTHONPATH:-}:" in
    *":${_target}:"*) ;;
    *) export PYTHONPATH="${_target}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac

  for _spec in \
    "decord:decord" \
    "imageio-ffmpeg:imageio_ffmpeg" \
    "librosa:librosa" \
    "audioread:audioread" \
    "pooch:pooch" \
    "lazy-loader:lazy_loader"; do
    _pkg="${_spec%%:*}"
    _mod="${_spec##*:}"
    if PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}" 2>/dev/null; then
      echo "[vlm-deps] ${_pkg} ok" >&2
      continue
    fi
    _missing+=("${_pkg}")
  done

  if ((${#_missing[@]} > 0)); then
    echo "[vlm-deps] installing container-missing deps: ${_missing[*]} → ${_target}" >&2
    "${_py}" -m pip install --no-cache-dir --no-deps --target "${_target}" --upgrade "${_missing[@]}" || {
      echo "[vlm-deps] error: pip install of ${_missing[*]} into ${_target} failed" >&2
      return 1
    }
  fi

  local _fail=0
  for _mod in decord imageio_ffmpeg librosa; do
    PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}" 2>/dev/null \
      || { echo "[vlm-deps] error: ${_mod} still not importable after install (${_target})" >&2; _fail=1; }
  done
  (( _fail == 0 )) || return 1

  _inference_vlm_ffmpeg_on_path
  echo "[vlm-deps] ok: extra deps (decord, imageio_ffmpeg, librosa) in ${_target}" >&2
}
