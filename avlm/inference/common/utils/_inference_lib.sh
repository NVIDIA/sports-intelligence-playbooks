#!/usr/bin/env bash
# Shared inference launch helpers.

INFERENCE_COMMON_UTILS_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_COMMON_DIR="$(cd -- "${INFERENCE_COMMON_UTILS_DIR}/.." && pwd)"
_INFERENCE_DIR="$(cd -- "${_COMMON_DIR}/.." && pwd)"
export AVLM_ROOT="$(cd -- "${_INFERENCE_DIR}/.." && pwd)"
export REPO_ROOT="$(cd -- "${AVLM_ROOT}/.." && pwd)"
export INFERENCE_COMMON_DIR="${_COMMON_DIR}"
export INFERENCE_COMMON_UTILS_DIR
export AVLM_UTILS_DIR="$(cd -- "${AVLM_ROOT}/utils" && pwd)"

source "${AVLM_UTILS_DIR}/_source_params.sh"

INFERENCE_COMMON_ENV_OVERRIDE_KEYS=(
  INFERENCE_CONFIG
  INFERENCE_NAME
  BASE_OUTPUT_DIR
  DATA_PATH
  MODEL_PATH
  ADAPTER_PATH
  IS_FSDP
  FSDP_SHARDING_STRATEGY
  MAX_INFERENCE_SAMPLES
  MAX_INFERENCE_SAMPLES_SEED
  RESUME
  CACHE_DIR
  CLUSTER_PARAMS
)

INFERENCE_SBATCH_ENV_OVERRIDE_KEYS=(
  num_nodes
  GPUS_PER_NODE
  NPROC_PER_NODE
  partition
  use_exclusive
  SLURM_ACCOUNTS
  SLURM_ACCOUNT_RACE
  SLURM_TIME_LIMIT
)

INFERENCE_SBATCH_ENV_ONLY_KEYS=(
  "${SBATCH_ENV_ONLY_KEYS[@]}"
)

INFERENCE_ORCHESTRATION_ENV_OVERRIDE_KEYS=(
  INFERENCE_BACKEND
  STAGES
  INFERENCE_DIR
  PIPELINE_LOG_DIR
  PIPELINE_POLL_SECONDS
  SUITE_CONFIG
  suite_name
  VLM_SCORER_CONFIG
  MAX_LLM_JUDGE_SAMPLES
  JUDGE_VENV_PATH
)

inference_all_env_override_keys() {
  printf '%s\n' \
    "${INFERENCE_COMMON_ENV_OVERRIDE_KEYS[@]}" \
    "${INFERENCE_SBATCH_ENV_OVERRIDE_KEYS[@]}" \
    "${INFERENCE_SBATCH_ENV_ONLY_KEYS[@]}" \
    "${INFERENCE_ORCHESTRATION_ENV_OVERRIDE_KEYS[@]}"
}

inference_reject_config_args() {
  (($# == 0)) && return 0
  local arg="$1"
  if [[ "${arg}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    echo "error: inference overrides are environment-only; place '${arg}' before 'bash <script>'" >&2
  else
    echo "error: unexpected inference argument: ${arg}" >&2
  fi
  return 2
}

inference_export_config_overrides() {
  unset INFERENCE_NAME_OVERRIDE BASE_OUTPUT_DIR_OVERRIDE DATA_PATH_OVERRIDE
  unset MODEL_PATH_OVERRIDE ADAPTER_PATH_OVERRIDE RESUME_OVERRIDE
  unset IS_FSDP_OVERRIDE FSDP_SHARDING_STRATEGY_OVERRIDE

  inference_env_override_is_explicit INFERENCE_NAME && export INFERENCE_NAME_OVERRIDE="${INFERENCE_NAME}"
  inference_env_override_is_explicit BASE_OUTPUT_DIR && export BASE_OUTPUT_DIR_OVERRIDE="${BASE_OUTPUT_DIR}"
  inference_env_override_is_explicit DATA_PATH && export DATA_PATH_OVERRIDE="${DATA_PATH}"
  inference_env_override_is_explicit MODEL_PATH && export MODEL_PATH_OVERRIDE="${MODEL_PATH}"
  inference_env_override_is_explicit ADAPTER_PATH && export ADAPTER_PATH_OVERRIDE="${ADAPTER_PATH}"
  inference_env_override_is_explicit RESUME && export RESUME_OVERRIDE="${RESUME}"
  inference_env_override_is_explicit IS_FSDP && export IS_FSDP_OVERRIDE="${IS_FSDP}"
  inference_env_override_is_explicit FSDP_SHARDING_STRATEGY && export FSDP_SHARDING_STRATEGY_OVERRIDE="${FSDP_SHARDING_STRATEGY}"
  return 0
}

# needed because in our sbatch_starer uses dependency singleton
# so we need to get the inference name
resolve_inference_job_name() {
  local inference_name="${INFERENCE_NAME_OVERRIDE:-}"
  local inference_config="${INFERENCE_CONFIG:-}"

  if [[ -z "${inference_name}" && -n "${inference_config}" ]]; then
    [[ "${inference_config}" == /* ]] || inference_config="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}/${inference_config}"
    inference_name="$(python3 - "${inference_config}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f) or {}

print(cfg.get("inference_name", ""))
PY
)"
  fi

  echo "${inference_name:-infer}"
}
