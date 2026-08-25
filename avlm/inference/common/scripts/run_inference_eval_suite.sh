#!/usr/bin/env bash
# Run one inference/eval pipeline per run in a suite YAML.
#
#   SUITE_CONFIG=avlm/inference/common/suites/my_suite.yaml \
#     bash avlm/inference/common/scripts/run_inference_eval_suite.sh
#
# Environment overrides: num_nodes=8 STAGES=mcq,judge,judge_eval ...
# Precedence: command-prefix environment > run YAML > suite YAML > launch or
# inference YAML > ordinary inherited shell exports.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_COMMON_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
_INFERENCE_DIR="$(cd -- "${_COMMON_DIR}/.." && pwd)"
_AVLM_DIR="$(cd -- "${_INFERENCE_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${_AVLM_DIR}/.." && pwd)"
BASE_REPO_ROOT="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_inference_eval_pipeline.sh"
SUITE_YAML_SCRIPT="${SCRIPT_DIR}/_load_inference_eval_suite.py"
source "${_COMMON_DIR}/utils/_inference_lib.sh"
cd "${REPO_ROOT}"

inference_reject_config_args "$@"
mapfile -t inference_env_keys < <(inference_all_env_override_keys)
inference_env_commit_overrides "${SCRIPT_DIR}" "${inference_env_keys[@]}"

: "${SUITE_CONFIG:?Set SUITE_CONFIG=path/to/suite.yaml}"
[[ "${SUITE_CONFIG}" == /* ]] || SUITE_CONFIG="${BASE_REPO_ROOT}/${SUITE_CONFIG}"
if [[ ! -f "${SUITE_CONFIG}" ]]; then
  echo "error: suite config not found: ${SUITE_CONFIG}" >&2
  exit 1
fi

suite_yaml() {
  python3 "${SUITE_YAML_SCRIPT}" "${SUITE_CONFIG}" "$@"
}

suite_name_yaml="$(suite_yaml name)"
if inference_env_override_is_explicit suite_name; then
  effective_suite_name="${suite_name}"
else
  effective_suite_name="${suite_name_yaml:-${suite_name:-}}"
fi
: "${effective_suite_name:?Set suite_name in suite YAML or command-prefix environment}"
suite_name="${effective_suite_name}"
logs_suites_base="${BASE_REPO_ROOT}/avlm/inference/logs/suites/${suite_name}"
outputs_suites_base="${BASE_REPO_ROOT}/avlm/inference/outputs/suites/${suite_name}"

external_env_overrides=()
for key in "${!_INFERENCE_ENV_OVERRIDES[@]}"; do
  case "${key}" in
    SUITE_CONFIG|suite_name) continue ;;
  esac
  external_env_overrides+=( "${key}=${_INFERENCE_ENV_OVERRIDES[${key}]}" )
done

run_count="$(suite_yaml count)"
for ((run_idx = 0; run_idx < run_count; run_idx++)); do
  run_name="$(suite_yaml run-name "${run_idx}")"
  run_ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="${logs_suites_base}/${run_name}/run_${run_ts}"

  run_env=( "BASE_OUTPUT_DIR=${outputs_suites_base}" )
  mapfile -t run_yaml_env < <(suite_yaml args "${run_idx}")
  run_env+=( "${run_yaml_env[@]}" )

  echo "Starting run $((run_idx + 1))/${run_count}: ${run_name}"
  (
    export AVLM_INFERENCE_EXPLICIT_ENV_KEYS=""
    explicit_run_keys=()
    for assignment in "${run_env[@]}" "${external_env_overrides[@]}"; do
      export "${assignment}"
      explicit_run_keys+=( "${assignment%%=*}" )
    done
    export PIPELINE_LOG_DIR="${run_dir}"
    explicit_run_keys+=(PIPELINE_LOG_DIR)
    export MB_BRIDGE_OVERLAY="${MB_BRIDGE_OVERLAY:+${MB_BRIDGE_OVERLAY}/suite_${run_idx}_${run_ts}}"
    inference_env_mark_explicit_keys "${explicit_run_keys[@]}"
    exec bash "${PIPELINE_SCRIPT}"
  )
done
