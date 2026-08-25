#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_COMMON_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
_INFERENCE_DIR="$(cd -- "${_COMMON_DIR}/.." && pwd)"
_AVLM_DIR="$(cd -- "${_INFERENCE_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${_AVLM_DIR}/.." && pwd)"
BASE_REPO_ROOT="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_inference_eval_pipeline.sh"
MCQ_EVAL_SCRIPT="${_AVLM_DIR}/evals/scripts/run_mcq_eval.sh"
QA_LLM_JUDGE_SCRIPT="${_AVLM_DIR}/evals/scripts/run_qa_llm_judge.sh"
QA_LLM_JUDGE_EVAL_SCRIPT="${_AVLM_DIR}/evals/scripts/run_qa_llm_judge_eval.sh"
QA_LLM_JUDGE_SUMMARY_SCRIPT="${_AVLM_DIR}/evals/scripts/run_qa_llm_judge_summary.sh"
source "${_COMMON_DIR}/utils/_inference_lib.sh"
cd "${REPO_ROOT}"

mapfile -t inference_env_keys < <(inference_all_env_override_keys)
inference_env_commit_overrides "${SCRIPT_DIR}" "${inference_env_keys[@]}"
inference_env_forward_overrides

inference_backend_path() {
  case "$1" in
    automodel)
      echo "avlm/inference/automodel"
      ;;
    megatron_bridge)
      echo "avlm/inference/megatron-bridge"
      ;;
    *)
      echo "error: backend must be one of: automodel, megatron_bridge" >&2
      return 1
      ;;
  esac
}

inference_sbatch_script() {
  local backend_path
  backend_path="$(inference_backend_path "$1")"
  echo "${REPO_ROOT}/${backend_path}/slurm/sbatch/sbatch_starter.sh"
}

resolve_base_path() {
  if [[ "$1" == /* ]]; then
    echo "$1"
  else
    echo "${BASE_REPO_ROOT}/$1"
  fi
}

resolve_inference_name() {
  local inference_config inference_dir inference_name
  inference_dir=""
  if inference_env_override_is_explicit INFERENCE_DIR || [[ -z "${INFERENCE_CONFIG:-}" ]]; then
    inference_dir="${INFERENCE_DIR:-}"
  fi
  if [[ -n "${inference_dir}" ]]; then
    basename "${inference_dir}"
    return 0
  fi

  if inference_env_override_is_explicit INFERENCE_NAME; then
    inference_name="${INFERENCE_NAME}"
    echo "${inference_name}"
    return 0
  fi

  inference_config="${INFERENCE_CONFIG:-}"
  if [[ -z "${inference_config}" ]]; then
    echo "${INFERENCE_NAME:-}"
    return 0
  fi
  inference_config="$(resolve_base_path "${inference_config}")"

  python3 - "${inference_config}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)

print(cfg.get("inference_name") or "")
PY
}

resolve_inference_dir() {
  local inference_config inference_name base_output_dir
  inference_config="${INFERENCE_CONFIG:-}"
  inference_name=""
  base_output_dir=""
  inference_env_override_is_explicit INFERENCE_NAME && inference_name="${INFERENCE_NAME}"
  inference_env_override_is_explicit BASE_OUTPUT_DIR && base_output_dir="${BASE_OUTPUT_DIR}"
  : "${inference_config:?Set INFERENCE_CONFIG}"
  inference_config="$(resolve_base_path "${inference_config}")"

  INFERENCE_NAME_OVERRIDE="${inference_name}" \
    BASE_OUTPUT_DIR_OVERRIDE="${base_output_dir}" \
    INFERENCE_NAME_FALLBACK="${INFERENCE_NAME:-}" \
    BASE_OUTPUT_DIR_FALLBACK="${BASE_OUTPUT_DIR:-}" \
    python3 - "${inference_config}" <<'PY'
import os
import sys
import yaml

with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)

inference_name = (
    os.environ.get("INFERENCE_NAME_OVERRIDE")
    or cfg.get("inference_name")
    or os.environ.get("INFERENCE_NAME_FALLBACK")
)
base_output_dir = (
    os.environ.get("BASE_OUTPUT_DIR_OVERRIDE")
    or cfg.get("base_output_dir")
    or os.environ.get("BASE_OUTPUT_DIR_FALLBACK")
)
if not inference_name or not base_output_dir:
    raise ValueError("inference_name and base_output_dir must be set in config or environment")
print(os.path.join(base_output_dir, inference_name))
PY
}

normalize_stages() {
  local stages="${STAGES:-inference,mcq,judge,judge_eval,judge_summary}"
  stages="${stages//,/ }"
  stages="${stages//;/ }"
  echo "${stages}"
}

has_stage() {
  local want="$1"
  local stage
  for stage in ${PIPELINE_STAGES}; do
    [[ "${stage}" == "${want}" ]] && return 0
  done
  return 1
}

PIPELINE_STAGES="$(normalize_stages)"
for stage in ${PIPELINE_STAGES}; do
  case "${stage}" in
    inference|mcq|judge|judge_eval|judge_summary) ;;
    *) echo "Unknown stage: ${stage}" >&2; exit 1 ;;
  esac
done

if [[ "${1:-}" == "stop" ]]; then
  (($# == 2)) || {
    echo "usage: bash ${PIPELINE_SCRIPT} stop <pipeline-run-dir>" >&2
    exit 2
  }
  run_dir="${2:?Set pipeline run dir}"
  [[ -f "${run_dir}/slurm_job_id" ]] && scancel "$(cat "${run_dir}/slurm_job_id")" 2>/dev/null || true
  if [[ -f "${run_dir}/pipeline.pid" ]] && kill -- "-$(cat "${run_dir}/pipeline.pid")" 2>/dev/null; then
    echo "Cancelled pipeline"
  else
    echo "No running pipeline found; not cancelled"
  fi
  exit 0
fi

if [[ "${1:-}" != "--run" ]]; then
  inference_reject_config_args "$@"
  inference_name="$(resolve_inference_name)"
  run_name="${inference_name:-run}"
  run_dir="${PIPELINE_LOG_DIR:-avlm/inference/logs/${run_name}_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${run_dir}"
  run_dir="$(cd -- "${run_dir}" && pwd)"

  nohup setsid bash "${PIPELINE_SCRIPT}" --run "${run_dir}" >"${run_dir}/pipeline.log" 2>&1 &
  pid="$!"
  echo "${pid}" >"${run_dir}/pipeline.pid"

  echo "Started pipeline"
  echo "  pid: ${pid}"
  echo "  log: ${run_dir}/pipeline.log"
  echo "  stop: bash ${PIPELINE_SCRIPT} stop ${run_dir}"
  exit 0
fi

(($# == 2)) || {
  echo "usage: bash ${PIPELINE_SCRIPT} --run <pipeline-run-dir>" >&2
  exit 2
}
run_dir="${2:?Set pipeline run dir}"
backend="${INFERENCE_BACKEND:-}"
if has_stage inference; then
  [[ -n "${backend}" ]] || {
    echo "usage: INFERENCE_BACKEND=<automodel|megatron_bridge> bash ${PIPELINE_SCRIPT} --run <run_dir>" >&2
    exit 1
  }
  backend_path="$(inference_backend_path "${backend}")"
  INFERENCE_SBATCH_SCRIPT="$(inference_sbatch_script "${backend}")"
  export CLUSTER_PARAMS="${CLUSTER_PARAMS:-${BASE_REPO_ROOT}/${backend_path}/slurm/launch_local.yaml}"
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_job() {
  local job_id="$1"
  local poll_seconds="${PIPELINE_POLL_SECONDS:-60}"
  local state final_state

  while [[ -n "$(squeue -j "${job_id}" -h 2>/dev/null || true)" ]]; do
    state="$(squeue -j "${job_id}" -h -o "%T" 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    log "Inference job ${job_id} state: ${state:-unknown}"
    sleep "${poll_seconds}"
  done

  final_state="$(sacct -j "${job_id}" -X -n -o State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
  log "Inference job ${job_id} final state: ${final_state:-unknown}"

  [[ "${final_state}" == "COMPLETED" ]]
}

mkdir -p "${run_dir}"
inference_dir=""
if inference_env_override_is_explicit INFERENCE_DIR || [[ -z "${INFERENCE_CONFIG:-}" ]]; then
  inference_dir="${INFERENCE_DIR:-}"
fi
if [[ -n "${inference_dir}" && "${inference_dir}" != /* ]]; then
  inference_dir="$(resolve_base_path "${inference_dir}")"
fi
if has_stage inference; then
  resolved_inference_dir="$(resolve_inference_dir)"
  if [[ -n "${inference_dir}" && "${inference_dir}" != "${resolved_inference_dir}" ]]; then
    echo "error: INFERENCE_DIR (${inference_dir}) does not match resolved inference output (${resolved_inference_dir})" >&2
    exit 1
  fi

  inference_logs="${run_dir}/inference_logs"
  mkdir -p "${inference_logs}"
  inference_dir="${resolved_inference_dir}"

  log "Submitting ${backend} inference"
  LOGS_DIR="${inference_logs}" bash "${INFERENCE_SBATCH_SCRIPT}"

  job_file="${inference_logs}/race_jobs.tsv"
  job_id="$(awk 'NR == 2 {print $2; exit}' "${job_file}")"
  : "${job_id:?Could not find Slurm job id}"
  echo "${job_id}" >"${run_dir}/slurm_job_id"
  log "Inference Slurm job id: ${job_id}"

  if ! wait_for_job "${job_id}"; then
    log "Inference did not complete successfully. Skipping remaining stages."
    exit 1
  fi

  echo "${inference_dir}" >"${run_dir}/inference_dir"
  log "Inference output: ${inference_dir}"
else
  if [[ -z "${inference_dir}" ]]; then
    inference_dir="$(resolve_inference_dir)"
  fi
fi

if has_stage mcq || has_stage judge || has_stage judge_eval; then
  [[ -f "${inference_dir}/predictions.jsonl" ]] || {
    log "Missing predictions.jsonl: ${inference_dir}/predictions.jsonl"
    exit 1
  }
fi

if has_stage mcq; then
  log "Running MCQ eval"
  INFERENCE_DIR="${inference_dir}" bash "${MCQ_EVAL_SCRIPT}"
fi

if has_stage judge; then
  log "Running LLM judge"
  INFERENCE_DIR="${inference_dir}" bash "${QA_LLM_JUDGE_SCRIPT}"
fi

if has_stage judge_eval; then
  log "Running LLM judge eval"
  INFERENCE_DIR="${inference_dir}" bash "${QA_LLM_JUDGE_EVAL_SCRIPT}"
fi

if has_stage judge_summary; then
  log "Running LLM judge summary"
  INFERENCE_DIR="${inference_dir}" bash "${QA_LLM_JUDGE_SUMMARY_SCRIPT}"
fi

log "Pipeline complete"
