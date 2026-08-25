#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set}"
: "${INFERENCE_VALIDATION_BACKEND:?INFERENCE_VALIDATION_BACKEND must be set}"
: "${INFERENCE_VALIDATION_CHECKPOINT_KIND:?INFERENCE_VALIDATION_CHECKPOINT_KIND must be set}"
: "${INFERENCE_VALIDATION_EVERY_STEPS:?INFERENCE_VALIDATION_EVERY_STEPS must be set}"
: "${INFERENCE_VALIDATION_INFERENCE_CONFIG:?INFERENCE_VALIDATION_INFERENCE_CONFIG must be set}"
: "${INFERENCE_VALIDATION_LOG_DIR:?INFERENCE_VALIDATION_LOG_DIR must be set}"
: "${INFERENCE_VALIDATION_OUTPUT_DIR:?INFERENCE_VALIDATION_OUTPUT_DIR must be set}"
: "${INFERENCE_VALIDATION_TRAIN_JOB_ID:?INFERENCE_VALIDATION_TRAIN_JOB_ID must be set}"

# Preserve only the checkpoint-specific inference values selected below across
# watcher -> pipeline -> launcher child boundaries.
source "${REPO_ROOT}/avlm/utils/_source_params.sh"

PIPELINE_SCRIPT="${REPO_ROOT}/avlm/inference/common/scripts/run_inference_eval_pipeline.sh"
METRICS_SCRIPT="${REPO_ROOT}/avlm/utils/log_inference_validation_metrics.py"
PIPELINES_TSV="${INFERENCE_VALIDATION_LOG_DIR}/pipelines.tsv"
PID_FILE="${INFERENCE_VALIDATION_LOG_DIR}/watcher.pid"
CLAIMS_DIR="${INFERENCE_VALIDATION_OUTPUT_DIR}/claims"
POLL_SECONDS="${INFERENCE_VALIDATION_POLL_SECONDS:-60}"
MIN_CHECKPOINT_AGE_SECONDS="${INFERENCE_VALIDATION_MIN_CHECKPOINT_AGE_SECONDS:-30}"
WATCHER_START_TIME=0
PIPELINE_PIDS=()

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
  log "watcher exiting"
}

trap cleanup EXIT
trap 'exit 0' INT TERM

mkdir -p "${INFERENCE_VALIDATION_LOG_DIR}" "${INFERENCE_VALIDATION_OUTPUT_DIR}" "${CLAIMS_DIR}"
echo "$$" >"${PID_FILE}"
if [[ ! -f "${PIPELINES_TSV}" ]]; then
  printf 'step\tcheckpoint_path\tinference_dir\tpipeline_log_dir\tpipeline_pid\tstarted_at\n' >"${PIPELINES_TSV}"
fi

step_from_checkpoint() {
  local base="$1" raw offset=0
  case "${base}" in
    epoch_*_step_*)
      raw="${base##*_step_}"
      # AutoModel checkpoint directory suffixes are zero-based: step_1 is the
      # checkpoint produced after completed training step 2.
      offset=1
      ;;
    iter_*)
      raw="${base#iter_}"
      ;;
    *)
      return 1
      ;;
  esac
  [[ "${raw}" =~ ^[0-9]+$ ]] || return 1
  printf '%d\n' "$((10#${raw} + offset))"
}

checkpoint_mtime() {
  local path="$1"
  [[ -d "${path}" ]] || return 1
  stat -c %Y "${path}" 2>/dev/null || echo 0
}

checkpoint_is_preexisting() {
  local path="$1" mtime
  mtime="$(checkpoint_mtime "${path}")"
  ((mtime < WATCHER_START_TIME))
}

checkpoint_is_ready() {
  local path="$1" mtime now

  # Require the final artifacts written by each checkpoint format.
  case "${INFERENCE_VALIDATION_CHECKPOINT_KIND}" in
    automodel_sft)
      [[ -s "${path}/model/consolidated/model.safetensors.index.json" ]] || return 1
      ;;
    automodel_lora)
      [[ -s "${path}/model/adapter_model.safetensors" ]] || return 1
      [[ -s "${path}/model/adapter_config.json" ]] || return 1
      ;;
    mbridge_sft)
      [[ -s "${path}/.metadata" ]] || return 1
      [[ -s "${path}/run_config.yaml" ]] || return 1
      ;;
    mbridge_lora)
      [[ -s "${path}/.metadata" ]] || return 1
      [[ -s "${path}/run_config.yaml" ]] || return 1
      ;;
    *)
      log "unknown checkpoint kind: ${INFERENCE_VALIDATION_CHECKPOINT_KIND}"
      exit 1
      ;;
  esac

  mtime="$(checkpoint_mtime "${path}")"
  now="$(date +%s)"
  ((now - mtime >= MIN_CHECKPOINT_AGE_SECONDS))
}

step_already_started() {
  local step="$1"
  awk -F '\t' -v step="${step}" 'NR > 1 && $1 == step {found=1} END {exit !found}' "${PIPELINES_TSV}" 2>/dev/null
}

training_is_active() {
  local job_id="${INFERENCE_VALIDATION_TRAIN_JOB_ID:-}"
  [[ -n "$(squeue -j "${job_id}" -h 2>/dev/null || true)" ]]
}

training_state() {
  squeue -j "${INFERENCE_VALIDATION_TRAIN_JOB_ID}" -h -o "%T" 2>/dev/null | awk 'NF {print $1; exit}' || true
}

wait_for_training_start() {
  local state
  while true; do
    state="$(training_state)"
    case "${state}" in
      RUNNING|COMPLETING)
        log "training job ${INFERENCE_VALIDATION_TRAIN_JOB_ID} is ${state}; starting checkpoint scan"
        WATCHER_START_TIME="$(date +%s)"
        return 0
        ;;
      "")
        log "training job ${INFERENCE_VALIDATION_TRAIN_JOB_ID} is no longer queued; watcher exiting"
        return 1
        ;;
    esac
    log "waiting for training job ${INFERENCE_VALIDATION_TRAIN_JOB_ID} to start (state=${state})"
    sleep "${POLL_SECONDS}"
  done
}

claim_step() {
  local step="$1"
  local checkpoint_path="$2"
  local claim_dir="${CLAIMS_DIR}/step_${step}"

  if ! mkdir "${claim_dir}" 2>/dev/null; then
    log "step ${step} already claimed: ${claim_dir}"
    return 1
  fi

  {
    printf 'checkpoint_path=%s\n' "${checkpoint_path}"
    printf 'watcher_pid=%s\n' "$$"
    printf 'claimed_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  } >"${claim_dir}/claim.txt"
}

metrics_python_cmd() {
  local env_root
  if [[ -n "${WANDB_VENV_PATH:-}" ]]; then
    env_root="${WANDB_VENV_PATH%/}"
    if [[ -x "${env_root}/bin/python" ]]; then
      printf '%s/bin/python' "${env_root}"
    else
      printf '%s/bin/python3' "${env_root}"
    fi
  else
    command -v python3
  fi
}

find_checkpoints() {
  local name_pattern
  case "${INFERENCE_VALIDATION_CHECKPOINT_KIND}" in
    automodel_sft|automodel_lora)
      name_pattern='epoch_*_step_*'
      ;;
    mbridge_sft|mbridge_lora)
      name_pattern='iter_*'
      ;;
    *)
      log "unknown checkpoint kind: ${INFERENCE_VALIDATION_CHECKPOINT_KIND}"
      return 1
      ;;
  esac
  [[ -d "${CHECKPOINT_DIR}" ]] || return 0
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "${name_pattern}" -print 2>/dev/null
}

launch_pipeline() {
  local step="$1"
  local checkpoint_path="$2"
  local step_name="step_${step}"
  local output_base="${INFERENCE_VALIDATION_OUTPUT_DIR}/inference_outputs"
  local inference_dir="${output_base}/${step_name}"
  local pipeline_log_dir="${INFERENCE_VALIDATION_LOG_DIR}/pipelines/${step_name}"
  local pipeline_log="${pipeline_log_dir}/pipeline.log"
  local metrics_jsonl="${INFERENCE_VALIDATION_OUTPUT_DIR}/metrics.jsonl"
  local pipeline_pid
  local metrics_python
  local vlm_scorer_config="${VLM_SCORER_CONFIG:-}"

  mkdir -p "${output_base}" "${pipeline_log_dir}"
  log "launching inference validation for step ${step}: ${checkpoint_path}"
  metrics_python="$(metrics_python_cmd)"

  (
    local pipeline_status

    set +e

    unset INFERENCE_DIR INFERENCE_CONFIG INFERENCE_NAME BASE_OUTPUT_DIR DATA_PATH
    unset MODEL_PATH ADAPTER_PATH MAX_INFERENCE_SAMPLES MAX_INFERENCE_SAMPLES_SEED RESUME CACHE_DIR CLUSTER_PARAMS
    unset num_nodes GPUS_PER_NODE NPROC_PER_NODE partition use_exclusive
    unset SLURM_ACCOUNTS SLURM_ACCOUNT_RACE SLURM_ACCOUNTS_RACE SLURM_MEM SLURM_TIME_LIMIT
    unset VLM_SCORER_CONFIG MAX_LLM_JUDGE_SAMPLES
    export AVLM_INFERENCE_EXPLICIT_ENV_KEYS=""

    export AVLM_BASE_REPO_ROOT="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}"
    export INFERENCE_BACKEND="${INFERENCE_VALIDATION_BACKEND}"
    export INFERENCE_CONFIG="${INFERENCE_VALIDATION_INFERENCE_CONFIG}"
    export BASE_OUTPUT_DIR="${output_base}"
    export INFERENCE_NAME="${step_name}"
    export STAGES="inference,mcq,judge,judge_eval"

    case "${INFERENCE_VALIDATION_CHECKPOINT_KIND}" in
      automodel_sft|mbridge_sft)
        export MODEL_PATH="${checkpoint_path}"
        ;;
      automodel_lora)
        export ADAPTER_PATH="${checkpoint_path}"
        ;;
      mbridge_lora)
        : "${PRETRAINED_CHECKPOINT:?PRETRAINED_CHECKPOINT is required for Megatron-Bridge LoRA validation}"
        export MODEL_PATH="${PRETRAINED_CHECKPOINT}"
        export ADAPTER_PATH="${checkpoint_path}"
        ;;
      *)
        echo "Unknown checkpoint kind: ${INFERENCE_VALIDATION_CHECKPOINT_KIND}" >&2
        exit 1
        ;;
    esac

    [[ -n "${INFERENCE_VALIDATION_DATA_PATH:-}" ]] && export DATA_PATH="${INFERENCE_VALIDATION_DATA_PATH}"
    if [[ -n "${INFERENCE_VALIDATION_MAX_SAMPLES:-}" ]]; then
      export MAX_INFERENCE_SAMPLES="${INFERENCE_VALIDATION_MAX_SAMPLES}"
      export MAX_LLM_JUDGE_SAMPLES="${INFERENCE_VALIDATION_MAX_SAMPLES}"
    fi
    [[ -n "${INFERENCE_VALIDATION_MAX_SAMPLES_SEED:-}" ]] && export MAX_INFERENCE_SAMPLES_SEED="${INFERENCE_VALIDATION_MAX_SAMPLES_SEED}"
    [[ -n "${INFERENCE_VALIDATION_CLUSTER_PARAMS:-}" ]] && export CLUSTER_PARAMS="${INFERENCE_VALIDATION_CLUSTER_PARAMS}"
    [[ -n "${INFERENCE_VALIDATION_NUM_NODES:-}" ]] && export num_nodes="${INFERENCE_VALIDATION_NUM_NODES}"
    [[ -n "${INFERENCE_VALIDATION_GPUS_PER_NODE:-}" ]] && export GPUS_PER_NODE="${INFERENCE_VALIDATION_GPUS_PER_NODE}"
    [[ -n "${INFERENCE_VALIDATION_NPROC_PER_NODE:-}" ]] && export NPROC_PER_NODE="${INFERENCE_VALIDATION_NPROC_PER_NODE}"
    [[ -n "${vlm_scorer_config}" ]] && export VLM_SCORER_CONFIG="${vlm_scorer_config}"

    inference_env_mark_explicit_keys \
      INFERENCE_BACKEND INFERENCE_CONFIG BASE_OUTPUT_DIR INFERENCE_NAME STAGES \
      MODEL_PATH ADAPTER_PATH DATA_PATH MAX_INFERENCE_SAMPLES MAX_INFERENCE_SAMPLES_SEED \
      MAX_LLM_JUDGE_SAMPLES CLUSTER_PARAMS num_nodes GPUS_PER_NODE NPROC_PER_NODE \
      VLM_SCORER_CONFIG

    bash "${PIPELINE_SCRIPT}" --run "${pipeline_log_dir}"
    pipeline_status=$?
    echo "pipeline exit status: ${pipeline_status}"

    if [[ "${pipeline_status}" == "0" ]]; then
      "${metrics_python}" "${METRICS_SCRIPT}" \
        --inference-dir "${inference_dir}" \
        --output-jsonl "${metrics_jsonl}" \
        --step "${step}" \
        --checkpoint-path "${checkpoint_path}"
    fi
    exit "${pipeline_status}"
  ) >>"${pipeline_log}" 2>&1 &
  pipeline_pid=$!
  PIPELINE_PIDS+=("${pipeline_pid}")
  printf '%s\n' "${pipeline_pid}" >"${pipeline_log_dir}/pipeline.pid"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${step}" "${checkpoint_path}" "${inference_dir}" "${pipeline_log_dir}" "${pipeline_pid}" "$(date '+%Y-%m-%d %H:%M:%S')" \
    >>"${PIPELINES_TSV}"
}

scan_once() {
  local every="${INFERENCE_VALIDATION_EVERY_STEPS}"
  local found_any=0

  while IFS=$'\t' read -r step path; do
    [[ -n "${step}" && -n "${path}" ]] || continue
    found_any=1
    if ((step % every != 0)); then
      continue
    fi
    if step_already_started "${step}"; then
      continue
    fi
    if checkpoint_is_preexisting "${path}"; then
      continue
    fi
    if ! checkpoint_is_ready "${path}"; then
      log "step ${step} checkpoint is not old enough yet: ${path}"
      continue
    fi
    if ! claim_step "${step}" "${path}"; then
      continue
    fi
    launch_pipeline "${step}" "${path}"
  done < <(
    while IFS= read -r checkpoint_path; do
      base="$(basename "${checkpoint_path}")"
      step="$(step_from_checkpoint "${base}" 2>/dev/null || true)"
      [[ -n "${step}" ]] || continue
      printf '%s\t%s\n' "${step}" "${checkpoint_path}"
    done < <(find_checkpoints) | sort -n -k1,1
  )

  ((found_any == 1)) || true
}

wait_for_pipelines() {
  local pid
  for pid in "${PIPELINE_PIDS[@]}"; do
    wait "${pid}" || true
  done
}

log "watching ${CHECKPOINT_DIR}"
log "backend=${INFERENCE_VALIDATION_BACKEND} checkpoint_kind=${INFERENCE_VALIDATION_CHECKPOINT_KIND} every_steps=${INFERENCE_VALIDATION_EVERY_STEPS}"
wait_for_training_start || exit 0

inactive_count=0
while true; do
  scan_once
  if training_is_active; then
    inactive_count=0
  else
    inactive_count=$((inactive_count + 1))
    if ((inactive_count >= 3)); then
      log "training is no longer active; final scan complete"
      break
    fi
  fi
  sleep "${POLL_SECONDS}"
done
wait_for_pipelines
