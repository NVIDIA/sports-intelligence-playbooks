#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cancel all Slurm jobs from a multi-account race (any training mode).
#
#   bash avlm/utils/cancel_slurm_race.sh \
#     --slurm-dir avlm/training/megatron-bridge/sft/slurm --latest
#
#   bash avlm/utils/cancel_slurm_race.sh \
#     --slurm-dir avlm/training/automodel/sft/slurm logs/run_20260528_190951
#
# Run dir may be omitted from --slurm-dir when the path contains .../slurm/logs/...
set -euo pipefail

_UTILS_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CANCEL_CMD="avlm/utils/cancel_slurm_race.sh"

cancel_slurm_race_usage() {
  cat <<EOF
Usage: bash ${_CANCEL_CMD} --slurm-dir PATH [--latest | run_dir | --jobs JOBID[,JOBID...]]

Cancel every Slurm job recorded for an account-race submit and kill the background watcher (if any).

Examples:
  bash ${_CANCEL_CMD} --slurm-dir avlm/training/megatron-bridge/sft/slurm --latest
  bash ${_CANCEL_CMD} --slurm-dir avlm/training/automodel/sft/slurm logs/run_20260528_190951
  bash ${_CANCEL_CMD} --slurm-dir avlm/training/automodel/lora/slurm --jobs 28317640,28317641
EOF
}

_cancel_resolve_slurm_dir() {
  local raw="${1:?--slurm-dir path required}"
  if [[ "${raw}" == /* && -d "${raw}" ]]; then
    cd -- "${raw}" && pwd
    return 0
  fi
  if [[ -d "${raw}" ]]; then
    cd -- "${raw}" && pwd
    return 0
  fi
  local repo="${REPO_ROOT:-}"
  if [[ -z "${repo}" ]]; then
    repo="$(cd -- "${_UTILS_DIR}/../../.." && pwd)"
  fi
  if [[ -d "${repo}/${raw}" ]]; then
    cd -- "${repo}/${raw}" && pwd
    return 0
  fi
  echo "error: slurm dir not found: ${raw}" >&2
  return 1
}

_cancel_infer_slurm_dir_from_run() {
  local path="$1"
  [[ "${path}" == */submit.log ]] && path="$(dirname "${path}")"
  if [[ "${path}" == *"/slurm/logs/"* ]]; then
    _cancel_resolve_slurm_dir "${path%%/logs/*}"
    return 0
  fi
  return 1
}

_cancel_slurm_race_init() {
  local slurm_dir_raw="" inferred=0
  local -a rest=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --slurm-dir)
        [[ $# -ge 2 ]] || {
          echo "error: --slurm-dir requires a path" >&2
          exit 1
        }
        slurm_dir_raw="$2"
        shift 2
        ;;
      -h | --help)
        cancel_slurm_race_usage
        exit 0
        ;;
      *)
        rest+=("$1")
        shift
        ;;
    esac
  done

  # shellcheck source=_source_params.sh
  source "${_UTILS_DIR}/_source_params.sh"

  if [[ -n "${slurm_dir_raw}" ]]; then
    SLURM_MODE_DIR="$(_cancel_resolve_slurm_dir "${slurm_dir_raw}")"
  elif ((${#rest[@]} > 0)) && [[ "${rest[0]}" != --latest && "${rest[0]}" != --jobs ]]; then
    if SLURM_MODE_DIR="$(_cancel_infer_slurm_dir_from_run "${rest[0]}")"; then
      inferred=1
    fi
  fi

  if [[ -z "${SLURM_MODE_DIR:-}" ]]; then
    cancel_slurm_race_usage >&2
    exit 1
  fi

  slurm_dir="${SLURM_MODE_DIR}"
  script_dir="${slurm_dir}/sbatch"
  logs_base="${slurm_dir}/logs"

  # shellcheck source=_slurm_account_race.sh
  source "${_UTILS_DIR}/_slurm_account_race.sh"
  set -- "${rest[@]}"
}

_cancel_resolve_run_dir() {
  local arg="$1"
  local path="${arg}"

  if [[ "${path}" == */submit.log ]]; then
    path="$(dirname "${path}")"
  elif [[ ! "${path}" == /* ]]; then
    if [[ -d "${script_dir}/${path}" ]]; then
      path="${script_dir}/${path}"
    elif [[ -d "${logs_base}/${path}" ]]; then
      path="${logs_base}/${path}"
    fi
  fi

  if [[ ! -d "${path}" ]]; then
    echo "error: run dir not found: ${arg}" >&2
    return 1
  fi
  echo "${path}"
}

_cancel_load_race_jobs() {
  local run_dir="$1"
  local jobs_file="${run_dir}/race_jobs.tsv"
  local submit_log="${run_dir}/submit.log"

  if [[ -f "${jobs_file}" ]]; then
    tail -n +2 "${jobs_file}" 2>/dev/null || true
    return 0
  fi

  if [[ -f "${submit_log}" ]]; then
    local parsed=""
    parsed="$(sed -n 's/^[[:space:]]*\([^[:space:]]\{1,\}\)[[:space:]]\{1,\}job \([0-9][0-9]*\)[[:space:]]*$/\1\t\2/p' "${submit_log}")"
    if [[ -n "${parsed}" ]]; then
      printf '%s\n' "${parsed}"
      return 0
    fi

    parsed="$(sed -n 's/^Submitted batch job \([0-9][0-9]*\)[[:space:]]*$/unknown\t\1/p' "${submit_log}")"
    if [[ -n "${parsed}" ]]; then
      printf '%s\n' "${parsed}"
      return 0
    fi

    local jobs_csv
    jobs_csv="$(sed -n 's/^Submitted [0-9][0-9]* jobs: \([0-9,][0-9,]*\)\..*/\1/p' "${submit_log}" | head -1)"
    if [[ -n "${jobs_csv}" ]]; then
      local id
      IFS=',' read -ra _legacy_ids <<<"${jobs_csv}"
      for id in "${_legacy_ids[@]}"; do
        id="${id// /}"
        [[ -n "${id}" ]] && printf 'unknown\t%s\n' "${id}"
      done
      return 0
    fi

    echo "error: submit.log found but no job IDs could be parsed: ${submit_log}" >&2
    return 1
  fi

  echo "error: no race_jobs.tsv or submit.log in ${run_dir}" >&2
  return 1
}

_cancel_stop_watcher() {
  local run_dir="$1"
  local pid_file="${run_dir}/race_watcher.pid"
  local pid=""

  if [[ ! -f "${pid_file}" ]]; then
    echo "  watcher: none (race_watcher.pid not found)"
    return 0
  fi

  pid="$(tr -d '[:space:]' <"${pid_file}")"
  if [[ -z "${pid}" ]]; then
    echo "  watcher: none (empty race_watcher.pid)"
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    echo "  watcher: stopped pid ${pid}"
  else
    echo "  watcher: pid ${pid} not running"
  fi
}

_cancel_relpath() {
  local p="$1"
  resolve_avlm_repo_roots_from_mode_dir "${slurm_dir}"
  if [[ "${p}" == "${REPO_ROOT}/"* ]]; then
    echo "${p#"${REPO_ROOT}/"}"
  else
    echo "${p}"
  fi
}

_cancel_race_in_dir() {
  local run_dir="$1"
  local -a job_ids=()
  local -a accounts=()
  local line account job_id

  echo "Run dir: $(_cancel_relpath "${run_dir}")"
  echo ""

  while IFS=$'\t' read -r account job_id; do
    [[ -z "${job_id}" ]] && continue
    accounts+=("${account}")
    job_ids+=("${job_id}")
  done < <(_cancel_load_race_jobs "${run_dir}")

  if ((${#job_ids[@]} == 0)); then
    echo "error: no job IDs found under ${run_dir}" >&2
    return 1
  fi

  echo "[cancel slurm]"
  local i
  for ((i = 0; i < ${#job_ids[@]}; i++)); do
    job_id="${job_ids[i]}"
    account="${accounts[i]}"
    if squeue -j "${job_id}" -h >/dev/null 2>&1; then
      scancel "${job_id}"
      printf "  cancelled  job %-8s  (%s)\n" "${job_id}" "${account}"
    else
      printf "  skipped    job %-8s  (%s) — not in queue\n" "${job_id}" "${account}"
    fi
  done

  sleep 3
  echo ""
  echo "[remove logs]"
  for ((i = 0; i < ${#job_ids[@]}; i++)); do
    _race_remove_job_logs "${job_ids[i]}" "${run_dir}"
    printf "  removed    job %-8s  logs\n" "${job_ids[i]}"
  done

  echo ""
  echo "[stop watcher]"
  _cancel_stop_watcher "${run_dir}"
}

_cancel_job_list() {
  local jobs_csv="$1"
  local job_id
  echo "[cancel slurm]"
  IFS=',' read -ra job_ids <<<"${jobs_csv}"
  for job_id in "${job_ids[@]}"; do
    job_id="${job_id// /}"
    [[ -z "${job_id}" ]] && continue
    if squeue -j "${job_id}" -h >/dev/null 2>&1; then
      scancel "${job_id}"
      echo "  cancelled  job ${job_id}"
    else
      echo "  skipped    job ${job_id} — not in queue"
    fi
  done
}

cancel_slurm_race_main() {
  if (($# == 0)); then
    cancel_slurm_race_usage >&2
    exit 1
  fi

  case "$1" in
    -h | --help)
      cancel_slurm_race_usage
      exit 0
      ;;
    --latest)
      local latest
      latest="$(find "${logs_base}" -maxdepth 1 -type d -name 'run_*' | sort | tail -1)"
      if [[ -z "${latest}" ]]; then
        echo "error: no run_* directories under ${logs_base}" >&2
        exit 1
      fi
      _cancel_race_in_dir "${latest}"
      ;;
    --jobs)
      if [[ $# -lt 2 ]]; then
        echo "error: --jobs requires a comma-separated job ID list" >&2
        exit 1
      fi
      _cancel_job_list "$2"
      ;;
    *)
      _cancel_race_in_dir "$(_cancel_resolve_run_dir "$1")"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  _cancel_slurm_race_init "$@"
  cancel_slurm_race_main "$@"
fi