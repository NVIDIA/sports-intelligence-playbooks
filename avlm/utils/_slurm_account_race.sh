#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Multi-account Slurm racing submission (login node only).
#
# Sourced by sbatch_starter.sh before training starts. Kept separate from
# mode _train_lib.sh, which runs inside the container at train time.
#
# Submit the same job to several Slurm accounts in parallel; the first job to
# reach RUNNING wins and the rest are cancelled.
#
# Requires submit_one(account, out_file) to be defined by the caller.
# Slurm --job-name is MODEL_NAME-<account> so racing jobs don't block each other under singleton.

# Remove srun/sbatch logs for a cancelled race job (srun.sh naming: *_<jobid>_*.log, slurm_<jobid>.out).
RACE_POLL_INTERVAL="${RACE_POLL_INTERVAL:-5}"

_snapshot_submission_yaml() {
  local var_name="$1"
  local destination="$2"
  local source="${!var_name:-}"
  local source_path destination_path

  [[ -n "${source}" ]] || return 0
  if [[ "${source}" != /* && -n "${AVLM_BASE_REPO_ROOT:-}" ]]; then
    source="${AVLM_BASE_REPO_ROOT}/${source}"
  fi
  [[ -f "${source}" ]] || {
    echo "error: ${var_name} not found: ${source}" >&2
    return 1
  }

  source_path="$(readlink -f -- "${source}")"
  destination_path="$(readlink -m -- "${destination}")"
  if [[ "${source_path}" != "${destination_path}" ]]; then
    cp -- "${source_path}" "${destination_path}"
  fi

  printf -v "${var_name}" '%s' "${destination_path}"
  export "${var_name}"
}

_snapshot_submission_yamls() {
  : "${LOGS_DIR:?LOGS_DIR must be set before Slurm submission}"
  local snapshot_dir="${LOGS_DIR}/submitted_config"
  mkdir -p "${snapshot_dir}"

  _snapshot_submission_yaml CLUSTER_PARAMS "${snapshot_dir}/launch_local.yaml"
  _snapshot_submission_yaml CONFIG_YAML "${snapshot_dir}/training.yaml"
  _snapshot_submission_yaml INFERENCE_CONFIG "${snapshot_dir}/inference.yaml"
}

_race_remove_job_logs() {
  local job_id="$1"
  local logs_dir="${2:-${LOGS_DIR:-}}"
  [[ -n "${job_id}" && -n "${logs_dir}" && -d "${logs_dir}" ]] || return 0

  rm -f "${logs_dir}/slurm_${job_id}.out"
  shopt -s nullglob
  local f
  for f in "${logs_dir}"/*_"${job_id}"_*.log; do
    rm -f "${f}"
  done
  shopt -u nullglob
}

_race_cancel_losers() {
  local winning_job_id="$1"
  shift
  local -a job_ids=("$@")
  local i

  for ((i = 0; i < ${#job_ids[@]}; i++)); do
    if [[ "${job_ids[i]}" != "${winning_job_id}" ]]; then
      scancel "${job_ids[i]}" 2>/dev/null || true
    fi
  done

  # Let srun/sbatch release log file handles before unlink.
  sleep 3

  for ((i = 0; i < ${#job_ids[@]}; i++)); do
    if [[ "${job_ids[i]}" != "${winning_job_id}" ]]; then
      _race_remove_job_logs "${job_ids[i]}"
    fi
  done
}

run_slurm_account_race() {
  _snapshot_submission_yamls

  local -a accounts=("$@")
  local num_accounts=${#accounts[@]}

  if ((num_accounts < 1)); then
    echo "error: need at least 1 Slurm account" >&2
    return 1
  fi

  if ((num_accounts == 1)); then
    echo "[submit]"
    local out_file
    out_file="$(mktemp)"
    trap 'rm -f "${out_file}"' RETURN

    submit_one "${accounts[0]}" "${out_file}"
    local job_id
    job_id="$(sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' "${out_file}" | head -1)"
    if [[ -z "${job_id}" ]]; then
      echo "error: failed to get job ID for account ${accounts[0]}" >&2
      cat "${out_file}" >&2
      return 1
    fi
    printf "  %-24s  job %s\n" "${accounts[0]}" "${job_id}"
    if [[ -n "${LOGS_DIR:-}" ]]; then
      printf 'account\tjob_id\n%s\t%s\n' "${accounts[0]}" "${job_id}" >"${LOGS_DIR}/race_jobs.tsv"
    fi
    return 0
  fi

  echo "[submit]"
  export RACE_QUIET_SUBMIT=1

  local -a out_files=()
  local i
  for ((i = 0; i < num_accounts; i++)); do
    out_files+=("$(mktemp)")
  done
  trap 'rm -f "${out_files[@]}"' RETURN

  for ((i = 0; i < num_accounts; i++)); do
    submit_one "${accounts[i]}" "${out_files[i]}" &
  done
  wait
  unset RACE_QUIET_SUBMIT

  local -a job_ids=()
  for ((i = 0; i < num_accounts; i++)); do
    local id
    id="$(sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' "${out_files[i]}" | head -1)"
    if [[ -z "${id}" ]]; then
      echo "error: failed to get job ID for account ${accounts[i]}" >&2
      cat "${out_files[i]}" >&2
      return 1
    fi
    job_ids+=("${id}")
    printf "  %-24s  job %s\n" "${accounts[i]}" "${id}"
  done

  if [[ -n "${LOGS_DIR:-}" ]]; then
    {
      printf 'account\tjob_id\n'
      for ((i = 0; i < num_accounts; i++)); do
        printf '%s\t%s\n' "${accounts[i]}" "${job_ids[i]}"
      done
    } >"${LOGS_DIR}/race_jobs.tsv"
  fi

  local job_list
  job_list="$(IFS=,; echo "${job_ids[*]}")"
  echo ""
  echo "[race]"
  echo "  polling every ${RACE_POLL_INTERVAL}s for first RUNNING among jobs ${job_list}"

  local winning_job_id=""
  local empty_count=0
  local max_empty=6
  while true; do
    local states found_any=""
    states="$(squeue -j "${job_list}" -h -o "%i %T" 2>/dev/null || true)"
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      found_any=1
      local jobid="${line%% *}"
      jobid="${jobid%%_*}"
      local state="${line#* }"
      if [[ "${state}" == "RUNNING" ]]; then
        winning_job_id="${jobid}"
        break 2
      fi
    done <<EOF
${states}
EOF
    if [[ -z "${found_any}" ]]; then
      empty_count=$((empty_count + 1))
      echo "  warning: squeue returned no jobs (${empty_count}/${max_empty})"
      if ((empty_count >= max_empty)); then
        break
      fi
    else
      empty_count=0
    fi
    sleep "${RACE_POLL_INTERVAL}"
  done

  echo ""
  if [[ -n "${winning_job_id}" ]]; then
    _race_cancel_losers "${winning_job_id}" "${job_ids[@]}"
    for ((i = 0; i < num_accounts; i++)); do
      if [[ "${job_ids[i]}" == "${winning_job_id}" ]]; then
        if [[ -n "${LOGS_DIR:-}" ]]; then
          printf 'account\tjob_id\n%s\t%s\n' "${accounts[i]}" "${winning_job_id}" >"${LOGS_DIR}/race_jobs.tsv"
        fi
        echo "[result]"
        echo "  winner   job ${winning_job_id}  (${accounts[i]})"
        echo "  cancelled jobs (logs removed):"
        local j
        for ((j = 0; j < num_accounts; j++)); do
          if [[ "${job_ids[j]}" != "${winning_job_id}" ]]; then
            printf "    %-22s  job %s\n" "${accounts[j]}" "${job_ids[j]}"
          fi
        done
        return 0
      fi
    done
  fi

  for ((i = 0; i < num_accounts; i++)); do
    scancel "${job_ids[i]}" 2>/dev/null || true
  done
  sleep 3
  for ((i = 0; i < num_accounts; i++)); do
    _race_remove_job_logs "${job_ids[i]}"
  done
  echo "[result]"
  echo "  error: no job reached RUNNING; all ${num_accounts} jobs cancelled." >&2
  return 1
}
