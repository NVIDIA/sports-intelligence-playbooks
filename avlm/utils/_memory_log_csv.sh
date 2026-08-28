#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Background host/GPU memory CSV logger for MEMORY_LOG=1 (invoked from _train_lib.sh).
# Usage: _memory_log_csv.sh TRAIN_LOG CSV_FILE INTERVAL_SEC TRAIN_PID
set -euo pipefail

[[ $# -ge 4 ]] || {
  echo "usage: _memory_log_csv.sh TRAIN_LOG CSV_FILE INTERVAL_SEC TRAIN_PID" >&2
  exit 1
}

train_log="$1"
csv_file="$2"
interval="${3:-${MONITOR_INTERVAL:-5}}"
train_pid="$4"

export MONITOR_GPU="${MONITOR_GPU:-1}"
export MONITOR_PSS="${MONITOR_PSS:-1}"

_kb_to_gib_num() {
  awk -v kb="${1:-0}" 'BEGIN{printf "%.2f", kb/1024/1024}'
}

_mib_to_gib_num() {
  awk -v mib="${1:-0}" 'BEGIN{printf "%.2f", mib/1024}'
}

_latest_train_iter() {
  local log="$1"
  [[ -n "$log" && -f "$log" ]] || return 0
  local iter=""
  iter="$(grep -Eo 'iteration[[:space:]]+[0-9]+' "$log" 2>/dev/null | awk '{print $2}' | tail -1 || true)"
  [[ -n "$iter" ]] && { echo "$iter"; return 0; }
  iter="$(grep -Eo 'validation loss at iteration [0-9]+' "$log" 2>/dev/null | awk '{print $NF}' | tail -1 || true)"
  [[ -n "$iter" ]] && { echo "$iter"; return 0; }
  grep -Eo 'Starting training loop at iteration [0-9]+' "$log" 2>/dev/null | awk '{print $NF}' | tail -1 || true
}

_pss_kb_for_pid() {
  local pid="$1"
  local pss
  pss="$(awk '/^Pss:/{print $2; exit}' "/proc/${pid}/smaps_rollup" 2>/dev/null || true)"
  if [[ -n "$pss" ]]; then
    echo "$pss"
  else
    awk '/^VmRSS:/{print $2; exit}' "/proc/${pid}/status" 2>/dev/null || echo 0
  fi
}

_collect_descendant_pids() {
  local root="$1"
  local -a queue=("$root")
  local -A seen=()
  local pid cpid
  while ((${#queue[@]})); do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    [[ -n "${seen[$pid]+x}" ]] && continue
    [[ -r "/proc/${pid}/stat" ]] || continue
    seen["$pid"]=1
    printf '%s\n' "$pid"
    while read -r cpid; do
      [[ -n "$cpid" ]] && queue+=("$cpid")
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  done
}

_train_job_mem_stats_rss() {
  local user="$1"
  ps -u "$user" -o pid=,ppid=,rss=,args= 2>/dev/null | awk '
    function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
    function is_torchrun_root(args) {
      return args ~ /python/ && args ~ /torch\.distributed\.(run|launch)/ && args ~ /run_recipe(_avlm)?\.py/
    }
    function is_rank(args) { return args ~ /run_recipe(_avlm)?\.py/ }
    function is_worker(args) {
      return args ~ /spawn_main/ || args ~ /multiprocessing\.(spawn|resource_tracker)/
    }
    function is_launcher(args) {
      return args ~ /torch\.distributed\.(run|launch)/ || args ~ /torchrun/
    }
    function classify(args,   k) {
      if (is_worker(args)) return "worker"
      if (is_rank(args)) return "rank"
      if (is_launcher(args)) return "launcher"
      return "other"
    }
    function bfs_count(root,    q, head, pid, cpid, n) {
      delete in_tree; n = 0; q[1] = root; head = 1
      while (head <= length(q)) {
        pid = q[head++]
        if (pid in in_tree) continue
        in_tree[pid] = 1; n++
        for (cpid in child_of) {
          if (child_of[cpid] == pid && !(cpid in in_tree)) q[length(q) + 1] = cpid
        }
      }
      return n
    }
    {
      pid = $1; ppid = $2; rss = $3
      args = $0; sub(/^[ \t]*[0-9]+[ \t]+[0-9]+[ \t]+[0-9]+[ \t]+/, "", args)
      pid = trim(pid); ppid = trim(ppid); rss = trim(rss) + 0
      child_of[pid] = ppid; rss_of[pid] = rss; args_of[pid] = args
      if (args !~ /python/) next
      if (is_torchrun_root(args)) roots[++nroots] = pid
    }
    END {
      if (nroots == 0) {
        for (pid in args_of) {
          if (args_of[pid] ~ /python/ && is_rank(args_of[pid])) roots[++nroots] = pid
        }
      }
      if (nroots == 0) { print "0 0 0 0 0 0 0"; exit }
      best_root = roots[1]; best_n = 0
      for (i = 1; i <= nroots; i++) {
        cn = bfs_count(roots[i])
        if (cn > best_n) { best_n = cn; best_root = roots[i] }
      }
      delete in_tree
      q[1] = best_root; head = 1
      n_total = 0; mem_total = 0; n_rank = 0; mem_rank = 0; n_worker = 0; mem_worker = 0
      while (head <= length(q)) {
        pid = q[head++]
        if (pid in in_tree) continue
        in_tree[pid] = 1
        args = args_of[pid]
        if (args !~ /python/) continue
        rss = rss_of[pid] + 0
        k = classify(args)
        if (k == "other") {
          for (cpid in child_of) {
            if (child_of[cpid] == pid && !(cpid in in_tree)) q[length(q) + 1] = cpid
          }
          continue
        }
        n_total++; mem_total += rss
        if (k == "worker") { n_worker++; mem_worker += rss }
        else { n_rank++; mem_rank += rss }
        for (cpid in child_of) {
          if (child_of[cpid] == pid && !(cpid in in_tree)) q[length(q) + 1] = cpid
        }
      }
      print n_total, mem_total, n_rank, mem_rank, n_worker, mem_worker, 1
    }'
}

_train_job_mem_stats_pss() {
  local user="$1"
  local -a roots=()
  local best_root="" best_n=0 root_n root pid
  local n_total=0 mem_total=0 n_rank=0 mem_rank=0 n_worker=0 mem_worker=0
  local cmd kind pss
  local -A counted=()

  mapfile -t roots < <(
    ps -u "$user" -o pid=,args= 2>/dev/null | awk '
      /python/ && /torch\.distributed\.(run|launch)/ && /run_recipe(_avlm)?\.py/ { print $1 }'
  )
  if ((${#roots[@]} == 0)); then
    mapfile -t roots < <(
      ps -u "$user" -o pid=,args= 2>/dev/null | awk '/python/ && /run_recipe(_avlm)?\.py/ { print $1 }'
    )
  elif ((${#roots[@]} > 1)); then
    for root in "${roots[@]}"; do
      root_n="$( (_collect_descendant_pids "$root"; echo "$root") | sort -u | wc -l | tr -d ' ')"
      if ((root_n > best_n)); then best_n=$root_n; best_root="$root"; fi
    done
    roots=("$best_root")
  fi
  if ((${#roots[@]} == 0)); then printf '0 0 0 0 0 0 0'; return 0; fi
  for root in "${roots[@]}"; do
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      [[ -n "${counted[$pid]+x}" ]] && continue
      counted["$pid"]=1
      cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      [[ "$cmd" == *python* ]] || continue
      pss="$(_pss_kb_for_pid "$pid")"
      if [[ "$cmd" == *spawn_main* || "$cmd" == *multiprocessing.* ]]; then kind=worker
      elif [[ "$cmd" == *run_recipe* || "$cmd" == *torch.distributed* ]]; then kind=rank
      else continue; fi
      n_total=$((n_total + 1)); mem_total=$((mem_total + pss))
      if [[ "$kind" == worker ]]; then n_worker=$((n_worker + 1)); mem_worker=$((mem_worker + pss))
      else n_rank=$((n_rank + 1)); mem_rank=$((mem_rank + pss)); fi
    done < <(_collect_descendant_pids "$root"; printf '%s\n' "$root")
  done
  printf '%d %d %d %d %d %d %d\n' "$n_total" "$mem_total" "$n_rank" "$mem_rank" "$n_worker" "$mem_worker" "${#roots[@]}"
}

_train_job_mem_stats() {
  local user="${MONITOR_USER:-$(id -un)}"
  if [[ "${MONITOR_PSS:-0}" == 1 ]]; then _train_job_mem_stats_pss "$user"
  else _train_job_mem_stats_rss "$user"; fi
}

_host_mem_kb() {
  awk '
    /^MemTotal:/     { total = $2 }
    /^MemAvailable:/ { avail = $2 }
    /^AnonPages:/    { anon = $2 }
    /^Cached:/       { cached = $2 }
    END { printf "%d %d %d %d\n", total - avail + 0, avail + 0, anon + 0, cached + 0 }' /proc/meminfo
}

_gpu_mem_totals_only() {
  if [[ "${MONITOR_GPU:-0}" != 1 ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '0 0 0\n'
    return 0
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F',' '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); gsub(/ MiB$/, "", s); return s }
    { u = trim($1); t = trim($2); if (u ~ /^[0-9]+$/ && t ~ /^[0-9]+$/) { used += u; total += t; n++ } }
    END { printf "%d %d %d", used + 0, total + 0, n + 0 }'
}

_write_csv_header() {
  echo "timestamp_utc,train_iter,mem_mode,py_procs,py_ranks,py_workers,train_mem_gib,rank_mem_gib,worker_mem_gib,mem_used_gib,mem_avail_gib,anon_gib,cached_gib,gpu_used_gib,gpu_total_gib,gpu_count" >"$csv_file"
}

_write_run_metadata() {
  cat >"${csv_file%.csv}_meta.env" <<EOF
# Memory log metadata ($(date -u '+%Y-%m-%dT%H:%M:%SZ'))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-}
USE_SEQUENCE_PACKING=${USE_SEQUENCE_PACKING:-}
MAX_VIDEO_FRAMES=${MAX_VIDEO_FRAMES:-}
SEQ_LENGTH=${SEQ_LENGTH:-}
PACK_SIZE=${PACK_SIZE:-}
MONITOR_INTERVAL=${interval}
MONITOR_PSS=${MONITOR_PSS}
MONITOR_GPU=${MONITOR_GPU}
CSV_FILE=${csv_file}
TRAIN_LOG=${train_log}
EOF
}

_sample_csv_row() {
  local iter mem_used_kb mem_avail_kb anon_kb cached_kb
  local npy mem_kb nrk rank_mem_kb nw worker_mem_kb n_jobs
  local gpu_used_mib gpu_total_mib gpu_count mem_mode csv_ts

  read -r mem_used_kb mem_avail_kb anon_kb cached_kb <<<"$(_host_mem_kb)"
  read -r npy mem_kb nrk rank_mem_kb nw worker_mem_kb n_jobs <<<"$(_train_job_mem_stats)"
  iter="$(_latest_train_iter "$train_log")"
  read -r gpu_used_mib gpu_total_mib gpu_count <<<"$(_gpu_mem_totals_only)"
  mem_mode="$([[ "$MONITOR_PSS" == 1 ]] && echo pss || echo rss)"
  csv_ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "${csv_ts},${iter:-},${mem_mode},${npy},${nrk},${nw},$(_kb_to_gib_num "${mem_kb}"),$(_kb_to_gib_num "${rank_mem_kb}"),$(_kb_to_gib_num "${worker_mem_kb}"),$(_kb_to_gib_num "${mem_used_kb}"),$(_kb_to_gib_num "${mem_avail_kb}"),$(_kb_to_gib_num "${anon_kb}"),$(_kb_to_gib_num "${cached_kb}"),$(_mib_to_gib_num "${gpu_used_mib}"),$(_mib_to_gib_num "${gpu_total_mib}"),${gpu_count:-0}" >>"$csv_file"
}

[[ -f "$csv_file" ]] || { _write_csv_header; _write_run_metadata; }

while kill -0 "$train_pid" 2>/dev/null; do
  _sample_csv_row
  sleep "$interval"
done
_sample_csv_row
