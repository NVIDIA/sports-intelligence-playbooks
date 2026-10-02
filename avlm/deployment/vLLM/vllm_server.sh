#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start / stop / status for a persistent vLLM OpenAI server (inference only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${DEPLOY_ROOT}/../../.." && pwd)"
STATE_DIR="${VLLM_STATE_DIR:-${REPO_ROOT}/.cache/vllm_server}"
STATE_FILE="${STATE_DIR}/state.json"
SERVE_BIN="${SERVE_BIN:-${DEPLOY_ROOT}/serve.py}"
PYTHON="${PYTHON:-python}"

# shellcheck source=utils/_serve_params.sh
source "${DEPLOY_ROOT}/utils/_serve_params.sh"
# Ignore sticky diagnosis exports from the interactive shell. Serve YAML fills
# the keys it defines; the rest stay unset. MODEL_PATH, port, and TP still
# honor an explicit environment override.
unset VLLM_ATTENTION_BACKEND VLLM_USE_DEEP_GEMM VLLM_MOE_USE_DEEP_GEMM \
  VLLM_KV_CACHE_DTYPE VLLM_COMPILATION_CONFIG VLLM_MOE_BACKEND \
  VLLM_ENABLE_CHUNKED_PREFILL VLLM_ASYNC_SCHEDULING VLLM_ENABLE_FLASHINFER_AUTOTUNE \
  VLLM_DISABLE_COMPILE_CACHE VLLM_ENFORCE_EAGER CUBLAS_WORKSPACE_CONFIG
source_serve_params "${DEPLOY_ROOT}" "${REPO_ROOT}"
vllm_guess_cache_dir "${REPO_ROOT}"
if [[ "${1:-}" == "start" ]]; then
  # shellcheck source=utils/_vllm_env.sh
  source "${DEPLOY_ROOT}/utils/_vllm_env.sh"
  if _vllm_truthy "${VLLM_INSTALL_EXTRA:-0}"; then
    _vllm_install_missing_extras || exit 1
  else
    _vllm_reuse_extra_venv
  fi
fi
vllm_resolve_serve_python || exit 1

MODEL_PATH="${MODEL_PATH:-${MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME:-nemotron_3_nano_omni}}"
VLLM_PORT="${VLLM_PORT:-12500}"
# tensor_parallel_size and data_parallel_size in serve YAML — not forced to GPUS_PER_NODE.
# Env overrides: VLLM_TENSOR_PARALLEL_SIZE, VLLM_DATA_PARALLEL_SIZE.
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-${TP_SIZE:-1}}"
DP_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
# Per DP rank: waiting + running before vLLM returns HTTP 503. Unset = no cap.
MAX_NUM_QUEUED_REQS="${VLLM_MAX_NUM_QUEUED_REQS:-}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-2000}"
NUM_FRAMES="${NUM_FRAMES:-128}"
FPS="${FPS:-2}"
# hub_checkpoint: Hub video_io.py index math (see utils/hub_checkpoint_video_loader.py). opencv: stock vLLM.
VLLM_VIDEO_BACKEND="${VLLM_VIDEO_BACKEND:-hub_checkpoint}"
VIDEO_PRUNING_RATE="${VIDEO_PRUNING_RATE:-0.0}"
VLLM_ENABLE_REASONING="${VLLM_ENABLE_REASONING:-0}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-nemotron_v3}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_COMPILATION_CONFIG="${VLLM_COMPILATION_CONFIG:-}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-}"
VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-}"
VLLM_ENABLE_FLASHINFER_AUTOTUNE="${VLLM_ENABLE_FLASHINFER_AUTOTUNE:-}"

_vllm_truthy() {
  case "${1:-0}" in
    1 | true | True | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  echo "Usage: $0 {start|stop|status}" >&2
  echo "  start  — launch vLLM server in background" >&2
  echo "  VLLM_INSTALL_EXTRA=1 start — if decord (or VLLM_EXTRA_PIP_SPEC) is missing from the image, install it into a separate venv and serve with that" >&2
  echo "  stop   — stop tracked server, vLLM workers, and free serve ports" >&2
  echo "  status — print server health" >&2
}

_write_state() {
  mkdir -p "${STATE_DIR}"
  python3 - "${STATE_FILE}" "${1}" "${VLLM_PORT}" "${MODEL_PATH}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys

path, pid, port, model, served_name = sys.argv[1:6]
payload = {
    "pid": int(pid),
    "url": f"http://127.0.0.1:{port}",
    "model": model,
    "served_model_name": served_name,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY
}

_cmd_start() {
  if [[ -f "${STATE_FILE}" ]]; then
    pid="$(python3 -c "import json; print(json.load(open('${STATE_FILE}')).get('pid',''))" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[INFO] Server already running (pid=${pid}). URL=$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['url'])")/v1"
      return 0
    fi
  fi

  mkdir -p "${STATE_DIR}"
  LOG_FILE="${STATE_DIR}/server.log"
  vllm_resolve_serve_python || exit 1
  vllm_preflight_serve_python "${STATE_DIR}/serve_import_error.txt" || exit 1
  local reasoning_flag=no
  _vllm_truthy "${VLLM_ENABLE_REASONING}" && reasoning_flag=yes

  if ! [[ "${TP_SIZE}" =~ ^[0-9]+$ && "${DP_SIZE}" =~ ^[0-9]+$ ]] || ((TP_SIZE < 1 || DP_SIZE < 1)); then
    echo "[ERROR] tensor_parallel_size and data_parallel_size must be positive integers (got tp=${TP_SIZE} dp=${DP_SIZE})" >&2
    exit 1
  fi
  local gpu_count=$((DP_SIZE * TP_SIZE))
  local serve_cvd=""
  serve_cvd="$(vllm_resolve_serve_cuda_visible_devices "${gpu_count}")" || exit 1
  vllm_validate_tp_and_visible_devices "${gpu_count}" "${serve_cvd}" || exit 1

  local gpu_info="all visible in shell"
  if [[ -n "${serve_cvd}" ]]; then
    gpu_info="CUDA_VISIBLE_DEVICES=${serve_cvd}"
  fi
  local eager_flag=no
  _vllm_truthy "${VLLM_ENFORCE_EAGER}" && eager_flag=yes
  local attn_info="${VLLM_ATTENTION_BACKEND:-auto}"
  local kv_info="${VLLM_KV_CACHE_DTYPE:-auto}"
  local moe_info="${VLLM_MOE_BACKEND:-auto}"
  local compile_info="${VLLM_COMPILATION_CONFIG:-default}"
  local queued_reqs_info=off
  [[ -n "${MAX_NUM_QUEUED_REQS}" ]] && queued_reqs_info="${MAX_NUM_QUEUED_REQS}"
  echo "[INFO] Starting vLLM server: model=${MODEL_PATH} tp=${TP_SIZE} dp=${DP_SIZE} port=${VLLM_PORT} max_num_queued_reqs=${queued_reqs_info} dtype=${VLLM_DTYPE} enforce_eager=${eager_flag} attention_backend=${attn_info} kv_cache_dtype=${kv_info} moe_backend=${moe_info} compilation_config=${compile_info} reasoning=${reasoning_flag} video_backend=${VLLM_VIDEO_BACKEND} gpus=${gpu_info}"

  local -a serve_args=(
    --model "${MODEL_PATH}"
    --trust-remote-code
    --dtype "${VLLM_DTYPE}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --data-parallel-size "${DP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --allowed-local-media-path /
    --max-num-seqs "${MAX_NUM_SEQS}"
    --video-pruning-rate "${VIDEO_PRUNING_RATE}"
  )
  if [[ -n "${MAX_NUM_QUEUED_REQS}" ]]; then
    if ! [[ "${MAX_NUM_QUEUED_REQS}" =~ ^[0-9]+$ ]] || ((MAX_NUM_QUEUED_REQS < 1)); then
      echo "[ERROR] max_num_queued_reqs must be a positive integer (got: ${MAX_NUM_QUEUED_REQS})" >&2
      exit 1
    fi
    serve_args+=(--max-num-queued-reqs "${MAX_NUM_QUEUED_REQS}")
  fi
  serve_args+=(
    --media-io-kwargs "{\"video\": {\"fps\": ${FPS}, \"num_frames\": ${NUM_FRAMES}, \"video_backend\": \"${VLLM_VIDEO_BACKEND}\"}}"
  )
  if _vllm_truthy "${VLLM_ENABLE_REASONING}"; then
    serve_args+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  fi
  if _vllm_truthy "${VLLM_ENFORCE_EAGER}"; then
    serve_args+=(--enforce-eager)
  fi
  if [[ -n "${VLLM_ATTENTION_BACKEND}" ]]; then
    serve_args+=(--attention-backend "${VLLM_ATTENTION_BACKEND}")
  fi
  if [[ -n "${VLLM_KV_CACHE_DTYPE}" ]]; then
    serve_args+=(--kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}")
  fi
  if [[ -n "${VLLM_COMPILATION_CONFIG}" ]]; then
    serve_args+=(--compilation-config "${VLLM_COMPILATION_CONFIG}")
  fi
  if [[ -n "${VLLM_MOE_BACKEND}" ]]; then
    serve_args+=(--moe-backend "${VLLM_MOE_BACKEND}")
  fi
  if [[ -n "${VLLM_ENABLE_CHUNKED_PREFILL}" ]]; then
    if _vllm_truthy "${VLLM_ENABLE_CHUNKED_PREFILL}"; then
      serve_args+=(--enable-chunked-prefill)
    else
      serve_args+=(--no-enable-chunked-prefill)
    fi
  fi
  if [[ -n "${VLLM_ASYNC_SCHEDULING}" ]]; then
    if _vllm_truthy "${VLLM_ASYNC_SCHEDULING}"; then
      serve_args+=(--async-scheduling)
    else
      serve_args+=(--no-async-scheduling)
    fi
  fi
  if [[ -n "${VLLM_ENABLE_FLASHINFER_AUTOTUNE}" ]]; then
    if _vllm_truthy "${VLLM_ENABLE_FLASHINFER_AUTOTUNE}"; then
      serve_args+=(--enable-flashinfer-autotune)
    else
      serve_args+=(--no-enable-flashinfer-autotune)
    fi
  fi

  if _vllm_truthy "${VLLM_DISABLE_COMPILE_CACHE:-0}"; then
    export VLLM_DISABLE_COMPILE_CACHE=1
  else
    unset VLLM_DISABLE_COMPILE_CACHE || true
  fi

  vllm_export_serve_runtime_caches || exit 1
  vllm_assert_serve_port_free "${VLLM_PORT}" || exit 1

  echo "[INFO] Launch: ${PYTHON} ${SERVE_BIN} (log=${LOG_FILE})" >&2
  if [[ -n "${serve_cvd}" ]]; then
    CUDA_VISIBLE_DEVICES="${serve_cvd}" \
      nohup "${PYTHON}" "${SERVE_BIN}" \
      "${serve_args[@]}" \
      > "${LOG_FILE}" 2>&1 &
  else
    nohup "${PYTHON}" "${SERVE_BIN}" \
      "${serve_args[@]}" \
      > "${LOG_FILE}" 2>&1 &
  fi

  pid=$!
  _write_state "${pid}"

  for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] vLLM server exited early (pid=${pid}). Tail of ${LOG_FILE}:" >&2
      if [[ ! -s "${LOG_FILE}" ]]; then
        echo "[ERROR] server.log is empty — run manually: CUDA_VISIBLE_DEVICES=${serve_cvd:-*} ${PYTHON} ${SERVE_BIN} ..." >&2
      fi
      tail -n 80 "${LOG_FILE}" >&2 || true
      rm -f "${STATE_FILE}"
      exit 1
    fi
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
      echo "[INFO] Server ready: http://127.0.0.1:${VLLM_PORT}/v1 (pid=${pid})"
      echo "[INFO] Log: ${LOG_FILE}"
      return 0
    fi
    sleep 1
  done

  echo "[ERROR] Server did not become healthy within ${STARTUP_TIMEOUT}s" >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  exit 1
}

_vllm_pkill_pattern() {
  local sig="$1"
  local pattern="$2"
  pkill "-${sig}" -f "${pattern}" 2>/dev/null || true
}

_vllm_kill_descendants() {
  local root_pid="$1"
  local child
  for child in $(pgrep -P "${root_pid}" 2>/dev/null || true); do
    _vllm_kill_descendants "${child}"
  done
  kill -TERM "${root_pid}" 2>/dev/null || true
}

_vllm_kill_tcp_listeners() {
  local port="$1"
  local sig_name="${2:-TERM}"
  local kill_flag="-${sig_name}"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
    return 0
  fi
  if command -v lsof >/dev/null 2>&1; then
    local pid
    for pid in $(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true); do
      kill "${kill_flag}" "${pid}" 2>/dev/null || true
    done
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    local pid
    while read -r pid; do
      [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]] || continue
      kill "${kill_flag}" "${pid}" 2>/dev/null || true
    done < <(ss -H -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p' || true)
    return 0
  fi
  python3 - "${port}" "${sig_name}" <<'PY'
import os
import signal
import sys

port = int(sys.argv[1])
sig_name = sys.argv[2].upper()
sig = signal.SIGKILL if sig_name == "KILL" else signal.SIGTERM
port_hex = format(port, "04X")
inodes: set[str] = set()
for fname in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(fname, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 10 or parts[3] != "0A":
                    continue
                local_port = parts[1].split(":")[1].upper()
                if local_port == port_hex:
                    inodes.add(parts[9])
    except OSError:
        continue
if not inodes:
    raise SystemExit(0)
my_pid = os.getpid()
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    if pid == my_pid:
        continue
    fd_dir = f"/proc/{entry}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                link = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if link.startswith("socket:[") and link[8:-1] in inodes:
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                break
    except OSError:
        continue
PY
}

_vllm_free_serve_ports() {
  local span="${VLLM_STOP_PORT_SPAN:-6}"
  local p end
  end=$((VLLM_PORT + span - 1))
  for ((p = VLLM_PORT; p <= end; p++)); do
    _vllm_kill_tcp_listeners "${p}" TERM
  done
  sleep 1
  for ((p = VLLM_PORT; p <= end; p++)); do
    _vllm_kill_tcp_listeners "${p}" KILL
  done
}

_vllm_stop_orphans() {
  local serve_pat="${SERVE_BIN}"
  _vllm_pkill_pattern TERM "${serve_pat}"
  sleep 2
  _vllm_pkill_pattern KILL "${serve_pat}"

  if _vllm_truthy "${VLLM_STOP_KILL_ALL_WORKERS:-1}"; then
    _vllm_pkill_pattern TERM 'VLLM::Worker'
    _vllm_pkill_pattern TERM 'VLLM::Engine'
    sleep 2
    _vllm_pkill_pattern KILL 'VLLM::Worker'
    _vllm_pkill_pattern KILL 'VLLM::Engine'
  fi

  _vllm_free_serve_ports
}

_cmd_stop() {
  local pid=""
  if [[ -f "${STATE_FILE}" ]]; then
    pid="$(python3 -c "import json; print(json.load(open('${STATE_FILE}')).get('pid',''))" 2>/dev/null || true)"
  fi

  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[INFO] Stopping vLLM server pid=${pid} (process tree)"
    _vllm_kill_descendants "${pid}"
    sleep 2
    kill -KILL "${pid}" 2>/dev/null || true
  elif [[ -n "${pid}" ]]; then
    echo "[INFO] Tracked pid=${pid} is not running"
  else
    echo "[INFO] No server tracked in ${STATE_FILE}"
  fi

  rm -f "${STATE_FILE}"

  local span="${VLLM_STOP_PORT_SPAN:-6}"
  echo "[INFO] Cleaning serve.py, vLLM workers, and TCP ports ${VLLM_PORT}-$((VLLM_PORT + span - 1))"
  _vllm_stop_orphans
  echo "[INFO] Server stopped"
}

_cmd_status() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo "[INFO] Server not running (no state file)"
    return 1
  fi
  pid="$(python3 -c "import json; print(json.load(open('${STATE_FILE}')).get('pid',''))")"
  url="$(python3 -c "import json; print(json.load(open('${STATE_FILE}'))['url'])")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[INFO] pid=${pid} url=${url}/v1"
    curl -sf "${url}/health" >/dev/null && echo "[INFO] health: OK" || echo "[WARN] health: FAIL"
    return 0
  fi
  echo "[INFO] State file exists but pid ${pid} is not running"
  return 1
}

case "${1:-}" in
  start) _cmd_start ;;
  stop) _cmd_stop ;;
  status) _cmd_status ;;
  *) usage; exit 2 ;;
esac
