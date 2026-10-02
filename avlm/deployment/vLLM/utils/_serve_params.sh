#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Serve setup: YAML → env, venv Python, and CUDA_VISIBLE_DEVICES vs data parallel × tensor parallel.
# Env exports override YAML.

# shellcheck shell=bash

_SERVE_PARAM_KEYS=(
  MODEL_PATH
  SERVED_MODEL_NAME
  VLLM_PORT
  VLLM_TENSOR_PARALLEL_SIZE
  VLLM_DATA_PARALLEL_SIZE
  VLLM_MAX_MODEL_LEN
  VLLM_MAX_NUM_SEQS
  VLLM_MAX_NUM_QUEUED_REQS
  VLLM_STARTUP_TIMEOUT
  VLLM_ENABLE_REASONING
  VLLM_REASONING_PARSER
  VLLM_DEFAULT_MAX_TOKENS
  VLLM_DEFAULT_STREAM
  VLLM_PIN_SERVE_GPUS
  VLLM_SERVE_CUDA_VISIBLE_DEVICES
  NUM_FRAMES
  FPS
  VLLM_VIDEO_BACKEND
  VIDEO_PRUNING_RATE
  VLLM_DTYPE
  VLLM_ENFORCE_EAGER
  VLLM_ATTENTION_BACKEND
  VLLM_KV_CACHE_DTYPE
  VLLM_COMPILATION_CONFIG
  VLLM_MOE_BACKEND
  VLLM_ENABLE_CHUNKED_PREFILL
  VLLM_ASYNC_SCHEDULING
  VLLM_ENABLE_FLASHINFER_AUTOTUNE
  CACHE_DIR
)

_expand_repo_root_in_var() {
  local key="$1" repo_root="$2"
  local val="${!key:-}"
  [[ -n "${val}" ]] || return 0
  val="${val//\$\{REPO_ROOT\}/${repo_root}}"
  export "${key}=${val}"
}

_default_serve_config_path() {
  local deploy_root="$1"
  local local_cfg="${deploy_root}/configs/vllm_server_params_local.yaml"
  local template_cfg="${deploy_root}/configs/vllm_server_params.yaml"
  if [[ -f "${local_cfg}" ]]; then
    printf '%s\n' "${local_cfg}"
  else
    printf '%s\n' "${template_cfg}"
  fi
}

source_serve_params() {
  local deploy_root="${1:?deploy_root required}"
  local repo_root="${2:?repo_root required}"
  local cfg="${SERVE_CONFIG:-$(_default_serve_config_path "${deploy_root}")}"

  if [[ "${cfg}" != /* ]]; then
    cfg="${repo_root}/${cfg}"
  fi
  if [[ ! -f "${cfg}" ]]; then
    echo "[WARN] serve config not found: ${cfg}; using env/defaults only" >&2
    return 0
  fi

  local avlm_utils _py yaml_exports key
  avlm_utils="$(cd -- "${deploy_root}/../../utils" && pwd)"
  _py=python3
  command -v python3 >/dev/null || {
    echo "[ERROR] python3 required to load serve config" >&2
    return 1
  }

  yaml_exports="$("${_py}" "${avlm_utils}/_load_params_yaml.py" "${cfg}" \
    MODEL_PATH=model_path \
    SERVED_MODEL_NAME=served_model_name \
    VLLM_PORT=vllm_port \
    VLLM_TENSOR_PARALLEL_SIZE=tensor_parallel_size \
    VLLM_DATA_PARALLEL_SIZE=data_parallel_size \
    VLLM_MAX_MODEL_LEN=max_model_len \
    VLLM_MAX_NUM_SEQS=max_num_seqs \
    VLLM_MAX_NUM_QUEUED_REQS=max_num_queued_reqs \
    VLLM_STARTUP_TIMEOUT=startup_timeout \
    VLLM_ENABLE_REASONING=enable_reasoning \
    VLLM_REASONING_PARSER=reasoning_parser \
    VLLM_DEFAULT_MAX_TOKENS=max_tokens \
    VLLM_DEFAULT_STREAM=stream \
    VLLM_PIN_SERVE_GPUS=pin_serve_gpus \
    VLLM_SERVE_CUDA_VISIBLE_DEVICES=cuda_visible_devices \
    NUM_FRAMES=num_frames \
    FPS=fps \
    VLLM_VIDEO_BACKEND=video_backend \
    VIDEO_PRUNING_RATE=video_pruning_rate \
    VLLM_DTYPE=vllm_dtype \
    VLLM_ENFORCE_EAGER=vllm_enforce_eager \
    VLLM_ATTENTION_BACKEND=vllm_attention_backend \
    VLLM_KV_CACHE_DTYPE=vllm_kv_cache_dtype \
    VLLM_COMPILATION_CONFIG=vllm_compilation_config \
    VLLM_MOE_BACKEND=vllm_moe_backend \
    VLLM_ENABLE_CHUNKED_PREFILL=vllm_enable_chunked_prefill \
    VLLM_ASYNC_SCHEDULING=vllm_async_scheduling \
    VLLM_ENABLE_FLASHINFER_AUTOTUNE=vllm_enable_flashinfer_autotune \
    CACHE_DIR=cache_dir)"

  declare -A _serve_saved=()
  for key in "${_SERVE_PARAM_KEYS[@]}"; do
    if [[ -v "${key}" ]]; then
      _serve_saved["${key}"]="${!key}"
    fi
  done

  # shellcheck disable=SC2086
  eval "${yaml_exports}"

  for key in "${_SERVE_PARAM_KEYS[@]}"; do
    if [[ -n "${_serve_saved[${key}]+x}" ]]; then
      export "${key}=${_serve_saved[${key}]}"
    fi
  done

  _expand_repo_root_in_var MODEL_PATH "${repo_root}"
  _expand_repo_root_in_var CACHE_DIR "${repo_root}"
  echo "[INFO] serve params from ${cfg}" >&2
}

_vllm_truthy() {
  case "${1:-0}" in
    1 | true | True | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

_vllm_count_csv_devices() {
  local csv="$1" n=0
  [[ -n "${csv}" ]] || {
    echo 0
    return
  }
  n=$((1 + $(grep -o ',' <<< "${csv}" | wc -l)))
  echo "${n}"
}

# Prints CUDA_VISIBLE_DEVICES to use, or empty to leave the shell env unchanged.
# gpu_count is data_parallel_size * tensor_parallel_size.
vllm_resolve_serve_cuda_visible_devices() {
  local gpu_count="$1"

  if [[ -n "${VLLM_SERVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s' "${VLLM_SERVE_CUDA_VISIBLE_DEVICES}"
    return 0
  fi

  if ! _vllm_truthy "${VLLM_PIN_SERVE_GPUS:-1}"; then
    return 0
  fi

  if ! [[ "${gpu_count}" =~ ^[0-9]+$ ]] || ((gpu_count < 1)); then
    echo "[ERROR] data_parallel_size * tensor_parallel_size must be a positive integer (got: ${gpu_count})" >&2
    return 1
  fi

  if ((gpu_count == 1)); then
    printf '0'
    return 0
  fi

  local ids=() i
  for ((i = 0; i < gpu_count; i++)); do
    ids+=("${i}")
  done
  local IFS=,
  printf '%s' "${ids[*]}"
}

vllm_validate_tp_and_visible_devices() {
  local gpu_count="$1" cvd="$2"
  [[ -n "${cvd}" ]] || return 0

  local n
  n="$(_vllm_count_csv_devices "${cvd}")"
  if ((n != gpu_count)); then
    echo "[ERROR] data_parallel_size * tensor_parallel_size=${gpu_count} but CUDA_VISIBLE_DEVICES='${cvd}' lists ${n} device(s)" >&2
    echo "[ERROR] Set cuda_visible_devices in serve YAML or export VLLM_SERVE_CUDA_VISIBLE_DEVICES to that many GPUs." >&2
    return 1
  fi
  return 0
}

vllm_guess_cache_dir() {
  [[ -n "${CACHE_DIR:-}" ]] && return 0
  local repo_root="${1:?repo_root required}"
  export CACHE_DIR="${repo_root}/.cache"
  echo "[INFO] CACHE_DIR=${CACHE_DIR}" >&2
}

# Serve temp and compile caches live under cache_dir (serve YAML). vLLM ZMQ IPC paths must be
# ≤107 bytes, so TMPDIR is a short symlink into <cache_dir>/serve_runtime/tmp.
vllm_export_serve_runtime_caches() {
  local base="${CACHE_DIR:-}"
  if [[ -z "${base}" ]]; then
    echo "[WARN] CACHE_DIR unset; serve may write compile caches under /tmp or \$HOME" >&2
    return 0
  fi
  local runtime_root="${base}/serve_runtime"
  local runtime_tmp="${VLLM_SERVE_TMPDIR:-${runtime_root}/tmp}"
  local tmp_tag="${USER:-run}"
  tmp_tag="${tmp_tag//[^a-zA-Z0-9_-]/_}"
  local ipc_tmp="${VLLM_SERVE_IPC_TMPDIR:-/tmp/vllm-serve-${tmp_tag}}"
  local triton_dir="${TRITON_CACHE_DIR:-${runtime_root}/triton}"
  local inductor_dir="${TORCHINDUCTOR_CACHE_DIR:-${runtime_root}/torch_inductor}"
  local xdg_dir="${XDG_CACHE_HOME:-${runtime_root}/xdg}"
  local hf_home="${HF_HOME:-${base}/huggingface}"
  local hf_hub="${HUGGINGFACE_HUB_CACHE:-${hf_home}/hub}"

  mkdir -p "${runtime_tmp}" "${triton_dir}" "${inductor_dir}" "${xdg_dir}" "${hf_hub}" || {
    echo "[ERROR] Failed to create serve runtime cache dirs under ${runtime_root}" >&2
    return 1
  }

  if [[ -d "${ipc_tmp}" && ! -L "${ipc_tmp}" ]]; then
    echo "[INFO] Replacing ${ipc_tmp} (old directory) with symlink to ${runtime_tmp}" >&2
    rm -rf "${ipc_tmp}"
  elif [[ -e "${ipc_tmp}" && ! -L "${ipc_tmp}" ]]; then
    echo "[ERROR] ${ipc_tmp} exists and is not a directory or symlink." >&2
    echo "[ERROR] Remove it or set VLLM_SERVE_IPC_TMPDIR to another short path under /tmp." >&2
    return 1
  fi
  ln -sfn "${runtime_tmp}" "${ipc_tmp}"

  export TMPDIR="${ipc_tmp}"
  export TEMP="${VLLM_SERVE_TEMP:-${TMPDIR}}"
  export TMP="${VLLM_SERVE_TMP:-${TMPDIR}}"
  export TRITON_CACHE_DIR="${triton_dir}"
  export TORCHINDUCTOR_CACHE_DIR="${inductor_dir}"
  export XDG_CACHE_HOME="${xdg_dir}"
  export HF_HOME="${hf_home}"
  export HUGGINGFACE_HUB_CACHE="${hf_hub}"

  local probe="${runtime_tmp}/.vllm_write_probe"
  if ! ( : >"${probe}" ) 2>/dev/null; then
    echo "[ERROR] Disk quota or permissions block writes under CACHE_DIR=${base} (tmp=${runtime_tmp})" >&2
    echo "[ERROR] Free quota on that filesystem or set VLLM_SERVE_TMPDIR to a writable path under cache_dir." >&2
    return 1
  fi
  rm -f "${probe}"

  echo "[INFO] serve runtime caches under ${runtime_root}: tmp=${runtime_tmp} (TMPDIR=${TMPDIR} symlink) triton=${TRITON_CACHE_DIR} torch_inductor=${TORCHINDUCTOR_CACHE_DIR} xdg=${XDG_CACHE_HOME}" >&2
  return 0
}

vllm_assert_serve_port_free() {
  local port="${1:?port required}"
  if python3 - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        raise SystemExit(1)
PY
  then
    return 0
  fi
  echo "[ERROR] Port ${port} is already in use (vLLM would bind another port and clients would fail)." >&2
  echo "[ERROR] Run: bash avlm/deployment/vLLM/vllm_server.sh stop" >&2
  return 1
}

_vllm_install_packages_enabled() {
  case "${INSTALL_VLLM_PACKAGES:-0}" in
    1 | true | True | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

vllm_resolve_serve_python() {
  if ! _vllm_install_packages_enabled; then
    if [[ -n "${VLLM_EXTRA_PIP_SPEC:-}" ]]; then
      local extra_py="${VLLM_EXTRA_VENV_DIR:-${CACHE_DIR}/vllm_extra_venv}/bin/python"
      if [[ -x "${extra_py}" ]]; then
        PYTHON="${extra_py}"
        export PYTHON
        echo "[INFO] serve PYTHON=${PYTHON} (slim extra venv)" >&2
        return 0
      fi
      echo "[ERROR] VLLM_EXTRA_PIP_SPEC is set but ${extra_py} is missing." >&2
      echo "[ERROR] Create it with: VLLM_INSTALL_EXTRA=1 bash avlm/deployment/vLLM/vllm_server.sh start" >&2
      return 1
    fi
    if [[ -n "${PYTHON:-}" && "${PYTHON}" != "python" && -x "${PYTHON}" && "${PYTHON}" != *"/vllm_serve_venv/"* && "${PYTHON}" != *"/vllm_extra_venv/"* ]]; then
      echo "[INFO] serve PYTHON=${PYTHON} (container; package install off)" >&2
      return 0
    fi
    if [[ -x /opt/venv/bin/python3 ]]; then
      PYTHON="/opt/venv/bin/python3"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON="$(command -v python3)"
    else
      echo "[ERROR] No container Python found. Set INSTALL_VLLM_PACKAGES=true or export PYTHON." >&2
      return 1
    fi
    export PYTHON
    echo "[INFO] serve PYTHON=${PYTHON} (container; package install off)" >&2
    return 0
  fi
  if [[ -n "${PYTHON:-}" && "${PYTHON}" != "python" && -x "${PYTHON}" ]]; then
    return 0
  fi
  local candidate=""
  local -a candidates=()
  if [[ -n "${VLLM_VENV_DIR:-}" ]]; then
    candidates+=("${VLLM_VENV_DIR}/bin/python" "${VLLM_VENV_DIR}/bin/python3")
  fi
  if [[ -n "${CACHE_DIR:-}" ]]; then
    candidates+=("${CACHE_DIR}/vllm_serve_venv/bin/python" "${CACHE_DIR}/vllm_serve_venv/bin/python3")
  fi
  if ((${#candidates[@]} > 0)); then
    for candidate in "${candidates[@]}"; do
      if [[ -x "${candidate}" ]]; then
        PYTHON="${candidate}"
        export PYTHON
        echo "[INFO] serve PYTHON=${PYTHON}" >&2
        return 0
      fi
    done
    echo "[ERROR] vLLM venv Python is not executable here: ${candidates[0]}" >&2
    echo "[ERROR] Start inside the NGC vLLM container (see avlm/deployment/vLLM/README.md), then: VLLM_INSTALL_EXTRA=1 bash avlm/deployment/vLLM/vllm_server.sh start" >&2
    return 1
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    export PYTHON
    echo "[WARN] serve PYTHON fallback=${PYTHON} (prefer vllm_serve_venv)" >&2
    return 0
  fi
  echo "[ERROR] No Python for vLLM serve. Export PYTHON=\${CACHE_DIR}/vllm_serve_venv/bin/python" >&2
  return 1
}

vllm_preflight_serve_python() {
  local err_file="${1:?err_file required}"
  if ! "${PYTHON}" -c "import vllm" 2>"${err_file}"; then
    echo "[ERROR] ${PYTHON} cannot import vllm:" >&2
    cat "${err_file}" >&2 || true
    return 1
  fi
  return 0
}
