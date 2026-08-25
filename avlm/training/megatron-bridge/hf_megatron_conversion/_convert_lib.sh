# Megatron-Bridge HF ↔ Megatron conversion. Sourced by convert_interactive.sh.

# shellcheck shell=bash
[[ -n "${_MB_CONVERT_LIB_LOADED:-}" && $(type -t mb_convert_run 2>/dev/null) == function ]] && return 0
_MB_CONVERT_LIB_LOADED=1

_MB_CONVERT_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MB_ROOT="$(cd -- "${_MB_CONVERT_LIB_DIR}/.." && pwd)"

# shellcheck source=../_prep_bridge_env.sh
source "${_MB_ROOT}/../../utils/_prep_bridge_env.sh"

_mb_convert_path_under_cache() {
  local p="$1"
  if [[ "${p}" == /* ]]; then
    printf '%s\n' "${p}"
    return 0
  fi
  : "${CACHE_DIR:?Set CACHE_DIR in conversion_local.yaml}"
  printf '%s\n' "${CACHE_DIR%/}/${p}"
}

_mb_convert_is_local_path() {
  [[ "${1}" == /* || "${1}" == ./* || "${1}" == ../* ]]
}

_mb_convert_resolve_hf_model() {
  local hf="${HF_MODEL_ID:?Set HF_MODEL_ID in conversion_local.yaml}"
  if _mb_convert_is_local_path "${hf}"; then
    if [[ "${hf}" != /* ]]; then
      hf="${CACHE_DIR%/}/${hf}"
    fi
    if [[ ! -d "${hf}" ]]; then
      echo "error: local HF checkpoint not found: ${hf}" >&2
      return 1
    fi
    export HF_MODEL_ID="${hf}"
    export HF_MODEL_BASENAME="$(basename "${hf}")"
  else
    export HF_MODEL_BASENAME="${HF_MODEL_BASENAME:-$(basename "${HF_MODEL_ID}")}"
  fi
  echo "info: hf_model=${HF_MODEL_ID}" >&2
}

mb_convert_clean_workspace() {
  local _ws="${WORKSPACE:?}"
  if [[ "${CLEAN_WORKSPACE:-1}" != "1" ]]; then
    echo "info: CLEAN_WORKSPACE=0 — keeping existing workspace (conversion may overwrite outputs)" >&2
    return 0
  fi

  case "${CONVERSION_DIRECTION:-HF_to_MEG}" in
    MEG_to_HF|MEG_FSDP_to_HF)
      echo "info: CLEAN_WORKSPACE=1 — skipping workspace clean for ${CONVERSION_DIRECTION} (checkpoint must remain)" >&2
      return 0
      ;;
  esac

  [[ -n "${_ws}" && "${_ws}" != "/" ]] || {
    echo "error: refusing CLEAN_WORKSPACE on empty or root path" >&2
    return 1
  }
  [[ "${_ws}" != "${CACHE_DIR}" && "${_ws}" != "${CACHE_DIR}/" ]] || {
    echo "error: refusing CLEAN_WORKSPACE on CACHE_DIR itself (${_ws})" >&2
    return 1
  }
  case "${_ws}" in
    "${CACHE_DIR}"/*) ;;
    *)
      echo "error: WORKSPACE must be under CACHE_DIR for CLEAN_WORKSPACE (${_ws})" >&2
      return 1
      ;;
  esac

  echo "info: cleaning workspace before conversion: ${_ws}" >&2
  if [[ -e "${_ws}" ]]; then
    rm -rf "${_ws}"
    echo "info: workspace removed; continuing with conversion" >&2
  else
    echo "info: workspace already empty; continuing with conversion" >&2
  fi
}

mb_convert_resolve_paths() {
  : "${CACHE_DIR:?Set CACHE_DIR in conversion_local.yaml}"
  _mb_convert_resolve_hf_model || return 1
  : "${WORKSPACE:?Set WORKSPACE in conversion_local.yaml}"
  : "${MEGATRON_CKPT_PATH:?Set MEGATRON_CKPT_PATH in conversion_local.yaml}"
  : "${HF_EXPORT_PATH:?Set HF_EXPORT_PATH in conversion_local.yaml}"
  export WORKSPACE="$(_mb_convert_path_under_cache "${WORKSPACE}")"
  export MEGATRON_CKPT_PATH="$(_mb_convert_path_under_cache "${MEGATRON_CKPT_PATH}")"
  export HF_EXPORT_PATH="$(_mb_convert_path_under_cache "${HF_EXPORT_PATH}")"
  if [[ -n "${CHECK_OUTPUT_DIR:-}" ]]; then
    export CHECK_OUTPUT_DIR="$(_mb_convert_path_under_cache "${CHECK_OUTPUT_DIR}")"
  fi
  if [[ -n "${CHECK_MEGATRON_SAVE_PATH:-}" ]]; then
    export CHECK_MEGATRON_SAVE_PATH="$(_mb_convert_path_under_cache "${CHECK_MEGATRON_SAVE_PATH}")"
  fi
  if [[ -n "${CHECK_MEGATRON_LOAD_PATH:-}" ]]; then
    export CHECK_MEGATRON_LOAD_PATH="$(_mb_convert_path_under_cache "${CHECK_MEGATRON_LOAD_PATH}")"
  fi
  mb_convert_clean_workspace || return 1
  mkdir -p "${WORKSPACE}/models"
  echo "info: cache_dir=${CACHE_DIR}" >&2
  echo "info: workspace=${WORKSPACE}" >&2
  echo "info: megatron_path=${MEGATRON_CKPT_PATH}" >&2
  echo "info: hf_export_path=${HF_EXPORT_PATH}" >&2
}

mb_convert_normalize_direction() {
  local d="${CONVERSION_DIRECTION:-HF_to_MEG}"
  case "${d}" in
    HF_to_MEG|MEG_to_HF|HF_to_MEG_FSDP|MEG_FSDP_to_HF)
      export CONVERSION_DIRECTION="${d}"
      ;;
    *)
      echo "error: invalid CONVERSION_DIRECTION: ${d}" >&2
      return 1
      ;;
  esac
}

mb_convert_hf_to_meg() {
  local _extra=()
  echo "info: HF_to_MEG — HuggingFace → Megatron import → ${MEGATRON_CKPT_PATH}" >&2
  [[ -n "${CONVERSION_TORCH_DTYPE:-}" ]] && _extra+=(--torch-dtype "${CONVERSION_TORCH_DTYPE}")
  [[ -n "${CONVERSION_DEVICE_MAP:-}" ]] && _extra+=(--device-map "${CONVERSION_DEVICE_MAP}")
  local -a _py; mapfile -t _py < <(mb_py_launch)
  "${_py[@]}" examples/conversion/convert_checkpoints.py import \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_CKPT_PATH}" \
    --trust-remote-code \
    "${_extra[@]}"
}

mb_convert_meg_to_hf() {
  local _extra=()
  if [[ ! -d "${MEGATRON_CKPT_PATH}" ]]; then
    echo "error: MEG_to_HF requires an existing Megatron checkpoint at MEGATRON_CKPT_PATH=${MEGATRON_CKPT_PATH}" >&2
    return 1
  fi
  echo "info: MEG_to_HF — Megatron → HuggingFace export → ${HF_EXPORT_PATH}" >&2
  [[ "${CONVERSION_EXPORT_NO_PROGRESS:-0}" == "1" ]] && _extra+=(--no-progress)
  [[ "${CONVERSION_EXPORT_STRICT:-0}" != "1" ]] && _extra+=(--not-strict)
  local -a _py; mapfile -t _py < <(mb_py_launch)
  "${_py[@]}" examples/conversion/convert_checkpoints.py export \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_CKPT_PATH}" \
    --hf-path "${HF_EXPORT_PATH}" \
    "${_extra[@]}"
}

mb_convert_fsdp() {
  local _command="$1" _label="$2"
  local _wrapper="${_MB_CONVERT_LIB_DIR}/_convert_fsdp.py"
  local _dtype="${CONVERSION_TORCH_DTYPE:-bfloat16}"
  local -a _extra=()

  if [[ "${_command}" == "export" ]]; then
    if [[ ! -d "${MEGATRON_CKPT_PATH}" ]]; then
      echo "error: ${_label} requires an existing checkpoint at MEGATRON_CKPT_PATH=${MEGATRON_CKPT_PATH}" >&2
      return 1
    fi
    _extra+=(--hf-path "${HF_EXPORT_PATH}" --distributed-save)
    [[ "${CONVERSION_EXPORT_NO_PROGRESS:-0}" == "1" ]] && _extra+=(--no-progress)
    [[ "${CONVERSION_EXPORT_STRICT:-0}" != "1" ]] && _extra+=(--not-strict)
  else
    _extra+=(--no-low-memory-save)
  fi

  local -a _py; mapfile -t _py < <(mb_py_launch)
  "${_py[@]}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${_wrapper}" \
    "${_command}" \
    --hf-model "${HF_MODEL_ID}" \
    --megatron-path "${MEGATRON_CKPT_PATH}" \
    --torch-dtype "${_dtype}" \
    --ckpt-format fsdp_dtensor \
    --trust-remote-code \
    "${_extra[@]}"
}

mb_convert_finalize_parity_hf() {
  local _staged="${1:?}"
  if [[ ! -d "${_staged}" ]]; then
    echo "error: parity HF export not found at ${_staged}" >&2
    return 1
  fi
  if [[ "${_staged}" == "${HF_EXPORT_PATH}" ]]; then
    echo "info: parity HF export at ${HF_EXPORT_PATH}" >&2
    return 0
  fi
  rm -rf "${HF_EXPORT_PATH}"
  mv "${_staged}" "${HF_EXPORT_PATH}"
  echo "info: parity HF export saved to ${HF_EXPORT_PATH}" >&2
}

mb_convert_roundtrip() {
  local _extra=(--trust-remote-code)
  local _stage_dir="${CHECK_OUTPUT_DIR:-$(dirname "${HF_EXPORT_PATH}")}"
  local _staged_hf="${_stage_dir}/${HF_MODEL_BASENAME}"

  echo "info: RUN_PARITY_CHECK=1 — HF↔Megatron round-trip weight check" >&2
  echo "info: parity HF export target: ${HF_EXPORT_PATH}" >&2
  [[ "${CONVERSION_EXPORT_STRICT:-0}" != "1" ]] && _extra+=(--not-strict)
  _extra+=(--tp "${TP:-1}" --pp "${PP:-1}" --ep "${EP:-1}" --etp "${ETP:-1}")
  _extra+=(--output-dir "${_stage_dir}")
  [[ -n "${CHECK_MEGATRON_SAVE_PATH:-}" ]] && _extra+=(--megatron-save-path "${CHECK_MEGATRON_SAVE_PATH}")
  [[ -n "${CHECK_MEGATRON_LOAD_PATH:-}" ]] && _extra+=(--megatron-load-path "${CHECK_MEGATRON_LOAD_PATH}")
  local -a _py; mapfile -t _py < <(mb_py_launch)
  "${_py[@]}" -m torch.distributed.run --nproc_per_node="${NPROC:-1}" \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id "${HF_MODEL_ID}" \
    "${_extra[@]}" || return 1
  mb_convert_finalize_parity_hf "${_staged_hf}"
}

mb_convert_run() {
  mb_sync_bridge_env
  mb_convert_resolve_paths
  mb_convert_normalize_direction || return 1

  cd "${MEGATRON_BRIDGE_ROOT}"

  echo "info: CONVERSION_DIRECTION=${CONVERSION_DIRECTION} RUN_PARITY_CHECK=${RUN_PARITY_CHECK:-0} CLEAN_WORKSPACE=${CLEAN_WORKSPACE:-1}" >&2

  case "${CONVERSION_DIRECTION}" in
    HF_to_MEG)
      mb_convert_hf_to_meg
      ;;
    MEG_to_HF)
      mb_convert_meg_to_hf
      ;;
    HF_to_MEG_FSDP)
      mb_convert_fsdp import "HF_to_MEG_FSDP — HuggingFace → Megatron-FSDP → ${MEGATRON_CKPT_PATH}"
      ;;
    MEG_FSDP_to_HF)
      mb_convert_fsdp export "MEG_FSDP_to_HF — Megatron-FSDP → HuggingFace → ${HF_EXPORT_PATH}"
      ;;
  esac

  if [[ "${RUN_PARITY_CHECK:-0}" == "1" ]]; then
    case "${CONVERSION_DIRECTION}" in
      HF_to_MEG|MEG_to_HF) mb_convert_roundtrip ;;
      *) echo "info: RUN_PARITY_CHECK applies only to standard Megatron conversion; skipping" >&2 ;;
    esac
  fi

  echo "info: conversion complete" >&2
}

mb_convert_init_launch() {
  mb_init_launch "${_MB_CONVERT_LIB_DIR}" "${1:-interactive}" conversion
  setup_cache_dir_env
  : "${HF_MODEL_ID:?Set HF_MODEL_ID in conversion_local.yaml}"
}
