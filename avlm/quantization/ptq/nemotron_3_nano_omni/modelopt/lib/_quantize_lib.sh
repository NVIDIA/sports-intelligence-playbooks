# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nemotron-3-Nano-Omni PTQ:
#   hf (default): quantize_omni.py — HF load, VLM calibration, export_hf_checkpoint
# shellcheck shell=bash
[[ -n "${_MB_QUANTIZE_LIB_LOADED:-}" && $(type -t mb_quantize_run 2>/dev/null) == function ]] && return 0
_MB_QUANTIZE_LIB_LOADED=1

_MB_QUANT_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MB_MODE_DIR="$(cd -- "${_MB_QUANT_LIB_DIR}/.." && pwd)"
# shellcheck source=hf_local_model.sh
source "${_MB_QUANT_LIB_DIR}/hf_local_model.sh"

_quant_resolve_avlm_root() {
  local d="$1"
  while [[ "${d}" != "/" ]]; do
    if [[ -f "${d}/utils/_source_params.sh" ]]; then
      printf '%s\n' "${d}"
      return 0
    fi
    d="$(dirname -- "${d}")"
  done
  echo "error: could not find avlm root (utils/_source_params.sh) from ${1}" >&2
  return 1
}

_AVLM_ROOT="$(_quant_resolve_avlm_root "${_MB_MODE_DIR}")" || return 1

# shellcheck source=../../../../../utils/_prep_bridge_env.sh
source "${_AVLM_ROOT}/utils/_prep_bridge_env.sh"

_MODELOPT_GIT_URL="https://github.com/NVIDIA/Model-Optimizer.git"
# MB_QUANT_CLI_OVERRIDE_KEYS — defined in utils/_prep_bridge_env.sh (sourced above).

# Optional first arg to quantize.sh: hf
_MB_QUANT_RECIPE_ARG=""

mb_quantize_peel_recipe_arg() {
  local arg
  _MB_QUANT_RECIPE_ARG=""
  _MB_QUANT_RECIPE_REMAINING_ARGS=()
  for arg in "$@"; do
    if [[ -z "${_MB_QUANT_RECIPE_ARG}" && "${arg}" != *=* ]]; then
      case "${arg}" in
        hf) _MB_QUANT_RECIPE_ARG=hf; continue ;;
        megatron_bridge|megatron-bridge|mb)
          echo "error: Megatron-Bridge PTQ is not included. Use: quantize.sh hf" >&2
          return 1
          ;;
      esac
    fi
    _MB_QUANT_RECIPE_REMAINING_ARGS+=("${arg}")
  done
}

mb_quantize_resolve_params_file() {
  local mode_dir="$1" recipe="${2:-hf}" f=""
  if [[ -n "${QUANTIZATION_PARAMS:-}" ]]; then
    for f in "${QUANTIZATION_PARAMS}" "${mode_dir}/${QUANTIZATION_PARAMS}" \
      "${mode_dir}/configs/${QUANTIZATION_PARAMS}"; do
      if [[ -f "${f}" ]]; then
        export QUANTIZATION_PARAMS="${f}"
        echo "info: using QUANTIZATION_PARAMS=${QUANTIZATION_PARAMS}" >&2
        return 0
      fi
    done
    echo "error: QUANTIZATION_PARAMS not found: ${QUANTIZATION_PARAMS}" >&2
    return 1
  fi
  case "${recipe}" in
    hf) f="${mode_dir}/configs/quantization_hf_tennis.yaml" ;;
    *)
      echo "error: unknown quant recipe '${recipe}' (expected hf)" >&2
      return 1
      ;;
  esac
  if [[ ! -f "${f}" ]]; then
    echo "error: missing ${f}" >&2
    return 1
  fi
  export QUANTIZATION_PARAMS="${f}"
  echo "info: quant recipe=${recipe} → ${QUANTIZATION_PARAMS}" >&2
}

# Resolve WORKSPACE / export paths:
#   /absolute/...     → as-is
#   @repo/relative    → ${REPO_ROOT}/relative (sports_intelligence checkout)
#   relative          → ${CACHE_DIR}/relative (default scratch under cache)
_mb_quant_resolve_path() {
  local p="$1"
  if [[ "${p}" == /* ]]; then
    printf '%s\n' "${p}"
    return 0
  fi
  if [[ "${p}" == @repo/* ]]; then
    : "${REPO_ROOT:?REPO_ROOT must be set (run from repo via quantize.sh)}"
    printf '%s\n' "${REPO_ROOT}/${p#@repo/}"
    return 0
  fi
  : "${CACHE_DIR:?Set CACHE_DIR in the quant params YAML}"
  printf '%s\n' "${CACHE_DIR%/}/${p}"
}

_mb_quant_path_under_cache() {
  _mb_quant_resolve_path "$1"
}

_mb_quant_expand_repo_root_in_var() {
  local key="$1"
  local val="${!key:-}"
  [[ -n "${val}" && -n "${REPO_ROOT:-}" ]] || return 0
  val="${val//\$\{REPO_ROOT\}/${REPO_ROOT}}"
  export "${key}=${val}"
}

# Slug for run/export dirs (QUANT_RUN_LABEL, else Hub repo name or local dir basename).
_mb_quant_artifact_slug() {
  if [[ -n "${QUANT_RUN_LABEL:-}" ]]; then
    printf '%s\n' "${QUANT_RUN_LABEL}"
    return 0
  fi
  printf '%s\n' "${HF_MODEL_BASENAME:?HF model must be resolved before path defaults}"
}

_mb_quantize_try_local_hf_dir() {
  local candidate="$1"
  [[ -n "${candidate}" && -d "${candidate}" ]] || return 1
  quant_has_model_weights "${candidate}" || return 1
  export HF_MODEL_ID="$(cd -- "${candidate}" && pwd)"
  export HF_MODEL_BASENAME="$(basename "${HF_MODEL_ID}")"
  echo "info: hf_model=${HF_MODEL_ID} (local checkpoint)" >&2
  return 0
}

# HF_MODEL_ID: Hub id (org/model) or absolute consolidated/. Hub ids materialize under WORKSPACE/base/.
_mb_quantize_materialize_hf_model() {
  local hf="${HF_MODEL_ID:?Set HF_MODEL_ID in the quant params YAML}"
  local dest local_dir="" hub_id=""

  if [[ "${hf}" == @repo/* ]]; then
    hf="$(_mb_quant_resolve_path "${hf}")"
    export HF_MODEL_ID="${hf}"
  fi

  if [[ "${hf}" == /* ]]; then
    _mb_quantize_try_local_hf_dir "${hf}" && return 0
    echo "error: HF_MODEL_ID local path not found or missing weights: ${hf}" >&2
    return 1
  fi

  if [[ ! "${hf}" =~ ^[^/]+/[^/]+$ ]]; then
    echo "error: HF_MODEL_ID must be a Hub id (org/model) or an absolute path: ${hf}" >&2
    return 1
  fi

  hub_id="${hf}"
  export HF_MODEL_HUB_ID="${hub_id}"
  if [[ -n "${HF_MODEL_LOCAL_DIR:-}" ]]; then
    local_dir="$(_mb_quant_path_under_cache "${HF_MODEL_LOCAL_DIR}")"
  fi
  dest="$(quant_default_base_model_dir "${WORKSPACE}" "${hub_id}" "${local_dir}")"
  mkdir -p "$(dirname "${dest}")"

  if quant_has_model_weights "${dest}"; then
    echo "info: base model at ${dest}" >&2
    _mb_quantize_try_local_hf_dir "${dest}" && return 0
  fi

  if [[ "${HF_HUB_OFFLINE:-0}" == "1" || "${HF_MODEL_AUTO_DOWNLOAD:-1}" != "1" ]]; then
    echo "error: base model weights missing at ${dest}" >&2
    echo "  Download is skipped because HF_HUB_OFFLINE=1 or HF_MODEL_AUTO_DOWNLOAD is not 1." >&2
    echo "  Place the BF16 weights at ${dest}, or allow quantize.sh to download the Hub checkpoint." >&2
    return 1
  fi

  echo "info: downloading ${hub_id} → ${dest} (WORKSPACE/base, not HF_HOME)" >&2
  quant_hf_download "${hub_id}" "${dest}" || return 1
  _mb_quantize_try_local_hf_dir "${dest}" || {
    echo "error: download finished but weights missing under ${dest}" >&2
    return 1
  }
}

_mb_quantize_bootstrap_modelopt() {
  local root ref
  root="$(_mb_quant_path_under_cache "${MODELOPT_ROOT:?Set MODELOPT_ROOT in the quant params YAML}")"
  export MODELOPT_ROOT="${root}"

  if [[ -f "${MODELOPT_ROOT}/examples/megatron_bridge/quantize.py" ]]; then
    echo "info: Model-Optimizer at ${MODELOPT_ROOT}" >&2
    return 0
  fi

  if [[ -f /opt/Model-Optimizer/examples/megatron_bridge/quantize.py ]]; then
    export MODELOPT_ROOT="/opt/Model-Optimizer"
    echo "info: using container Model-Optimizer at ${MODELOPT_ROOT}" >&2
    return 0
  fi

  if [[ "${MODELOPT_GIT_BOOTSTRAP:-0}" != "1" ]]; then
    echo "error: Model-Optimizer not found at ${MODELOPT_ROOT} and MODELOPT_GIT_BOOTSTRAP!=1" >&2
    echo "  Clone https://github.com/NVIDIA/Model-Optimizer or set MODELOPT_GIT_BOOTSTRAP=1" >&2
    return 1
  fi

  ref="${MODELOPT_GIT_REF:-main}"
  echo "info: cloning Model-Optimizer (${ref}) → ${MODELOPT_ROOT}" >&2
  mkdir -p "$(dirname "${MODELOPT_ROOT}")"
  if [[ -d "${MODELOPT_ROOT}/.git" ]]; then
    git -C "${MODELOPT_ROOT}" fetch --depth 1 origin "${ref}"
    git -C "${MODELOPT_ROOT}" checkout "${ref}"
  else
    rm -rf "${MODELOPT_ROOT}"
    git clone --depth 1 --branch "${ref}" "${_MODELOPT_GIT_URL}" "${MODELOPT_ROOT}"
  fi
  [[ -f "${MODELOPT_ROOT}/examples/megatron_bridge/quantize.py" ]] || {
    echo "error: Model-Optimizer checkout incomplete at ${MODELOPT_ROOT}" >&2
    return 1
  }
}

_mb_quantize_install_modelopt_editable() {
  # NeMo containers ship modelopt; mount/clone the repo and overlay site-packages when using a git checkout.
  if [[ "${MODELOPT_ROOT}" == /opt/Model-Optimizer ]]; then
    return 0
  fi
  local -a _py
  mapfile -t _py < <(mb_py_launch)
  if [[ -d "${MODELOPT_ROOT}/modelopt" ]]; then
    echo "info: pip install Model-Optimizer from ${MODELOPT_ROOT} (editable)" >&2
    "${_py[@]}" -m pip install -q -e "${MODELOPT_ROOT}" || {
      echo "warn: editable install failed; continuing with PYTHONPATH overlay" >&2
      export PYTHONPATH="${MODELOPT_ROOT}:${MODELOPT_ROOT}/modelopt_recipes:${PYTHONPATH:-}"
    }
  fi
}

_mb_quantize_resolve_hf_quant_config() {
  if [[ -n "${QUANTIZE_CONFIG:-}" ]]; then
    export _MB_HF_QUANT_CONFIG="${QUANTIZE_CONFIG}"
    return 0
  fi
  local fmt="${QUANT_FORMAT:-fp8}"
  case "${fmt}" in
    fp8) export _MB_HF_QUANT_CONFIG="fp8_fp8_kv" ;;
    nvfp4) export _MB_HF_QUANT_CONFIG="nvfp4_fp8_mixed_fp8_kv" ;;
    *)
      echo "error: QUANT_FORMAT=${fmt} — set QUANTIZE_CONFIG explicitly" >&2
      return 1
      ;;
  esac
}

_mb_quantize_resolve_preset() {
  local fmt="${QUANT_FORMAT:-fp8}"
  case "${fmt}" in
    fp8)
      export _MB_QUANT_RECIPE="${QUANT_RECIPE:-general/ptq/fp8_default-kv_fp8}"
      export _MB_QUANT_CFG=""
      ;;
    nvfp4)
      export _MB_QUANT_RECIPE=""
      export _MB_QUANT_CFG="${QUANT_CFG:-MAMBA_MOE_NVFP4_CONSERVATIVE_CFG}"
      ;;
    custom)
      export _MB_QUANT_RECIPE="${QUANT_RECIPE:-}"
      export _MB_QUANT_CFG="${QUANT_CFG:-}"
      if [[ -z "${_MB_QUANT_RECIPE}" && -z "${_MB_QUANT_CFG}" ]]; then
        echo "error: QUANT_FORMAT=custom requires QUANT_RECIPE and/or QUANT_CFG" >&2
        return 1
      fi
      ;;
    *)
      echo "error: invalid QUANT_FORMAT: ${fmt} (expected fp8, nvfp4, or custom)" >&2
      return 1
      ;;
  esac
}

_mb_quantize_copy_latest_checkpoints() {
  local work_root="$1" dest="$2"
  local latest=""
  latest="$(find "${work_root}" -type d -name checkpoints 2>/dev/null -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- || true)"
  if [[ -z "${latest}" || ! -d "${latest}" ]]; then
    echo "error: no checkpoints/ directory under ${work_root}" >&2
    return 1
  fi
  echo "info: publishing ${latest} → ${dest}" >&2
  rm -rf "${dest}"
  mkdir -p "${dest}"
  cp -a "${latest}/." "${dest}/"
}

# Tokenizer/processor/code from BF16 base (export may omit or overwrite some paths).
_mb_quantize_sync_export_sidecars() {
  local src="$1" dest="$2"
  local f copied=0
  [[ -d "${src}" && -d "${dest}" ]] || return 0
  for f in \
    audio_model.py chat_template.jinja configuration_nemotron_h.py configuration.py \
    configuration_radio.py evs.py image_processing.py modeling_nemotron_h.py modeling.py \
    preprocessor_config.json processing.py processing_utils.py processor_config.json \
    special_tokens_map.json tokenizer_config.json tokenizer.json video_io.py video_processing.py; do
    if [[ -f "${src}/${f}" ]]; then
      cp -a "${src}/${f}" "${dest}/${f}"
      copied=$((copied + 1))
    fi
  done
  echo "info: synced ${copied} HF sidecar files from ${src} → ${dest}" >&2
}

# Nemotron VLM calib uses huggingface_hub (hf_hub_download), not WORKSPACE/base.
_mb_quantize_bind_calib_hf_cache() {
  local root
  if [[ -n "${QUANT_DATASETS_DIR:-}" ]]; then
    root="$(_mb_quant_path_under_cache "${QUANT_DATASETS_DIR}")"
  else
    root="${WORKSPACE}/datasets"
  fi
  mkdir -p "${root}/hub" "${root}/hf_datasets"
  export QUANT_DATASETS_DIR="${root}"
  # PTQ: keep hub/datasets/modules under WORKSPACE, not inherited HF_HOME (stale partial
  # transformers_modules there breaks trust_remote_code loads from WORKSPACE/base).
  export HF_HUB_CACHE="${root}/hub"
  export HF_DATASETS_CACHE="${root}/hf_datasets"
  # HF_MODULES_CACHE must be the "modules" dir (transformers appends transformers_modules/ itself).
  export HF_HOME="${WORKSPACE}/hf_home"
  export HF_MODULES_CACHE="${HF_HOME}/modules"
  mkdir -p "${HF_MODULES_CACHE}"
  echo "info: VLM calib downloads → ${root} (HF_HUB_CACHE=${HF_HUB_CACHE})" >&2
  echo "info: HF_HOME=${HF_HOME} (modules cache under ${HF_MODULES_CACHE})" >&2
}

mb_quantize_resolve_paths() {
  local slug fmt rel_ws="${WORKSPACE:-@repo/quantization_workspace}"

  : "${CACHE_DIR:?Set CACHE_DIR in the quant params YAML}"

  fmt="${QUANT_FORMAT:-fp8}"
  export WORKSPACE="$(_mb_quant_path_under_cache "${rel_ws}")"
  mkdir -p "${WORKSPACE}/base"
  _mb_quantize_materialize_hf_model || return 1
  slug="$(_mb_quant_artifact_slug)"

  export HF_QUANT_EXPORT_PATH="${HF_QUANT_EXPORT_PATH:-${rel_ws}/models/${slug}-${fmt}-hf}"
  export HF_QUANT_EXPORT_PATH="$(_mb_quant_path_under_cache "${HF_QUANT_EXPORT_PATH}")"
  export QUANT_WORK_DIR="${QUANT_WORK_DIR:-${rel_ws}/quant_runs/${slug}-${fmt}}"
  export QUANT_WORK_DIR="$(_mb_quant_path_under_cache "${QUANT_WORK_DIR}")"
  export MEGATRON_QUANT_PATH="${MEGATRON_QUANT_PATH:-${rel_ws}/models/${slug}-${fmt}-megatron}"
  export MEGATRON_QUANT_PATH="$(_mb_quant_path_under_cache "${MEGATRON_QUANT_PATH}")"

  mkdir -p "${WORKSPACE}/models" "${WORKSPACE}/datasets" "${QUANT_WORK_DIR}" \
    "$(dirname "${HF_QUANT_EXPORT_PATH}")" "$(dirname "${MEGATRON_QUANT_PATH}")"
  if [[ -n "${CALIB_JSONL:-}" ]]; then
    export CALIB_JSONL="$(_mb_quant_path_under_cache "${CALIB_JSONL}")"
  fi
  _mb_quantize_bind_calib_hf_cache

  echo "info: workspace=${WORKSPACE}" >&2
  echo "info: quant_work_dir=${QUANT_WORK_DIR}" >&2
  echo "info: hf_quant_export_path=${HF_QUANT_EXPORT_PATH}" >&2
  if [[ "${QUANT_BACKEND:-hf}" == "megatron_bridge" ]]; then
    echo "info: megatron_quant_path=${MEGATRON_QUANT_PATH}" >&2
  fi
}

_mb_quantize_torchrun_nproc() {
  local tp="${TP_SIZE:-1}" pp="${PP_SIZE:-1}" ep="${EP_SIZE:-1}" cp="${CP_SIZE:-1}"
  local n="${NPROC_PER_NODE:?Set NPROC_PER_NODE}"
  local min=$(( tp * pp * cp * ep ))
  if (( n < min )); then
    echo "error: NPROC_PER_NODE=${n} < tp*pp*cp*ep=${min}" >&2
    return 1
  fi
  if (( n % min != 0 )); then
    echo "warn: NPROC_PER_NODE=${n} is not a multiple of tp*pp*cp*ep=${min}; ModelOpt may reject this layout" >&2
  fi
  printf '%s\n' "${n}"
}

_mb_quantize_common_args() {
  local -n _out=$1
  _out=(--hf_model_name_or_path "${HF_MODEL_ID}")
  [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]] && _out+=(--trust_remote_code)
}

mb_quantize_run_ptq() {
  local script="${MODELOPT_ROOT}/examples/megatron_bridge/quantize.py"
  local -a cmd extra=() quant_flag=()
  local nproc

  _mb_quantize_resolve_preset || return 1
  nproc="$(_mb_quantize_torchrun_nproc)" || return 1

  _mb_quantize_common_args extra
  extra+=(--tp_size "${TP_SIZE:-1}" --pp_size "${PP_SIZE:-1}" --ep_size "${EP_SIZE:-1}" --cp_size "${CP_SIZE:-1}")
  extra+=(--calib_num_samples "${CALIB_NUM_SAMPLES:-1024}")
  extra+=(--calib_batch_size "${CALIB_BATCH_SIZE:-1}")
  extra+=(--seq_length "${SEQ_LENGTH:-4096}")
  extra+=(--export_megatron_path "${MEGATRON_QUANT_PATH}")

  if [[ -n "${_MB_QUANT_RECIPE}" ]]; then
    extra+=(--recipe "${_MB_QUANT_RECIPE}")
  elif [[ -n "${_MB_QUANT_CFG}" ]]; then
    extra+=(--quant_cfg "${_MB_QUANT_CFG}")
  fi

  if [[ -n "${CALIB_DATASET_NAME:-}" ]]; then
    extra+=(--calib_dataset_name "${CALIB_DATASET_NAME}")
  fi
  [[ "${COMPRESS:-1}" == "1" ]] && extra+=(--compress)
  [[ "${SKIP_GENERATE:-1}" == "1" ]] && extra+=(--skip_generate)

  echo "info: PTQ — torchrun nproc=${nproc} quant_cfg/recipe format=${QUANT_FORMAT:-fp8}" >&2
  local -a _py
  mapfile -t _py < <(mb_py_launch)
  cmd=("${_py[@]}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "${script}" "${extra[@]}")
  printf 'info: %q ' "${cmd[@]}" >&2; echo >&2
  "${cmd[@]}"
}

mb_quantize_run_export() {
  local script="${MODELOPT_ROOT}/examples/megatron_bridge/export_quantized_megatron_to_hf.py"
  local -a cmd extra=()
  local nproc="${NPROC_PER_NODE:-1}"
  local export_pp="${EXPORT_PP_SIZE:-1}"

  if [[ ! -d "${MEGATRON_QUANT_PATH}" ]]; then
    echo "error: Megatron quantized checkpoint not found: ${MEGATRON_QUANT_PATH}" >&2
    return 1
  fi

  _mb_quantize_common_args extra
  extra+=(--megatron_path "${MEGATRON_QUANT_PATH}")
  extra+=(--pp_size "${export_pp}")
  extra+=(--export_unified_hf_path "${HF_QUANT_EXPORT_PATH}")

  echo "info: export unified HF → ${HF_QUANT_EXPORT_PATH}" >&2
  local -a _py
  mapfile -t _py < <(mb_py_launch)
  cmd=("${_py[@]}" -m torch.distributed.run --standalone --nproc_per_node="${nproc}" "${script}" "${extra[@]}")
  printf 'info: %q ' "${cmd[@]}" >&2; echo >&2
  "${cmd[@]}"
}

mb_quantize_run_hf() {
  _mb_quantize_bootstrap_modelopt || return 1
  _mb_quantize_resolve_hf_quant_config || return 1

  export OMNI_QUANT_OUT_ROOT="${QUANT_WORK_DIR}"
  export MODEL_OPTIMIZER_SRC="${MODELOPT_ROOT}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

  local -a cmd=(
    "${_MB_QUANT_LIB_DIR}/run_quantize.sh"
    "--quantize-config=${_MB_HF_QUANT_CONFIG}"
    "--num-samples=${NUM_SAMPLES:-512}"
    "--model-path=${HF_MODEL_ID}"
    "--video-pruning-rate=${VIDEO_PRUNING_RATE:-0}"
  )
  if [[ -n "${CALIB_JSONL:-}" ]]; then
    cmd+=("--calib-jsonl=${CALIB_JSONL}")
    cmd+=("--video-root=${VIDEO_ROOT:?Set VIDEO_ROOT when CALIB_JSONL is set}")
    cmd+=("--calib-video-fps=${CALIB_VIDEO_FPS:-2}")
    cmd+=("--calib-max-video-frames=${CALIB_MAX_VIDEO_FRAMES:-128}")
  fi

  echo "info: HF PTQ (${_MB_HF_QUANT_CONFIG}, num_samples=${NUM_SAMPLES:-512})" >&2
  printf 'info: %q ' "${cmd[@]}" >&2; echo >&2
  "${cmd[@]}"

  _mb_quantize_copy_latest_checkpoints "${QUANT_WORK_DIR}" "${HF_QUANT_EXPORT_PATH}" || return 1
  _mb_quantize_sync_export_sidecars "${HF_MODEL_ID}" "${HF_QUANT_EXPORT_PATH}"
}

mb_quantize_run() {
  mb_quantize_resolve_paths || return 1

  if [[ "${QUANT_BACKEND:-hf}" == "megatron_bridge" ]]; then
    mb_sync_bridge_env
    _mb_quantize_bootstrap_modelopt || return 1
    _mb_quantize_install_modelopt_editable || return 1

    local step="${QUANTIZE_STEP:-both}"
    case "${step}" in
      quantize|ptq) mb_quantize_run_ptq ;;
      export) mb_quantize_run_export ;;
      both|all)
        mb_quantize_run_ptq || return 1
        [[ "${RUN_EXPORT:-1}" == "1" ]] && mb_quantize_run_export
        ;;
      *)
        echo "error: invalid QUANTIZE_STEP: ${step}" >&2
        return 1
        ;;
    esac
  else
    case "${QUANTIZE_STEP:-both}" in
      export)
        echo "error: HF backend runs PTQ+export in one step; use QUANTIZE_STEP=both or quantize" >&2
        return 1
        ;;
      *)
        mb_quantize_run_hf
        ;;
    esac
  fi

  echo "info: quantization complete" >&2
  if [[ -d "${HF_QUANT_EXPORT_PATH}" ]]; then
    echo "info: deploy with vLLM — model_path=${HF_QUANT_EXPORT_PATH}; FP8/NVFP4 use quantization=modelopt / modelopt_fp4; kv-cache-dtype fp8 for quants" >&2
  fi
}

mb_quantize_init_launch() {
  # shellcheck source=../../../../../utils/_source_params.sh
  source "${_AVLM_ROOT}/utils/_source_params.sh"
  resolve_avlm_repo_roots_from_mode_dir "${_MB_MODE_DIR}"
  source_quantization_params "${_MB_MODE_DIR}"
  _mb_quant_expand_repo_root_in_var CACHE_DIR
  : "${CACHE_DIR:?Set CACHE_DIR in the quant params YAML}"
  setup_cache_dir_env
  : "${HF_MODEL_ID:?Set HF_MODEL_ID in the quant params YAML}"

  if [[ "${QUANT_BACKEND:-hf}" == "megatron_bridge" ]]; then
    echo "error: Megatron-Bridge PTQ is not included. Set QUANT_BACKEND=hf." >&2
    return 1
  fi
  # HF PTQ is a single process. Slurm GPU count lives in slurm/launch_local.yaml for
  # slurm/launch_interactive_session.sh only.
  export NPROC_PER_NODE=1
  _mb_quantize_bootstrap_modelopt || return 1
  export MODEL_OPTIMIZER_SRC="${MODELOPT_ROOT}"
}
