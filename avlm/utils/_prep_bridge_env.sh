# Megatron-Bridge environment setup: git checkout, Python env, runtime exports, launch init.
# Used on the login node (git bootstrap) and inside the NeMo container (env prep + train/convert).
# Default: container source (/opt/Megatron-Bridge) + /opt/venv + a few extra pip wheels.
# Clone path (MEGATRON_BRIDGE_GIT_BOOTSTRAP=1): uv lock/sync into ${MEGATRON_BRIDGE_ROOT}/.venv.
# shellcheck shell=bash

_MB_COMMON_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_MEGATRON_BRIDGE_GIT_URL="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
# Pin matches nemo:26.06 (/opt/Megatron-Bridge @ fcbb6031).
_MEGATRON_BRIDGE_GIT_REF="fcbb6031103d0ca845c1a54d4fee55ecfcca17b6"

# nemo 26.06+ ships the full checkout here (src/, 3rdparty/Megatron-LM, scripts/, .git).
_MEGATRON_BRIDGE_CONTAINER_ROOT_DEFAULT="/opt/Megatron-Bridge"

# Keys accepted by Megatron-Bridge conversion launchers.
MB_BRIDGE_CLI_OVERRIDE_KEYS=(
  MODEL_NAME GPUS_PER_NODE NUM_GPUS NPROC_PER_NODE MEGATRON_BRIDGE_ROOT HF_MODEL_ID RECIPE CONFIG_YAML_REL
  CONVERSION_DIRECTION MEGATRON_CKPT_PATH HF_EXPORT_PATH WORKSPACE RUN_PARITY_CHECK CLEAN_WORKSPACE NPROC ETP
  CONVERSION_TORCH_DTYPE CONVERSION_DEVICE_MAP CONVERSION_EXPORT_NO_PROGRESS CONVERSION_EXPORT_STRICT
  CHECK_OUTPUT_DIR CHECK_MEGATRON_SAVE_PATH CHECK_MEGATRON_LOAD_PATH
  PRETRAINED_CHECKPOINT pretrained_checkpoint
  TP EP CP PP
  USE_SEQUENCE_PACKING PACKED_SEQ SEQ_LENGTH
  MAX_STEPS TRAIN_ITERS VAL_EVERY_STEPS EVAL_INTERVAL CKPT_EVERY_STEPS SAVE_INTERVAL
  GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE EVAL_ITERS
  LOG_INTERVAL WANDB_PROJECT WANDB_MODE WANDB_ENTITY WANDB_NAME
  AVLM_HF_RESIZE
  CACHE_DIR CONTAINER_IMAGE
  CONTAINER_MOUNT_HOST JOB_NAME SKIP_UV_SYNC partition SLURM_TIME_LIMIT CKPT_LAYOUT_TAG CHECKPOINT_DIR
  EXIT_MINS_BEFORE_LIMIT
  TRAIN_JSONL VAL_JSONL VIDEO_ROOT MAX_VIDEO_FRAMES VIDEO_SAMPLE_FPS
  DATALOADER_NUM_WORKERS GC_EVERY_STEPS GC_EVERY_ITERS
  FREEZE_LANGUAGE_MODEL FREEZE_VISION_MODEL FREEZE_VISION_PROJECTION FREEZE_SOUND_ENCODER FREEZE_SOUND_PROJECTION
  OPTIMIZER_LR OPTIMIZER_WEIGHT_DECAY OPTIMIZER_ADAM_BETA1 OPTIMIZER_ADAM_BETA2 CLIP_GRAD_MAX_NORM
  SCHEDULER_LR_DECAY_STYLE SCHEDULER_MIN_LR SCHEDULER_LR_WARMUP_ITERS SCHEDULER_LR_DECAY_ITERS RNG_SEED
  EMPTY_UNUSED_MEMORY_LEVEL OPTIMIZER_CPU_OFFLOAD OPTIMIZER_OFFLOAD_FRACTION OVERLAP_CPU_OPTIMIZER_D2H_H2D
  USE_PRECISION_AWARE_OPTIMIZER SAVE_OPTIM
  RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS RECOMPUTE_MODULES
  FINE_GRAINED_ACTIVATION_OFFLOADING OFFLOAD_MODULES
  MAX_TRAIN_SAMPLES MAX_VAL_SAMPLES
  MEGATRON_BRIDGE_GIT_DIR MEGATRON_BRIDGE_GIT_REF SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP
  MEGATRON_BRIDGE_GIT_BOOTSTRAP MEGATRON_BRIDGE_CONTAINER_ROOT
  MB_BRIDGE_VENV MB_BRIDGE_UV_CACHE MB_BRIDGE_PYTHON MB_BRIDGE_USE_CONTAINER_TE MB_BRIDGE_USE_CONTAINER_PKGS MB_BRIDGE_BACKUP_VENV
  MB_BRIDGE_FORCE_UV_SYNC
  num_nodes SLURM_ACCOUNTS SLURM_ACCOUNT_RACE use_exclusive partition OUTPUT_BASE MEMORY_LOG
  LORA_DIM LORA_ALPHA RESUME_CHECKPOINT
  INFERENCE_VALIDATION_ENABLED INFERENCE_VALIDATION_EVERY_STEPS INFERENCE_VALIDATION_INFERENCE_CONFIG
  INFERENCE_VALIDATION_DATA_PATH INFERENCE_VALIDATION_MAX_SAMPLES INFERENCE_VALIDATION_CLUSTER_PARAMS
  INFERENCE_VALIDATION_NUM_NODES INFERENCE_VALIDATION_GPUS_PER_NODE INFERENCE_VALIDATION_NPROC_PER_NODE
  VLM_SCORER_CONFIG
)

mb_log_section() {
  echo "[mb-bridge] ============================================================" >&2
  printf '[mb-bridge] %s\n' "$*" >&2
  echo "[mb-bridge] ============================================================" >&2
}

mb_log_step() {
  printf '[mb-bridge] -- %s\n' "$*" >&2
}

mb_log_info() {
  printf '[mb-bridge]    %s\n' "$*" >&2
}

mb_log_ok() {
  printf '[mb-bridge] ok %s\n' "$*" >&2
}

mb_log_warn() {
  printf '[mb-bridge] WARN %s\n' "$*" >&2
}

# Where the container ships the Megatron-Bridge checkout (override if the image layout changes).
mb_container_root() {
  printf '%s\n' "${MEGATRON_BRIDGE_CONTAINER_ROOT:-${_MEGATRON_BRIDGE_CONTAINER_ROOT_DEFAULT}}"
}

# The container's own venv (has torch, TE, mamba_ssm, causal_conv1d, librosa, and Bridge editable).
_MEGATRON_BRIDGE_CONTAINER_VENV="/opt/venv"

# True when the resolved source is the container checkout (under /opt): use /opt/venv directly
# and install only the handful of packages the image doesn't ship — no uv sync, no project venv.
# Lustre clones / explicit external checkouts keep the uv-synced ${ROOT}/.venv path instead.
mb_container_env_active() {
  [[ "${MEGATRON_BRIDGE_ROOT:-}" == /opt || "${MEGATRON_BRIDGE_ROOT:-}" == /opt/* ]]
}

# Repo cloning toggle (mirrors AUTOMODEL_GIT_BOOTSTRAP):
#   MEGATRON_BRIDGE_GIT_BOOTSTRAP=1 → clone/update into CACHE_DIR (legacy; needs network).
#   unset / 0 (default)            → use the container checkout (mb_container_root); no clone.
# SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP=1 is still honored as an explicit "do not clone".
_mb_git_bootstrap_enabled() {
  if [[ "${SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP:-0}" == "1" ]]; then
    return 1
  fi
  [[ "${MEGATRON_BRIDGE_GIT_BOOTSTRAP:-0}" == "1" ]]
}

_mb_bridge_default_root() {
  printf '%s\n' "${MEGATRON_BRIDGE_GIT_DIR:-${CACHE_DIR}/Megatron-Bridge}"
}

_mb_bridge_verify_tree() {
  local root="${1:?root required}"
  [[ -d "${root}/.git" ]] || return 1
  [[ -f "${root}/scripts/training/run_recipe.py" ]] || return 1
  [[ -d "${root}/src/megatron/bridge" ]] || return 1
  [[ -d "${root}/3rdparty/Megatron-LM/megatron/core" ]] || return 1
  return 0
}

_mb_bridge_clear_stale_git_locks() {
  local git_dir="$1/.git"
  [[ -d "${git_dir}" ]] || return 0
  local f
  for f in shallow.lock index.lock HEAD.lock config.lock; do
    [[ -f "${git_dir}/${f}" ]] && rm -f "${git_dir}/${f}"
  done
}

_mb_bridge_is_commit_sha() {
  [[ "${1:-}" =~ ^[0-9a-fA-F]{7,40}$ ]]
}

_mb_bridge_head_matches_ref() {
  local root="$1" want="$2" head target=""
  head="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
  [[ -n "${head}" ]] || return 1
  target="$(git -C "${root}" rev-parse "${want}^{commit}" 2>/dev/null || true)"
  if [[ -n "${target}" ]]; then
    [[ "${head}" == "${target}" ]]
    return
  fi
  [[ "${head}" == "${want}" ]] || [[ "${head}" == ${want}* ]] || [[ "${want}" == ${head}* ]]
}

_mb_bridge_update_submodules() {
  local root="$1"
  mb_log_info "submodule update (shallow): ${root}"
  git -C "${root}" submodule update --init --recursive --depth 1
}

_mb_bridge_fetch_ref() {
  local root="$1" ref="$2"
  if _mb_bridge_is_commit_sha "${ref}" && [[ ${#ref} -lt 40 ]]; then
    echo "error: Megatron-Bridge pin ${ref} is a short commit SHA; use the full 40-character hash" >&2
    return 1
  fi
  if ! git -C "${root}" fetch --depth 1 origin "${ref}"; then
    echo "error: Megatron-Bridge git fetch failed in ${root} (ref=${ref})" >&2
    return 1
  fi
  if ! git -C "${root}" reset --hard FETCH_HEAD; then
    echo "error: Megatron-Bridge git reset failed in ${root}" >&2
    return 1
  fi
  _mb_bridge_update_submodules "${root}"
}

_mb_bridge_clone_ref() {
  local dir="$1" url="$2" ref="$3"
  rm -rf "${dir}"
  if _mb_bridge_is_commit_sha "${ref}"; then
    git init "${dir}" || return 1
    git -C "${dir}" remote add origin "${url}" || return 1
    _mb_bridge_fetch_ref "${dir}" "${ref}" || return 1
  elif ! git clone --depth 1 --branch "${ref}" --recurse-submodules --shallow-submodules "${url}" "${dir}" 2>/dev/null; then
    rm -rf "${dir}"
    git init "${dir}" || return 1
    git -C "${dir}" remote add origin "${url}" || return 1
    _mb_bridge_fetch_ref "${dir}" "${ref}" || return 1
  else
    _mb_bridge_update_submodules "${dir}"
  fi
}

# Clone or update Megatron-Bridge under ${CACHE_DIR}/Megatron-Bridge (flock on Lustre).
# uv lock/sync for clone checkouts runs in mb_sync_bridge_env (skipped for container source).
bootstrap_megatron_bridge_git_checkout() {
  local dir url ref lockfile _bootstrap_rc=0 _croot

  # Explicit override always wins (must be a complete checkout).
  if [[ -n "${MEGATRON_BRIDGE_ROOT:-}" ]]; then
    if ! _mb_bridge_verify_tree "${MEGATRON_BRIDGE_ROOT}"; then
      echo "error: MEGATRON_BRIDGE_ROOT is set but not a complete Megatron-Bridge checkout: ${MEGATRON_BRIDGE_ROOT}" >&2
      return 1
    fi
    export MEGATRON_BRIDGE_ROOT="$(cd -- "${MEGATRON_BRIDGE_ROOT}" && pwd)"
    mb_log_ok "using MEGATRON_BRIDGE_ROOT=${MEGATRON_BRIDGE_ROOT}"
    return 0
  fi

  # Default: use the checkout shipped inside the container (no clone, no network).
  if ! _mb_git_bootstrap_enabled; then
    _croot="$(mb_container_root)"
    if ! _mb_bridge_verify_tree "${_croot}"; then
      echo "error: container Megatron-Bridge not found or incomplete at ${_croot}" >&2
      echo "hint: use a nemo image that ships Megatron-Bridge, set MEGATRON_BRIDGE_CONTAINER_ROOT," >&2
      echo "      or set MEGATRON_BRIDGE_GIT_BOOTSTRAP=1 to clone into CACHE_DIR" >&2
      return 1
    fi
    export MEGATRON_BRIDGE_ROOT="$(cd -- "${_croot}" && pwd)"
    mb_log_ok "using container Megatron-Bridge (no clone): ${MEGATRON_BRIDGE_ROOT}"
    return 0
  fi

  # Legacy git bootstrap: clone/update under CACHE_DIR.
  : "${CACHE_DIR:?CACHE_DIR must be set before Megatron-Bridge bootstrap}"

  dir="$(_mb_bridge_default_root)"
  url="${MEGATRON_BRIDGE_GIT_URL:-${_MEGATRON_BRIDGE_GIT_URL}}"
  ref="${MEGATRON_BRIDGE_GIT_REF:-${_MEGATRON_BRIDGE_GIT_REF}}"
  lockfile="${CACHE_DIR}/.megatron_bridge_bootstrap.lock"

  mkdir -p "${CACHE_DIR}"

  _mb_bridge_sync() {
    if [[ -d "${dir}/.git" ]] && _mb_bridge_verify_tree "${dir}" && _mb_bridge_head_matches_ref "${dir}" "${ref}"; then
      mb_log_ok "git @ ${ref} ($(git -C "${dir}" rev-parse --short HEAD)) — up to date"
      return 0
    fi
    _mb_bridge_clear_stale_git_locks "${dir}"
    if [[ -d "${dir}/.git" ]]; then
      mb_log_info "fetching ${ref} into ${dir}"
      _mb_bridge_fetch_ref "${dir}" "${ref}" || return 1
    else
      mb_log_info "cloning ${url} (${ref}) → ${dir}"
      _mb_bridge_clone_ref "${dir}" "${url}" "${ref}" || return 1
    fi
    _mb_bridge_verify_tree "${dir}"
  }

  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 200 || exit 1
      _mb_bridge_sync
    ) 200>"${lockfile}" || _bootstrap_rc=$?
  else
    mb_log_warn "flock not found; git bootstrap may race on multi-node jobs"
    _mb_bridge_sync || _bootstrap_rc=$?
  fi

  if ((_bootstrap_rc != 0)) || ! _mb_bridge_verify_tree "${dir}"; then
    echo "error: Megatron-Bridge git bootstrap failed (${dir})." >&2
    echo "hint: unset MEGATRON_BRIDGE_GIT_BOOTSTRAP to use the container checkout (default)," >&2
    echo "      or set MEGATRON_BRIDGE_ROOT=/path/to/checkout to use an existing tree" >&2
    return 1
  fi

  export MEGATRON_BRIDGE_ROOT="$(cd -- "${dir}" && pwd)"
  export MEGATRON_BRIDGE_GIT_DIR="${MEGATRON_BRIDGE_ROOT}"
  mb_log_ok "checkout ready: ${MEGATRON_BRIDGE_ROOT} ($(git -C "${MEGATRON_BRIDGE_ROOT}" rev-parse --short HEAD))"
}

# Login-node resolution (runs OUTSIDE the container): choose the source and, only for the
# legacy git path, clone into shared Lustre before srun. The container checkout lives at /opt
# inside the image, so it is recorded here but validated during container prep.
mb_resolve_bridge_root_login() {
  if [[ -n "${MEGATRON_BRIDGE_ROOT:-}" ]]; then
    [[ -d "${MEGATRON_BRIDGE_ROOT}" ]] && export MEGATRON_BRIDGE_ROOT="$(cd -- "${MEGATRON_BRIDGE_ROOT}" && pwd)"
    export SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP=1
    mb_log_info "Megatron-Bridge source (preset): ${MEGATRON_BRIDGE_ROOT}"
    return 0
  fi
  if _mb_git_bootstrap_enabled; then
    bootstrap_megatron_bridge_git_checkout || return 1
    export SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP=1
    return 0
  fi
  export MEGATRON_BRIDGE_ROOT="$(mb_container_root)"
  export SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP=1
  mb_log_info "Megatron-Bridge source: container ${MEGATRON_BRIDGE_ROOT} (validated at container prep)"
  return 0
}

# NeMo container: symlink imageio-ffmpeg as ffmpeg (Bridge qwen25_omni README).
mb_ensure_ffmpeg_on_path() {
  command -v ffmpeg >/dev/null 2>&1 && return 0
  local _py="${UV_PROJECT_ENVIRONMENT}/bin/python" _bin _dir
  [[ -x "${_py}" ]] || _py="$(command -v python3)"
  _bin="$("${_py}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)" || return 0
  for _dir in /usr/local/bin "${CACHE_DIR}/bin"; do
    mkdir -p "${_dir}" && ln -sf "${_bin}" "${_dir}/ffmpeg" && export PATH="${_dir}:${PATH}"
    command -v ffmpeg >/dev/null && { mb_log_ok "ffmpeg → ${_dir}/ffmpeg"; return 0; }
  done
}

mb_ensure_bridge_audio_deps() {
  local _py="${UV_PROJECT_ENVIRONMENT}/bin/python"
  [[ -x "${_py}" ]] || {
    echo "error: bridge venv python not found: ${_py}" >&2
    return 1
  }
  if "${_py}" -c "import librosa" >/dev/null 2>&1; then
    mb_log_ok "librosa available in bridge venv"
    return 0
  fi
  mb_log_info "installing librosa into bridge venv (required by ParakeetFeatureExtractor)"
  "${_py}" -m pip install librosa
  "${_py}" -c "import librosa; print('librosa', librosa.__version__)" >/dev/null
  mb_log_ok "librosa installed in bridge venv"
}

# Full Megatron-Bridge prep inside the NeMo container: source verify + env setup + import check.
# Container source: /opt/venv + extra deps. Clone source: uv sync into ${ROOT}/.venv (cached by git HEAD).
mb_prepare_bridge_env() {
  local _mode_dir="${1:?mode dir required}"
  local _head _cached=0

  mb_log_section "Megatron-Bridge container prep"
  mb_export_runtime_env

  mb_log_step "Load launch config"
  # shellcheck source=_source_params.sh
  source "${_MB_COMMON_DIR}/_source_params.sh"
  source_cluster_params "${_mode_dir}"
  resolve_avlm_repo_roots_from_mode_dir "${_mode_dir}"
  setup_cache_dir_env
  mb_log_info "CACHE_DIR=${CACHE_DIR}"

  mb_log_step "Megatron-Bridge source"
  if [[ -n "${MEGATRON_BRIDGE_ROOT:-}" ]] && _mb_bridge_verify_tree "${MEGATRON_BRIDGE_ROOT}"; then
    export MEGATRON_BRIDGE_ROOT="$(cd -- "${MEGATRON_BRIDGE_ROOT}" && pwd)"
    _head="$(git -C "${MEGATRON_BRIDGE_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    mb_log_ok "using source: ${MEGATRON_BRIDGE_ROOT} (${_head})"
  else
    bootstrap_megatron_bridge_git_checkout || return 1
    _head="$(git -C "${MEGATRON_BRIDGE_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    mb_log_info "source=${MEGATRON_BRIDGE_ROOT} commit=${_head}"
  fi

  mb_log_step "Python environment"
  mb_sync_bridge_env || return 1
  [[ "${MB_BRIDGE_ENV_CACHED:-0}" == "1" ]] && _cached=1
  # Clone/explicit venv may lack librosa; container /opt/venv already has it (no-op there).
  mb_container_env_active || mb_ensure_bridge_audio_deps || return 1

  if ((_cached == 0)); then
    mb_log_step "Verify imports (megatron.core / megatron.bridge)"
    mb_verify_bridge_imports
  fi

  mb_log_section "Ready"
  if mb_container_env_active; then
    mb_log_info "using container /opt/venv (no uv sync)"
  elif ((_cached == 1)); then
    mb_log_info "venv cached @ git ${_head} — skipped uv lock/sync and import checks"
  fi
  mb_log_info "MEGATRON_BRIDGE_ROOT=${MEGATRON_BRIDGE_ROOT}"
  mb_log_info "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
  mb_log_info "UV_CACHE_DIR=${UV_CACHE_DIR}"
  mb_export_bridge_uv_env
  mb_ensure_ffmpeg_on_path
  mb_log_info "shell python: $(command -v python3)"
}

mb_init_launch() {
  local _mode_dir="${1:?mode dir required}"
  local _launch_kind="${2:-interactive}"
  local _load_conversion="${3:-}"

  export CLUSTER_PARAMS="${CLUSTER_PARAMS:-${_mode_dir}/launch_local.yaml}"
  [[ "${CLUSTER_PARAMS}" == /* ]] || CLUSTER_PARAMS="${_mode_dir}/${CLUSTER_PARAMS#"${_mode_dir}"/}"
  export CLUSTER_PARAMS

  # shellcheck source=_source_params.sh
  source "${_MB_COMMON_DIR}/_source_params.sh"
  if [[ "${_load_conversion}" == "conversion" ]]; then
    training_cli_commit_env_overrides "${_mode_dir}" "${MB_BRIDGE_CLI_OVERRIDE_KEYS[@]}"
  elif [[ "${_launch_kind}" == "local" ]]; then
    training_cli_commit_env_overrides "${_mode_dir}" "${GENERIC_CLI_OVERRIDE_KEYS[@]}"
  else
    training_cli_commit_env_overrides "${_mode_dir}" "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}"
  fi
  source_cluster_params "${_mode_dir}"
  if [[ "${_load_conversion}" == "conversion" ]]; then
    source_conversion_params "${_mode_dir}"
  fi
  resolve_avlm_repo_roots_from_mode_dir "${_mode_dir}"

  setup_cache_dir_env
  bootstrap_megatron_bridge_git_checkout

  if [[ "${_launch_kind}" == "local" ]]; then
    generic_resolve_num_gpus
    export GPUS_PER_NODE="${NPROC_PER_NODE}"
  else
    : "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
    export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"
  fi

  if [[ -n "${HF_MODEL_ID:-}" ]]; then
    export HF_MODEL_BASENAME="${HF_MODEL_BASENAME:-$(basename "${HF_MODEL_ID}")}"
    export WORKSPACE="${WORKSPACE:-${CACHE_DIR}/megatron_bridge_workspace}"
    export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/${HF_MODEL_BASENAME}-megatron}"
  fi
  export WORKDIR="${WORKDIR:-${REPO_ROOT}}"

  if [[ "${_launch_kind}" == "interactive" || "${_launch_kind}" == "local" ]]; then
    export num_nodes=1
  elif [[ "${SLURM_NNODES:-0}" -gt 0 ]]; then
    export num_nodes="${SLURM_NNODES}"
  else
    export num_nodes="${num_nodes:-1}"
  fi

  mb_export_runtime_env
}

# NeMo container may pre-set PYTHONWARNINGS; merge our filters instead of skipping them.
# NOTE: PYTHONWARNINGS entries must be module-based (no spaces / no regex `.*`): the value is
# converted to -W options and word-split, so message filters with spaces break parsing (and bare
# `.*` glob-expands in the shell). Message-text suppression lives in run_recipe_avlm.py via
# warnings.filterwarnings(), which handles spaced regexes correctly.
_mb_merge_python_warning_filters() {
  local _filter _clean="" _tok _filters=(
    ignore::FutureWarning
    ignore::UserWarning:importlib.metadata
    ignore::UserWarning:modelopt.torch
    ignore::UserWarning:megatron.core
    ignore::UserWarning:comet_ml.error_tracking.api
    ignore::UserWarning:sentry_sdk
  )
  # Sanitize any inherited value: drop tokens with spaces or `*` (they break -W and glob-expand,
  # and a bad value would otherwise re-merge into itself every run within a live session).
  # read -ra (quoted) splits on comma only — no pathname/glob expansion of `*` in the tokens.
  local -a _toks=()
  IFS=',' read -ra _toks <<< "${PYTHONWARNINGS:-}"
  for _tok in "${_toks[@]}"; do
    [[ -z "${_tok}" || "${_tok}" == *" "* || "${_tok}" == *"*"* ]] && continue
    [[ -n "${_clean}" ]] && _clean+=","
    _clean+="${_tok}"
  done
  for _filter in "${_filters[@]}"; do
    if [[ -z "${_clean}" || "${_clean}" != *"${_filter}"* ]]; then
      [[ -n "${_clean}" ]] && _clean+=","
      _clean+="${_filter}"
    fi
  done
  printf '%s\n' "${_clean}"
}

mb_export_runtime_env() {
  export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export HTTPX_LOG_LEVEL="${HTTPX_LOG_LEVEL:-WARNING}"
  export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
  # vLLM is pulled in transitively (modelopt/model import) but unused for training; its import-time
  # "Failed to import Triton kernels (matmul_ogs)" is logged at ERROR. Hide vLLM noise by default.
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-CRITICAL}"
  # Filtering stderr through grep makes isatty(2) false, so color-on-TTY libraries drop ANSI.
  # Force color back on so the grep filter doesn't change log appearance.
  export FORCE_COLOR="${FORCE_COLOR:-1}"
  export CLICOLOR_FORCE="${CLICOLOR_FORCE:-1}"
  export PYTHONWARNINGS="$(_mb_merge_python_warning_filters)"
  export COMET_DISABLE_AUTO_LOGGING="${COMET_DISABLE_AUTO_LOGGING:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
  # PyTorch 2.7+ renamed the allocator env var; set both for compatibility.
  export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
  # Megatron sequence parallelism recommendation (also silences SP layer warning).
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}

# Persistent dir (on Lustre) for the few packages the container doesn't ship (decord, imageio-ffmpeg).
mb_container_extra_pkgs_dir() {
  printf '%s\n' "${MB_BRIDGE_EXTRA_PKGS_DIR:-${CACHE_DIR}/mbridge_container_pkgs}"
}

# Python launcher tokens for the active env: container → /opt/venv python directly (no uv);
# clone/explicit → `uv run --no-sync python` against the project venv. Read with mapfile:
#   local -a _py; mapfile -t _py < <(mb_py_launch); "${_py[@]}" script.py ...
mb_py_launch() {
  if mb_container_env_active; then
    printf '%s\n' "${_MEGATRON_BRIDGE_CONTAINER_VENV}/bin/python"
  else
    printf '%s\n' uv run --no-sync python
  fi
}

mb_export_bridge_uv_env() {
  : "${MEGATRON_BRIDGE_ROOT:?MEGATRON_BRIDGE_ROOT must be set}"
  : "${CACHE_DIR:?CACHE_DIR must be set}"

  # Container-source mode: use the image's own /opt/venv directly (it already has torch, TE,
  # mamba_ssm, causal_conv1d, librosa and Bridge editable). No project venv, no uv sync — only
  # the container-missing deps go into a persistent Lustre dir added to PYTHONPATH.
  if mb_container_env_active; then
    export UV_PROJECT_ENVIRONMENT="${_MEGATRON_BRIDGE_CONTAINER_VENV}"
    export VIRTUAL_ENV="${_MEGATRON_BRIDGE_CONTAINER_VENV}"
    export UV_CACHE_DIR="${MB_BRIDGE_UV_CACHE:-${CACHE_DIR}/uv-cache}"
    mkdir -p "${UV_CACHE_DIR}"
    local _extra; _extra="$(mb_container_extra_pkgs_dir)"
    mkdir -p "${_extra}"
    case ":${PYTHONPATH:-}:" in
      *":${_extra}:"*) ;;
      *) export PYTHONPATH="${_extra}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
    export PATH="${_MEGATRON_BRIDGE_CONTAINER_VENV}/bin:${PATH//${_MEGATRON_BRIDGE_CONTAINER_VENV}\/bin:/}"
    return 0
  fi

  # Clone / explicit external checkout: build a uv-synced project venv next to the source.
  # NeMo container pre-sets UV_PROJECT_ENVIRONMENT=/opt/venv and UV_CACHE_DIR=/opt/uv_cache.
  export UV_PROJECT_ENVIRONMENT="${MB_BRIDGE_VENV:-${MEGATRON_BRIDGE_ROOT}/.venv}"
  export UV_CACHE_DIR="${MB_BRIDGE_UV_CACHE:-${CACHE_DIR}/uv-cache}"
  [[ "${UV_PROJECT_ENVIRONMENT}" == /opt || "${UV_PROJECT_ENVIRONMENT}" == /opt/* ]] && {
    echo "error: UV_PROJECT_ENVIRONMENT must not be under /opt: ${UV_PROJECT_ENVIRONMENT}" >&2
    return 1
  }
  mkdir -p "${UV_CACHE_DIR}"
  export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
  export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH//\/opt\/venv\/bin:/}"
}

_mb_venv_uses_system_site_packages() {
  local _venv="${1:?venv required}"
  [[ -f "${_venv}/pyvenv.cfg" ]] && grep -q 'include-system-site-packages = true' "${_venv}/pyvenv.cfg"
}

_mb_venv_python_imports_torch() {
  local _venv="${1:?venv required}"
  [[ -x "${_venv}/bin/python" ]] && "${_venv}/bin/python" -c "import torch" 2>/dev/null
}

# NeMo torch lives in /opt/venv or /usr/local — not in uv's managed CPython.
_mb_find_torch_python() {
  local _py _candidates=()
  if [[ -n "${MB_BRIDGE_PYTHON:-}" ]]; then
    _candidates=("${MB_BRIDGE_PYTHON}")
  else
    _candidates=(
      /opt/venv/bin/python3
      /opt/venv/bin/python
      /usr/local/bin/python3.12
      /usr/bin/python3.12
      python3.12
      python3
    )
  fi
  for _py in "${_candidates[@]}"; do
    [[ -n "${_py}" ]] || continue
    if [[ -x "${_py}" ]] && "${_py}" -c "import torch" 2>/dev/null; then
      printf '%s\n' "${_py}"
      return 0
    fi
  done
  return 1
}

# Megatron-Bridge uv.lock excludes torch (expects NeMo container system packages).
mb_ensure_bridge_venv() {
  local _venv="${UV_PROJECT_ENVIRONMENT}"
  local _py _marker="${_venv}/.mb_bridge_uv_sync_ok"

  _py="$(_mb_find_torch_python)" || {
    echo "error: no container Python with torch found (tried /opt/venv/bin/python, /usr/bin/python3.12, …)" >&2
    echo "hint: set MB_BRIDGE_PYTHON to a python that can 'import torch'" >&2
    return 1
  }
  mb_log_info "base python (torch): ${_py}"
  export UV_PYTHON="${_py}"

  if [[ -d "${_venv}" ]]; then
    if ! _mb_venv_uses_system_site_packages "${_venv}" || ! _mb_venv_python_imports_torch "${_venv}"; then
      mb_log_warn "removing .venv (needs --system-site-packages + container torch)"
      [[ "${_venv}" == /opt || "${_venv}" == /opt/* ]] && {
        echo "error: refusing to remove path under /opt: ${_venv}" >&2
        return 1
      }
      rm -rf "${_venv}"
      rm -f "${_marker}"
    fi
  fi

  if [[ ! -f "${_venv}/pyvenv.cfg" ]]; then
    mb_log_info "creating .venv with --system-site-packages"
    uv venv "${_venv}" --python "${_py}" --system-site-packages
    rm -f "${_marker}"
  fi

  if ! _mb_venv_python_imports_torch "${_venv}"; then
    echo "error: torch not importable from ${_venv}/bin/python after venv create" >&2
    "${_venv}/bin/python" -c "import sys; print('sys.path:', sys.path)" >&2 || true
    return 1
  fi
  mb_log_ok "torch available in project venv"
}

# Skip uv install only when the module resolves outside the project venv (container copy).
_mb_import_from_container() {
  local _venv_py="${1:?}" _mod="${2:?}" _venv="${3:?}" _path
  _path="$("${_venv_py}" -c "import ${_mod}; print(${_mod}.__file__)" 2>/dev/null)" || return 1
  [[ "${_path}" != "${_venv}/"* ]]
}

_mb_uv_container_skip_args() {
  local _venv_py="${1:?}" _venv="${2:?}" _spec _uv_pkg _import_mod
  [[ "${MB_BRIDGE_USE_CONTAINER_PKGS:-${MB_BRIDGE_USE_CONTAINER_TE:-1}}" == "1" ]] || return 0
  for _spec in \
    "transformer-engine:transformer_engine" \
    "mamba-ssm:mamba_ssm" \
    "causal-conv1d:causal_conv1d"; do
    _uv_pkg="${_spec%%:*}"
    _import_mod="${_spec#*:}"
    if _mb_import_from_container "${_venv_py}" "${_import_mod}" "${_venv}"; then
      printf '%s\n' --no-install-package "${_uv_pkg}"
      mb_log_info "${_uv_pkg} from container (--no-install-package)"
    fi
  done
}

_mb_verify_bridge_native_imports() {
  local _venv="${1:?}" _imp _path _fail=0
  for _imp in transformer_engine mamba_ssm causal_conv1d; do
    _path="$("${_venv}/bin/python" -c "import ${_imp}; print(${_imp}.__file__)" 2>/dev/null)" || {
      echo "error: import ${_imp} failed after uv sync" >&2
      _fail=1
      continue
    }
    mb_log_ok "${_imp} → ${_path}"
  done
  (( _fail == 0 ))
}

_mb_bridge_uv_sync_marker() {
  printf '%s\n' "${1:?venv required}/.mb_bridge_uv_sync_ok"
}

# True when venv was synced for the current Megatron-Bridge commit (marker stores full git SHA).
_mb_bridge_env_cached() {
  local _root="$1" _venv="$2" _marker _saved _head
  _marker="$(_mb_bridge_uv_sync_marker "${_venv}")"
  [[ -f "${_marker}" ]] || return 1
  _saved="$(tr -d '[:space:]' < "${_marker}" 2>/dev/null || true)"
  if [[ -z "${_saved}" ]] && _mb_venv_python_imports_torch "${_venv}"; then
    # Legacy empty marker (touch-only): stamp current commit and treat as cached.
    _head="$(git -C "${_root}" rev-parse HEAD 2>/dev/null || true)"
    if [[ -n "${_head}" ]]; then
      printf '%s\n' "${_head}" > "${_marker}"
      _saved="${_head}"
    fi
  fi
  [[ -n "${_saved}" ]] || return 1
  _head="$(git -C "${_root}" rev-parse HEAD 2>/dev/null || true)"
  [[ -n "${_head}" && "${_saved}" == "${_head}" ]] || return 1
  _mb_venv_python_imports_torch "${_venv}"
}

# Install (once, to a persistent Lustre dir) the packages the container image doesn't ship.
# librosa/mamba_ssm/causal_conv1d/TE/torch are already in /opt/venv; only these are missing.
mb_ensure_container_extra_deps() {
  local _py="${_MEGATRON_BRIDGE_CONTAINER_VENV}/bin/python"
  local _target; _target="$(mb_container_extra_pkgs_dir)"
  local _spec _pkg _mod _missing=()
  [[ -x "${_py}" ]] || { echo "error: container python not found: ${_py}" >&2; return 1; }
  mkdir -p "${_target}"
  for _spec in "decord:decord" "imageio-ffmpeg:imageio_ffmpeg"; do
    _pkg="${_spec%%:*}"; _mod="${_spec#*:}"
    PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}" 2>/dev/null || _missing+=("${_pkg}")
  done
  if ((${#_missing[@]} > 0)); then
    mb_log_info "installing container-missing deps: ${_missing[*]} → ${_target}"
    # --no-deps: decord's only dep is numpy (already in /opt/venv). Pulling transitive deps
    # into a PYTHONPATH dir would shadow the container's pinned numpy and break mamba/TE/etc.
    "${_py}" -m pip install --no-cache-dir --no-deps --target "${_target}" --upgrade "${_missing[@]}" || {
      echo "error: pip install of ${_missing[*]} into ${_target} failed" >&2
      return 1
    }
  fi
  local _fail=0
  for _mod in decord imageio_ffmpeg; do
    PYTHONPATH="${_target}:${PYTHONPATH:-}" "${_py}" -c "import ${_mod}" 2>/dev/null \
      || { echo "error: ${_mod} still not importable after install (${_target})" >&2; _fail=1; }
  done
  (( _fail == 0 )) || return 1
  mb_log_ok "container env ready: /opt/venv + extra deps (decord, imageio_ffmpeg) in ${_target}"
}

mb_sync_bridge_env() {
  local _root="${MEGATRON_BRIDGE_ROOT}"
  local _uv_quiet=() _no_install=()
  local _venv _marker _head
  export MB_BRIDGE_ENV_CACHED=0
  if [[ ! -d "${_root}" ]]; then
    echo "error: MEGATRON_BRIDGE_ROOT not found: ${_root}" >&2
    echo "hint: set CACHE_DIR and run bootstrap, or set MEGATRON_BRIDGE_ROOT explicitly" >&2
    return 1
  fi
  # Container-source mode: no uv sync; use /opt/venv and top up the missing deps only.
  if mb_container_env_active; then
    mb_export_bridge_uv_env
    mb_ensure_container_extra_deps || return 1
    export MB_BRIDGE_ENV_CACHED=1
    return 0
  fi
  if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
    mb_log_info "SKIP_UV_SYNC=1 — skipping uv lock/sync"
    mb_export_bridge_uv_env
    return 0
  fi
  mb_export_bridge_uv_env
  _venv="${UV_PROJECT_ENVIRONMENT}"
  _marker="$(_mb_bridge_uv_sync_marker "${_venv}")"
  if [[ "${MB_BRIDGE_FORCE_UV_SYNC:-0}" != "1" ]] && _mb_bridge_env_cached "${_root}" "${_venv}"; then
    _head="$(git -C "${_root}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    mb_log_ok "venv ready (cached @ ${_head}) — skipping uv lock/sync"
    export MB_BRIDGE_ENV_CACHED=1
    return 0
  fi
  mb_log_info "project venv: ${_venv}"
  mb_log_info "package cache: ${UV_CACHE_DIR}"
  if [[ "${MB_BRIDGE_BACKUP_VENV:-0}" == "1" && -d "${_venv}" ]]; then
    local _bak="${_venv}.bak-$(date -u +%Y%m%dT%H%M%SZ)"
    mb_log_info "backing up venv → ${_bak}"
    mv "${_venv}" "${_bak}"
  fi
  mb_ensure_bridge_venv
  export UV_PYTHON="${UV_PYTHON:-$(_mb_find_torch_python)}"
  mapfile -t _no_install < <(_mb_uv_container_skip_args "${_venv}/bin/python" "${_venv}")
  if [[ -f "${_marker}" ]]; then
    _uv_quiet=(--quiet)
    mb_log_info "running uv lock + sync (incremental, quiet)"
  else
    mb_log_info "running uv lock + sync (first run — may take several minutes)"
  fi
  mb_export_runtime_env
  (cd "${_root}" && uv lock "${_uv_quiet[@]}" && uv sync "${_uv_quiet[@]}" "${_no_install[@]}")
  [[ -d "${_venv}" ]] || {
    echo "error: uv sync finished but venv missing: ${_venv}" >&2
    return 1
  }
  mb_log_step "Verify native imports (transformer_engine / mamba_ssm / causal_conv1d)"
  _mb_verify_bridge_native_imports "${_venv}" || return 1
  _head="$(git -C "${_root}" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf '%s\n' "${_head}" > "${_marker}"
  mb_log_ok "venv ready: ${_venv} (@ $(git -C "${_root}" rev-parse --short HEAD 2>/dev/null || echo unknown))"
}

# Print megatron.core / megatron.bridge paths (stdout); warnings suppressed.
mb_verify_bridge_imports_paths() {
  local _venv="${UV_PROJECT_ENVIRONMENT:?UV_PROJECT_ENVIRONMENT must be set}"
  "${_venv}/bin/python" -W ignore -c "
import megatron.core, megatron.bridge
print(megatron.core.__path__[0])
print(megatron.bridge.__path__[0])
" 2>/dev/null
}

mb_verify_bridge_imports() {
  local _root="${MEGATRON_BRIDGE_ROOT}" _core _bridge
  local _expected_core="${_root}/3rdparty/Megatron-LM/megatron/core"
  local _expected_bridge="${_root}/src/megatron/bridge"
  local _paths=()

  mapfile -t _paths < <(mb_verify_bridge_imports_paths)
  _core="${_paths[0]:-}"
  _bridge="${_paths[1]:-}"
  if [[ -z "${_core}" || -z "${_bridge}" ]]; then
    echo "error: import verification failed (megatron.core / megatron.bridge)" >&2
    return 1
  fi

  mb_log_ok "megatron.core  → ${_core}"
  mb_log_ok "megatron.bridge → ${_bridge}"

  if [[ "${_core}" != "${_expected_core}" ]]; then
    mb_log_warn "megatron.core is not the checkout submodule (expected ${_expected_core})"
  fi
  if [[ "${_bridge}" != "${_expected_bridge}" ]]; then
    if [[ -n "${MB_BRIDGE_OVERLAY:-}" ]]; then
      mb_log_ok "megatron.bridge overlay paths injected from ${MB_BRIDGE_OVERLAY}"
    else
      mb_log_warn "megatron.bridge is not the checkout src tree (expected ${_expected_bridge})"
    fi
  fi
}

mb_validate_parallelism_layout() {
  local _nodes="${1:?nodes required}"
  local _gpus="${2:?gpus per node required}"
  local _tp="${TP:-1}"
  local _ep="${EP:-1}"
  local _world=$((_nodes * _gpus))
  local _required=$(( _tp > _ep ? _tp : _ep ))

  if ((_world < _required)); then
    echo "error: world_size=${_world} < required=${_required} (max(TP,EP)=max(${_tp},${_ep}))" >&2
    return 1
  fi
  if ((_world % _required != 0)); then
    mb_log_warn "world_size=${_world} is not a multiple of required=${_required}"
  fi
  mb_log_info "parallelism TP=${_tp} EP=${_ep} (world=${_world}, required=${_required})"
}
