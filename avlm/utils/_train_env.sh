# Shared env for AutoModel SFT/LoRA launch scripts.
# Set CONFIG_YAML or CONFIG_YAML_REL and _TRAIN_SCRIPT_DIR before sourcing
# (legacy: CORD_V2_YAML_REL, _CORD_SCRIPT_DIR).
#
# nemo_automodel ships in the nemo-automodel container at /opt/Automodel (26_06+).
# resolve_automodel_code_root() sets AUTOMODEL_CODE_ROOT (default /opt/Automodel).
# Legacy git checkout: AUTOMODEL_GIT_BOOTSTRAP=1 (requires CACHE_DIR + network).
# avlm + YAML from REPO_ROOT.

CONFIG_YAML_REL="${CONFIG_YAML_REL:-${CORD_V2_YAML_REL:-}}"
_TRAIN_SCRIPT_DIR="${_TRAIN_SCRIPT_DIR:-${_CORD_SCRIPT_DIR:-}}"
if [[ "${CORD_TRAIN_KEEP_CLUSTER_TMPDIR:-0}" == "1" ]]; then
  export TRAIN_KEEP_CLUSTER_TMPDIR=1
fi
: "${_TRAIN_SCRIPT_DIR:?}"

_train_env_utils_dir() {
  local d="$1"
  while [[ "${d}" != "/" ]]; do
    if [[ -f "${d}/utils/_source_params.sh" ]]; then
      cd -- "${d}/utils" && pwd
      return 0
    fi
    d="$(dirname -- "${d}")"
  done
  echo "error: avlm/utils/_source_params.sh not found relative to ${1}" >&2
  return 1
}
_AVLM_UTILS_DIR="$(_train_env_utils_dir "${_TRAIN_SCRIPT_DIR}")"

# shellcheck source=_source_params.sh
source "${_AVLM_UTILS_DIR}/_source_params.sh"
training_require_config_path
resolve_avlm_repo_roots_from_mode_dir "${_TRAIN_SCRIPT_DIR}"

if [[ "${TRAIN_KEEP_CLUSTER_TMPDIR:-0}" != "1" ]]; then
  export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
fi

# Runtime policy (hardcoded; override on CLI if needed — not in cluster_params).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRITON_DISABLE_AUTOTUNE="${TRITON_DISABLE_AUTOTUNE:-1}"

setup_cache_dir_env
resolve_automodel_code_root || {
  echo "error: Automodel resolve failed; use nemo-automodel_26_06+ or set AUTOMODEL_CODE_ROOT" >&2
  exit 1
}

AUTOMODEL_CONTAINER_ROOT="${AUTOMODEL_CONTAINER_ROOT:-/opt/Automodel}"
[[ -d "${AUTOMODEL_CONTAINER_ROOT}" ]] || {
  echo "error: ${AUTOMODEL_CONTAINER_ROOT} missing (use nemo-automodel container)." >&2
  exit 1
}

if [[ -z "${CONFIG_YAML:-}" && -f "${_REPO_ROOT}/${CONFIG_YAML_REL}" ]]; then
  export CONFIG_YAML="${_REPO_ROOT}/${CONFIG_YAML_REL}"
fi

_walk_up_for_path() {
  local d="$1" kind="$2" arg="$3"
  while [[ "${d}" != "/" ]]; do
    if [[ "${kind}" == "file" && -f "${d}/${arg}" ]]; then
      printf "%s/%s\n" "$(cd -- "${d}" && pwd)" "${arg}"
      return 0
    elif [[ "${kind}" == "dir" && -d "${d}/${arg}" ]]; then
      cd -- "${d}" && pwd
      return 0
    fi
    d="$(dirname -- "${d}")"
  done
  return 1
}

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ -n "${CONFIG_YAML:-}" ]]; then
  [[ -f "${CONFIG_YAML}" ]] || {
    echo "error: not a file: ${CONFIG_YAML}" >&2
    exit 1
  }
elif [[ -f "${AUTOMODEL_CONTAINER_ROOT}/${CONFIG_YAML_REL}" ]]; then
  CONFIG_YAML="${AUTOMODEL_CONTAINER_ROOT}/${CONFIG_YAML_REL}"
elif [[ -f "${_REPO_ROOT}/${CONFIG_YAML_REL}" ]]; then
  CONFIG_YAML="${_REPO_ROOT}/${CONFIG_YAML_REL}"
else
  CONFIG_YAML="$(_walk_up_for_path "${_TRAIN_SCRIPT_DIR}" "file" "${CONFIG_YAML_REL}")" || true
  if [[ -z "${CONFIG_YAML}" && -n "${AUTOMODEL_CHECKOUT:-}" ]]; then
    CONFIG_YAML="$(_walk_up_for_path "$(cd -- "${AUTOMODEL_CHECKOUT}" && pwd)" "file" "${CONFIG_YAML_REL}")" || true
  fi
  [[ -n "${CONFIG_YAML}" ]] || {
    echo "error: missing ${CONFIG_YAML_REL}. Set CONFIG_YAML or AUTOMODEL_CHECKOUT." >&2
    exit 1
  }
  echo "info: YAML resolved: ${CONFIG_YAML}" >&2
fi

CONFIG_YAML="$(cd -- "$(dirname "${CONFIG_YAML}")" && pwd)/$(basename "${CONFIG_YAML}")"
export CONFIG_YAML

_NEMO_CODE_ROOT="${AUTOMODEL_CODE_ROOT}"
export PYTHONPATH="${_SPORTS_INTEL_ROOT}:${_NEMO_CODE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "${_NEMO_CODE_ROOT}"

REQUIRE_CUDA_DEVICES="${REQUIRE_CUDA_DEVICES:-8}"
if [[ "${SKIP_CUDA_DEVICE_CHECK:-0}" != "1" ]]; then
  _train_env_python() {
    if [[ -x /opt/venv/bin/python3 ]]; then
      echo /opt/venv/bin/python3
    else
      command -v python3
    fi
  }
  _n="$("$(_train_env_python)" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  [[ "${_n}" -ge "${REQUIRE_CUDA_DEVICES}" ]] || {
    echo "error: CUDA devices ${_n} < ${REQUIRE_CUDA_DEVICES}. Use SKIP_CUDA_DEVICE_CHECK=1 to skip." >&2
    exit 1
  }
fi

# Ctrl+C (automatic — no env var to set):
#   interactive / local launch → kill the full torchrun tree (press twice if NCCL is stuck)
#   sbatch batch jobs          → finish the current step, then exit
_train_ctrl_c_kills_tree() {
  case "${TRAIN_LAUNCH_KIND:-}" in
    interactive|local) return 0 ;;
    batch) return 1 ;;
  esac
  [[ "${INSIDE_INTERACTIVE_SESSION:-0}" == "1" ]]
}
_train_kill_process_tree() {
  local root_pid="${1:-0}"
  local child
  if (( root_pid <= 0 )); then
    return 0
  fi
  while read -r child; do
    [[ -n "${child}" && "${child}" != "${root_pid}" ]] || continue
    _train_kill_process_tree "${child}"
  done < <(pgrep -P "${root_pid}" 2>/dev/null || true)
  kill -KILL "${root_pid}" 2>/dev/null || true
}

_train_force_kill_torchrun() {
  local leader_pid="${1:-0}"
  if (( leader_pid != 0 )); then
    kill -KILL -- -"${leader_pid}" 2>/dev/null || kill -KILL "${leader_pid}" 2>/dev/null || true
    _train_kill_process_tree "${leader_pid}"
  fi
  # Orphan ranks after a partial crash often leave the session; pattern-kill the rest.
  pkill -9 -f '[t]orchrun.*automodel' 2>/dev/null || true
  pkill -9 -f '[/]opt/venv/bin/automodel' 2>/dev/null || true
}

train_run_with_interrupt() {
  if ! _train_ctrl_c_kills_tree; then
    "$@"
    return $?
  fi

  local _leader_pid=0
  local _interrupt_count=0
  _train_fast_interrupt_cleanup() {
    _interrupt_count=$((_interrupt_count + 1))
    if (( _interrupt_count >= 2 )); then
      echo "info: second Ctrl+C — force killing torchrun/automodel processes" >&2
      _train_force_kill_torchrun "${_leader_pid}"
      trap - INT TERM
      exit 130
    fi

    echo "info: Ctrl+C — stopping torchrun (press Ctrl+C again to force kill)" >&2
    if (( _leader_pid != 0 )); then
      kill -INT -- -"${_leader_pid}" 2>/dev/null || kill -INT "${_leader_pid}" 2>/dev/null || true
      sleep 1
      kill -TERM -- -"${_leader_pid}" 2>/dev/null || kill -TERM "${_leader_pid}" 2>/dev/null || true
      sleep 1
      _train_force_kill_torchrun "${_leader_pid}"
    fi

    # Keep trap armed so a second Ctrl+C during wait still force-kills.
    if command -v timeout >/dev/null 2>&1; then
      timeout 10 wait "${_leader_pid}" 2>/dev/null || _train_force_kill_torchrun "${_leader_pid}"
    else
      wait "${_leader_pid}" 2>/dev/null || _train_force_kill_torchrun "${_leader_pid}"
    fi
    trap - INT TERM
    exit 130
  }

  trap _train_fast_interrupt_cleanup INT TERM
  setsid "$@" &
  _leader_pid=$!
  wait "${_leader_pid}"
  local _rc=$?
  trap - INT TERM
  return "${_rc}"
}
