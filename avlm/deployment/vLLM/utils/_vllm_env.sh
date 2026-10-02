# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Container Python is the serve interpreter.
# VLLM_EXTRA_PIP_SPEC, when set, installs only those packages into a slim venv
# that inherits the image (--system-site-packages).
# INSTALL_VLLM_PACKAGES=true still builds the full ${CACHE_DIR}/vllm_serve_venv.
#
# shellcheck shell=bash

_vllm_base_python() {
  if [[ -n "${PYTHON:-}" && -x "${PYTHON}" && "${INSIDE_VLLM_DEPLOY_SESSION:-0}" != "1" ]]; then
    printf '%s\n' "${PYTHON}"
  elif [[ -x /opt/venv/bin/python3 ]]; then
    printf '%s\n' /opt/venv/bin/python3
  else
    command -v python3
  fi
}

_vllm_install_packages_enabled() {
  case "${INSTALL_VLLM_PACKAGES:-0}" in
    1 | true | True | TRUE | yes | YES | on | ON) return 0 ;;
    *) return 1 ;;
  esac
}

_vllm_container_python() {
  if [[ -x /opt/venv/bin/python3 ]]; then
    printf '%s\n' /opt/venv/bin/python3
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  fi
}

# Install only VLLM_EXTRA_PIP_SPEC into a venv that sees the image site-packages.
_vllm_spec_imports_ok() {
  local py="$1" spec="$2" token name
  # shellcheck disable=SC2086
  for token in ${spec}; do
    name="${token%%[=<>!~]*}"
    name="${name//-/_}"
    [[ -n "${name}" ]] || continue
    if ! "${py}" -c "import ${name}" >/dev/null 2>&1; then
      return 1
    fi
  done
  return 0
}

# Later starts reuse the venv created by VLLM_INSTALL_EXTRA=1.
_vllm_reuse_extra_venv() {
  [[ -n "${VLLM_EXTRA_PIP_SPEC:-}" ]] && return 0
  local venv_dir py stamp spec
  if [[ -n "${VLLM_EXTRA_VENV_DIR:-}" ]]; then
    venv_dir="${VLLM_EXTRA_VENV_DIR}"
  elif [[ -n "${CACHE_DIR:-}" ]]; then
    venv_dir="${CACHE_DIR}/vllm_extra_venv"
  else
    return 0
  fi
  py="${venv_dir}/bin/python"
  [[ -x "${py}" ]] || return 0
  stamp="${venv_dir}/.vllm_extra_pip_spec"
  if [[ -f "${stamp}" ]]; then
    spec="$(<"${stamp}")"
  else
    spec="decord==0.6.0"
  fi
  [[ -n "${spec}" ]] || return 0
  export VLLM_EXTRA_VENV_DIR="${venv_dir}"
  export VLLM_EXTRA_PIP_SPEC="${spec}"
  echo "[vllm-env] reusing extra venv ${py}" >&2
}

# Used by vllm_server.sh start when VLLM_INSTALL_EXTRA=1.
# Skips the venv when the image Python can already import the extra packages.
_vllm_install_missing_extras() {
  local py spec
  py="$(_vllm_container_python)"
  [[ -n "${py}" && -x "${py}" ]] || {
    echo "[vllm-env] error: container Python not found" >&2
    return 1
  }
  spec="${VLLM_EXTRA_PIP_SPEC:-decord==0.6.0}"
  if _vllm_spec_imports_ok "${py}" "${spec}"; then
    echo "[vllm-env] extra packages already importable from ${py}" >&2
    unset VLLM_EXTRA_PIP_SPEC
    return 0
  fi
  export VLLM_EXTRA_PIP_SPEC="${spec}"
  if [[ -z "${VLLM_EXTRA_VENV_DIR:-}" ]]; then
    if [[ -n "${CACHE_DIR:-}" ]]; then
      export VLLM_EXTRA_VENV_DIR="${CACHE_DIR}/vllm_extra_venv"
    else
      : "${REPO_ROOT:?REPO_ROOT must be set to place the extra venv}"
      export VLLM_EXTRA_VENV_DIR="${REPO_ROOT}/.cache/vllm_extra_venv"
    fi
  fi
  _vllm_ensure_extra_venv
}

_vllm_ensure_extra_venv() {
  local spec="${VLLM_EXTRA_PIP_SPEC:-}"
  [[ -n "${spec}" ]] || return 0

  local venv_dir base_py py stamp
  venv_dir="${VLLM_EXTRA_VENV_DIR:-}"
  if [[ -z "${venv_dir}" ]]; then
    : "${CACHE_DIR:?Set CACHE_DIR or VLLM_EXTRA_VENV_DIR before installing extra packages}"
    venv_dir="${CACHE_DIR}/vllm_extra_venv"
  fi
  base_py="$(_vllm_container_python)"
  [[ -n "${base_py}" && -x "${base_py}" ]] || {
    echo "[vllm-env] error: container Python not found" >&2
    return 1
  }
  stamp="${venv_dir}/.vllm_extra_pip_spec"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "[vllm-env] creating slim venv at ${venv_dir}" >&2
    "${base_py}" -m venv --system-site-packages "${venv_dir}"
  fi
  py="${venv_dir}/bin/python"
  if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${spec}" ]]; then
    echo "[vllm-env] slim venv ok (cached): ${venv_dir} (${spec})" >&2
  else
    echo "[vllm-env] installing extra packages into ${venv_dir}: ${spec}" >&2
    # shellcheck disable=SC2086
    "${py}" -m pip install ${spec}
    printf '%s\n' "${spec}" > "${stamp}"
  fi
  export VLLM_EXTRA_VENV_DIR="${venv_dir}"
  export PYTHON="${py}"
  export PATH="${venv_dir}/bin:${PATH}"
}

ensure_vllm_serve_env() {
  if [[ "${SKIP_VLLM_SERVE_ENV:-0}" == "1" ]] || ! _vllm_install_packages_enabled; then
    PYTHON="$(_vllm_container_python)"
    [[ -n "${PYTHON}" && -x "${PYTHON}" ]] || {
      echo "[vllm-env] error: container Python not found" >&2
      return 1
    }
    export PYTHON
    if [[ -n "${VLLM_EXTRA_PIP_SPEC:-}" ]]; then
      _vllm_ensure_extra_venv || return 1
    else
      echo "[vllm-env] using container Python ${PYTHON}" >&2
    fi
    return 0
  fi
  : "${CACHE_DIR:?CACHE_DIR must be set before ensure_vllm_serve_env}"

  local venv_dir spec tf_spec stamp expected_stamp base_py py lock_file
  venv_dir="${VLLM_VENV_DIR:-${CACHE_DIR}/vllm_serve_venv}"
  spec="${VLLM_PIP_SPEC:-vllm[audio]==0.20.0}"
  # Nemotron Omni consolidated checkpoints use tokenizer_class TokenizersBackend (Transformers v5+).
  tf_spec="${VLLM_TRANSFORMERS_PIP_SPEC:-transformers>=5.0.0,<6}"
  stamp="${venv_dir}/.vllm_pip_spec"
  expected_stamp="${spec}"$'\n'"${tf_spec}"
  lock_file="${CACHE_DIR}/.vllm_serve_venv.lock"

  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_DIR}/pip}"
  mkdir -p "${CACHE_DIR}" "${PIP_CACHE_DIR}"

  base_py="$(_vllm_base_python)"
  [[ -x "${base_py}" ]] || {
    echo "[vllm-env] error: base python not found (${base_py})" >&2
    return 1
  }

  (
    flock -w "${VLLM_VENV_LOCK_TIMEOUT:-7200}" 9 || {
      echo "[vllm-env] error: timed out waiting for ${lock_file}" >&2
      exit 1
    }

    if [[ ! -x "${venv_dir}/bin/python" ]]; then
      echo "[vllm-env] creating venv at ${venv_dir}" >&2
      "${base_py}" -m venv "${venv_dir}"
    fi

    py="${venv_dir}/bin/python"
    if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${expected_stamp}" ]] \
      && "${py}" -c "import vllm" 2>/dev/null \
      && "${py}" -c "import transformers; assert int(transformers.__version__.split('.')[0]) >= 5" 2>/dev/null; then
      echo "[vllm-env] ok (cached): ${venv_dir} (${spec}; ${tf_spec})" >&2
    else
      echo "[vllm-env] installing ${spec} + ${tf_spec} into ${venv_dir} (pip cache: ${PIP_CACHE_DIR})" >&2
      "${py}" -m pip install --upgrade pip
      "${py}" -m pip install "${spec}"
      "${py}" -m pip install "${tf_spec}"
      printf '%s\n' "${expected_stamp}" > "${stamp}"
      "${py}" -c "import vllm, transformers; print('[vllm-env] vllm', vllm.__version__, 'transformers', transformers.__version__)" >&2
    fi
  ) 9>"${lock_file}"

  py="${venv_dir}/bin/python"
  export VLLM_VENV_DIR="${venv_dir}"
  export PYTHON="${py}"
  export PATH="${venv_dir}/bin:${PATH}"
}
