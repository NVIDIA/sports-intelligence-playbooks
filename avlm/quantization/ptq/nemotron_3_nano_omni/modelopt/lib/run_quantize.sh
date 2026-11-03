#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

_QUANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset RANK WORLD_SIZE LOCAL_RANK
unset PIP_CONSTRAINT

_REPO_ROOT="$(cd -- "${_QUANT_DIR}/../../../../../../" && pwd)"
export PYTHONPATH="${_REPO_ROOT}:${_QUANT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

_run_with_python() {
  local py="$1"
  shift
  exec "${py}" "${_QUANT_DIR}/quantize_omni.py" "$@"
}

_quant_python_has_omni_deps() {
  local py="$1"
  [[ -n "${py}" && -x "${py}" ]] || return 1
  "${py}" -c "import torch, mamba_ssm, causal_conv1d" 2>/dev/null
}

_quant_find_nemo_python() {
  local py _torch_only="" _candidates=()
  if [[ -n "${QUANT_CONTAINER_PYTHON:-}" ]]; then
    _candidates=("${QUANT_CONTAINER_PYTHON}")
  elif [[ -n "${MB_BRIDGE_PYTHON:-}" ]]; then
    _candidates=("${MB_BRIDGE_PYTHON}")
  else
    _candidates=(
      /opt/venv/bin/python3
      /opt/venv/bin/python
    )
  fi
  for py in "${_candidates[@]}"; do
    if _quant_python_has_omni_deps "${py}"; then
      printf '%s\n' "${py}"
      return 0
    fi
  done
  # shellcheck source=../../../../../utils/_prep_bridge_env.sh
  source "${_QUANT_DIR}/../../../../../utils/_prep_bridge_env.sh"
  py="$(_mb_find_torch_python 2>/dev/null || true)"
  if [[ -n "${py}" ]]; then
    if _quant_python_has_omni_deps "${py}"; then
      printf '%s\n' "${py}"
      return 0
    fi
    _torch_only="${py}"
  fi
  if [[ -n "${_torch_only}" ]]; then
    echo "error: ${_torch_only} has torch but not mamba_ssm/causal_conv1d (Omni PTQ needs the NeMo image)" >&2
    echo "hint: bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/slurm/launch_interactive_session.sh" >&2
    echo "hint: then export QUANT_CONTAINER_PYTHON=/opt/venv/bin/python and rerun quantize.sh" >&2
  else
    echo "error: no Python with torch+mamba_ssm (set QUANT_CONTAINER_PYTHON=/opt/venv/bin/python inside the NeMo container)" >&2
  fi
  return 1
}

if [[ "${QUANT_USE_CONTAINER_PKGS:-1}" == "1" ]]; then
  _container_py="$(_quant_find_nemo_python)" || exit 1
  export QUANT_CONTAINER_PYTHON="${_container_py}"
  # shellcheck source=setup_quantize_container.sh
  source "${_QUANT_DIR}/setup_quantize_container.sh"
  _run_with_python "${QUANT_CONTAINER_PYTHON}" "$@"
fi

# Fallback: persistent venv under CACHE_DIR (downloads mamba wheels + editable modelopt; no torch if --system-site-packages + base py).
if [[ -z "${VENV_DIR:-}" ]]; then
  if [[ -n "${CACHE_DIR:-}" ]]; then
    VENV_DIR="${CACHE_DIR%/}/venv_quant"
  else
    VENV_DIR="/tmp/venv_quant"
  fi
fi
export VENV_DIR
if [[ -d "${VENV_DIR}/bin" ]] && ! "${VENV_DIR}/bin/python" -c "from transformers import initialization" 2>/dev/null; then
  echo "info: removing stale quantize venv (transformers missing initialization module)" >&2
  rm -rf "${VENV_DIR}"
fi
if [[ ! -d "${VENV_DIR}/bin" ]]; then
  echo "info: creating quantize venv at ${VENV_DIR} (QUANT_USE_CONTAINER_PKGS=0)" >&2
  bash "${_QUANT_DIR}/setup_quantize_env.sh"
else
  echo "info: reusing quantize venv at ${VENV_DIR}" >&2
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
_run_with_python python "$@"
