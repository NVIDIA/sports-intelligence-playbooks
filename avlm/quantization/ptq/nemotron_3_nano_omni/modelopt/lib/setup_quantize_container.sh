#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal deps on top of NeMo /opt/venv (torch, mamba, etc. from the image). Source from run_quantize.sh.
set -euo pipefail

_SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../../../utils/_decord_pip.sh
source "${_SCRIPT_DIR}/../../../../../utils/_decord_pip.sh"

: "${MODEL_OPTIMIZER_SRC:?Set MODEL_OPTIMIZER_SRC}"
: "${CACHE_DIR:?Set CACHE_DIR}"

_QUANT_CONTAINER_PY="${QUANT_CONTAINER_PYTHON:-/opt/venv/bin/python}"
_QUANT_EXTRA="${QUANT_CONTAINER_PKGS_DIR:-${CACHE_DIR}/quant_container_pkgs}"
_MARKER="${_QUANT_EXTRA}/.hf_quant_deps_ok"
# Hub Nemotron-3-Nano-Omni modeling uses transformers.initialization (Transformers 5.x).
_QUANT_TRANSFORMERS_SPEC="${QUANT_TRANSFORMERS_SPEC:-transformers>=5.5.0,<6}"

_quant_transformers_ok() {
  PYTHONPATH="${PYTHONPATH}" "${_QUANT_CONTAINER_PY}" -c "from transformers import initialization" 2>/dev/null
}

export QUANT_CONTAINER_PYTHON="${_QUANT_CONTAINER_PY}"
export PYTHONPATH="${_QUANT_EXTRA}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${_QUANT_CONTAINER_PY}" ]]; then
  echo "error: container python not found: ${_QUANT_CONTAINER_PY}" >&2
  return 1 2>/dev/null || exit 1
fi

if ! "${_QUANT_CONTAINER_PY}" -c "import torch" 2>/dev/null; then
  echo "error: ${_QUANT_CONTAINER_PY} cannot import torch — set QUANT_USE_CONTAINER_PKGS=0" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "${_QUANT_EXTRA}"

_quant_datasets_ok() {
  PYTHONPATH="${PYTHONPATH}" "${_QUANT_CONTAINER_PY}" -c "from datasets import load_dataset" 2>/dev/null
}

_quant_overlay_ok() {
  _quant_transformers_ok \
    && PYTHONPATH="${PYTHONPATH}" "${_QUANT_CONTAINER_PY}" -c "import modelopt.torch.quantization" 2>/dev/null \
    && PYTHONPATH="${PYTHONPATH}" "${_QUANT_CONTAINER_PY}" -c "import decord" 2>/dev/null \
    && _quant_datasets_ok
}

if [[ -f "${_MARKER}" ]] && _quant_overlay_ok; then
  echo "info: reusing container quant deps (${_MARKER})" >&2
  return 0 2>/dev/null || exit 0
fi
if [[ -f "${_MARKER}" ]]; then
  echo "info: refreshing container quant deps (transformers/modelopt overlay incomplete)" >&2
  rm -f "${_MARKER}"
fi

echo "info: HF PTQ overlays → ${_QUANT_EXTRA} (python=${_QUANT_CONTAINER_PY})" >&2

"${_QUANT_CONTAINER_PY}" -m pip install -q --no-deps -e "${MODEL_OPTIMIZER_SRC}"

_install_if_missing() {
  local mod="$1" spec="$2"
  if ! PYTHONPATH="${PYTHONPATH}" "${_QUANT_CONTAINER_PY}" -c "import ${mod}" 2>/dev/null; then
    echo "info: pip install ${spec} (--no-deps → overlay)" >&2
    "${_QUANT_CONTAINER_PY}" -m pip install -q --no-deps --target "${_QUANT_EXTRA}" "${spec}"
  fi
}

# Hub nemotron_vlm_dataset_v2 calib needs `datasets` (not sports JSONL calib).
# Install with deps into overlay — a bare `datasets` + old /opt/venv huggingface_hub breaks on httpx import.
if ! _quant_datasets_ok; then
  echo "info: installing httpx + huggingface_hub + datasets for Hub VLM calib" >&2
  "${_QUANT_CONTAINER_PY}" -m pip install -q --upgrade --target "${_QUANT_EXTRA}" httpx "huggingface_hub>=0.26.0"
  "${_QUANT_CONTAINER_PY}" -m pip install -q --target "${_QUANT_EXTRA}" "datasets>=2.19.0"
fi
_install_if_missing ftfy ftfy
_install_if_missing regex regex
# Sports JSONL calib (avlm.inference video_io) decodes clips with decord; not in NeMo /opt/venv.
_install_if_missing "$(avlm_decord_import_module)" "$(avlm_decord_pip_pkg_unpinned)"

if ! _quant_transformers_ok; then
  echo "info: installing ${_QUANT_TRANSFORMERS_SPEC} into overlay (Hub needs transformers.initialization)" >&2
  "${_QUANT_CONTAINER_PY}" -m pip install -q --upgrade --target "${_QUANT_EXTRA}" "${_QUANT_TRANSFORMERS_SPEC}"
fi
if ! _quant_transformers_ok; then
  echo "error: transformers still missing initialization after install; set QUANT_TRANSFORMERS_SPEC" >&2
  return 1 2>/dev/null || exit 1
fi

for mod in mamba_ssm causal_conv1d; do
  if ! "${_QUANT_CONTAINER_PY}" -c "import ${mod}" 2>/dev/null; then
    echo "error: NeMo image missing ${mod}; use QUANT_USE_CONTAINER_PKGS=0" >&2
    return 1 2>/dev/null || exit 1
  fi
done

if ! PYTHONPATH="${PYTHONPATH}" "${_QUANT_CONTAINER_PY}" -c "import modelopt.torch.quantization" 2>/dev/null; then
  echo "error: modelopt not importable after install from ${MODEL_OPTIMIZER_SRC}" >&2
  return 1 2>/dev/null || exit 1
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"${_MARKER}"
echo "info: container quant deps ready" >&2
