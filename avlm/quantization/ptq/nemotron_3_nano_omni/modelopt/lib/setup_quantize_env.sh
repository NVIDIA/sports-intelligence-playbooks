#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fallback venv when QUANT_USE_CONTAINER_PKGS=0 (NeMo: prefer container mode in run_quantize.sh).
set -euo pipefail

_QUANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODEL_OPTIMIZER_SRC:?Set MODEL_OPTIMIZER_SRC to the Model-Optimizer checkout}"

: "${VENV_DIR:?Set VENV_DIR or run via run_quantize.sh (defaults under CACHE_DIR)}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -q python-is-python3 git git-lfs || true
fi

ARCH=$(uname -m)
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found; use NGC PyTorch 26.02+ or install uv" >&2
  exit 1
fi

_base_py=""
# shellcheck source=../../../../../utils/_prep_bridge_env.sh
source "${_QUANT_DIR}/../../../../../utils/_prep_bridge_env.sh"
_base_py="$(_mb_find_torch_python 2>/dev/null || true)"

if [[ -n "${_base_py}" ]]; then
  uv venv "${VENV_DIR}" --python "${_base_py}" --system-site-packages
else
  uv venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if [[ "${ARCH}" == "aarch64" ]]; then
  MAMBA_WHL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu13torch26.02cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"
  CAUSAL_CONV1D_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu13torch26.02cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"
elif [[ "${ARCH}" == "x86_64" ]]; then
  MAMBA_WHL="https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu13torch26.02cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  CAUSAL_CONV1D_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu13torch26.02cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
else
  echo "error: unsupported architecture: ${ARCH}" >&2
  exit 1
fi

if ! python -c "import mamba_ssm" 2>/dev/null; then
  uv pip install "${MAMBA_WHL}" "${CAUSAL_CONV1D_WHL}"
fi

pip uninstall -y nvidia-modelopt 2>/dev/null || true
uv pip install --no-deps -e "${MODEL_OPTIMIZER_SRC}"
uv pip install --no-deps datasets pillow timm librosa ftfy regex
uv pip install --no-deps open_clip_torch

_QUANT_TRANSFORMERS_SPEC="${QUANT_TRANSFORMERS_SPEC:-transformers>=5.5.0,<6}"
if ! python -c "from transformers import initialization" 2>/dev/null; then
  echo "info: installing ${_QUANT_TRANSFORMERS_SPEC} (Hub Nemotron-3-Nano-Omni modeling)" >&2
  uv pip install "${_QUANT_TRANSFORMERS_SPEC}"
fi
if ! python -c "from transformers import initialization" 2>/dev/null; then
  echo "error: venv transformers missing initialization; rm -rf ${VENV_DIR} and retry" >&2
  exit 1
fi

echo "quantize venv ready: ${VENV_DIR}" >&2
