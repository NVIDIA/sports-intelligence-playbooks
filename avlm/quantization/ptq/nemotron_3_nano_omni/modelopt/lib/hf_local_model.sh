# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Hub checkpoints as plain local dirs under WORKSPACE/base (not HF_HOME cache).
# shellcheck shell=bash

[[ -n "${_QUANT_HF_LOCAL_MODEL_LOADED:-}" ]] && return 0
_QUANT_HF_LOCAL_MODEL_LOADED=1

quant_hub_repo_slug() {
  printf '%s\n' "${1##*/}" | tr '[:upper:]' '[:lower:]'
}

quant_has_model_weights() {
  local d="$1"
  [[ -f "${d}/model.safetensors.index.json" ]] || compgen -G "${d}/*.safetensors" >/dev/null
}

quant_hf_download() {
  local repo_id="$1" dest="$2"
  if [[ "${FORCE_DOWNLOAD:-0}" != "1" ]] && quant_has_model_weights "${dest}"; then
    echo "info: skipping download (weights already present) — ${dest}" >&2
    return 0
  fi
  mkdir -p "${dest}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo_id}" --local-dir "${dest}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo_id}" --local-dir "${dest}"
  else
    echo "error: need hf or huggingface-cli on PATH to download ${repo_id}" >&2
    return 1
  fi
}

# Default: ${workspace}/base/<hub-repo-name-lowercase>
quant_default_base_model_dir() {
  local workspace="$1" hub_id="$2" local_override="${3:-}"
  if [[ -n "${local_override}" ]]; then
    printf '%s\n' "${local_override}"
    return 0
  fi
  printf '%s\n' "${workspace}/base/$(quant_hub_repo_slug "${hub_id}")"
}
