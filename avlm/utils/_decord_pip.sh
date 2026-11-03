# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Platform-aware pip specs for video decode (always `import decord` at runtime).
# Original `decord` publishes no Linux aarch64 wheels; `decord2` does and uses the same module name.
#
# shellcheck shell=bash

avlm_decord_pip_spec() {
  if [[ -n "${AVLM_DECORD_PIP_SPEC:-}" ]]; then
    printf '%s\n' "${AVLM_DECORD_PIP_SPEC}"
    return 0
  fi
  case "$(uname -m)" in
    aarch64 | arm64) printf '%s\n' "decord2" ;;
    *) printf '%s\n' "decord==0.6.0" ;;
  esac
}

# Unpinned package name for --target installs (training / inference vlm-deps).
avlm_decord_pip_pkg_unpinned() {
  if [[ -n "${AVLM_DECORD_PIP_PKG:-}" ]]; then
    printf '%s\n' "${AVLM_DECORD_PIP_PKG}"
    return 0
  fi
  case "$(uname -m)" in
    aarch64 | arm64) printf '%s\n' "decord2" ;;
    *) printf '%s\n' "decord" ;;
  esac
}

avlm_decord_import_module() {
  printf '%s\n' "decord"
}

# Map a pip token (e.g. decord2, decord==0.6.0) to the Python import name to verify.
avlm_pip_spec_token_import_name() {
  local token="$1"
  local name="${token%%[=<>!~]*}"
  case "${name}" in
    decord2) printf 'decord\n' ;;
    PyYAML) printf 'yaml\n' ;;
    imageio-ffmpeg) printf 'imageio_ffmpeg\n' ;;
    *)
      name="${name//-/_}"
      printf '%s\n' "${name}"
      ;;
  esac
}
