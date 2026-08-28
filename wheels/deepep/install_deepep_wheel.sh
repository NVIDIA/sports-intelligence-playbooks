#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install a prebuilt deep_ep wheel over the nemo-automodel container default.
#
# Run INSIDE the container after it is loaded (lustre mount must include this repo).
# Docs: wheels/deepep/DEEPEP.MD
#
#   bash wheels/deepep/install_deepep_wheel.sh
#
# Skips automatically unless recipe/ env uses dispatcher: deepep (set CONFIG_YAML or
# CONFIG_YAML_REL + REPO_ROOT). Force: DEEPEP_FORCE_INSTALL=1
#
# Pick a specific wheel:
#   DEEPEP_WHEEL=/path/to/deep_ep-1.2.1+7febc6e.prehopper8089-cp312-cp312-linux_x86_64.whl \
#     bash wheels/deepep/install_deepep_wheel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
  :
elif [[ -x /opt/venv/bin/python3 ]]; then
  PYTHON=/opt/venv/bin/python3
else
  PYTHON="$(command -v python3)"
fi
echo "[deepep-install] python=${PYTHON}" >&2

if [[ "${SKIP_DEEPEP_INSTALL:-${SKIP_DEEPEP_INSTALL_ON_LAUNCH:-${SKIP_DEEPEP_INSTALL_ON_BATCH:-0}}}" == "1" ]]; then
  echo "[deepep-install] skip (SKIP_DEEPEP_INSTALL)" >&2
  exit 0
fi

if [[ -z "${CONFIG_YAML:-}" && -n "${CONFIG_YAML_REL:-}" && -n "${REPO_ROOT:-}" ]]; then
  CONFIG_YAML="${REPO_ROOT}/${CONFIG_YAML_REL}"
fi
export CONFIG_YAML

if [[ "${DEEPEP_FORCE_INSTALL:-0}" != "1" && -n "${CONFIG_YAML:-}" ]]; then
  if ! "${PYTHON}" "${SCRIPT_DIR}/check_deepep_wheel.py" --needs-deepep; then
    echo "[deepep-install] skip (dispatcher is not deepep)" >&2
    exit 0
  fi
  echo "[deepep-install] dispatcher=deepep" >&2
fi

if [[ "${DEEPEP_FORCE_INSTALL:-0}" != "1" ]]; then
  if "${PYTHON}" "${SCRIPT_DIR}/check_deepep_wheel.py" 2>/dev/null; then
    echo "[deepep-install] ok (wheel already matches GPU)" >&2
    exit 0
  fi
  echo "[deepep-install] reinstalling (missing or wrong GPU kernels)" >&2
fi

PRE_HOPPER_PROFILE="pre-hopper-sm80-sm89"
POST_HOPPER_PROFILE="post-hopper-sm90-sm100-sm120"
PRE_HOPPER_WHEEL_LABEL="prehopper8089"
POST_HOPPER_WHEEL_LABEL="posthopper90120"

normalize_wheel_profile() {
  case "${1}" in
    all) printf '%s' "all" ;;
    ampere | "${PRE_HOPPER_PROFILE}") printf '%s' "${PRE_HOPPER_PROFILE}" ;;
    hopper | post-hopper-sm90-sm120 | "${POST_HOPPER_PROFILE}") printf '%s' "${POST_HOPPER_PROFILE}" ;;
    *) printf '%s' "${1}" ;;
  esac
}

wheel_local_version_label_for_profile() {
  case "${1}" in
    "${PRE_HOPPER_PROFILE}" | ampere) printf '%s' "${PRE_HOPPER_WHEEL_LABEL}" ;;
    "${POST_HOPPER_PROFILE}" | hopper | post-hopper-sm90-sm120) printf '%s' "${POST_HOPPER_WHEEL_LABEL}" ;;
    *) return 1 ;;
  esac
}

infer_wheel_profile_from_gpu() {
  "${PYTHON}" -c "
import torch
if not torch.cuda.is_available():
    print('${PRE_HOPPER_PROFILE}')
else:
    major, _ = torch.cuda.get_device_capability(0)
    print('${POST_HOPPER_PROFILE}' if major >= 9 else '${PRE_HOPPER_PROFILE}')
" 2>/dev/null || echo "${PRE_HOPPER_PROFILE}"
}

pip_can_install_wheel() {
  "${PYTHON}" - "${1}" <<'PY'
import sys
from pathlib import Path

try:
    from pip._internal.utils.wheel import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.utils import parse_wheel_filename

try:
    parse_wheel_filename(Path(sys.argv[1]).name)
except ValueError as exc:
    print(f"invalid: {exc}", file=sys.stderr)
    sys.exit(1)
print("ok", file=sys.stderr)
PY
}

# Normalize any legacy lustre filename to +{commit}.{label}-cp312-...
normalize_wheel_filename_for_profile() {
  local wheel="${1}"
  local profile="${2}"
  local label base target
  [[ -f "${wheel}" ]] || return 1
  label="$(wheel_local_version_label_for_profile "${profile}")" || return 1
  if pip_can_install_wheel "${wheel}" 2>/dev/null; then
    printf '%s' "${wheel}"
    return 0
  fi
  base="$(basename "${wheel}")"
  target=""
  case "${base}" in
    *".${label}-"*) target="${wheel}" ;;
    *"+${label}-"*) target="${wheel}" ;;
    deep_ep-*+7febc6e-prehopper_sm80_sm89-cp*)
      target="${wheel/+7febc6e-prehopper_sm80_sm89-/+7febc6e.${label}-}"
      ;;
    deep_ep-*+7febc6e-posthopper_sm90_sm100_sm120-cp*)
      target="${wheel/+7febc6e-posthopper_sm90_sm100_sm120-/+7febc6e.${label}-}"
      ;;
    deep_ep-*+7febc6e-cp*)
      target="${wheel/+7febc6e-/+7febc6e.${label}-}"
      ;;
    deep_ep-*-cp*)
      if [[ "${base}" =~ ^(deep_ep-.+\+[^./-]+)(-cp.*)$ ]]; then
        target="${wheel%/*}/${BASH_REMATCH[1]}.${label}${BASH_REMATCH[2]}"
      fi
      ;;
  esac
  [[ -n "${target}" && "${target}" != "${wheel}" ]] || return 1
  echo "[deepep-install] renaming to pip-compatible: $(basename "${target}")" >&2
  mv -f "${wheel}" "${target}"
  pip_can_install_wheel "${target}" || return 1
  printf '%s' "${target}"
}

find_wheel_for_profile() {
  local profile="${1}"
  local label wheel=""
  label="$(wheel_local_version_label_for_profile "${profile}")" || return 1
  for wheel in $(ls -t "${SCRIPT_DIR}"/deep_ep-*"${label}"*.whl 2>/dev/null); do
    if pip_can_install_wheel "${wheel}" 2>/dev/null; then
      printf '%s' "${wheel}"
      return
    fi
    wheel="$(normalize_wheel_filename_for_profile "${wheel}" "${profile}")" && {
      printf '%s' "${wheel}"
      return
    }
  done
  wheel=""
  for wheel in $(ls -t "${SCRIPT_DIR}"/deep_ep-*.whl 2>/dev/null); do
    case "${profile}" in
      "${PRE_HOPPER_PROFILE}")
        [[ "${wheel}" == *ampere* || "${wheel}" == *pre-hopper* || "${wheel}" == *prehopper* ]] || continue
        ;;
      "${POST_HOPPER_PROFILE}")
        [[ "${wheel}" == *hopper* || "${wheel}" == *posthopper* ]] || continue
        ;;
    esac
    wheel="$(normalize_wheel_filename_for_profile "${wheel}" "${profile}")" && {
      printf '%s' "${wheel}"
      return
    }
  done
  return 1
}

pick_deepep_wheel() {
  local profile="${DEEPEP_WHEEL_PROFILE:-}"
  local wheel=""
  if [[ -z "${profile}" ]]; then
    profile="$(infer_wheel_profile_from_gpu)"
    echo "[deepep-install] auto-selected profile=${profile} from GPU compute capability" >&2
  else
    profile="$(normalize_wheel_profile "${profile}")"
  fi
  if [[ "${profile}" == "all" ]]; then
    profile="$(infer_wheel_profile_from_gpu)"
    echo "[deepep-install] DEEPEP_WHEEL_PROFILE=all is build-only; installing profile=${profile} for this GPU" >&2
  fi
  if [[ -n "${profile}" ]]; then
    wheel="$(find_wheel_for_profile "${profile}")" || wheel=""
  fi
  if [[ -z "${wheel}" ]]; then
    wheel="$(ls -t "${SCRIPT_DIR}"/deep_ep-*.whl 2>/dev/null | head -1 || true)"
  fi
  printf '%s' "${wheel}"
}

if [[ -n "${DEEPEP_WHEEL:-}" ]]; then
  WHEEL="${DEEPEP_WHEEL}"
else
  WHEEL="$(pick_deepep_wheel)"
fi

[[ -n "${WHEEL}" && -f "${WHEEL}" ]] || {
  echo "error: no deep_ep wheel found in ${SCRIPT_DIR}" >&2
  echo "Build one first: bash wheels/deepep/building/build_deepep_wheel.sh" >&2
  exit 1
}

if ! pip_can_install_wheel "${WHEEL}" 2>/dev/null; then
  _profile="$(normalize_wheel_profile "${DEEPEP_WHEEL_PROFILE:-$(infer_wheel_profile_from_gpu)}")"
  WHEEL="$(normalize_wheel_filename_for_profile "${WHEEL}" "${_profile}")" || {
    echo "error: wheel filename is not pip-compatible: ${WHEEL}" >&2
    echo "Expected form: deep_ep-1.2.1+{commit}.{label}-cp312-cp312-linux_x86_64.whl" >&2
    echo "  pre-Hopper label: ${PRE_HOPPER_WHEEL_LABEL}" >&2
    echo "  post-Hopper label: ${POST_HOPPER_WHEEL_LABEL}" >&2
    exit 1
  }
fi

echo "[deepep-install] wheel=${WHEEL}"

if [[ -f "${SCRIPT_DIR}/building/deepep_wheel_build_manifest.txt" ]]; then
  echo "[deepep-install] manifest:"
  sed 's/^/  /' "${SCRIPT_DIR}/building/deepep_wheel_build_manifest.txt"
elif [[ -f "${SCRIPT_DIR}/deepep_wheel_build_manifest.txt" ]]; then
  echo "[deepep-install] manifest:"
  sed 's/^/  /' "${SCRIPT_DIR}/deepep_wheel_build_manifest.txt"
fi

# Venv can retain a broken deep_ep entry (metadata without files). Wipe before reinstall.
"${PYTHON}" -m pip uninstall -y deep_ep deep-ep 2>/dev/null || true
_SITE="$("${PYTHON}" -c "import site; print(site.getsitepackages()[0])")"
rm -rf \
  "${_SITE}/deep_ep" \
  "${_SITE}"/deep_ep-*.dist-info \
  "${_SITE}"/deep_ep*.so \
  "${_SITE}"/hybrid_ep*.so \
  2>/dev/null || true

"${PYTHON}" -m pip install --force-reinstall --no-deps "${WHEEL}"

# Do not pip-install nvidia-ml-py by default: it can shadow container NVML and break
# nvidia-smi / NCCL (Driver/library version mismatch) on some nodes. The image already
# ships a compatible stack. Set DEEPEP_INSTALL_NVIDIA_ML_PY=1 only if deep_ep import fails.
if [[ "${DEEPEP_INSTALL_NVIDIA_ML_PY:-0}" == "1" ]]; then
  "${PYTHON}" -m pip install "nvidia-ml-py>=12.0.0"
fi

"${PYTHON}" "${SCRIPT_DIR}/check_deepep_wheel.py" || {
  echo "[deepep-install] error: wheel install did not pass check_deepep_wheel.py." >&2
  exit 1
}

if "${PYTHON}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  "${PYTHON}" -c "import torch; print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
fi

echo "[deepep-install] done. Set model.backend.dispatcher: deepep in your training YAML."
