#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${AVLM_RUN_COMMIT:?AVLM_RUN_COMMIT is required}"
LAUNCHER="${1:?launcher path is required}"
shift

RUN_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify "${AVLM_RUN_COMMIT}^{commit}")"
WORKTREE_ROOT="${REPO_ROOT}/avlm-worktrees"
WORKTREE="${WORKTREE_ROOT}/${RUN_COMMIT}"

mkdir -p "${WORKTREE_ROOT}"

exec 9>"${WORKTREE_ROOT}/.create.lock"
flock -x 9
if [[ ! -d "${WORKTREE}" ]]; then
  git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE}" "${RUN_COMMIT}"
fi
flock -u 9
exec 9>&-

WORKTREE_COMMIT="$(git -C "${WORKTREE}" rev-parse --verify HEAD 2>/dev/null || true)"
if [[ "${WORKTREE_COMMIT}" != "${RUN_COMMIT}" ]]; then
  echo "error: worktree ${WORKTREE} is not at requested commit ${RUN_COMMIT}" >&2
  exit 1
fi

if [[ -n "$(git -C "${WORKTREE}" status --porcelain --untracked-files=all)" ]]; then
  echo "error: worktree ${WORKTREE} has source or non-ignored changes" >&2
  exit 1
fi

INFERENCE_TARGET=0
case "${LAUNCHER}" in
  avlm/inference/*|avlm/evals/*)
    INFERENCE_TARGET=1
    # Capture command-prefix inference env before the worktree exec creates a
    # new parent boundary.
    source "${SCRIPT_DIR}/inference/common/utils/_inference_lib.sh"
    mapfile -t WORKTREE_INFERENCE_ENV_KEYS < <(inference_all_env_override_keys)
    inference_env_commit_overrides "${SCRIPT_DIR}" "${WORKTREE_INFERENCE_ENV_KEYS[@]}"
    ;;
esac

if ! git -C "${WORKTREE}" ls-files --error-unmatch -- "${LAUNCHER}" >/dev/null 2>&1 \
  || [[ ! -f "${WORKTREE}/${LAUNCHER}" ]]; then
  echo "error: launcher is not a tracked file at ${RUN_COMMIT}: ${LAUNCHER}" >&2
  exit 1
fi

export AVLM_BASE_REPO_ROOT="${WORKTREE}"

if [[ -n "${CLUSTER_PARAMS:-}" && "${CLUSTER_PARAMS}" != /* ]]; then
  export CLUSTER_PARAMS="${WORKTREE}/${CLUSTER_PARAMS}"
fi

LAUNCH_ARGS=()
for arg in "$@"; do
  if [[ "${INFERENCE_TARGET}" == "1" && "${arg}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    echo "error: inference overrides are environment-only; place '${arg}' before 'bash avlm/launch_from_worktree.sh'" >&2
    exit 2
  fi
  case "${arg}" in
    CLUSTER_PARAMS=|CLUSTER_PARAMS=/*) ;;
    CLUSTER_PARAMS=*) arg="CLUSTER_PARAMS=${WORKTREE}/${arg#CLUSTER_PARAMS=}" ;;
  esac
  LAUNCH_ARGS+=("${arg}")
done

if [[ "${AVLM_USE_WORKTREE_MB_BRIDGE_OVERLAY:-0}" =~ ^(1|true)$ ]]; then
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)_$$"
  WORKTREE_ROOT_PHYSICAL="$(cd -- "${WORKTREE_ROOT}" && pwd -P)"
  export MB_BRIDGE_OVERLAY="${MB_BRIDGE_OVERLAY:-${WORKTREE_ROOT_PHYSICAL}/.mb-overlays/${RUN_COMMIT}/${RUN_TIMESTAMP}}"
fi

echo "AVLM source commit: ${RUN_COMMIT}"
echo "AVLM worktree:      ${WORKTREE}"

cd "${WORKTREE}"

if [[ "${INFERENCE_TARGET}" == "1" ]]; then
  inference_env_exec_sanitized "${WORKTREE}/${LAUNCHER}" "${LAUNCH_ARGS[@]}"
  exit $?
fi

exec bash "${WORKTREE}/${LAUNCHER}" "${LAUNCH_ARGS[@]}"
