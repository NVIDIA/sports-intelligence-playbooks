#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Interactive Slurm session: start the vLLM container and drop into a shell.
# Does not install packages. Serve with vllm_server.sh inside the session.
#
#   bash avlm/deployment/vLLM/slurm/interactive/launch_interactive_session.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
_DEPLOY_ROOT="$(cd -- "${_SLURM_DIR}/.." && pwd)"
export REPO_ROOT="$(cd -- "${_DEPLOY_ROOT}/../../.." && pwd)"
export AVLM_UTILS_DIR="${REPO_ROOT}/avlm/utils"

# shellcheck source=/dev/null
source "${AVLM_UTILS_DIR}/_source_params.sh"
source_cluster_params "${_SLURM_DIR}"

export JOB_NAME="${JOB_NAME:-avlm_vllm_deploy_interactive}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
export WORKDIR="${WORKDIR:-${REPO_ROOT}}"

SLURM_ACCOUNT="$(primary_slurm_account)"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
PARTITION="$(interactive_slurm_partition)"

_MOUNT="${CONTAINER_MOUNT_HOST:-/lustre}"

echo "[INFO] mode              = deployment/vLLM (interactive, 1 node)"
echo "[INFO] SLURM_ACCOUNT     = ${SLURM_ACCOUNT}"
echo "[INFO] PARTITION         = ${PARTITION}"
echo "[INFO] num_nodes         = 1"
echo "[INFO] CONTAINER_IMAGE   = ${CONTAINER_IMAGE}"
echo "[INFO] JOB_NAME          = ${JOB_NAME}"
echo "[INFO] WORKDIR           = ${WORKDIR}"
echo "[INFO] GPUS_PER_NODE     = ${GPUS_PER_NODE}"
echo "[INFO] SLURM_TIME_LIMIT  = ${SLURM_TIME_LIMIT:-4:00:00}"

srun \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --job-name="${JOB_NAME}" \
  --gpus="${GPUS_PER_NODE}" \
  --ntasks=1 \
  --time="${SLURM_TIME_LIMIT:-4:00:00}" \
  --pty \
  --container-image="${CONTAINER_IMAGE}" \
  --container-name="${JOB_NAME}" \
  --container-workdir="${WORKDIR}" \
  --container-mounts="${WORKDIR}:${WORKDIR},${_MOUNT}:${_MOUNT}" \
  bash -c "
    unset CACHE_DIR HF_HOME XDG_CACHE_HOME TRITON_CACHE_DIR
    unset INSTALL_VLLM_PACKAGES VLLM_PIP_SPEC VLLM_EXTRA_PIP_SPEC
    unset VLLM_VENV_DIR VLLM_EXTRA_VENV_DIR VLLM_INSTALL_EXTRA
    export REPO_ROOT='${REPO_ROOT}'
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:\${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export TOKENIZERS_PARALLELISM=\${TOKENIZERS_PARALLELISM:-false}
    export PYTHONPATH='${REPO_ROOT}':\${PYTHONPATH:-}
    echo '[INFO] Serve: bash avlm/deployment/vLLM/vllm_server.sh start  (auto-installs decord + client deps when needed)'
    exec bash
  "
