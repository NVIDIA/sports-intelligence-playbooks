#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start an interactive Slurm container session for Megatron-Bridge inference.
#
#   bash avlm/inference/megatron-bridge/slurm/interactive/launch_interactive_session.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
_BACKEND_DIR="$(cd -- "${_SLURM_DIR}/.." && pwd)"
_INFERENCE_DIR="$(cd -- "${_BACKEND_DIR}/.." && pwd)"

source "${_INFERENCE_DIR}/common/utils/_inference_lib.sh"
inference_reject_config_args "$@"
inference_env_commit_overrides \
  "${_SLURM_DIR}" \
  "${INFERENCE_COMMON_ENV_OVERRIDE_KEYS[@]}" \
  "${INFERENCE_SBATCH_ENV_OVERRIDE_KEYS[@]}" \
  "${INFERENCE_SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${_SLURM_DIR}"
inference_env_forward_overrides

export JOB_NAME="${JOB_NAME:-avlm_infer_mbridge_interactive}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
export WORKDIR="${WORKDIR:-$REPO_ROOT}"

SLURM_ACCOUNT="$(primary_slurm_account)"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
PARTITION="$(interactive_slurm_partition)"

setup_cache_dir_env

_MOUNT="${CONTAINER_MOUNT_HOST:-/lustre}"

echo "[INFO] mode              = inference/megatron-bridge"
echo "[INFO] SLURM_ACCOUNT     = ${SLURM_ACCOUNT}"
echo "[INFO] PARTITION         = ${PARTITION}"
echo "[INFO] CONTAINER_IMAGE   = ${CONTAINER_IMAGE}"
echo "[INFO] JOB_NAME          = ${JOB_NAME}"
echo "[INFO] WORKDIR           = ${WORKDIR}"
echo "[INFO] GPUS_PER_NODE     = ${GPUS_PER_NODE}"
echo "[INFO] SLURM_TIME_LIMIT  = ${SLURM_TIME_LIMIT:-4:00:00}"
echo "[INFO] CACHE_DIR         = ${CACHE_DIR}"
echo "[INFO] HF_HOME           = ${HF_HOME}"
echo "[INFO] TMPDIR            = ${TMPDIR:-/tmp}"

srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$PARTITION" \
  --nodes=1 \
  --job-name="$JOB_NAME" \
  --gpus="$GPUS_PER_NODE" \
  --ntasks=1 \
  --time="${SLURM_TIME_LIMIT:-4:00:00}" \
  --pty \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$JOB_NAME" \
  --container-workdir="$WORKDIR" \
  --container-mounts="$WORKDIR:$WORKDIR,${_MOUNT}:${_MOUNT}" \
  bash -c "
    export INSIDE_AVLM_INFERENCE_SESSION=1
    export AVLM_INFERENCE_BACKEND=megatron_bridge
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:\${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export PYTHONPATH='${WORKDIR}':\${PYTHONPATH:-}
    export TOKENIZERS_PARALLELISM=\${TOKENIZERS_PARALLELISM:-false}
    source \"${AVLM_UTILS_DIR}/_prep_bridge_env.sh\"
    mb_prepare_bridge_env \"${_SLURM_DIR}\"
    exec bash
  "
