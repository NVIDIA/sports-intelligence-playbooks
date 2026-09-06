#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start a plain interactive Slurm container for testing generic training.
#
# Example:
#   SLURM_ACCOUNT=ai4m_video \
#   CONTAINER_IMAGE=/path/to/container.sqsh \
#     bash avlm/utils/launch_vanilla_interactive_session.sh
set -euo pipefail

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE}"

PARTITION="${PARTITION:-interactive_singlenode}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:00:00}"
JOB_NAME="${JOB_NAME:-generic_training_interactive}"
WORKDIR="${WORKDIR:-$(pwd)}"
CONTAINER_MOUNT_HOST="${CONTAINER_MOUNT_HOST:-/lustre}"

srun \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus="${GPUS_PER_NODE}" \
  --time="${SLURM_TIME_LIMIT}" \
  --job-name="${JOB_NAME}" \
  --pty \
  --container-image="${CONTAINER_IMAGE}" \
  --container-workdir="${WORKDIR}" \
  --container-mounts="${WORKDIR}:${WORKDIR},${CONTAINER_MOUNT_HOST}:${CONTAINER_MOUNT_HOST}" \
  bash
