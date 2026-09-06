#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run AutoModel SFT for a submitted Slurm batch job.

#SBATCH -t 4:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --overcommit

set -euo pipefail

_SFT_DIR_REL="avlm/training/automodel/sft/slurm"
_SFT_DIR="${REPO_ROOT}/${_SFT_DIR_REL}"
_TRAIN_LIB="${_SFT_DIR}/_train_lib.sh"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "============================================"
echo "SLURM Job Info"
echo "============================================"
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES       = ${SLURM_NNODES}"
echo "MASTER_ADDR        = ${MASTER_ADDR}"
echo "MASTER_PORT        = ${MASTER_PORT}"
echo "REPO_ROOT          = ${REPO_ROOT:-<unset>}"
echo "LOGS_DIR           = ${LOGS_DIR:-<unset>}"
echo "============================================"

: "${CONTAINER_IMAGE:?CONTAINER_IMAGE must be set (sbatch_starter.sh)}"
: "${REPO_ROOT:?REPO_ROOT must be set (sbatch_starter.sh)}"
: "${LOGS_DIR:?LOGS_DIR must be set (sbatch_starter.sh)}"
[[ -f "${_TRAIN_LIB}" ]] || {
  echo "error: train lib not found: ${_TRAIN_LIB}" >&2
  exit 1
}

datetime="$(date +%Y%m%d-%H%M%S)"
log_file="${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${datetime}.log"

# One task per node; each runs torchrun with the same c10d rendezvous (PyTorch multi-node Slurm pattern).
srun --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  -l \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS:-${CONTAINER_MOUNT_HOST:-/lustre},${CONTAINER_MOUNT_HOST:-/lustre},${REPO_ROOT},${REPO_ROOT}}" \
  --container-workdir="${REPO_ROOT}" \
  bash -c "
    set -euo pipefail
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:\${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export PYTHONUNBUFFERED=1
    export MALLOC_ARENA_MAX=\${MALLOC_ARENA_MAX:-2}
    export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-1}
    export OPENBLAS_NUM_THREADS=\${OPENBLAS_NUM_THREADS:-1}
    export NUMEXPR_NUM_THREADS=\${NUMEXPR_NUM_THREADS:-1}
    export MASTER_ADDR='${MASTER_ADDR}'
    export MASTER_PORT='${MASTER_PORT}'
    export SLURM_NNODES=${SLURM_NNODES}
    export SLURM_JOB_ID=${SLURM_JOB_ID}
    # shellcheck source=_train_lib.sh
    source '${_TRAIN_LIB}'
    sft_init_launch '${_SFT_DIR}' batch
    sft_train
  " &> "${log_file}"
