#!/usr/bin/env bash
# Submit a non-interactive Slurm job that builds the deep_ep wheel inside
# nemo-automodel:26.06 and writes it to wheels/deepep/.
#
# Usage (login node):
#   bash wheels/deepep/building/run_build_in_container.sh
#
# Docs: wheels/deepep/DEEPEP.MD
#
# Override arch list (A100-only, faster):
#   DEEPEP_CUDA_ARCH_LIST="8.0" bash wheels/deepep/building/run_build_in_container.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../../../avlm/utils/_source_params.sh
source "${SCRIPT_DIR}/../../../avlm/utils/_source_params.sh"
resolve_avlm_repo_roots_from_avlm_root "$(cd "${SCRIPT_DIR}/../../../avlm" && pwd)"
PROJECT_ROOT="${REPO_ROOT}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-ai4m_sportscaster}"
PARTITION="${PARTITION:-interactive_singlenode}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs11/portfolios/ai4m/projects/ai4m_sportscaster/docker_images/nemo-automodel_26_06.sqsh}"
WORKDIR="${WORKDIR:-${PROJECT_ROOT}}"
JOB_NAME="${JOB_NAME:-deepep_wheel_build}"
BUILD_TIME="${BUILD_TIME:-4:00:00}"

echo "[INFO] SLURM_ACCOUNT = ${SLURM_ACCOUNT}"
echo "[INFO] PARTITION     = ${PARTITION}"
echo "[INFO] CONTAINER_IMAGE = ${CONTAINER_IMAGE}"
echo "[INFO] WORKDIR       = ${WORKDIR}"
echo "[INFO] BUILD_TIME    = ${BUILD_TIME}"
echo "[INFO] wheel output  = ${DEEPEP_DIR}"

srun \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --job-name="${JOB_NAME}" \
  --gpus=1 \
  --ntasks=1 \
  --time="${BUILD_TIME}" \
  --container-image="${CONTAINER_IMAGE}" \
  --container-name=deepep_wheel_build \
  --container-workdir="${WORKDIR}" \
  --container-mounts="${WORKDIR}:${WORKDIR},/lustre:/lustre" \
  bash -lc '
    set -euo pipefail
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export DEEPEP_WHEEL_PROFILE="${DEEPEP_WHEEL_PROFILE:-all}"
    export DISABLE_AGGRESSIVE_PTX_INSTRS="${DISABLE_AGGRESSIVE_PTX_INSTRS:-1}"
    export MAX_JOBS="${MAX_JOBS:-8}"
    bash "'"${SCRIPT_DIR}"'/build_deepep_wheel.sh"
  '

echo "[INFO] build job finished; check ${SCRIPT_DIR}/deepep_wheel_build_manifest.txt"
