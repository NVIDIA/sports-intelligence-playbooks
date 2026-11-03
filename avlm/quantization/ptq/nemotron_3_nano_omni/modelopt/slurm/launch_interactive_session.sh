#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Slurm + enroot interactive shell (GPU + container only). No PTQ setup here.
#
#   bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/slurm/launch_interactive_session.sh
# Then inside the container:
#   bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="${SCRIPT_DIR}"

_quant_resolve_avlm_root() {
  local d="$1"
  while [[ "${d}" != "/" ]]; do
    if [[ -f "${d}/utils/_source_params.sh" ]]; then
      printf '%s\n' "${d}"
      return 0
    fi
    d="$(dirname -- "${d}")"
  done
  echo "error: could not find avlm root (utils/_source_params.sh) from ${1}" >&2
  return 1
}

_AVLM_ROOT="$(_quant_resolve_avlm_root "${_SLURM_DIR}")" || exit 1

# shellcheck source=../../../../../utils/_source_params.sh
source "${_AVLM_ROOT}/utils/_source_params.sh"
training_cli_apply_arg_overrides "$@"
set -- "${_TRAINING_REMAINING_ARGS[@]}"
training_cli_commit_env_overrides "${_SLURM_DIR}" "${GENERIC_CLI_OVERRIDE_KEYS[@]}"
source_cluster_params "${_SLURM_DIR}"
resolve_avlm_repo_roots_from_mode_dir "${_SLURM_DIR}"

export JOB_NAME="${JOB_NAME:-omni_quant_interactive}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
export WORKDIR="${WORKDIR:-$REPO_ROOT}"

SLURM_ACCOUNT="$(primary_slurm_account)"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
PARTITION="$(interactive_slurm_partition)"
_MOUNT="${CONTAINER_MOUNT_HOST:-/lustre}"

echo "info: Slurm interactive container (account=${SLURM_ACCOUNT} partition=${PARTITION} gpus=${GPUS_PER_NODE})" >&2
echo "info: image=${CONTAINER_IMAGE}" >&2
echo "info: after allocate, run: bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh" >&2

srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$PARTITION" \
  --nodes=1 \
  --job-name="$JOB_NAME" \
  --gpus="$GPUS_PER_NODE" \
  --ntasks=1 \
  --time="${SLURM_TIME_LIMIT:-8:00:00}" \
  --pty \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="${JOB_NAME}" \
  --container-workdir="$WORKDIR" \
  --container-mounts="$WORKDIR:$WORKDIR,${_MOUNT}:${_MOUNT}" \
  bash -c "
    set -euo pipefail
    export INSIDE_INTERACTIVE_SESSION=1
    export REPO_ROOT=\"${REPO_ROOT}\"
    export QUANT_CONTAINER_PYTHON=/opt/venv/bin/python
    export MB_BRIDGE_PYTHON=/opt/venv/bin/python
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:\${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    exec bash
  "
