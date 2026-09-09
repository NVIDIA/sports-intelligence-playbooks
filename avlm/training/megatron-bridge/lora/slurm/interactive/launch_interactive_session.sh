#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start an interactive Slurm container session for Megatron-Bridge LoRA (PEFT).
#
#   bash avlm/training/megatron-bridge/lora/slurm/interactive/launch_interactive_session.sh
#   bash avlm/training/megatron-bridge/lora/slurm/interactive/train_interactive.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
_MB_ROOT="$(cd "${_SLURM_DIR}/../.." && pwd)"

# shellcheck source=../../../../utils/_source_params.sh
source "${_SLURM_DIR}/../../../../utils/_source_params.sh"
# shellcheck source=../../_prep_bridge_env.sh
source "${_MB_ROOT}/../../utils/_prep_bridge_env.sh"
training_cli_commit_env_overrides "${_SLURM_DIR}" "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${_SLURM_DIR}"
resolve_avlm_repo_roots_from_mode_dir "${_SLURM_DIR}"

export JOB_NAME="${JOB_NAME:-mb_lora_interactive}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
export WORKDIR="${WORKDIR:-$REPO_ROOT}"

SLURM_ACCOUNT="$(primary_slurm_account)"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
PARTITION="$(interactive_slurm_partition)"

setup_cache_dir_env
# Resolve the source on the login node: container checkout (default) or a Lustre clone
# (MEGATRON_BRIDGE_GIT_BOOTSTRAP=1). Clone path runs uv sync inside the container.
mb_resolve_bridge_root_login
: "${MEGATRON_BRIDGE_ROOT:?Megatron-Bridge source unresolved; set MEGATRON_BRIDGE_ROOT or MEGATRON_BRIDGE_GIT_BOOTSTRAP=1}"

_MOUNT="${CONTAINER_MOUNT_HOST:-/lustre}"

mb_log_section "Slurm interactive session (LoRA / PEFT)"
mb_log_info "account:   ${SLURM_ACCOUNT}"
mb_log_info "partition: ${PARTITION}"
mb_log_info "container: ${CONTAINER_IMAGE}"
mb_log_info "bridge:    ${MEGATRON_BRIDGE_ROOT}"
mb_log_info "cache:     ${CACHE_DIR}"
mb_log_info "time limit: ${SLURM_TIME_LIMIT:-4:00:00}"
mb_log_info "after allocate → container prep then shell"

_PREP_BRIDGE_ENV="${_MB_ROOT}/../../utils/_prep_bridge_env.sh"

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
  --container-name="${JOB_NAME}" \
  --container-workdir="$WORKDIR" \
  --container-mounts="$WORKDIR:$WORKDIR,${_MOUNT}:${_MOUNT}" \
  bash -c "
    set -euo pipefail
    export INSIDE_INTERACTIVE_SESSION=1
    export CACHE_DIR=\"${CACHE_DIR}\"
    export MEGATRON_BRIDGE_ROOT=\"${MEGATRON_BRIDGE_ROOT}\"
    export EXIT_MINS_BEFORE_LIMIT=\"${EXIT_MINS_BEFORE_LIMIT:-10}\"
    export SKIP_MEGATRON_BRIDGE_GIT_BOOTSTRAP=1
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:\${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export TOKENIZERS_PARALLELISM=\${TOKENIZERS_PARALLELISM:-false}
    echo '[mb-bridge] waiting for GPU allocation...' >&2
    source \"${_PREP_BRIDGE_ENV}\"
    mb_prepare_bridge_env \"${_SLURM_DIR}\"
    echo '[mb-bridge] starting interactive shell' >&2
    exec bash
  "
