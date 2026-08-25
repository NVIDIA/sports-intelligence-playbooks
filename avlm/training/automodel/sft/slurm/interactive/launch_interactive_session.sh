#!/usr/bin/env bash
# Start an interactive Slurm container session for SFT training.
#
#   bash avlm/training/automodel/sft/slurm/interactive/launch_interactive_session.sh
#   bash avlm/training/automodel/sft/slurm/interactive/train_interactive.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SLURM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=../../../../utils/_source_params.sh
source "${_SLURM_DIR}/../../../../utils/_source_params.sh"
training_cli_commit_env_overrides "${_SLURM_DIR}" "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}"
source_cluster_params "${_SLURM_DIR}"
resolve_avlm_repo_roots_from_mode_dir "${_SLURM_DIR}"

export JOB_NAME="${JOB_NAME:-sft_interactive}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
training_require_config_path
export WORKDIR="${WORKDIR:-$REPO_ROOT}"
_RECIPE_YAML="${CONFIG_YAML:-${REPO_ROOT}/${CONFIG_YAML_REL}}"
[[ "${_RECIPE_YAML}" == /* ]] || _RECIPE_YAML="${REPO_ROOT}/${_RECIPE_YAML}"
_SOURCE_PARAMS="${REPO_ROOT}/avlm/utils/_source_params.sh"

SLURM_ACCOUNT="$(primary_slurm_account)"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in launch_local.yaml}"
PARTITION="$(interactive_slurm_partition)"

setup_cache_dir_env

_MOUNT="${CONTAINER_MOUNT_HOST:-/lustre}"

echo "[INFO] mode              = sft"
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
echo "[INFO] recipe_yaml       = ${_RECIPE_YAML}"
[[ -n "${DEEPEP_WHEEL_PROFILE:-}" ]] && echo "[INFO] DEEPEP_WHEEL_PROFILE = ${DEEPEP_WHEEL_PROFILE}"

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
    export INSIDE_INTERACTIVE_SESSION=1
    export REPO_ROOT='${REPO_ROOT}'
    export CONFIG_YAML='${_RECIPE_YAML}'
    export CACHE_DIR='${CACHE_DIR}'
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:\${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export TOKENIZERS_PARALLELISM=\${TOKENIZERS_PARALLELISM:-false}
    if [[ -f '${_SOURCE_PARAMS}' ]]; then
      source '${_SOURCE_PARAMS}'
      ${DEEPEP_WHEEL_PROFILE:+export DEEPEP_WHEEL_PROFILE='${DEEPEP_WHEEL_PROFILE}'; }
      ensure_deepep
      ensure_vlm_training_deps || echo '[WARN] VLM python deps install skipped or failed' >&2
    fi
    exec bash
  "
