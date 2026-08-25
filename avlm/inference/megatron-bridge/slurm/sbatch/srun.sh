#!/usr/bin/env bash
# Slurm batch script: containerized Megatron-Bridge inference.

#SBATCH -t 4:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --overcommit

set -euo pipefail

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

: "${REPO_ROOT:?REPO_ROOT must be set}"
: "${LOGS_DIR:?LOGS_DIR must be set}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE must be set}"
: "${CONTAINER_MOUNTS:?CONTAINER_MOUNTS must be set}"
: "${NPROC_PER_NODE:?NPROC_PER_NODE must be set}"
: "${INFERENCE_CONFIG:?INFERENCE_CONFIG must be set}"

LOG_FILE="${LOGS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S).log"

echo "============================================"
echo "SLURM Job Info"
echo "============================================"
echo "SLURM_JOB_ID       = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES       = ${SLURM_NNODES}"
echo "MASTER_ADDR        = ${MASTER_ADDR}"
echo "MASTER_PORT        = ${MASTER_PORT}"
echo "REPO_ROOT          = ${REPO_ROOT}"
echo "LOGS_DIR           = ${LOGS_DIR}"
echo "============================================"

srun --nodes="${SLURM_NNODES}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  -l \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${AVLM_BASE_REPO_ROOT:-${REPO_ROOT}}" \
  bash -c '
    set -euo pipefail

    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH:-}
    export TMPDIR=/tmp TEMP=/tmp TMP=/tmp
    export PYTHONUNBUFFERED=1
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

    source "${REPO_ROOT}/avlm/utils/_prep_bridge_env.sh"
    mb_prepare_bridge_env "${REPO_ROOT}/avlm/inference/megatron-bridge/slurm"

    export MB_BRIDGE_OVERLAY="${MB_BRIDGE_OVERLAY:-${CACHE_DIR}/megatron_bridge_workspace/megatron_bridge_overlay}"
    python "${REPO_ROOT}/avlm/training/megatron-bridge/training_data_processing/bridge_integration/apply_patches.py" \
      --overlay-dir "${MB_BRIDGE_OVERLAY}" \
      --recipe-name conversation_jsonl_sft_config

    infer_args=(
      --config "${INFERENCE_CONFIG}"
    )

    [[ -n "${MAX_INFERENCE_SAMPLES:-}" ]] && infer_args+=(--max-inference-samples "${MAX_INFERENCE_SAMPLES}")
    [[ -n "${MAX_INFERENCE_SAMPLES_SEED:-}" ]] && infer_args+=(--max-inference-samples-seed "${MAX_INFERENCE_SAMPLES_SEED}")
    [[ -n "${INFERENCE_NAME_OVERRIDE:-}" ]] && infer_args+=(--inference-name "${INFERENCE_NAME_OVERRIDE}")
    [[ -n "${BASE_OUTPUT_DIR_OVERRIDE:-}" ]] && infer_args+=(--base-output-dir "${BASE_OUTPUT_DIR_OVERRIDE}")
    [[ -n "${DATA_PATH_OVERRIDE:-}" ]] && infer_args+=(--data-path "${DATA_PATH_OVERRIDE}")
    [[ -n "${MODEL_PATH_OVERRIDE:-}" && "${MODEL_PATH_OVERRIDE}" != "null" ]] && infer_args+=(--model-path "${MODEL_PATH_OVERRIDE}")
    [[ -n "${ADAPTER_PATH_OVERRIDE:-}" && "${ADAPTER_PATH_OVERRIDE}" != "null" ]] && infer_args+=(--adapter-path "${ADAPTER_PATH_OVERRIDE}")
    [[ -n "${IS_FSDP_OVERRIDE:-}" ]] && infer_args+=(--is-fsdp "${IS_FSDP_OVERRIDE}")
    [[ -n "${FSDP_SHARDING_STRATEGY_OVERRIDE:-}" ]] && infer_args+=(--fsdp-sharding-strategy "${FSDP_SHARDING_STRATEGY_OVERRIDE}")
    [[ "${RESUME_OVERRIDE:-0}" == "1" || "${RESUME_OVERRIDE:-0}" == "true" || "${RESUME_OVERRIDE:-0}" == "True" || "${RESUME_OVERRIDE:-0}" == "TRUE" ]] && infer_args+=(--resume)

    torchrun \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${SLURM_PROCID}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      "${REPO_ROOT}/avlm/inference/megatron-bridge/entrypoint.py" \
      "${infer_args[@]}"
  ' &> "${LOG_FILE}"
