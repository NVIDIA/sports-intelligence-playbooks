# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared VLM LoRA launch (NeMo AutoModel). Sourced by interactive/train_interactive.sh and sbatch/srun.sh.

# shellcheck shell=bash
[[ -n "${_TRAIN_LIB_LOADED:-}" && $(type -t lora_train 2>/dev/null) == function ]] && return 0
_TRAIN_LIB_LOADED=1

_LORA_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load training knobs from the NeMo recipe YAML (always; not overridable from launch_local.yaml).
lora_load_recipe_env() {
  local recipe_yaml="${1:?recipe YAML path required}"
  local _py _yaml_params _recipe_tp=1 _recipe_pp=1 _recipe_cp=1
  if [[ $(type -t apply_yaml_exports_preserving_cli 2>/dev/null) != function ]]; then
    # shellcheck source=../../../../utils/_source_params.sh
    source "${_LORA_LIB_DIR}/../../../../utils/_source_params.sh"
  fi
  if [[ $(type -t training_cli_commit_recipe_overrides 2>/dev/null) == function ]]; then
    training_cli_commit_recipe_overrides
  fi
  if [[ $(type -t training_cli_pin_slurm_recipe_env 2>/dev/null) == function ]]; then
    training_cli_pin_slurm_recipe_env
  fi
  _py=/opt/venv/bin/python3
  [[ -x "${_py}" ]] || _py=$(command -v python3)
  _yaml_params="$("${_py}" "${_LORA_LIB_DIR}/../../../../utils/_load_params_yaml.py" "${recipe_yaml}" \
    _recipe_tp=distributed.tp_size \
    _recipe_pp=distributed.pp_size \
    _recipe_cp=distributed.cp_size \
    EP=distributed.ep_size \
    GLOBAL_BATCH_SIZE=step_scheduler.global_batch_size \
    USE_SEQUENCE_PACKING=dataloader.use_sequence_packing \
    PACK_SIZE=dataloader.packed_sequence.pack_size \
    GC_EVERY_STEPS=step_scheduler.gc_every_steps \
    CKPT_EVERY_STEPS=step_scheduler.ckpt_every_steps \
    VAL_EVERY_STEPS=step_scheduler.val_every_steps \
    DISPATCHER=model.backend.dispatcher \
    COLLATE_MAX_LENGTH=dataloader.collate_fn.max_length \
    MAX_VIDEO_FRAMES=dataset.max_video_frames \
    VAL_SAMPLE_RATIO=validation_dataset.sample_ratio \
    "${INFERENCE_VALIDATION_YAML_SELECTOR_ARGS[@]}")"
  apply_yaml_exports_preserving_cli "${_yaml_params}"
  inference_validation_apply_defaults
  finalize_wandb_mode_env "${recipe_yaml}"
  if [[ "${_recipe_tp}" != "${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}" ||
        "${_recipe_pp}" != "${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}" ||
        "${_recipe_cp}" != "${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[cp]}" ]]; then
    echo "error: Nemotron Omni AutoModel requires TP=${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}, PP=${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}, CP=${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[cp]} (recipe has TP=${_recipe_tp}, PP=${_recipe_pp}, CP=${_recipe_cp})" >&2
    return 1
  fi
  : "${EP:?distributed.ep_size missing in ${recipe_yaml}}"
  : "${GLOBAL_BATCH_SIZE:?step_scheduler.global_batch_size missing in ${recipe_yaml}}"
  : "${USE_SEQUENCE_PACKING:?dataloader.use_sequence_packing missing in ${recipe_yaml}}"
  training_normalize_bool USE_SEQUENCE_PACKING
  : "${GC_EVERY_STEPS:?step_scheduler.gc_every_steps missing in ${recipe_yaml}}"
  : "${CKPT_EVERY_STEPS:?step_scheduler.ckpt_every_steps missing in ${recipe_yaml}}"
  : "${VAL_EVERY_STEPS:?step_scheduler.val_every_steps missing in ${recipe_yaml}}"
  : "${DISPATCHER:?model.backend.dispatcher missing in ${recipe_yaml}}"
  : "${VAL_SAMPLE_RATIO:?validation_dataset.sample_ratio missing in ${recipe_yaml}}"
  : "${MAX_VIDEO_FRAMES:?dataset.max_video_frames missing in ${recipe_yaml}}"
  if [[ "${USE_SEQUENCE_PACKING}" == "1" ]]; then
    : "${PACK_SIZE:?dataloader.packed_sequence.pack_size missing in ${recipe_yaml} (required when use_sequence_packing: true)}"
  else
    : "${COLLATE_MAX_LENGTH:?dataloader.collate_fn.max_length missing in ${recipe_yaml} (required when use_sequence_packing: false)}"
  fi
  export VAL_SAMPLE_RATIO USE_SEQUENCE_PACKING PACK_SIZE GC_EVERY_STEPS CKPT_EVERY_STEPS VAL_EVERY_STEPS
  export DISPATCHER COLLATE_MAX_LENGTH MAX_VIDEO_FRAMES
}

lora_validate_parallelism_layout() {
  local nodes="${1:?nodes required}"
  local gpus_per_node="${2:?gpus_per_node required}"
  local _world=$((nodes * gpus_per_node))
  local _tp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}"
  local _pp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}"
  local _cp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[cp]}"
  local _non_pp=$((_world / _pp))

  if [[ -n "${TP:-}" && "${TP}" != "${_tp}" ]]; then
    echo "warn: TP=${TP} requested but Nemotron Omni AutoModel only supports TP=${_tp}" >&2
  fi
  if ((EP > 8)); then
    echo "warn: EP=${EP} > 8 needs DeepEP NVSHMEM (Hopper+); pre-hopper A100 wheels require EP≤8" >&2
  fi
  if ((_non_pp % EP != 0)); then
    echo "error: EP=${EP} must divide non_pp=${_non_pp} (nodes=${nodes}, GPUs/node=${gpus_per_node}, PP=${_pp})" >&2
    exit 1
  fi
  if ((_world % (_tp * _cp * _pp) != 0)); then
    echo "error: world_size=${_world} must be divisible by TP×PP×CP (${_tp}×${_pp}×${_cp})" >&2
    exit 1
  fi
}

lora_export_cpu_memory_env() {
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}

lora_resolve_output_dirs() {
  local _nodes="${SLURM_NNODES:-${num_nodes:-1}}"
  local _gpus="${NPROC_PER_NODE:-${GPUS_PER_NODE:?GPUS_PER_NODE must be set}}"
  lora_validate_parallelism_layout "${_nodes}" "${_gpus}"
  : "${MODEL_NAME:?Set MODEL_NAME in launch_local.yaml}"
  local _layout_nodes="${SLURM_NNODES:-${num_nodes:-1}}"
  export CKPT_LAYOUT_TAG="${CKPT_LAYOUT_TAG:-n${_layout_nodes}_tp${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}_pp${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}_ep${EP}}"
  export OUTPUT_BASE="${OUTPUT_BASE:-${_LORA_LIB_DIR}/outputs}"
  export OUTPUT="${OUTPUT:-${OUTPUT_BASE}/${MODEL_NAME}}"
  export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT}/checkpoints_${CKPT_LAYOUT_TAG}}"
  export WANDB_DIR="${WANDB_DIR:-${OUTPUT}/wandb}"
  mkdir -p "${OUTPUT}" "${CHECKPOINT_DIR}"
  [[ "${WANDB_MODE:-disabled}" == "disabled" ]] || mkdir -p "${WANDB_DIR}"
  echo "info: checkpoint_dir=${CHECKPOINT_DIR}" >&2
}

lora_launch_automodel() {
  local -a _torchrun_args=(--nproc_per_node="${NPROC_PER_NODE}")
  local _automodel_bin _rdzv_id

  _automodel_bin="$(command -v automodel)"
  if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node Slurm}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    _rdzv_id="${SLURM_JOB_ID:-${RDZV_ID:-lora_train}}"
    _torchrun_args+=(
      --nnodes="${SLURM_NNODES}"
      --rdzv_backend=c10d
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
      --rdzv_id="${_rdzv_id}"
    )
    echo "info: torchrun → automodel (${_automodel_bin}) nnodes=${SLURM_NNODES} nproc_per_node=${NPROC_PER_NODE} rdzv=${MASTER_ADDR}:${MASTER_PORT}" >&2
  else
    _torchrun_args+=(--nnodes=1)
    echo "info: torchrun → automodel (${_automodel_bin}) nproc_per_node=${NPROC_PER_NODE}" >&2
  fi

  train_run_with_interrupt torchrun "${_torchrun_args[@]}" \
    "${_automodel_bin}" \
    "${CONFIG_YAML}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    "${_extra_args[@]}"
}

_train_prepare_deepep() {
  ensure_deepep
  [[ "${SKIP_DEEPEP_WHEEL_CHECK:-0}" == "1" ]] || {
    local _py=/opt/venv/bin/python3
    [[ -x "${_py}" ]] || _py=$(command -v python3)
    "${_py}" "${REPO_ROOT}/wheels/deepep/check_deepep_wheel.py" --pretrain
  }
}

lora_train() {
  training_require_config_path
  : "${NPROC_PER_NODE:?NPROC_PER_NODE must be set}"
  : "${GPUS_PER_NODE:?GPUS_PER_NODE must be set}"
  _TRAIN_SCRIPT_DIR="${_LORA_LIB_DIR}"
  # shellcheck source=../../../../utils/_train_env.sh
  source "${_LORA_LIB_DIR}/../../../../utils/_train_env.sh"

  lora_load_recipe_env "${CONFIG_YAML}"
  ensure_vlm_training_deps
  _train_prepare_deepep

  lora_export_cpu_memory_env
  lora_resolve_output_dirs
  {
    local _nodes="${SLURM_NNODES:-${num_nodes:-1}}"
    local _world=$((NPROC_PER_NODE * (_nodes > 0 ? _nodes : 1)))
    local _tp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[tp]}"
    local _pp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[pp]}"
    local _cp="${AUTOMODEL_NEMOTRON_OMNI_PARALLELISM[cp]}"
    local _dp=$((_world / (_tp * _cp * _pp)))
    local _local_bs=1
    local _ga=0
    if [[ "${_dp}" -gt 0 && "${GLOBAL_BATCH_SIZE}" -gt 0 ]]; then
      _ga=$((GLOBAL_BATCH_SIZE / (_local_bs * _dp)))
    fi
    echo "info: global_batch_size=${GLOBAL_BATCH_SIZE} (DP=${_dp}, ${_world} GPU ranks)" >&2
    echo "info: grad_accum_steps=${_ga} (local_batch_size=${_local_bs} × dp_size=${_dp})" >&2
  }

  local -a _extra_args=(
    --step_scheduler.global_batch_size="${GLOBAL_BATCH_SIZE}"
    --dataset.max_video_frames="${MAX_VIDEO_FRAMES}"
    --validation_dataset.max_video_frames="${MAX_VIDEO_FRAMES}"
  )
  if [[ "${USE_SEQUENCE_PACKING}" == "1" ]]; then
    _extra_args+=(
      --dataloader.use_sequence_packing=true
      --dataloader.packed_sequence.max_length="${PACK_SIZE}"
      --dataloader.packed_sequence.pack_size="${PACK_SIZE}"
      --dataloader.packed_sequence.collate_max_length="${PACK_SIZE}"
      --model.llm_config.output_hidden_states=true
      --loss_fn._target_=nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
    )
  else
    _extra_args+=(
      --packed_sequence.pack_size=0
      --packed_sequence.max_length=0
      --dataloader.use_sequence_packing=false
      --dataloader.collate_fn.max_length="${COLLATE_MAX_LENGTH}"
      --dataloader.collate_fn.max_video_frames="${MAX_VIDEO_FRAMES}"
      --validation_dataloader.collate_fn.max_length="${COLLATE_MAX_LENGTH}"
      --validation_dataloader.collate_fn.max_video_frames="${MAX_VIDEO_FRAMES}"
      --model.llm_config.output_hidden_states=false
      --loss_fn._target_=nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy
    )
  fi
  _extra_args+=(
    --step_scheduler.gc_every_steps="${GC_EVERY_STEPS}"
    --step_scheduler.ckpt_every_steps="${CKPT_EVERY_STEPS}"
    --step_scheduler.val_every_steps="${VAL_EVERY_STEPS}"
  )
  if [[ ${#WANDB_OVERRIDES[@]} -gt 0 ]]; then
    _extra_args+=("${WANDB_OVERRIDES[@]}")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    _extra_args+=(--wandb.mode="${WANDB_MODE}")
  fi
  if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    _extra_args+=(--checkpoint.checkpoint_dir="${CHECKPOINT_DIR}")
  fi
  if [[ -n "${MAX_STEPS:-}" ]]; then
    _extra_args+=(--step_scheduler.max_steps="${MAX_STEPS}")
  fi
  if [[ -n "${DISPATCHER:-}" ]]; then
    _extra_args+=(--model.backend.dispatcher="${DISPATCHER}")
  fi
  if [[ "${FAKE_BALANCED_GATE:-0}" == "1" ]]; then
    _extra_args+=(--model.backend.fake_balanced_gate=true)
  fi
  if [[ -n "${VAL_SAMPLE_RATIO:-}" ]]; then
    _extra_args+=(--validation_dataset.sample_ratio="${VAL_SAMPLE_RATIO}")
  fi
  if [[ -n "${VAL_MAX_PACKS:-}" ]]; then
    _extra_args+=(--validation_dataloader.packed_sequence.max_packs="${VAL_MAX_PACKS}")
  fi
  if [[ "${USE_SEQUENCE_PACKING}" == "1" ]]; then
    echo "info: sequence packing enabled (pack_size=${PACK_SIZE})" >&2
  else
    echo "info: sequence packing disabled (collate max_length=${COLLATE_MAX_LENGTH}, frames=${MAX_VIDEO_FRAMES})" >&2
  fi
  cd "${AUTOMODEL_CODE_ROOT}"

  if [[ "${NPROC_PER_NODE}" -gt 1 ]] || [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
    lora_launch_automodel
    return $?
  fi

  automodel "${CONFIG_YAML}" --nproc-per-node "${NPROC_PER_NODE}" "${_extra_args[@]}"
}

lora_init_launch() {
  local _mode_dir="${1:?mode dir required}"
  local _launch_kind="${2:-interactive}"

  export CLUSTER_PARAMS="${CLUSTER_PARAMS:-${_mode_dir}/launch_local.yaml}"
  [[ "${CLUSTER_PARAMS}" == /* ]] || CLUSTER_PARAMS="${_mode_dir}/${CLUSTER_PARAMS#"${_mode_dir}"/}"
  export CLUSTER_PARAMS
  # shellcheck source=../../../../utils/_source_params.sh
  source "${_mode_dir}/../../../../utils/_source_params.sh"
  training_cli_commit_env_overrides "${_mode_dir}" "${SBATCH_CLI_OVERRIDE_KEYS[@]}" "${SBATCH_ENV_ONLY_KEYS[@]}"
  source_cluster_params "${_mode_dir}"

  training_require_config_path
  : "${MODEL_NAME:?Set MODEL_NAME in launch_local.yaml}"
  : "${GPUS_PER_NODE:?Set GPUS_PER_NODE in launch_local.yaml}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"

  if [[ "${_launch_kind}" == "interactive" ]]; then
    export num_nodes=1
    export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    export WANDB_NAME="${WANDB_NAME:-${MODEL_NAME}_${RUN_TIMESTAMP}}"
    WANDB_OVERRIDES=()
    [[ -n "${WANDB_NAME:-}" ]] && WANDB_OVERRIDES+=(--wandb.name="${WANDB_NAME}")
    [[ -n "${WANDB_RUN_ID:-}" ]] && WANDB_OVERRIDES+=(--wandb.id="${WANDB_RUN_ID}")
    [[ -n "${WANDB_RESUME:-}" ]] && WANDB_OVERRIDES+=(--wandb.resume="${WANDB_RESUME}")
    export TRAIN_LAUNCH_KIND=interactive
  else
    if [[ "${SLURM_NNODES:-0}" -gt 0 ]]; then
      export num_nodes="${SLURM_NNODES}"
    else
      export num_nodes=1
    fi
    WANDB_OVERRIDES=()
    [[ -n "${WANDB_NAME:-}" ]] && WANDB_OVERRIDES+=(--wandb.name="${WANDB_NAME}")
    [[ -n "${WANDB_RUN_ID:-}" ]] && WANDB_OVERRIDES+=(--wandb.id="${WANDB_RUN_ID}")
    [[ -n "${WANDB_RESUME:-}" ]] && WANDB_OVERRIDES+=(--wandb.resume="${WANDB_RESUME}")
    export TRAIN_LAUNCH_KIND=batch
  fi
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
}
