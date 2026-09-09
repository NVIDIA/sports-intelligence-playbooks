# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Megatron-Bridge VLM SFT launch. Sourced by interactive/train_interactive.sh and sbatch/srun.sh.

# shellcheck shell=bash
[[ -n "${_MB_SFT_TRAIN_LIB_LOADED:-}" && $(type -t mb_sft_train 2>/dev/null) == function ]] && return 0
_MB_SFT_TRAIN_LIB_LOADED=1

_MB_SFT_LIB_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MB_ROOT="$(cd -- "${_MB_SFT_LIB_DIR}/../.." && pwd)"

# shellcheck source=../../_prep_bridge_env.sh
source "${_MB_ROOT}/../../utils/_prep_bridge_env.sh"

# Keys loaded from Megatron-Bridge recipe YAML that may be overridden from the environment.
MB_SFT_RECIPE_CLI_OVERRIDE_KEYS=(
  MODEL_NAME CONFIG_YAML_REL HF_MODEL_ID RECIPE PRETRAINED_CHECKPOINT pretrained_checkpoint
  TRAIN_JSONL VAL_JSONL TRAIN_JSONL_FORMAT VAL_JSONL_FORMAT VIDEO_ROOT VAL_VIDEO_ROOT
  TRAIN_SAMPLE_RATIO VAL_SAMPLE_RATIO
  GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE MAX_STEPS
  GC_EVERY_STEPS CKPT_EVERY_STEPS VAL_EVERY_STEPS USE_SEQUENCE_PACKING SEQ_LENGTH
  MAX_VIDEO_FRAMES VAL_MAX_VIDEO_FRAMES VIDEO_SAMPLE_FPS VAL_VIDEO_SAMPLE_FPS DATALOADER_NUM_WORKERS
  FREEZE_LANGUAGE_MODEL FREEZE_VISION_MODEL FREEZE_VISION_PROJECTION FREEZE_SOUND_ENCODER FREEZE_SOUND_PROJECTION
  OPTIMIZER_LR OPTIMIZER_WEIGHT_DECAY OPTIMIZER_ADAM_BETA1 OPTIMIZER_ADAM_BETA2
  SCHEDULER_LR_DECAY_STYLE SCHEDULER_MIN_LR SCHEDULER_LR_WARMUP_ITERS SCHEDULER_LR_DECAY_ITERS
  CLIP_GRAD_MAX_NORM EVAL_ITERS LOG_INTERVAL WANDB_MODE WANDB_ENTITY WANDB_PROJECT WANDB_NAME RNG_SEED
  AVLM_HF_RESIZE
  EMPTY_UNUSED_MEMORY_LEVEL OPTIMIZER_CPU_OFFLOAD OPTIMIZER_OFFLOAD_FRACTION OVERLAP_CPU_OPTIMIZER_D2H_H2D
  USE_PRECISION_AWARE_OPTIMIZER SAVE_OPTIM
  RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS RECOMPUTE_MODULES
  FINE_GRAINED_ACTIVATION_OFFLOADING OFFLOAD_MODULES
  EXIT_MINS_BEFORE_LIMIT
  CKPT_LAYOUT_TAG CHECKPOINT_DIR
  MAX_TRAIN_SAMPLES MAX_VAL_SAMPLES
  LORA_DIM LORA_ALPHA LORA_TARGET_MODULES RESUME_CHECKPOINT
  TRAIN_ITERS EVAL_INTERVAL SAVE_INTERVAL PACKED_SEQ GC_EVERY_ITERS
  INFERENCE_VALIDATION_ENABLED INFERENCE_VALIDATION_EVERY_STEPS INFERENCE_VALIDATION_INFERENCE_CONFIG
  INFERENCE_VALIDATION_DATA_PATH INFERENCE_VALIDATION_MAX_SAMPLES INFERENCE_VALIDATION_MAX_SAMPLES_SEED INFERENCE_VALIDATION_CLUSTER_PARAMS
  INFERENCE_VALIDATION_NUM_NODES INFERENCE_VALIDATION_GPUS_PER_NODE INFERENCE_VALIDATION_NPROC_PER_NODE
  VLM_SCORER_CONFIG
)

_mb_sft_normalize_cli_alias_overrides() {
  # Map alternate CLI names onto canonical recipe env keys before YAML load.
  declare -A _from_to=(
    [TRAIN_ITERS]=MAX_STEPS
    [EVAL_INTERVAL]=VAL_EVERY_STEPS
    [SAVE_INTERVAL]=CKPT_EVERY_STEPS
    [PACKED_SEQ]=USE_SEQUENCE_PACKING
    [GC_EVERY_ITERS]=GC_EVERY_STEPS
  )
  local from to
  for from in "${!_from_to[@]}"; do
    to="${_from_to[${from}]}"
    if [[ -n "${_TRAINING_CLI_OVERRIDES[${from}]+x}" && -z "${_TRAINING_CLI_OVERRIDES[${to}]+x}" ]]; then
      _TRAINING_CLI_OVERRIDES["${to}"]="${_TRAINING_CLI_OVERRIDES[${from}]}"
    fi
  done
}

mb_sft_commit_recipe_cli_overrides() {
  # shellcheck source=../../utils/_source_params.sh
  source "${_MB_ROOT}/../../utils/_source_params.sh"
  training_cli_detect_prefix_overrides "${MB_SFT_RECIPE_CLI_OVERRIDE_KEYS[@]}"
  training_cli_record_bashrc_fallback "${MB_SFT_RECIPE_CLI_OVERRIDE_KEYS[@]}"
  training_cli_pin_slurm_env "${MB_SFT_RECIPE_CLI_OVERRIDE_KEYS[@]}"
  _mb_sft_normalize_cli_alias_overrides
}

_mb_sft_cli_override_tag() {
  local key="$1"
  if [[ -n "${_TRAINING_CLI_OVERRIDES[${key}]+x}" ]]; then
    printf ' (CLI override)'
  else
    printf ' (recipe YAML)'
  fi
}

mb_sft_log_training_knobs() {
  echo "info: effective training knobs (CLI overrides beat recipe YAML):" >&2
  echo "info:   MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}$(_mb_sft_cli_override_tag MICRO_BATCH_SIZE)" >&2
  echo "info:   MAX_STEPS=${MAX_STEPS}$(_mb_sft_cli_override_tag MAX_STEPS)" >&2
  echo "info:   GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}$(_mb_sft_cli_override_tag GLOBAL_BATCH_SIZE)" >&2
  echo "info:   SEQ_LENGTH=${SEQ_LENGTH}$(_mb_sft_cli_override_tag SEQ_LENGTH)" >&2
  echo "info:   USE_SEQUENCE_PACKING=${USE_SEQUENCE_PACKING}$(_mb_sft_cli_override_tag USE_SEQUENCE_PACKING)" >&2
  echo "info:   PACKING_RATIO=${PACKING_RATIO:-1.0}$(_mb_sft_cli_override_tag PACKING_RATIO)" >&2
  echo "info:   MAX_VIDEO_FRAMES=${MAX_VIDEO_FRAMES}$(_mb_sft_cli_override_tag MAX_VIDEO_FRAMES)" >&2
  echo "info:   VAL_MAX_VIDEO_FRAMES=${VAL_MAX_VIDEO_FRAMES}$(_mb_sft_cli_override_tag VAL_MAX_VIDEO_FRAMES)" >&2
  echo "info:   FREEZE_LANGUAGE_MODEL=${FREEZE_LANGUAGE_MODEL:-false}$(_mb_sft_cli_override_tag FREEZE_LANGUAGE_MODEL)" >&2
  echo "info:   EMPTY_UNUSED_MEMORY_LEVEL=${EMPTY_UNUSED_MEMORY_LEVEL:-0}$(_mb_sft_cli_override_tag EMPTY_UNUSED_MEMORY_LEVEL)" >&2
  echo "info:   OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-0}$(_mb_sft_cli_override_tag OPTIMIZER_CPU_OFFLOAD)" >&2
  echo "info:   OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0}$(_mb_sft_cli_override_tag OPTIMIZER_OFFLOAD_FRACTION)" >&2
  echo "info:   OVERLAP_CPU_OPTIMIZER_D2H_H2D=${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-0}$(_mb_sft_cli_override_tag OVERLAP_CPU_OPTIMIZER_D2H_H2D)" >&2
  echo "info:   RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-null}$(_mb_sft_cli_override_tag RECOMPUTE_GRANULARITY)" >&2
  echo "info:   RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-null}$(_mb_sft_cli_override_tag RECOMPUTE_METHOD)" >&2
  echo "info:   RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-null}$(_mb_sft_cli_override_tag RECOMPUTE_NUM_LAYERS)" >&2
  echo "info:   RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-null}$(_mb_sft_cli_override_tag RECOMPUTE_MODULES)" >&2
  echo "info:   FINE_GRAINED_ACTIVATION_OFFLOADING=${FINE_GRAINED_ACTIVATION_OFFLOADING:-0}$(_mb_sft_cli_override_tag FINE_GRAINED_ACTIVATION_OFFLOADING)" >&2
  echo "info:   OFFLOAD_MODULES=${OFFLOAD_MODULES:-null}$(_mb_sft_cli_override_tag OFFLOAD_MODULES)" >&2
  echo "info:   USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-0}$(_mb_sft_cli_override_tag USE_PRECISION_AWARE_OPTIMIZER)" >&2
  echo "info:   BF16_OPTIMIZER_STATES=${BF16_OPTIMIZER_STATES:-0}$(_mb_sft_cli_override_tag BF16_OPTIMIZER_STATES)" >&2
  echo "info:   USE_MEGATRON_FSDP=${USE_MEGATRON_FSDP:-0}$(_mb_sft_cli_override_tag USE_MEGATRON_FSDP)" >&2
  echo "info:   PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}$(_mb_sft_cli_override_tag PYTORCH_CUDA_ALLOC_CONF)" >&2
  echo "info:   NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}$(_mb_sft_cli_override_tag NCCL_NVLS_ENABLE)" >&2
  echo "info:   TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}$(_mb_sft_cli_override_tag TORCH_NCCL_AVOID_RECORD_STREAMS)" >&2
  echo "info:   CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}$(_mb_sft_cli_override_tag CUDA_DEVICE_MAX_CONNECTIONS)" >&2
  echo "info:   MODEL_NAME=${MODEL_NAME}$(_mb_sft_cli_override_tag MODEL_NAME)" >&2
}

mb_sft_validate_training_knobs() {
  local _packed _mbs _fine_grained
  _packed="$(_mb_sft_bool_cli "${USE_SEQUENCE_PACKING:-false}")"
  _mbs="${MICRO_BATCH_SIZE:-1}"
  if [[ "${_packed}" == "True" && "${_mbs}" -ne 1 ]]; then
    echo "error: USE_SEQUENCE_PACKING=1 requires MICRO_BATCH_SIZE=1 (each micro-batch item is one completed pack)." >&2
    echo "hint: set MICRO_BATCH_SIZE=1, or disable packing with USE_SEQUENCE_PACKING=0" >&2
    return 1
  fi

  _fine_grained="$(_mb_sft_bool_enabled "${FINE_GRAINED_ACTIVATION_OFFLOADING:-0}")"
  if [[ "${_fine_grained}" == "1" && -z "${OFFLOAD_MODULES:-}" ]]; then
    echo "error: fine-grained activation offloading requires OFFLOAD_MODULES." >&2
    return 1
  fi

  if [[ "$(_mb_sft_bool_enabled "${OPTIMIZER_CPU_OFFLOAD:-0}")" == "1" ]]; then
    if [[ "$(_mb_sft_bool_enabled "${USE_PRECISION_AWARE_OPTIMIZER:-0}")" != "1" ]]; then
      echo "error: optimizer CPU offloading requires USE_PRECISION_AWARE_OPTIMIZER=true." >&2
      return 1
    fi
  fi

  if [[ "$(_mb_sft_bool_enabled "${BF16_OPTIMIZER_STATES:-0}")" == "1" ]]; then
    if [[ "$(_mb_sft_bool_enabled "${USE_PRECISION_AWARE_OPTIMIZER:-0}")" != "1" ]]; then
      echo "error: precision.bf16_optimizer_states=true requires optimizer.use_precision_aware_optimizer=true." >&2
      return 1
    fi
  fi
}

_mb_sft_bool_enabled() {
  case "${1:-1}" in
    1 | true | True | TRUE | yes | YES | on | ON) echo "1" ;;
    *) echo "0" ;;
  esac
}

_mb_sft_bool_cli() {
  case "${1:-}" in
    1 | true | True | TRUE | yes | YES | on | ON) echo "True" ;;
    0 | false | False | FALSE | no | NO | off | OFF) echo "False" ;;
    *) echo "${1}" ;;
  esac
}

_mb_sft_modules_cli() {
  local raw="${1:-}" item joined=""
  raw="${raw#[}"
  raw="${raw%]}"
  raw="${raw//,/ }"
  for item in ${raw}; do
    [[ -n "${joined}" ]] && joined+=","
    joined+="${item}"
  done
  printf '[%s]' "${joined}"
}

_mb_sft_is_peft_recipe() {
  [[ "${RECIPE:-}" == *peft* ]]
}

# Environment: pretrained_checkpoint= (recipe key) or PRETRAINED_CHECKPOINT= (launcher name).
mb_sft_resolve_pretrained_checkpoint() {
  if [[ -n "${_TRAINING_CLI_OVERRIDES[pretrained_checkpoint]+x}" && -z "${_TRAINING_CLI_OVERRIDES[PRETRAINED_CHECKPOINT]+x}" ]]; then
    PRETRAINED_CHECKPOINT="${_TRAINING_CLI_OVERRIDES[pretrained_checkpoint]}"
  elif [[ -n "${pretrained_checkpoint:-}" && -z "${_TRAINING_CLI_OVERRIDES[PRETRAINED_CHECKPOINT]+x}" && -z "${_TRAINING_CLI_OVERRIDES[pretrained_checkpoint]+x}" ]]; then
    PRETRAINED_CHECKPOINT="${pretrained_checkpoint}"
  fi
  PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${WORKSPACE}/models/${HF_MODEL_BASENAME}-megatron}"
  export PRETRAINED_CHECKPOINT
  : "${PRETRAINED_CHECKPOINT:?checkpoint.pretrained_checkpoint missing (recipe YAML or PRETRAINED_CHECKPOINT environment override)}"
}

# Resolve supported alternate environment names.
mb_sft_resolve_cli_aliases() {
  MAX_STEPS="${MAX_STEPS:-${TRAIN_ITERS:-}}"
  VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-${EVAL_INTERVAL:-}}"
  CKPT_EVERY_STEPS="${CKPT_EVERY_STEPS:-${SAVE_INTERVAL:-}}"
  USE_SEQUENCE_PACKING="${USE_SEQUENCE_PACKING:-${PACKED_SEQ:-}}"
  GC_EVERY_STEPS="${GC_EVERY_STEPS:-${GC_EVERY_ITERS:-}}"
  export MAX_STEPS VAL_EVERY_STEPS CKPT_EVERY_STEPS
  export USE_SEQUENCE_PACKING SEQ_LENGTH GC_EVERY_STEPS WANDB_NAME
}

mb_sft_resolve_config_yaml() {
  if [[ -n "${CONFIG_YAML:-}" && -f "${CONFIG_YAML}" ]]; then
    :
  elif [[ -n "${CONFIG_YAML_REL:-}" && -f "${REPO_ROOT}/${CONFIG_YAML_REL}" ]]; then
    export CONFIG_YAML="${REPO_ROOT}/${CONFIG_YAML_REL}"
  else
    echo "error: set CONFIG_YAML or CONFIG_YAML_REL" >&2
    return 1
  fi
  CONFIG_YAML="$(cd -- "$(dirname "${CONFIG_YAML}")" && pwd)/$(basename "${CONFIG_YAML}")"
  export CONFIG_YAML CONFIG_YAML_REL
  echo "info: recipe YAML ${CONFIG_YAML}" >&2
}

# Training settings from the selected Megatron-Bridge recipe.
mb_sft_load_recipe_env() {
  mb_sft_commit_recipe_cli_overrides
  mb_sft_resolve_config_yaml
  local _py _yaml_params
  _py=$(command -v python3)
  _yaml_params="$("${_py}" "${_MB_ROOT}/../../utils/_load_params_yaml.py" "${CONFIG_YAML}" \
    HF_MODEL_ID=run.hf_model_id \
    RECIPE=run.recipe \
    MODEL_NAME=run.model_name \
    TRAIN_JSONL=dataset.path_or_dataset \
    VAL_JSONL=validation_dataset.path_or_dataset \
    TRAIN_JSONL_FORMAT=dataset.jsonl_format \
    VAL_JSONL_FORMAT=validation_dataset.jsonl_format \
    VIDEO_ROOT=dataset.video_root \
    VIDEO_METADATA_PATH=dataset.video_metadata_path \
    PACKING_RATIO=dataset.packing_ratio \
    TRAIN_SAMPLE_RATIO=dataset.sample_ratio \
    MAX_VIDEO_FRAMES=dataset.max_video_frames \
    VIDEO_SAMPLE_FPS=dataset.video_sample_fps \
    VAL_VIDEO_ROOT=validation_dataset.video_root \
    VAL_SAMPLE_RATIO=validation_dataset.sample_ratio \
    VAL_MAX_VIDEO_FRAMES=validation_dataset.max_video_frames \
    VAL_VIDEO_SAMPLE_FPS=validation_dataset.video_sample_fps \
    TP=distributed.tp_size \
    EP=distributed.ep_size \
    USE_MEGATRON_FSDP=distributed.use_megatron_fsdp \
    BF16_OPTIMIZER_STATES=precision.bf16_optimizer_states \
    PYTORCH_CUDA_ALLOC_CONF=environment.pytorch_cuda_alloc_conf \
    NCCL_NVLS_ENABLE=environment.nccl_nvls_enable \
    TORCH_NCCL_AVOID_RECORD_STREAMS=environment.torch_nccl_avoid_record_streams \
    CUDA_DEVICE_MAX_CONNECTIONS=environment.cuda_device_max_connections \
    PRETRAINED_CHECKPOINT=checkpoint.pretrained_checkpoint \
    SAVE_OPTIM=checkpoint.save_optim \
    GLOBAL_BATCH_SIZE=step_scheduler.global_batch_size \
    MICRO_BATCH_SIZE=step_scheduler.micro_batch_size \
    MAX_STEPS=step_scheduler.max_steps \
    GC_EVERY_STEPS=step_scheduler.gc_every_iters \
    CKPT_EVERY_STEPS=step_scheduler.ckpt_every_iters \
    VAL_EVERY_STEPS=step_scheduler.val_every_iters \
    USE_SEQUENCE_PACKING=dataset.pack_sequences_in_batch \
    SEQ_LENGTH=dataset.seq_length \
    DATALOADER_NUM_WORKERS=dataset.num_workers \
    FREEZE_LANGUAGE_MODEL=freeze_config.freeze_language_model \
    FREEZE_VISION_MODEL=freeze_config.freeze_vision_model \
    FREEZE_VISION_PROJECTION=freeze_config.freeze_vision_projection \
    FREEZE_SOUND_ENCODER=freeze_config.freeze_sound_encoder \
    FREEZE_SOUND_PROJECTION=freeze_config.freeze_sound_projection \
    OPTIMIZER_LR=optimizer.lr \
    OPTIMIZER_WEIGHT_DECAY=optimizer.weight_decay \
    OPTIMIZER_ADAM_BETA1=optimizer.adam_beta1 \
    OPTIMIZER_ADAM_BETA2=optimizer.adam_beta2 \
    OPTIMIZER_CPU_OFFLOAD=optimizer.optimizer_cpu_offload \
    OPTIMIZER_OFFLOAD_FRACTION=optimizer.optimizer_offload_fraction \
    OVERLAP_CPU_OPTIMIZER_D2H_H2D=optimizer.overlap_cpu_optimizer_d2h_h2d \
    USE_PRECISION_AWARE_OPTIMIZER=optimizer.use_precision_aware_optimizer \
    RECOMPUTE_GRANULARITY=model.recompute_granularity \
    RECOMPUTE_METHOD=model.recompute_method \
    RECOMPUTE_NUM_LAYERS=model.recompute_num_layers \
    RECOMPUTE_MODULES=model.recompute_modules \
    FINE_GRAINED_ACTIVATION_OFFLOADING=model.fine_grained_activation_offloading \
    OFFLOAD_MODULES=model.offload_modules \
    EMPTY_UNUSED_MEMORY_LEVEL=train.empty_unused_memory_level \
    SCHEDULER_LR_DECAY_STYLE=lr_scheduler.lr_decay_style \
    SCHEDULER_MIN_LR=lr_scheduler.min_lr \
    SCHEDULER_LR_WARMUP_ITERS=lr_scheduler.lr_warmup_iters \
    SCHEDULER_LR_DECAY_ITERS=lr_scheduler.lr_decay_iters \
    CLIP_GRAD_MAX_NORM=clip_grad_norm.max_norm \
    EVAL_ITERS=validation.eval_iters \
    LOG_INTERVAL=logger.log_interval \
    WANDB_MODE=wandb.mode \
    WANDB_ENTITY=wandb.entity \
    WANDB_PROJECT=wandb.project \
    WANDB_NAME=wandb.name \
    RNG_SEED=rng.seed \
    AVLM_HF_RESIZE=task_encoder.resize_to_512 \
    LORA_DIM=peft.dim \
    LORA_ALPHA=peft.alpha \
    LORA_TARGET_MODULES=peft.target_modules \
    "${INFERENCE_VALIDATION_YAML_SELECTOR_ARGS[@]}")"
  # shellcheck source=../../utils/_source_params.sh
  source "${_MB_ROOT}/../../utils/_source_params.sh"
  apply_yaml_exports_preserving_cli "${_yaml_params}"
  inference_validation_apply_defaults
  # Re-apply environment overrides after loading the recipe.
  for key in "${!_TRAINING_CLI_OVERRIDES[@]}"; do
    export "${key}=${_TRAINING_CLI_OVERRIDES[${key}]}"
  done
  : "${TRAIN_JSONL:?dataset.path_or_dataset missing in ${CONFIG_YAML}}"
  : "${VAL_JSONL:?validation_dataset.path_or_dataset missing in ${CONFIG_YAML}}"
  : "${VIDEO_ROOT:?dataset.video_root missing in ${CONFIG_YAML}}"
  : "${TP:?distributed.tp_size missing in ${CONFIG_YAML}}"
  : "${EP:?distributed.ep_size missing in ${CONFIG_YAML}}"
  mb_sft_resolve_cli_aliases
  training_normalize_bool USE_SEQUENCE_PACKING
  BF16_OPTIMIZER_STATES="${BF16_OPTIMIZER_STATES:-false}"
  USE_MEGATRON_FSDP="${USE_MEGATRON_FSDP:-false}"
  training_normalize_bool BF16_OPTIMIZER_STATES
  training_normalize_bool USE_MEGATRON_FSDP
  : "${HF_MODEL_ID:?run.hf_model_id missing in ${CONFIG_YAML}}"
  : "${RECIPE:?run.recipe missing in ${CONFIG_YAML}}"
  : "${MODEL_NAME:?run.model_name missing in ${CONFIG_YAML}}"
  : "${MAX_STEPS:?step_scheduler.max_steps missing in ${CONFIG_YAML}}"
  export HF_MODEL_BASENAME="${HF_MODEL_BASENAME:-$(basename "${HF_MODEL_ID}")}"
  export WORKSPACE="${WORKSPACE:-${CACHE_DIR}/megatron_bridge_workspace}"
  VAL_VIDEO_ROOT="${VAL_VIDEO_ROOT:-${VIDEO_ROOT}}"
  # Apply a training frame override to validation unless validation is overridden separately.
  if [[ -n "${_TRAINING_CLI_OVERRIDES[MAX_VIDEO_FRAMES]+x}" ]]; then
    if [[ -z "${_TRAINING_CLI_OVERRIDES[VAL_MAX_VIDEO_FRAMES]+x}" ]]; then
      VAL_MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES}"
    fi
  else
    VAL_MAX_VIDEO_FRAMES="${VAL_MAX_VIDEO_FRAMES:-${MAX_VIDEO_FRAMES}}"
  fi
  VAL_VIDEO_SAMPLE_FPS="${VAL_VIDEO_SAMPLE_FPS:-${VIDEO_SAMPLE_FPS}}"
  TRAIN_JSONL_FORMAT="${TRAIN_JSONL_FORMAT:-llava}"
  VAL_JSONL_FORMAT="${VAL_JSONL_FORMAT:-${TRAIN_JSONL_FORMAT}}"
  RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-null}"
  FINE_GRAINED_ACTIVATION_OFFLOADING="${FINE_GRAINED_ACTIVATION_OFFLOADING:-false}"
  mb_sft_resolve_pretrained_checkpoint
  export TRAIN_JSONL VAL_JSONL TRAIN_JSONL_FORMAT VAL_JSONL_FORMAT VIDEO_ROOT TRAIN_SAMPLE_RATIO MAX_VIDEO_FRAMES VIDEO_SAMPLE_FPS
  export VIDEO_METADATA_PATH PACKING_RATIO
  export VAL_VIDEO_ROOT VAL_SAMPLE_RATIO VAL_MAX_VIDEO_FRAMES VAL_VIDEO_SAMPLE_FPS
  export TP EP GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE MAX_STEPS
  export HF_MODEL_ID RECIPE MODEL_NAME HF_MODEL_BASENAME
  export GC_EVERY_STEPS CKPT_EVERY_STEPS VAL_EVERY_STEPS USE_SEQUENCE_PACKING SEQ_LENGTH DATALOADER_NUM_WORKERS
  export FREEZE_LANGUAGE_MODEL FREEZE_VISION_MODEL FREEZE_VISION_PROJECTION
  export FREEZE_SOUND_ENCODER FREEZE_SOUND_PROJECTION
  export OPTIMIZER_LR OPTIMIZER_WEIGHT_DECAY OPTIMIZER_ADAM_BETA1 OPTIMIZER_ADAM_BETA2
  export OPTIMIZER_CPU_OFFLOAD OPTIMIZER_OFFLOAD_FRACTION OVERLAP_CPU_OPTIMIZER_D2H_H2D
  export USE_PRECISION_AWARE_OPTIMIZER EMPTY_UNUSED_MEMORY_LEVEL SAVE_OPTIM
  export BF16_OPTIMIZER_STATES USE_MEGATRON_FSDP
  export PYTORCH_CUDA_ALLOC_CONF NCCL_NVLS_ENABLE TORCH_NCCL_AVOID_RECORD_STREAMS
  export CUDA_DEVICE_MAX_CONNECTIONS
  export RECOMPUTE_GRANULARITY RECOMPUTE_METHOD RECOMPUTE_NUM_LAYERS RECOMPUTE_MODULES
  export FINE_GRAINED_ACTIVATION_OFFLOADING OFFLOAD_MODULES
  export SCHEDULER_LR_DECAY_STYLE SCHEDULER_MIN_LR SCHEDULER_LR_WARMUP_ITERS SCHEDULER_LR_DECAY_ITERS
  export CLIP_GRAD_MAX_NORM EVAL_ITERS LOG_INTERVAL
  export WANDB_MODE WANDB_ENTITY WANDB_PROJECT WANDB_NAME RNG_SEED
  export AVLM_HF_RESIZE
  export LORA_DIM LORA_ALPHA LORA_TARGET_MODULES
  mb_sft_validate_training_knobs || return
  mb_sft_apply_processed_prompt_logging_defaults
  mb_sft_log_training_knobs
}

mb_sft_apply_processed_prompt_logging_defaults() {
  case "${AVLM_LOG_PROCESSED_PROMPT:-}" in
    1|true|True|yes|on)
      export DATALOADER_NUM_WORKERS=0
      echo "info: AVLM_LOG_PROCESSED_PROMPT=1 → num_workers=0, logging to \${OUTPUT}/processed_prompt_logs/" >&2
      ;;
  esac
  echo "info: AVLM_HF_RESIZE=${AVLM_HF_RESIZE:-<unset>}" >&2
}

mb_sft_resolve_output_dirs() {
  local _nodes="${SLURM_NNODES:-${num_nodes:-1}}"
  local _gpus="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"
  mb_validate_parallelism_layout "${_nodes}" "${_gpus}"

  export OUTPUT_BASE="${OUTPUT_BASE:-${MB_TRAIN_SLURM_DIR:-${_MB_SFT_LIB_DIR}}/outputs}"
  export OUTPUT="${OUTPUT:-${OUTPUT_BASE}/${MODEL_NAME}}"
  export WANDB_DIR="${WANDB_DIR:-${OUTPUT}/wandb}"

  local _layout_tag="n${_nodes}_pp1_tp${TP}_ep${EP}"
  export CKPT_LAYOUT_TAG="${CKPT_LAYOUT_TAG:-${_layout_tag}}"

  export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT}/checkpoints_${CKPT_LAYOUT_TAG}}"
  export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT}/tb_logs}"
  mkdir -p "${OUTPUT}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"
  [[ "${WANDB_MODE:-disabled}" == "disabled" ]] || mkdir -p "${WANDB_DIR}"
  echo "info: output=${OUTPUT}" >&2
  echo "info: ckpt_layout_tag=${CKPT_LAYOUT_TAG}" >&2
  echo "info: checkpoint_dir=${CHECKPOINT_DIR}" >&2
  echo "info: pretrained_checkpoint=${PRETRAINED_CHECKPOINT}" >&2
}

mb_sft_build_cli_overrides() {
  mb_sft_validate_training_knobs || return
  local _packed _gc_manual=false _checkpoint_finetune=True _sequence_parallel=False
  _packed="$(_mb_sft_bool_cli "${USE_SEQUENCE_PACKING:-false}")"
  if [[ "${TP:-2}" -gt 1 ]]; then
    _sequence_parallel=True
  fi
  # Pre-pad packed batches to TP multiple in collate (cu_seqlens kept in sync) so MoE+TP
  # can keep sequence_parallel=True without LlavaModel adding orphan tail padding.
  if [[ "${_packed}" == "True" ]]; then
    export AVLM_PACK_PAD_MULTIPLE="${TP:-2}"
  else
    unset AVLM_PACK_PAD_MULTIPLE
  fi
  if [[ -n "${GC_EVERY_STEPS:-}" && "${GC_EVERY_STEPS}" -gt 0 ]]; then
    _gc_manual=true
  fi
  if ! _mb_sft_is_peft_recipe && [[ -f "${CHECKPOINT_DIR}/latest_train_state.pt" || -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    _checkpoint_finetune=False
  fi

  MB_SFT_CLI_OVERRIDES=(
    "checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
    "checkpoint.save=${CHECKPOINT_DIR}"
    "checkpoint.save_interval=${CKPT_EVERY_STEPS:-500}"
    "checkpoint.finetune=${_checkpoint_finetune}"
    "logger.tensorboard_dir=${TENSORBOARD_DIR}"
    "model.seq_length=${SEQ_LENGTH:-4096}"
    "dataset.seq_length=${SEQ_LENGTH:-4096}"
    "model.tensor_model_parallel_size=${TP:-2}"
    "model.expert_model_parallel_size=${EP:-4}"
    "model.expert_tensor_parallel_size=1"
    "model.context_parallel_size=1"
    "model.pipeline_model_parallel_size=1"
    "model.sequence_parallel=${_sequence_parallel}"
    "model.recompute_granularity=${RECOMPUTE_GRANULARITY:-null}"
    "model.fine_grained_activation_offloading=$(_mb_sft_bool_cli "${FINE_GRAINED_ACTIVATION_OFFLOADING:-false}")"
    "model.freeze_language_model=$(_mb_sft_bool_cli "${FREEZE_LANGUAGE_MODEL:-false}")"
    "model.freeze_vision_model=$(_mb_sft_bool_cli "${FREEZE_VISION_MODEL:-false}")"
    "model.freeze_vision_projection=$(_mb_sft_bool_cli "${FREEZE_VISION_PROJECTION:-false}")"
    "model.freeze_sound_encoder=$(_mb_sft_bool_cli "${FREEZE_SOUND_ENCODER:-false}")"
    "model.freeze_sound_projection=$(_mb_sft_bool_cli "${FREEZE_SOUND_PROJECTION:-false}")"
    "train.train_iters=${MAX_STEPS:-4000}"
    "train.global_batch_size=${GLOBAL_BATCH_SIZE:-16}"
    "train.micro_batch_size=${MICRO_BATCH_SIZE:-1}"
    "train.manual_gc=${_gc_manual}"
    "train.empty_unused_memory_level=${EMPTY_UNUSED_MEMORY_LEVEL:-0}"
  )

  if [[ "${RECOMPUTE_GRANULARITY:-null}" == "selective" ]]; then
    if [[ -n "${RECOMPUTE_MODULES:-}" ]]; then
      MB_SFT_CLI_OVERRIDES+=("model.recompute_modules=$(_mb_sft_modules_cli "${RECOMPUTE_MODULES}")")
    fi
  elif [[ "${RECOMPUTE_GRANULARITY:-null}" == "full" ]]; then
    if [[ -n "${RECOMPUTE_METHOD:-}" ]]; then
      MB_SFT_CLI_OVERRIDES+=("model.recompute_method=${RECOMPUTE_METHOD}")
    fi
    if [[ -n "${RECOMPUTE_NUM_LAYERS:-}" ]]; then
      MB_SFT_CLI_OVERRIDES+=("model.recompute_num_layers=${RECOMPUTE_NUM_LAYERS}")
    fi
  fi
  if [[ "$(_mb_sft_bool_enabled "${FINE_GRAINED_ACTIVATION_OFFLOADING:-0}")" == "1" ]]; then
    export NVTE_CPU_OFFLOAD_V1=1
    MB_SFT_CLI_OVERRIDES+=("model.offload_modules=$(_mb_sft_modules_cli "${OFFLOAD_MODULES}")")
  fi

  if [[ -n "${OPTIMIZER_CPU_OFFLOAD:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.optimizer_cpu_offload=$(_mb_sft_bool_cli "${OPTIMIZER_CPU_OFFLOAD}")")
  elif ! _mb_sft_is_peft_recipe; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.optimizer_cpu_offload=$(_mb_sft_bool_cli "${OPTIMIZER_CPU_OFFLOAD:-false}")")
  fi
  MB_SFT_CLI_OVERRIDES+=(
    "dataset.trust_remote_code=True"
    "dataset.pack_sequences_in_batch=False"
    "dataset.capacity_pack_sequences=${_packed}"
    "validation.eval_interval=${VAL_EVERY_STEPS:-50}"
    "validation.eval_iters=${EVAL_ITERS:-10}"
    "logger.log_interval=${LOG_INTERVAL:-1}"
    "logger.wandb_project=${WANDB_PROJECT:-megatron-bridge-${MODEL_NAME}}"
    "logger.wandb_exp_name=${WANDB_NAME:-${MODEL_NAME}$(_mb_sft_is_peft_recipe && echo _lora || echo _sft)}"
    "logger.wandb_save_dir=${WANDB_DIR}"
  )

  if _mb_sft_is_peft_recipe; then
    if [[ "$(_mb_sft_bool_enabled "${RESUME_CHECKPOINT:-0}")" == "1" ]]; then
      MB_SFT_CLI_OVERRIDES+=("checkpoint.load=${CHECKPOINT_DIR}")
    fi
  else
    MB_SFT_CLI_OVERRIDES+=("checkpoint.load=${CHECKPOINT_DIR}")
  fi

  if [[ "${_gc_manual}" == "true" ]]; then
    MB_SFT_CLI_OVERRIDES+=("train.manual_gc_interval=${GC_EVERY_STEPS}")
  fi
  if [[ -n "${OPTIMIZER_LR:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.lr=${OPTIMIZER_LR}")
  fi
  if [[ -n "${OPTIMIZER_WEIGHT_DECAY:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=(
      "optimizer.weight_decay=${OPTIMIZER_WEIGHT_DECAY}"
      "scheduler.start_weight_decay=${OPTIMIZER_WEIGHT_DECAY}"
      "scheduler.end_weight_decay=${OPTIMIZER_WEIGHT_DECAY}"
      "scheduler.weight_decay_incr_style=constant"
    )
  fi
  if [[ -n "${OPTIMIZER_ADAM_BETA1:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.adam_beta1=${OPTIMIZER_ADAM_BETA1}")
  fi
  if [[ -n "${OPTIMIZER_ADAM_BETA2:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.adam_beta2=${OPTIMIZER_ADAM_BETA2}")
  fi
  if [[ -n "${OPTIMIZER_OFFLOAD_FRACTION:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}")
  fi
  if [[ -n "${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.overlap_cpu_optimizer_d2h_h2d=$(_mb_sft_bool_cli "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}")")
  fi
  if [[ -n "${USE_PRECISION_AWARE_OPTIMIZER:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.use_precision_aware_optimizer=$(_mb_sft_bool_cli "${USE_PRECISION_AWARE_OPTIMIZER}")")
  fi
  if [[ -n "${SAVE_OPTIM:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("checkpoint.save_optim=$(_mb_sft_bool_cli "${SAVE_OPTIM}")")
  fi
  # peft.* → LORA_* env applied in run_recipe_avlm.py (OmegaConf struct excludes PEFT object)
  if [[ -n "${CLIP_GRAD_MAX_NORM:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.clip_grad=${CLIP_GRAD_MAX_NORM}")
  fi
  if [[ -n "${SCHEDULER_LR_DECAY_STYLE:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("scheduler.lr_decay_style=${SCHEDULER_LR_DECAY_STYLE}")
  fi
  if [[ -n "${SCHEDULER_MIN_LR:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("optimizer.min_lr=${SCHEDULER_MIN_LR}")
  fi
  if [[ -n "${SCHEDULER_LR_WARMUP_ITERS:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("scheduler.lr_warmup_iters=${SCHEDULER_LR_WARMUP_ITERS}")
  fi
  if [[ -n "${SCHEDULER_LR_DECAY_ITERS:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("scheduler.lr_decay_iters=${SCHEDULER_LR_DECAY_ITERS}")
  fi
  if [[ -n "${RNG_SEED:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=(
      "rng.seed=${RNG_SEED}"
      "dataset.rng_seed=${RNG_SEED}"
    )
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("logger.wandb_entity=${WANDB_ENTITY}")
  fi
  # wandb.mode → WANDB_MODE env (Bridge has no logger.wandb_mode); exported in mb_sft_load_recipe_env.

  # Dataset paths loaded from the recipe.
  if [[ -n "${TRAIN_JSONL:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.train_jsonl=${TRAIN_JSONL}")
  fi
  if [[ -n "${VAL_JSONL:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.val_jsonl=${VAL_JSONL}")
  fi
  if [[ -n "${TRAIN_JSONL_FORMAT:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.train_jsonl_format=${TRAIN_JSONL_FORMAT}")
  fi
  if [[ -n "${VAL_JSONL_FORMAT:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.val_jsonl_format=${VAL_JSONL_FORMAT}")
  fi
  if [[ -n "${VAL_VIDEO_ROOT:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.val_video_root=${VAL_VIDEO_ROOT}")
  fi
  if [[ -n "${VAL_MAX_VIDEO_FRAMES:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.val_max_video_frames=${VAL_MAX_VIDEO_FRAMES}")
  fi
  if [[ -n "${VAL_VIDEO_SAMPLE_FPS:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.val_video_sample_fps=${VAL_VIDEO_SAMPLE_FPS}")
  fi
  if [[ -n "${TRAIN_SAMPLE_RATIO:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.train_sample_ratio=${TRAIN_SAMPLE_RATIO}")
  fi
  if [[ -n "${VIDEO_METADATA_PATH:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.video_metadata_path=${VIDEO_METADATA_PATH}")
  fi
  if [[ -n "${PACKING_RATIO:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.packing_ratio=${PACKING_RATIO}")
  fi
  if [[ -n "${VAL_SAMPLE_RATIO:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.val_sample_ratio=${VAL_SAMPLE_RATIO}")
  fi
  if [[ -n "${VIDEO_ROOT:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.video_root=${VIDEO_ROOT}")
  fi
  if [[ -n "${MAX_VIDEO_FRAMES:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.max_video_frames=${MAX_VIDEO_FRAMES}")
  fi
  if [[ -n "${VIDEO_SAMPLE_FPS:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.video_sample_fps=${VIDEO_SAMPLE_FPS}")
  fi
  if [[ -n "${DATALOADER_NUM_WORKERS:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.num_workers=${DATALOADER_NUM_WORKERS}")
  fi
  if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.max_train_samples=${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
    MB_SFT_CLI_OVERRIDES+=("dataset.max_val_samples=${MAX_VAL_SAMPLES}")
  fi

  local _before="${EXIT_MINS_BEFORE_LIMIT:-10}" _exit_mins
  if [[ "${_before}" =~ ^[0-9]+$ && "${_before}" -gt 0 \
    && -n "${SLURM_TIMELIMIT:-}" && "${SLURM_TIMELIMIT}" =~ ^[0-9]+$ ]]; then
    _exit_mins=$((SLURM_TIMELIMIT - _before))
    if [[ "${_exit_mins}" -ge 1 ]]; then
      MB_SFT_CLI_OVERRIDES+=("train.exit_duration_in_mins=${_exit_mins}")
      echo "info: EXIT_MINS_BEFORE_LIMIT=${_before} (SLURM_TIMELIMIT=${SLURM_TIMELIMIT} min)" >&2
    else
      echo "warning: SLURM_TIMELIMIT (${SLURM_TIMELIMIT} min) - EXIT_MINS_BEFORE_LIMIT (${_before}) < 1; graceful exit disabled" >&2
    fi
  fi
}

mb_apply_bridge_sft_patches() {
  : "${MEGATRON_BRIDGE_ROOT:?MEGATRON_BRIDGE_ROOT must be set}"
  : "${RECIPE:?run.recipe missing in recipe YAML (call mb_sft_load_recipe_env first)}"
  local _patch_script="${_MB_ROOT}/training_data_processing/bridge_integration/apply_patches.py"
  [[ -f "${_patch_script}" ]] || {
    echo "error: missing ${_patch_script}" >&2
    return 1
  }
  export MB_BRIDGE_OVERLAY="${MB_BRIDGE_OVERLAY:-${WORKSPACE}/megatron_bridge_overlay}"
  python3 "${_patch_script}" \
    --overlay-dir "${MB_BRIDGE_OVERLAY}" \
    --recipe-name "${RECIPE}"
  # Overlay modules are injected via __path__ in run_recipe_avlm.py (not PYTHONPATH).
  # Putting overlay/megatron/* on PYTHONPATH breaks mb_verify_bridge_imports because
  # megatron.core.optimizer eagerly imports hybrid_optimizer.
  export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  echo "info: AVLM overlay staged at ${MB_BRIDGE_OVERLAY} (injected at train entrypoint)" >&2
}

_mb_sft_exec_torchrun() {
  local _cli="$1"
  local _step_func="$2"
  local -a _torchrun
  if mb_container_env_active; then
    # Container /opt/venv directly (no uv); PYTHONPATH carries decord/imageio-ffmpeg to workers.
    _torchrun=("${_MEGATRON_BRIDGE_CONTAINER_VENV}/bin/python" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}")
  else
    _torchrun=(uv run --no-sync python -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE}")
  fi
  local _nnodes="${SLURM_NNODES:-${num_nodes:-1}}"
  if [[ "${_nnodes}" -gt 1 ]]; then
    : "${MASTER_ADDR:?MASTER_ADDR must be set for multi-node Slurm}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    _torchrun+=(
      --nnodes="${_nnodes}"
      --rdzv_backend=c10d
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
      --rdzv_id="${SLURM_JOB_ID:-${RDZV_ID:-mb_sft}}"
    )
    echo "info: torchrun nnodes=${_nnodes} nproc_per_node=${NPROC_PER_NODE} rdzv=${MASTER_ADDR}:${MASTER_PORT}" >&2
  else
    _torchrun+=(--nnodes=1)
  fi
  "${_torchrun[@]}" \
    "${_MB_ROOT}/training_data_processing/bridge_integration/run_recipe_avlm.py" \
    --recipe "${RECIPE}" \
    --hf_path "${HF_MODEL_ID}" \
    --step_func "${_step_func}" \
    ${_cli} \
    2> >(grep -Ev --line-buffered 'duplicates an ancestor field|Child types should not re-register inherited fields' >&2)
}

mb_sft_export_cpu_memory_env() {
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}

mb_sft_run_with_memory_log() {
  local _cli="$1"
  local _step_func="$2"

  local _monitor_script="${_MB_ROOT}/../../utils/_memory_log_csv.sh"
  [[ -f "${_monitor_script}" ]] || {
    echo "error: missing ${_monitor_script}" >&2
    return 1
  }

  local _monitor_dir="${OUTPUT}/cpu_mem_monitor"
  local _log_file="${MEMORY_LOG_FILE:-${_monitor_dir}/train.log}"
  local _csv_file="${MEMORY_CSV_FILE:-${_monitor_dir}/host_ram.csv}"
  local _monitor_log="${_monitor_dir}/monitor.log"
  local _interval="${MONITOR_INTERVAL:-5}"
  local _train_pid="" _monitor_pid="" _train_exit=0

  mkdir -p "${_monitor_dir}"
  export MONITOR_GPU="${MONITOR_GPU:-1}"
  export MONITOR_PSS="${MONITOR_PSS:-1}"

  echo "info: MEMORY_LOG=1 train log=${_log_file}" >&2
  echo "info: MEMORY_LOG=1 RAM CSV=${_csv_file}" >&2
  echo "info: MEMORY_LOG=1 monitor log=${_monitor_log} (interval=${_interval}s)" >&2

  _mb_sft_memory_log_cleanup() {
    [[ -n "${_monitor_pid}" ]] && kill "${_monitor_pid}" 2>/dev/null || true
    [[ -n "${_train_pid}" ]] && kill "${_train_pid}" 2>/dev/null || true
  }
  trap _mb_sft_memory_log_cleanup INT TERM

  (
    _mb_sft_exec_torchrun "${_cli}" "${_step_func}"
  ) 2>&1 | tee "${_log_file}" &
  _train_pid=$!

  sleep 2
  if ! kill -0 "${_train_pid}" 2>/dev/null; then
    wait "${_train_pid}" || true
    trap - INT TERM
    return 1
  fi

  bash "${_monitor_script}" "${_log_file}" "${_csv_file}" "${_interval}" "${_train_pid}" \
    >>"${_monitor_log}" 2>&1 &
  _monitor_pid=$!
  echo "info: memory monitor pid=${_monitor_pid} (stops when training exits)" >&2

  wait "${_train_pid}" || _train_exit=$?
  kill "${_monitor_pid}" 2>/dev/null || true
  wait "${_monitor_pid}" 2>/dev/null || true
  trap - INT TERM

  echo "info: RAM summary (last 5 samples):" >&2
  tail -5 "${_csv_file}" >&2 || true
  return "${_train_exit}"
}

mb_sft_train() {
  : "${RECIPE:?run.recipe missing in recipe YAML}"

  mb_sync_bridge_env
  mb_ensure_ffmpeg_on_path
  mb_apply_bridge_sft_patches
  mb_verify_bridge_imports
  mb_sft_resolve_output_dirs
  mb_sft_build_cli_overrides

  local _cli="" _key
  for _key in "${MB_SFT_CLI_OVERRIDES[@]}"; do
    _cli="${_cli} ${_key}"
  done

  echo "info: recipe=${RECIPE} hf_path=${HF_MODEL_ID}" >&2
  echo "info: AVLM overlay → run_recipe_avlm.py (nproc=${NPROC_PER_NODE}, nodes=${num_nodes:-1})" >&2

  cd "${MEGATRON_BRIDGE_ROOT}"
  mb_export_runtime_env
  export MEGATRON_BRIDGE_ROOT
  export LORA_DIM LORA_ALPHA LORA_TARGET_MODULES
  export MBRIDGE_VLM_STEP_BASE="${MBRIDGE_VLM_STEP_BASE:-${STEP_FUNC_BASE:-nemotron_omni_step}}"
  local _step_func="${MBRIDGE_VLM_STEP_BASE}"
  echo "info: PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" >&2
  echo "info: step_func=${_step_func}" >&2

  if [[ "$(_mb_sft_bool_enabled "${MEMORY_LOG:-0}")" == "1" ]]; then
    mb_sft_run_with_memory_log "${_cli}" "${_step_func}"
    return $?
  fi

  # Interactive Slurm starts one task; torchrun creates one process per GPU.
  _mb_sft_exec_torchrun "${_cli}" "${_step_func}"
}

mb_sft_init_launch() {
  local _kind="${1:-interactive}"
  local _slurm_dir="${MB_TRAIN_SLURM_DIR:-${_MB_SFT_LIB_DIR}}"
  mb_init_launch "${_slurm_dir}" "${_kind}"
  if [[ "${_kind}" == "interactive" ]]; then
    export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    export WANDB_NAME="${WANDB_NAME:-${MODEL_NAME}_${RUN_TIMESTAMP}}"
  elif [[ "${_kind}" == "local" ]]; then
    mb_sft_export_cpu_memory_env
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  else
    mb_sft_export_cpu_memory_env
    export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    export WANDB_NAME="${WANDB_NAME:-${MODEL_NAME}_${RUN_TIMESTAMP}}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  fi
  setup_cache_dir_env
  mb_sft_load_recipe_env
  finalize_wandb_mode_env "${CONFIG_YAML}"
}
