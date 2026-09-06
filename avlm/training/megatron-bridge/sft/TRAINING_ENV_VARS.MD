# Megatron-Bridge SFT — training environment variables

Verified **environment / CLI overrides** for Megatron-Bridge SFT launchers. These CLI override variables override the defaults from the recipe YAML file.

## Override precedence

Recipe-backed overrides use this precedence:

1. Command-line environment
2. `launch_local.yaml` / recipe YAML
3. Inherited shell exports

If both `CONFIG_YAML` and `CONFIG_YAML_REL` are set, `CONFIG_YAML` takes precedence.

Command-line examples:

```bash
MAX_STEPS=10 WANDB_MODE=offline \
  bash avlm/training/megatron-bridge/sft/generic/train_sft.sh

num_nodes=2 MAX_STEPS=10 \
  bash avlm/training/megatron-bridge/sft/slurm/sbatch/sbatch_starter.sh
```

## Training knobs

| Variable | Default / source | Description |
|---|---|---|
| `MODEL_NAME` | `launch_local.yaml` | Experiment name used in checkpoint and log paths |
| `CONFIG_YAML_REL` / `CONFIG_YAML` | `launch_local.yaml` | Recipe path. Set one: `CONFIG_YAML_REL` for a repo-relative path or `CONFIG_YAML` for an explicit path |
| `MAX_STEPS` | Recipe | Maximum training steps |
| `GLOBAL_BATCH_SIZE` | Recipe | Total optimizer batch size |
| `MICRO_BATCH_SIZE` | Recipe | Per-rank micro-batch size; packing requires `1` |
| `GC_EVERY_STEPS` | Recipe | Python garbage-collection interval |
| `CKPT_EVERY_STEPS` | Recipe | Checkpoint interval |
| `PRETRAINED_CHECKPOINT` | Recipe | Megatron checkpoint used to initialize training |
| `SAVE_OPTIM` | Recipe | Save optimizer state with checkpoints |
| `WANDB_MODE` | Recipe | `online`, `offline`, or `disabled` |
| `WANDB_ENTITY` / `WANDB_PROJECT` | Recipe | W&B destination |
| `WANDB_NAME` | Slurm: `${MODEL_NAME}_${timestamp}`; generic: recipe | W&B run name |
| `WANDB_RUN_ID` / `WANDB_RESUME` | Unset | W&B run identity and resume behavior |
| `WANDB_DIR` | `${OUTPUT}/wandb` | Local W&B files for online and offline runs |
| `OUTPUT_BASE` | Launcher output directory | Root for experiment outputs |
| `OUTPUT` | `${OUTPUT_BASE}/${MODEL_NAME}` | Experiment output directory |
| `CHECKPOINT_DIR` | `${OUTPUT}/checkpoints_${CKPT_LAYOUT_TAG}` | Checkpoint directory |
| `CKPT_LAYOUT_TAG` | Generated | `n<N>_pp1_tp<TP>_ep<EP>`, derived from node count and recipe TP/EP |
| `FREEZE_LANGUAGE_MODEL` / `FREEZE_VISION_MODEL` / `FREEZE_VISION_PROJECTION` | Recipe | Freeze model components |
| `FREEZE_SOUND_ENCODER` / `FREEZE_SOUND_PROJECTION` | Recipe | Freeze audio components |

## Packing and validation

| Variable | Default / source | Description |
|---|---|---|
| `USE_SEQUENCE_PACKING` | Recipe | `1`/`true` enables packing; `0`/`false` uses per-sample batching |
| `SEQ_LENGTH` | Recipe | Overrides `model.seq_length` and `dataset.seq_length`: the per-sample limit when unpacked and full pack limit when packed |
| `MAX_VIDEO_FRAMES` | Recipe | Maximum sampled frames for both training and validation in packed and non-packed modes |
| `VAL_MAX_VIDEO_FRAMES` | Training value | Separate validation frame override |
| `VIDEO_SAMPLE_FPS` | Recipe | Training video sampling rate |
| `VAL_VIDEO_SAMPLE_FPS` | Validation recipe or training value | Validation sampling rate; defaults to `VIDEO_SAMPLE_FPS` when unset |
| `VAL_EVERY_STEPS` | Recipe | Validation interval |
| `VAL_SAMPLE_RATIO` | Recipe | Fraction of the validation dataset to use |
| `EVAL_ITERS` | Recipe | Validation iterations; `-1` evaluates the full validation set |
| `DATALOADER_NUM_WORKERS` | Recipe | Data-loader workers per rank |
| `MAX_TRAIN_SAMPLES` / `MAX_VAL_SAMPLES` | Unset | Optional sample limits |

## Runtime and Megatron-Bridge source

| Variable | Default / source | Description |
|---|---|---|
| `CACHE_DIR` | `launch_local.yaml` | Writable root for Hugging Face, Bridge workspaces, and runtime dependencies |
| `MEGATRON_BRIDGE_ROOT` | `/opt/Megatron-Bridge` | Megatron-Bridge source tree used for training |
| `MEGATRON_BRIDGE_GIT_BOOTSTRAP` | `0` | Set `1` to use a pinned git checkout instead of the container copy |
| `MEGATRON_BRIDGE_GIT_DIR` | `${CACHE_DIR}/Megatron-Bridge` | Git checkout directory |
| `MEGATRON_BRIDGE_GIT_REF` | Repository pin | Git revision used by bootstrap |
| `HF_HUB_OFFLINE` | Unset | Set `1` to restrict Hugging Face to cached files |

## Slurm and launch settings

| Variable | Default / source | Description |
|---|---|---|
| `num_nodes` | `launch_local.yaml` | Batch node count |
| `GPUS_PER_NODE` | `launch_local.yaml` | GPUs per Slurm node |
| `NPROC_PER_NODE` | GPU count | Training processes per node |
| `NUM_GPUS` | Auto-detected | Visible GPUs for generic training |
| `CONTAINER_IMAGE` | `launch_local.yaml` | Slurm container image |
| `CONTAINER_MOUNT_HOST` | `launch_local.yaml` | Host filesystem mounted into the container |
| `partition` | Unset | Direct batch partition override |
| `PARTITIONS` | `launch_local.yaml` | Batch partition candidates |
| `FEW_NODES_PARTITIONS` | `launch_local.yaml` | Partition candidates for two- or three-node jobs |
| `SINGLE_NODE_PARTITION` | `launch_local.yaml` | Partition for interactive and one-node jobs |
| `use_exclusive` | `1` | `1` requests exclusive nodes; `0` allows node sharing |
| `SLURM_ACCOUNTS` | `launch_local.yaml` | Accounts for normal submission |
| `SLURM_ACCOUNT_RACE` | `launch_local.yaml` | Enable multi-account racing |
| `SLURM_ACCOUNTS_RACE` | `launch_local.yaml` | Accounts used for racing |
| `SLURM_TIME_LIMIT` | `launch_local.yaml` | Allocation time limit |
| `EXIT_MINS_BEFORE_LIMIT` | `10` | Save and exit before the allocation ends |
| `SUBMIT_DETACH` | `1` | For multi-account racing, `1` polls in the background and `0` polls in the foreground |
| `RACE_POLL_INTERVAL` | `5` | Seconds between Slurm account-race polls |
| `LOGS_DIR` | Launcher log directory | Batch log directory |
| `CLUSTER_PARAMS` | Mode `launch_local.yaml` | Alternate launcher configuration path |
| `DRY_RUN` | `0` | Set `1` to print the resolved batch configuration and exit without submitting |
| `JOB_NAME` | `mb_sft_interactive` | Interactive Slurm job and container name |
| `WORKDIR` | Repository root | Interactive container working directory |

## Inference validation

Inference validation overrides apply to Slurm batch training.

| Variable | Default / source | Description |
|---|---|---|
| `INFERENCE_VALIDATION_ENABLED` | Recipe | Enable inference on new checkpoints |
| `INFERENCE_VALIDATION_EVERY_STEPS` | Recipe | Checkpoint interval for inference |
| `INFERENCE_VALIDATION_INFERENCE_CONFIG` | Recipe | Inference configuration path |
| `INFERENCE_VALIDATION_DATA_PATH` | Recipe | Evaluation dataset path |
| `INFERENCE_VALIDATION_MAX_SAMPLES` | Recipe | Maximum evaluated samples |
| `INFERENCE_VALIDATION_MAX_SAMPLES_SEED` | Recipe | Sampling seed |
| `INFERENCE_VALIDATION_CLUSTER_PARAMS` | Inference launch config | Inference Slurm configuration path |
| `INFERENCE_VALIDATION_NUM_NODES` | Recipe or inference launch config | Inference nodes |
| `INFERENCE_VALIDATION_GPUS_PER_NODE` | Recipe or inference launch config | GPUs per inference node |
| `INFERENCE_VALIDATION_NPROC_PER_NODE` | Recipe or inference launch config | Inference processes per node |
| `VLM_SCORER_CONFIG` | Recipe | Judge configuration path |

## Advanced troubleshooting

| Variable | Default | Description |
|---|---|---|
| `RECOMPUTE_GRANULARITY` / `RECOMPUTE_METHOD` / `RECOMPUTE_NUM_LAYERS` | Recipe | Activation recomputation |
| `RECOMPUTE_MODULES` | Recipe | Modules used for selective recomputation |
| `FINE_GRAINED_ACTIVATION_OFFLOADING` / `OFFLOAD_MODULES` | Recipe | Activation offloading |
| `OPTIMIZER_CPU_OFFLOAD` / `OPTIMIZER_OFFLOAD_FRACTION` | Recipe | Optimizer CPU offloading |
| `MEMORY_LOG` | `0` | Write training output and host-memory samples under `OUTPUT` |
| `SKIP_UV_SYNC` | `0` | Skip Megatron-Bridge environment synchronization |
| `MB_BRIDGE_FORCE_UV_SYNC` | `0` | Force environment synchronization |
| `MB_BRIDGE_VENV` | Under `MEGATRON_BRIDGE_ROOT` | Alternate project virtual environment |
| `MB_BRIDGE_UV_CACHE` | Under `CACHE_DIR` | Alternate uv cache directory |
| `AVLM_LOG_PROCESSED_PROMPT` | `0` | Log processed prompts for data diagnostics |
| `INFERENCE_VALIDATION_POLL_SECONDS` | `60` | Checkpoint polling interval |
| `INFERENCE_VALIDATION_MIN_CHECKPOINT_AGE_SECONDS` | `30` | Minimum checkpoint age before inference |
