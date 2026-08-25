# AutoModel LoRA — training environment variables

Verified **environment / CLI overrides** for NeMo AutoModel LoRA launchers. These CLI override variables override the defaults from the recipe YAML file.

## Override precedence

Recipe-backed overrides use this precedence:

1. Command-line environment
2. `launch_local.yaml` / recipe YAML
3. Inherited shell exports

If both `CONFIG_YAML` and `CONFIG_YAML_REL` are set, `CONFIG_YAML` takes precedence.

Command-line examples:

```bash
MAX_STEPS=10 WANDB_MODE=offline \
  bash avlm/training/automodel/lora/generic/train_lora.sh

num_nodes=2 MAX_STEPS=10 \
  bash avlm/training/automodel/lora/slurm/sbatch/sbatch_starter.sh
```

## Training knobs

| Variable | Default / source | Description |
|---|---|---|
| `MODEL_NAME` | `launch_local.yaml` | Experiment name used in checkpoint and log paths |
| `CONFIG_YAML_REL` / `CONFIG_YAML` | `launch_local.yaml` | Recipe path. Set one: `CONFIG_YAML_REL` for a repo-relative path or `CONFIG_YAML` for an explicit path |
| `MAX_STEPS` | Recipe | Maximum training steps |
| `GLOBAL_BATCH_SIZE` | Recipe | Total optimizer batch size |
| `GC_EVERY_STEPS` | Recipe | Python garbage-collection interval |
| `CKPT_EVERY_STEPS` | Recipe | Checkpoint interval |
| `DISPATCHER` | Recipe | MoE dispatcher |
| `WANDB_MODE` | Recipe | `online`, `offline`, or `disabled` |
| `WANDB_NAME` | Slurm: `${MODEL_NAME}_${timestamp}`; generic: recipe | W&B run name |
| `WANDB_RUN_ID` / `WANDB_RESUME` | Unset | W&B run identity and resume behavior |
| `WANDB_DIR` | `${OUTPUT}/wandb` | Local W&B files for online and offline runs |
| `OUTPUT_BASE` | Launcher output directory | Root for experiment outputs |
| `OUTPUT` | `${OUTPUT_BASE}/${MODEL_NAME}` | Experiment output directory |
| `CHECKPOINT_DIR` | Under `OUTPUT` | Checkpoint directory |
| `CKPT_LAYOUT_TAG` | Generated | Tag used in the default checkpoint path |
| `FAKE_BALANCED_GATE` | `0` | Set `1` to force a balanced MoE gate |

## Packing and validation

| Variable | Default / source | Description |
|---|---|---|
| `USE_SEQUENCE_PACKING` | Recipe | `1`/`true` enables packing; `0`/`false` uses per-sample collation |
| `PACK_SIZE` | Recipe | When packing is enabled, overrides `dataloader.packed_sequence.max_length`, `dataloader.packed_sequence.pack_size`, and `dataloader.packed_sequence.collate_max_length` |
| `COLLATE_MAX_LENGTH` | Recipe | Training and validation collation limit when packing is disabled |
| `MAX_VIDEO_FRAMES` | Recipe | Maximum sampled frames for both training and validation in packed and non-packed modes |
| `VAL_EVERY_STEPS` | Recipe | Validation interval |
| `VAL_SAMPLE_RATIO` | Recipe | Fraction of the validation dataset to use |
| `VAL_MAX_PACKS` | Unset | Maximum packed validation sequences |
| `VAL_MAX_BATCHES` | Unset | Maximum validation batches per rank |
| `SKIP_VALIDATION` | `0` | Set `1` to disable validation |

## Grad accumulation

With `local_batch_size=1` (recipe default):

`grad_accum = GLOBAL_BATCH_SIZE / (1 × num_nodes × GPUs_per_node)`

## Runtime and AutoModel source

| Variable | Default / source | Description |
|---|---|---|
| `CACHE_DIR` | `launch_local.yaml` | Writable root for Hugging Face, Triton, and installed runtime dependencies |
| `AUTOMODEL_CODE_ROOT` | `/opt/Automodel` | AutoModel source tree used for training |
| `AUTOMODEL_GIT_BOOTSTRAP` | `0` | Set `1` to use a pinned git checkout instead of the container copy |
| `AUTOMODEL_GIT_DIR` | `${CACHE_DIR}/Automodel` | Git checkout directory |
| `AUTOMODEL_GIT_REF` | Repository pin | Git revision used by bootstrap |
| `HF_HUB_OFFLINE` | `1` | Hugging Face offline mode |

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
| `SUBMIT_DETACH` | `1` | For multi-account racing, `1` polls in the background and `0` polls in the foreground |
| `RACE_POLL_INTERVAL` | `5` | Seconds between Slurm account-race polls |
| `LOGS_DIR` | Launcher log directory | Batch log directory |
| `CLUSTER_PARAMS` | Mode `launch_local.yaml` | Alternate launcher configuration path |
| `DRY_RUN` | `0` | Set `1` to print the resolved batch configuration and exit without submitting |
| `JOB_NAME` | `lora_interactive` | Interactive Slurm job and container name |
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
| `DEEPEP_WHEEL_PROFILE` | Auto-detected | DeepEP wheel profile |
| `SKIP_DEEPEP_INSTALL` | `0` | Skip automatic DeepEP installation |
| `DEEPEP_FORCE_INSTALL` | `0` | Force DeepEP installation |
| `SKIP_DEEPEP_WHEEL_CHECK` | `0` | Skip the DeepEP compatibility check |
| `SKIP_VLM_TRAINING_DEPS` | `0` | Skip VLM runtime dependency installation |
| `AUTOMODEL_EXTRA_PKGS_DIR` | Under `CACHE_DIR` | Directory for runtime dependencies missing from the container |
| `TRAIN_KEEP_CLUSTER_TMPDIR` | `0` | Preserve cluster temporary-directory variables |
| `REQUIRE_CUDA_DEVICES` | `8` | Minimum visible CUDA devices |
| `SKIP_CUDA_DEVICE_CHECK` | `0` | Skip the visible-device check |
| `AVLM_VERBOSE_OMNI_LOGS` | `0` | Set `1` to retain verbose Nemotron Omni logs |
| `INFERENCE_VALIDATION_POLL_SECONDS` | `60` | Checkpoint polling interval |
| `INFERENCE_VALIDATION_MIN_CHECKPOINT_AGE_SECONDS` | `30` | Minimum checkpoint age before inference |
