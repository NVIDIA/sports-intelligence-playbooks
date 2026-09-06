# Inference — environment variables

Verified **environment-only overrides** for Nemotron Omni inference launchers
(AutoModel and Megatron-Bridge) and the inference + eval pipeline
and suite scripts. Place overrides before the command. Trailing `KEY=VALUE`
arguments and positional eval configuration are rejected.

## Override precedence

1. Command-prefix environment
2. Run YAML
3. Suite YAML
4. `launch_local.yaml` / inference config YAML
5. Ordinary inherited shell exports

Examples:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh

INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_local.yaml \
num_nodes=2 \
bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

The common, Slurm, and pipeline/suite tables contain
precedence-managed variables. Sections explicitly labeled runtime, plus the
ordinary passthrough section, do not compete with supported YAML defaults.

## Common inference overrides

| Variable | Default / source | Description |
|---|---|---|
| `INFERENCE_CONFIG` | Required | Inference config YAML path |
| `INFERENCE_NAME` | Config `inference_name` | Experiment id (output directory name) |
| `BASE_OUTPUT_DIR` | Config `base_output_dir` | Root directory for inference outputs |
| `DATA_PATH` | Config `data_path` | JSONL dataset to run inference over |
| `MODEL_PATH` | Config `model_path` | Base or SFT model checkpoint path |
| `ADAPTER_PATH` | Config `adapter_path` | LoRA adapter checkpoint path |
| `MAX_INFERENCE_SAMPLES` | Unset (full dataset) | Limit inference to the first N (or a random N, with a seed) dataset records |
| `MAX_INFERENCE_SAMPLES_SEED` | `-1` (disabled; take the first N) | Seed for randomly sampling `MAX_INFERENCE_SAMPLES` records |
| `RESUME` | `0` | Set `1` to resume an interrupted run |
| `CACHE_DIR` | `launch_local.yaml` | Writable root for Hugging Face, Bridge workspaces, and runtime dependencies |
| `CLUSTER_PARAMS` | `launch_local.yaml` | Alternate launcher configuration path |

## Slurm and launch settings

| Variable | Default / source | Description |
|---|---|---|
| `num_nodes` | `launch_local.yaml` | Batch node count |
| `GPUS_PER_NODE` | `launch_local.yaml` | GPUs per Slurm node |
| `NPROC_PER_NODE` | `GPUS_PER_NODE` | Inference processes per node |
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
| `SLURM_TIME_LIMIT` | `4:00:00` | Allocation time limit |
| `RACE_POLL_INTERVAL` | `5` | Seconds between Slurm account-race polls |

## Shared model-loading runtime

| Variable | Default / source | Description |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | Hugging Face offline mode |
| `TRANSFORMERS_OFFLINE` | `1` | Transformers offline mode |
| `LOCAL_FILES_ONLY` | `1` | Restrict model/processor loading to local files |

These settings apply to AutoModel and Megatron-Bridge inference.

## AutoModel runtime

| Variable | Default / source | Description |
|---|---|---|
| `HF_ENABLE_PARALLEL_LOADING` | `true` | Parallelize checkpoint shard loading |
| `HF_PARALLEL_LOADING_WORKERS` | `2` | Worker count for parallel checkpoint loading |
| `AUTOMODEL_CODE_ROOT` | `/opt/Automodel` | AutoModel source tree used for inference |
| `AUTOMODEL_GIT_BOOTSTRAP` | `0` | Set `1` to use a pinned git checkout instead of the container copy |
| `AUTOMODEL_GIT_DIR` | `${CACHE_DIR}/Automodel` | Git checkout directory |
| `AUTOMODEL_GIT_REF` | Repository pin | Git revision used by bootstrap |
| `SKIP_VLM_INFERENCE_DEPS` | `0` | Skip automatic install of VLM runtime deps (`decord`, `imageio-ffmpeg`, `librosa`, …) |
| `AUTOMODEL_EXTRA_PKGS_DIR` | `${CACHE_DIR}/automodel_container_pkgs` | Directory for runtime dependencies missing from the container |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (batch only) | CUDA allocator config; avoids fragmentation OOM on high-resolution video prefill |

## Megatron-Bridge runtime

| Variable | Default / source | Description |
|---|---|---|
| `MEGATRON_BRIDGE_ROOT` | `/opt/Megatron-Bridge` | Megatron-Bridge source tree used for inference |
| `MEGATRON_BRIDGE_GIT_BOOTSTRAP` | `0` | Set `1` to use a pinned git checkout instead of the container copy |
| `MB_BRIDGE_VENV` | Under `MEGATRON_BRIDGE_ROOT` | Alternate project virtual environment (clone/explicit checkout only) |
| `MB_BRIDGE_UV_CACHE` | Under `CACHE_DIR` | Alternate uv cache directory |
| `SKIP_UV_SYNC` | `0` | Skip Megatron-Bridge environment synchronization |
| `MB_BRIDGE_FORCE_UV_SYNC` | `0` | Force environment synchronization |
| `MB_BRIDGE_OVERLAY` | `${CACHE_DIR}/megatron_bridge_workspace/megatron_bridge_overlay` | Bridge recipe overlay directory patched in before inference |
| `IS_FSDP` | Config `is_fsdp` | Override whether the checkpoint uses `fsdp_dtensor` format |
| `FSDP_SHARDING_STRATEGY` | Config `fsdp_sharding_strategy` | Override the Megatron FSDP inference sharding strategy |

## Inference + eval pipeline and suites

| Variable | Default / source | Description |
|---|---|---|
| `INFERENCE_BACKEND` | Required for the `inference` stage | `automodel` or `megatron_bridge` |
| `STAGES` | `inference,mcq,judge,judge_eval,judge_summary` | Comma- or semicolon-separated pipeline stages to run |
| `INFERENCE_DIR` | Derived from `INFERENCE_CONFIG` / `INFERENCE_NAME` | Existing inference output directory; use with eval-only stages |
| `PIPELINE_LOG_DIR` | `avlm/inference/logs/<name>_<timestamp>/` | Pipeline run log directory |
| `PIPELINE_POLL_SECONDS` | `60` | Slurm job status poll interval while the pipeline waits for inference |
| `SUITE_CONFIG` | Required for suites | Suite YAML path |
| `suite_name` | Suite YAML `suite_name` | Overrides the suite name used in output/log paths |
| `VLM_SCORER_CONFIG` | Recipe / judge default | LLM judge configuration path |
| `MAX_LLM_JUDGE_SAMPLES` | Unset (all samples) | Limit LLM judge inference to the first N predictions |
| `JUDGE_VENV_PATH` | Base Python | Python virtual environment root used by judge scripts |

## Ordinary runtime passthrough

Values such as `SLURM_MEM`, `LOGS_DIR`, `JOB_NAME`, `WORKDIR`, credentials,
bootstrap controls, and backend-specific tuning remain ordinary inherited
environment variables. They are not precedence-managed inference overrides.

`AVLM_INFERENCE_EXPLICIT_ENV_KEYS` is an internal names-only provenance marker
used across detached and child-process launches. It is not user configuration.

## Launch from a worktree

| Variable | Default / source | Description |
|---|---|---|
| `AVLM_RUN_COMMIT` | Required | Commit checked out into `avlm-worktrees/<commit>/` |
| `AVLM_BASE_REPO_ROOT` | Worktree path | Base repo root used for container mounts and relative-path resolution |
| `AVLM_USE_WORKTREE_MB_BRIDGE_OVERLAY` | `0` | Set `1` to give each worktree run its own `MB_BRIDGE_OVERLAY` |

The worktree wrapper preserves command-prefix environment precedence while
launching the selected commit. Inference and eval targets reject trailing
`KEY=VALUE` arguments; legitimate structural arguments are preserved.
