# AutoModel LoRA guide

Sports video and audio carry signals—player motion, ball trajectory, timing, commentary—that a base multimodal model has not been specialized for. Unlike SFT, which updates a large share of the language model (and typically the multimodal projectors), LoRA trains a small set of adapters while leaving most base weights frozen. That cuts GPU memory and checkpoint size and makes iteration faster, at the cost of less capacity than full-parameter SFT.

This runbook covers LoRA (PEFT) fine-tuning for Nemotron Omni with NeMo AutoModel: launch config, video metadata, generic and Slurm training, and optional inference validation. For full-parameter SFT, see the [AutoModel SFT guide](../sft/SFT_GUIDE.MD).

Run all commands from the **repo root**.

> **Path note:** It is safest to resolve absolute host paths with `realpath` before adding them to a recipe or launch configuration. On some systems, paths contain symlinks—for example, `/home/...` may resolve to a storage path such as `/lustre/...`. Using the resolved path helps ensure it is accessible inside the container.
>
> ```bash
> realpath /path/to/file-or-directory
> ```
>
> Keep explicitly relative fields such as `CONFIG_YAML_REL` relative.

## Setup

1. **Copy and edit launch templates.** In each launcher directory you use, copy `launch.yaml` to `launch_local.yaml` and edit the fields you need:

   ```bash
   cp <launcher_dir>/launch.yaml <launcher_dir>/launch_local.yaml
   ```

   - [Generic training](#generic-training) — `avlm/training/automodel/lora/generic/`
   - [Slurm training](#slurm-training) — `avlm/training/automodel/lora/slurm/`
   - [Optional inference validation](#optional-inference-validation) — `avlm/inference/automodel/slurm/` (batch only)

2. **Choose and edit the recipe** pointed to by `CONFIG_YAML_REL` in the training `launch_local.yaml`:

   - `nemotron_omni_automodel_lora_singlenode.yaml` — single-node debugging with up to 32 video frames, an 8,192-token limit, and sequence packing disabled
   - `nemotron_omni_automodel_lora_multinode.yaml` — multinode training with up to 128 video frames, a 21,504-token limit, and sequence packing enabled

   If you move a recipe, update `CONFIG_YAML_REL` in the training `launch_local.yaml` to its new repo-relative path.

   Otherwise, the launch templates use the single-node debugging recipe by default.

   Set the training and validation dataset paths:

   ```yaml
   dataset:
     path_or_dataset: /path/to/train.jsonl
     video_metadata_path: /path/to/train_video_metadata.json
     video_root: /path/to/videos

   validation_dataset:
     path_or_dataset: /path/to/validation.jsonl
     video_metadata_path: /path/to/validation_video_metadata.json
     video_root: /path/to/videos
   ```

   See [Precompute clip duration and FPS](#precompute-clip-duration-and-fps) for the video metadata files.

   Before your first run, search the selected training recipe for `REPLACE_ME` and review each marked setting. At minimum, update:

   - `wandb.entity`, `wandb.project`, and `wandb.name` — only if W&B logging is enabled (online or offline)
   - `inference_validation.data_path` and `inference_validation.cluster_params` — only if [inference validation](#optional-inference-validation) is enabled

3. **Configure Weights & Biases (optional).** Choose the logging mode when launching training:

   - `WANDB_MODE=offline` — store logs locally without an API key
   - `WANDB_MODE=online` — sync logs to W&B
   - `WANDB_MODE=disabled` — disable W&B logging

   To use `online` mode, export your W&B API key before launching training:

   ```bash
   export WANDB_API_KEY="<your-wandb-api-key>"
   ```

   For `online` or `offline` mode, replace `wandb.entity`, `wandb.project`, and `wandb.name` in the selected training recipe.

   W&B run naming depends on the launch method:

   - Generic training uses `wandb.name` from the training recipe.
   - Slurm interactive and batch training require `MODEL_NAME`, set in `launch_local.yaml` or in the launch command, and override `wandb.name` with `<MODEL_NAME>_<timestamp>`.
   - Set `WANDB_NAME` when launching to use a different W&B run name.

## Precompute clip duration and FPS

Training needs each clip’s duration and FPS for frame sampling and sequence packing. Probing every video when the dataloader starts is slow, so build a small sidecar JSON once per train/val JSONL; later train runs read those cached stats instead of opening every file at startup. Requires `decord`.

```bash
# Activate a Python environment with `decord` installed before running this command.
python avlm/utils/build_video_metadata.py \
  --dataset_path /path/to/{train|validation}.jsonl \
  --video_root /path/to/video_root \
  --workers NUM_WORKERS \
  --output-dir /path/to/output_dir
```

Point the recipe `video_metadata_path` fields at the generated `*_video_metadata.json` files.

## First run

If the Hugging Face model is not cached, prefix the training command with `HF_HUB_OFFLINE=0`. For example, for [interactive training](#interactive-quick-start):

```bash
HF_HUB_OFFLINE=0 \
  bash avlm/training/automodel/lora/slurm/interactive/train_interactive.sh
```

## Generic training

Single-node training when GPUs are already available in `nemo-automodel:26.06` — no Slurm. Requires `lora/generic/launch_local.yaml`.

```bash
MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_lora_generic_smoke \
  bash avlm/training/automodel/lora/generic/train_lora.sh
```

## Slurm training

Interactive and batch share the same recipe and `lora/slurm/launch_local.yaml`.

### Interactive (quick start)

1. From the login node:

   ```bash
   bash avlm/training/automodel/lora/slurm/interactive/launch_interactive_session.sh
   ```

2. Inside the container — default (uses `launch_local.yaml` and the recipe):

   ```bash
   bash avlm/training/automodel/lora/slurm/interactive/train_interactive.sh
   ```

   With overrides:

   ```bash
   MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=my_lora_smoke \
     bash avlm/training/automodel/lora/slurm/interactive/train_interactive.sh
   ```

### Batch

Unattended and multinode jobs via `sbatch_starter.sh`.

```bash
num_nodes=1 MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_lora_1n_smoke \
  bash avlm/training/automodel/lora/slurm/sbatch/sbatch_starter.sh
```

**Dry run (no submission):**

```bash
DRY_RUN=1 bash avlm/training/automodel/lora/slurm/sbatch/sbatch_starter.sh
```

Smoke with sequence packing:

```bash
# `PACK_SIZE=16384` overrides `dataloader.packed_sequence.max_length`, `dataloader.packed_sequence.pack_size`, and `dataloader.packed_sequence.collate_max_length`

USE_SEQUENCE_PACKING=1 PACK_SIZE=16384 MAX_VIDEO_FRAMES=16 GLOBAL_BATCH_SIZE=16 \
  num_nodes=1 MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_lora_smoke_pack \
  bash avlm/training/automodel/lora/slurm/sbatch/sbatch_starter.sh
```

Full training (omit `MAX_STEPS`):

```bash
CONFIG_YAML_REL=avlm/training/automodel/lora/configs/nemotron_omni_automodel_lora_multinode.yaml \
  num_nodes=8 WANDB_MODE=online MODEL_NAME=nemotron_omni_lora_8n \
  bash avlm/training/automodel/lora/slurm/sbatch/sbatch_starter.sh
```

For multi-account racing, `SUBMIT_DETACH=0` polls in the foreground until one job starts.

## Launch from a worktree

Run launcher code from another commit without switching your checkout. To use launch and training configurations outside that commit, pass their absolute paths explicitly.

```bash
AVLM_RUN_COMMIT="<commit>" \
CLUSTER_PARAMS="/absolute/path/to/launch_local.yaml" \
CONFIG_YAML="/absolute/path/to/training_config.yaml" \
num_nodes=8 \
MODEL_NAME="<model-name>" \
bash avlm/launch_from_worktree.sh \
  avlm/training/automodel/lora/slurm/sbatch/sbatch_starter.sh
```

The launcher code comes from the selected commit, while the absolute configuration paths remain outside the generated worktree. Relative paths resolve under `avlm-worktrees/<commit>/` and must exist in that commit.

## Default outputs

| Launch | Location |
|--------|----------|
| Slurm | Adapter checkpoints: `avlm/training/automodel/lora/slurm/outputs/<MODEL_NAME>/checkpoints_<layout>/`. Batch logs: `avlm/training/automodel/lora/slurm/logs/<MODEL_NAME>_<timestamp>/`. |
| Generic | Adapter checkpoints: `avlm/training/automodel/lora/generic/outputs/<MODEL_NAME>/checkpoints_<layout>/`. |

## Optional: inference validation

During batch training, run generation on new adapter checkpoints. Enable it in the recipe:

```yaml
inference_validation:
  enabled: true
  every_steps: 100  # Must be a multiple of the checkpoint interval.
  inference_config: avlm/inference/automodel/configs/default_lora_local.yaml  # Provided default; replace only to use another inference config.
  data_path: /path/to/validation.jsonl  # Inference-validation JSONL path.
  cluster_params: /path/to/inference_launch_local.yaml  # Inference launch_local.yaml path.
  # max_samples: 512  # Optional maximum number of validation samples.
  # inference_max_samples_seed: -1  # Sampling seed; -1 uses non-random, iterative sampling.
```

Configure `avlm/inference/automodel/slurm/launch_local.yaml` (see [Setup](#setup)). For inference, see the [Inference guide](../../../inference/README.md).

Logging inference-validation metrics to W&B requires a Python environment with `wandb` installed (`pip install wandb`). Set `WANDB_VENV_PATH` to the virtual environment root. If unset, the metrics script uses base `python3`.

LLM judge inference requires `CLIENT_API_KEY` and additional Python packages; see [LLM Judge Inference](../../../evals/README.MD#llm-judge-inference). If using a virtual environment, set `JUDGE_VENV_PATH` to the virtual environment root. If unset, the judge script uses base `python3`.

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export WANDB_VENV_PATH=<PATH_TO_WANDB_VENV>
export CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>
export JUDGE_VENV_PATH=<PATH_TO_JUDGE_VENV>
```

## Resume

To continue an interrupted or finished run, launch again with the same `MODEL_NAME` and the same topology (`num_nodes` and recipe `ep_size`). AutoModel finds the latest checkpoint in that experiment's checkpoint directory and restores the LoRA adapter, optimizer, scheduler, and training state.

If you change `num_nodes` or `ep_size`, use a new `MODEL_NAME` (or a new checkpoint dir) rather than resuming into the old layout.

## Common overrides

| Variable | Typical use |
|----------|-------------|
| `CONFIG_YAML_REL` / `CONFIG_YAML` | Relative or explicit recipe path; only one is required |
| `MODEL_NAME` | Experiment id (checkpoint and log paths) |
| `num_nodes` | Slurm batch node count |
| `MAX_STEPS` | Cap steps for a smoke; omit for a full recipe run |
| `WANDB_MODE` | `offline` / `online` / `disabled` |

> **Recipe-path precedence:** If both `CONFIG_YAML` and `CONFIG_YAML_REL` are set, `CONFIG_YAML` takes precedence.
>
> **Interactive session:** Sets both variables. Override with `CONFIG_YAML`, or unset `CONFIG_YAML` to use `CONFIG_YAML_REL`.

Full list: [TRAINING_ENV_VARS.MD](TRAINING_ENV_VARS.MD).

## Notes

See also: [SFT guide](../sft/SFT_GUIDE.MD#notes) — DeepEP, host CPU RAM, and validation packing.

- **DeepEP** — when `dispatcher: deepep`, launch installs and checks the wheel for generic, interactive, and batch. On A100 keep **EP≤8**. See [`wheels/deepep/DEEPEP.MD`](../../../../wheels/deepep/DEEPEP.MD).
- **Host CPU RAM** — packing can be CPU-heavy; lower `GLOBAL_BATCH_SIZE` or cap validation if the host OOMs.
- **Validation** — when packing is enabled, training and validation use the same packed pretokenize path. Override validation packing in the recipe.
