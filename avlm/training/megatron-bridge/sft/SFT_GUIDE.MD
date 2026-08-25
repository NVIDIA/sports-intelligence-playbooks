# Megatron-Bridge SFT guide

Sports video and audio carry signals—player motion, ball trajectory, timing, commentary—that a base multimodal model has not been specialized for. Unlike LoRA, which updates a small adapter set, SFT updates a large share of the language model, which gives more capacity to fit the task at the cost of higher GPU memory and much larger checkpoints. Vision and audio encoders are often left frozen at first to reduce forgetting of pretrained features; that mix is controlled with the recipe `freeze_*` fields.

This runbook covers full-parameter supervised fine-tuning (SFT) for Nemotron Omni with Megatron-Bridge: checkpoint conversion, launch configuration, video metadata, generic and Slurm training, and optional inference validation. For LoRA, see the [Megatron-Bridge LoRA guide](../lora/LORA_GUIDE.MD).

Run all commands from the **repo root**.

> **Path note:** It is safest to resolve absolute host paths with `realpath` before adding them to a recipe or launch configuration. On some systems, paths contain symlinks—for example, `/home/...` may resolve to a storage path such as `/lustre/...`. Using the resolved path helps ensure it is accessible inside the container.
>
> ```bash
> realpath /path/to/file-or-directory
> ```
>
> Keep explicitly relative fields such as `CONFIG_YAML_REL` relative.

## Provided recipes

| Recipe | Intended use | Packing | Limits | TP / EP |
|--------|--------------|---------|--------|---------|
| [`nemotron_omni_mbridge_sft_singlenode.yaml`](configs/nemotron_omni_mbridge_sft_singlenode.yaml) | Single-node debugging | Disabled | 32 frames / 8,192 tokens | 1 / 8 |
| [`nemotron_omni_mbridge_sft_multinode.yaml`](configs/nemotron_omni_mbridge_sft_multinode.yaml) | Multinode training | Enabled | 128 frames / 21,504 tokens | 1 / 8 |

## Setup

1. **Convert HF → Megatron.** See [MODEL_CONVERSION.MD](../hf_megatron_conversion/MODEL_CONVERSION.MD).

2. **Copy and edit launch templates.** In each launcher directory you use, copy `launch.yaml` to `launch_local.yaml` and edit the fields you need:

   ```bash
   cp <launcher_dir>/launch.yaml <launcher_dir>/launch_local.yaml
   ```

   - [Generic training](#generic-training) — `avlm/training/megatron-bridge/sft/generic/`
   - [Slurm training](#slurm-training) — `avlm/training/megatron-bridge/sft/slurm/`
   - [Optional inference validation](#optional-inference-validation) — `avlm/inference/megatron-bridge/slurm/` (batch only)

3. **Choose and edit the training recipe.** If you move a recipe, update `CONFIG_YAML_REL` in the training `launch_local.yaml` to its new repo-relative path.

   Otherwise, the launch templates use the single-node debugging recipe by default.

   Set the training and validation dataset paths:

   ```yaml
   dataset:
     path_or_dataset: /path/to/train.jsonl
     video_root: /path/to/videos
     video_metadata_path: /path/to/train_video_metadata.json

   validation_dataset:
     path_or_dataset: /path/to/validation.jsonl
     video_root: /path/to/videos
   ```

   See [Precompute clip duration and FPS](#precompute-clip-duration-and-fps) for the training video metadata file.

   Before your first run, search the selected recipe for `REPLACE_ME` and review every marked setting. At minimum, update:

   - `checkpoint.pretrained_checkpoint` — Megatron base checkpoint created by [model conversion](../hf_megatron_conversion/MODEL_CONVERSION.MD)
   - `wandb.entity`, `wandb.project`, and `wandb.name` — only if W&B logging is enabled
   - `inference_validation.data_path` and `inference_validation.cluster_params` — only if [inference validation](#optional-inference-validation) is enabled

4. **Configure Weights & Biases (optional).** Choose the logging mode when launching training:

   - `WANDB_MODE=offline` — store logs locally without an API key
   - `WANDB_MODE=online` — sync logs to W&B
   - `WANDB_MODE=disabled` — disable W&B logging

   To use `online` mode, export your W&B API key before launching training:

   ```bash
   export WANDB_API_KEY="<your-wandb-api-key>"
   ```

   For `online` or `offline` mode, replace `wandb.entity`, `wandb.project`, and `wandb.name` in the selected recipe. Local W&B files are stored under `${OUTPUT}/wandb` for both modes.

   W&B run naming depends on the launch method:

   - Generic training uses `wandb.name` from the training recipe.
   - Slurm interactive and batch training use `<MODEL_NAME>_<timestamp>`.
   - Set `WANDB_NAME` when launching to use a different W&B run name.

## Precompute clip duration and FPS

Packed training estimates each sample's video-token cost before constructing packs. Probing every video when the dataloader starts is slow, so build a sidecar JSON once for the training JSONL. Later packed runs read those cached duration and FPS values instead of opening every video during pack planning. Requires `decord`.

```bash
# Activate a Python environment with `decord` installed before running this command.
python avlm/utils/build_video_metadata.py \
  --dataset_path /path/to/train.jsonl \
  --video_root /path/to/video_root \
  --workers NUM_WORKERS \
  --output-dir /path/to/output_dir
```

Set `dataset.video_metadata_path` to the generated `*_video_metadata.json` file. MBridge validation is always non-packed and does not consume a validation metadata sidecar.

## Generic training

Single-node training when GPUs are already available in a `nemo:26.06` container — no Slurm. Requires `sft/generic/launch_local.yaml`; the supplied single-node recipes expect eight GPUs.

```bash
MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_mbridge_sft_generic_smoke \
  bash avlm/training/megatron-bridge/sft/generic/train_sft.sh
```

## Slurm training

Interactive and batch training use `sft/slurm/launch_local.yaml`.

### Interactive (quick start)

1. From the login node:

   ```bash
   bash avlm/training/megatron-bridge/sft/slurm/interactive/launch_interactive_session.sh
   ```

2. Inside the container:

   ```bash
   bash avlm/training/megatron-bridge/sft/slurm/interactive/train_interactive.sh
   ```

   With overrides:

   ```bash
   MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_mbridge_sft_interactive_smoke \
     bash avlm/training/megatron-bridge/sft/slurm/interactive/train_interactive.sh
   ```

### Batch

Submit unattended and multinode jobs with `sbatch_starter.sh`.

```bash
num_nodes=1 MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_mbridge_sft_1n_smoke \
  bash avlm/training/megatron-bridge/sft/slurm/sbatch/sbatch_starter.sh
```

Dry run (no submission):

```bash
DRY_RUN=1 bash avlm/training/megatron-bridge/sft/slurm/sbatch/sbatch_starter.sh
```

Packed smoke test:

`SEQ_LENGTH=16384` overrides both `model.seq_length` and `dataset.seq_length`; with packing enabled, it is the full pack limit. The recipe's `micro_batch_size` must remain `1`.

```bash
USE_SEQUENCE_PACKING=1 SEQ_LENGTH=16384 MAX_VIDEO_FRAMES=16 GLOBAL_BATCH_SIZE=16 \
  num_nodes=1 MAX_STEPS=2 WANDB_MODE=offline MODEL_NAME=nemotron_omni_mbridge_sft_smoke_pack \
  bash avlm/training/megatron-bridge/sft/slurm/sbatch/sbatch_starter.sh
```

Full multinode training (omit `MAX_STEPS`):

```bash
CONFIG_YAML_REL=avlm/training/megatron-bridge/sft/configs/nemotron_omni_mbridge_sft_multinode.yaml \
  num_nodes=8 WANDB_MODE=online MODEL_NAME=nemotron_omni_mbridge_sft_8n \
  bash avlm/training/megatron-bridge/sft/slurm/sbatch/sbatch_starter.sh
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
  avlm/training/megatron-bridge/sft/slurm/sbatch/sbatch_starter.sh
```

The launcher code comes from the selected commit, while the absolute configuration paths remain outside the generated worktree. Relative paths resolve under `avlm-worktrees/<commit>/` and must exist in that commit.

## Default outputs

The default checkpoint directory includes the node count and the recipe's TP/EP layout.

| Launch | Location |
|--------|----------|
| Slurm | Experiment: `avlm/training/megatron-bridge/sft/slurm/outputs/<MODEL_NAME>/`; checkpoints: `checkpoints_n<N>_pp1_tp<TP>_ep<EP>/`; W&B: `wandb/`; batch logs: `avlm/training/megatron-bridge/sft/slurm/logs/<MODEL_NAME>_<timestamp>/`. |
| Generic | Experiment: `avlm/training/megatron-bridge/sft/generic/outputs/<MODEL_NAME>/`; checkpoints: `checkpoints_n1_pp1_tp<TP>_ep<EP>/`; W&B: `wandb/`. |

## Optional: inference validation

Inference validation is available only for Slurm batch training. It launches inference on completed checkpoints at the configured interval.

Enable it in the training recipe:

```yaml
inference_validation:
  enabled: true
  every_steps: 100  # Must be a multiple of the checkpoint interval.
  inference_config: avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml  # Provided default; replace only to use another inference config.
  data_path: /path/to/validation.jsonl
  cluster_params: /path/to/inference_launch_local.yaml
  # max_samples: 512  # Optional maximum number of validation samples.
  # inference_max_samples_seed: -1  # -1 uses non-random, iterative sampling.
```

Configure `avlm/inference/megatron-bridge/slurm/launch_local.yaml` as described in [Setup](#setup). For inference, see the [Inference guide](../../../inference/README.md).

Logging inference-validation metrics to W&B requires a Python environment with `wandb` installed (`pip install wandb`). Set `WANDB_VENV_PATH` to the virtual environment root; if unset, the metrics script uses base `python3`.

LLM judge inference requires `CLIENT_API_KEY` and additional Python packages; see [LLM Judge Inference](../../../evals/README.MD#llm-judge-inference). If using a virtual environment, set `JUDGE_VENV_PATH` to its root; if unset, the judge scripts use base `python3`.

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export WANDB_VENV_PATH=<PATH_TO_WANDB_VENV>
export CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>
export JUDGE_VENV_PATH=<PATH_TO_JUDGE_VENV>
```

## Resume

To continue an interrupted or finished SFT run, use the same `MODEL_NAME`, checkpoint directory, and topology (`num_nodes`, `tp_size`, and `ep_size`). The default topology-tagged checkpoint directory then resolves to the existing run, and training loads its latest checkpoint. Optimizer state resumes when `checkpoint.save_optim` is `true`.

If you change topology or intend to start a new experiment, use a new `MODEL_NAME` or explicitly select a new checkpoint directory.

## Common overrides

| Variable | Typical use |
|----------|-------------|
| `CONFIG_YAML_REL` / `CONFIG_YAML` | Relative or explicit recipe path; only one is required |
| `MODEL_NAME` | Experiment id used in checkpoint and log paths |
| `num_nodes` | Slurm batch node count |
| `MAX_STEPS` | Cap steps for a smoke test; omit for a full recipe run |
| `SEQ_LENGTH` | Per-sample limit when unpacked; full pack limit when packed |
| `WANDB_MODE` | `offline` / `online` / `disabled` |

> **Recipe-path precedence:** If both `CONFIG_YAML` and `CONFIG_YAML_REL` are set, `CONFIG_YAML` takes precedence.

Full list: [TRAINING_ENV_VARS.MD](TRAINING_ENV_VARS.MD).

## Notes

- **Checkpoint conversion** — HF ↔ Megatron: [MODEL_CONVERSION.MD](../hf_megatron_conversion/MODEL_CONVERSION.MD).
- **Packing** — packed runs require `MICRO_BATCH_SIZE=1`; each micro-batch item is one completed pack.
