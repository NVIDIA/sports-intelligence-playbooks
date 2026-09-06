# Multimodal inference guide

Run inference with a trained multimodal checkpoint over a JSONL dataset. Train a
model first: for full-parameter SFT see the
[AutoModel SFT guide](../training/automodel/sft/SFT_GUIDE.MD) or
[Megatron-Bridge SFT guide](../training/megatron-bridge/sft/SFT_GUIDE.MD); for adapter
fine-tuning see the [AutoModel LoRA guide](../training/automodel/lora/LORA_GUIDE.MD) or
[Megatron-Bridge LoRA guide](../training/megatron-bridge/lora/LORA_GUIDE.MD).

Run all commands from the **repo root**.

## Setup

Copy and edit the launch template for each backend you use:

```bash
cp avlm/inference/automodel/slurm/launch.yaml avlm/inference/automodel/slurm/launch_local.yaml
cp avlm/inference/megatron-bridge/slurm/launch.yaml avlm/inference/megatron-bridge/slurm/launch_local.yaml
```

Copy and edit the inference config template for the backend and training mode you use (examples for SFT):

```bash
cp avlm/inference/automodel/configs/default_sft.yaml avlm/inference/automodel/configs/default_sft_local.yaml
cp avlm/inference/megatron-bridge/configs/default_sft_fsdp.yaml avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml
```

For LoRA inference, use `default_lora.yaml` with AutoModel or `default_lora_fsdp.yaml` with Megatron-Bridge. Point `INFERENCE_CONFIG` at the `*_local.yaml` file you edited.

## Configs

Inference settings are loaded from a YAML config. Tracked `default_*.yaml` files are templates only; runtime configs live in `default_*_local.yaml` (copy from the template and edit).

### AutoModel

```text
avlm/inference/automodel/configs/default_sft.yaml
avlm/inference/automodel/configs/default_lora.yaml
```

### Megatron-Bridge

```text
avlm/inference/megatron-bridge/configs/default_sft_fsdp.yaml
avlm/inference/megatron-bridge/configs/default_lora_fsdp.yaml
```

### Edit the config

Copy the template to `default_*_local.yaml`, then update:

- `data_path` and `video_root` — dataset to run inference over
- `model_path` — base or trained SFT checkpoint
- `adapter_path` — trained adapter checkpoint for LoRA inference
- `inference_name` — output directory name

Runtime overrides can be passed for `INFERENCE_NAME`, `DATA_PATH`,
`MODEL_PATH`, and `ADAPTER_PATH`.
`MAX_INFERENCE_SAMPLES` limits inference to the first N dataset records.

If the HF model or processor is not cached, allow the first run to download it:

```bash
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export LOCAL_FILES_ONLY=0
```

All inference and eval overrides are environment-only and must precede the
command, as shown below. Trailing `KEY=VALUE` arguments and positional eval
configuration are rejected.

Full list of overrides: [INFERENCE_ENV_VARS.MD](INFERENCE_ENV_VARS.MD).

## AutoModel

### Interactive

Start an interactive session:

```bash
bash avlm/inference/automodel/slurm/interactive/launch_interactive_session.sh
```

Run SFT inference inside the container:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh
```

With SFT overrides:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
INFERENCE_NAME=automodel_sft_interactive \
DATA_PATH=/path/to/data.jsonl \
MODEL_PATH=/path/to/model \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh
```

Run LoRA inference:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_lora_local.yaml \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh
```

With LoRA overrides:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_lora_local.yaml \
INFERENCE_NAME=automodel_lora_interactive \
DATA_PATH=/path/to/data.jsonl \
ADAPTER_PATH=/path/to/adapter \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/automodel/slurm/interactive/infer_interactive.sh
```

### Batch

Submit SFT inference:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
num_nodes=2 \
bash avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

With SFT overrides:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
INFERENCE_NAME=automodel_sft_batch \
DATA_PATH=/path/to/data.jsonl \
MODEL_PATH=/path/to/model \
num_nodes=2 \
bash avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

Submit LoRA inference:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_lora_local.yaml \
num_nodes=2 \
bash avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

With LoRA overrides:

```bash
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_lora_local.yaml \
INFERENCE_NAME=automodel_lora_batch \
DATA_PATH=/path/to/data.jsonl \
ADAPTER_PATH=/path/to/adapter \
num_nodes=2 \
bash avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

### Launch from a worktree

Use `avlm/launch_from_worktree.sh` with `AVLM_RUN_COMMIT` to run a launcher from another commit without switching your checkout.

Without overrides:

```bash
AVLM_RUN_COMMIT="<commit>" \
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
bash avlm/launch_from_worktree.sh \
  avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

With overrides:

```bash
AVLM_RUN_COMMIT="<commit>" \
INFERENCE_CONFIG="/absolute/path/to/inference_config.yaml" \
INFERENCE_NAME="<inference-name>" \
MODEL_PATH="/absolute/path/to/model" \
DATA_PATH="/absolute/path/to/dataset.jsonl" \
MAX_INFERENCE_SAMPLES=100 \
bash avlm/launch_from_worktree.sh \
  avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

Absolute config paths are used as supplied. Relative config paths resolve from the worktree and must exist in that commit.

Environment overrides before the worktree command retain their precedence in
the selected commit. Do not place inference overrides after the launcher path;
only a launcher's documented structural arguments are accepted there.

### Resume

Rerun the same SFT or LoRA command with `RESUME=1`. Keep the same config,
inference name, data, model and adapter paths, sample limit and seed, and number
of ranks used by the interrupted run.

```bash
RESUME=1 \
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
num_nodes=2 \
bash avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
```

## Megatron-Bridge

### Interactive

Start an interactive session:

```bash
bash avlm/inference/megatron-bridge/slurm/interactive/launch_interactive_session.sh
```

Run SFT inference inside the container:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/megatron-bridge/slurm/interactive/infer_interactive.sh
```

With SFT overrides:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
INFERENCE_NAME=mbridge_sft_interactive \
DATA_PATH=/path/to/data.jsonl \
MODEL_PATH=/path/to/model \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/megatron-bridge/slurm/interactive/infer_interactive.sh
```

Run LoRA inference:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_lora_fsdp_local.yaml \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/megatron-bridge/slurm/interactive/infer_interactive.sh
```

With LoRA overrides:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_lora_fsdp_local.yaml \
INFERENCE_NAME=mbridge_lora_interactive \
DATA_PATH=/path/to/data.jsonl \
ADAPTER_PATH=/path/to/adapter \
MAX_INFERENCE_SAMPLES=16 \
bash avlm/inference/megatron-bridge/slurm/interactive/infer_interactive.sh
```

### Batch

Submit SFT inference:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
num_nodes=2 \
bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

With SFT overrides:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
INFERENCE_NAME=mbridge_sft_batch \
DATA_PATH=/path/to/data.jsonl \
MODEL_PATH=/path/to/model \
num_nodes=2 \
bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

Submit LoRA inference:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_lora_fsdp_local.yaml \
num_nodes=2 \
bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

With LoRA overrides:

```bash
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_lora_fsdp_local.yaml \
INFERENCE_NAME=mbridge_lora_batch \
DATA_PATH=/path/to/data.jsonl \
ADAPTER_PATH=/path/to/adapter \
num_nodes=2 \
bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

### Launch from a worktree

Use `avlm/launch_from_worktree.sh` with `AVLM_RUN_COMMIT` to run a launcher from another commit without switching your checkout.

Without overrides:

```bash
AVLM_RUN_COMMIT="<commit>" \
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
bash avlm/launch_from_worktree.sh \
  avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

With overrides:

```bash
AVLM_RUN_COMMIT="<commit>" \
INFERENCE_CONFIG="/absolute/path/to/inference_config.yaml" \
INFERENCE_NAME="<inference-name>" \
MODEL_PATH="/absolute/path/to/model" \
DATA_PATH="/absolute/path/to/dataset.jsonl" \
MAX_INFERENCE_SAMPLES=100 \
bash avlm/launch_from_worktree.sh \
  avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

Absolute config paths are used as supplied. Relative config paths resolve from the worktree and must exist in that commit.

Environment overrides before the worktree command retain their precedence in
the selected commit. Do not place inference overrides after the launcher path;
only a launcher's documented structural arguments are accepted there.

### Resume

Rerun the same SFT or LoRA command with `RESUME=1`. Keep the same config,
inference name, data, model and adapter paths, sample limit and seed, and number
of ranks used by the interrupted run.

```bash
RESUME=1 \
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
num_nodes=2 \
bash avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh
```

## Outputs

```text
avlm/inference/outputs/<INFERENCE_NAME>/predictions.jsonl
avlm/inference/outputs/<INFERENCE_NAME>/predictions.json
```

Batch and pipeline logs are written under `avlm/inference/logs/...`.

## Inference + Eval Pipeline

Run inference, MCQ eval, LLM judge, LLM judge eval, and LLM judge summary:

LLM judge stages require `CLIENT_API_KEY`; see [LLM Judge Inference](../evals/README.MD#llm-judge-inference).

```bash
export CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>
```

If using a judge virtual environment, also set:

```bash
export JUDGE_VENV_PATH=/path/to/judge_venv
```

AutoModel:

```bash
INFERENCE_BACKEND=automodel \
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh
```

Megatron-Bridge:

```bash
INFERENCE_BACKEND=megatron_bridge \
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh
```

For LoRA inference, use `default_lora_local.yaml` with AutoModel or `default_lora_fsdp_local.yaml` with Megatron-Bridge.

With AutoModel overrides:

```bash
INFERENCE_BACKEND=automodel \
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
INFERENCE_NAME=automodel_sft_eval \
DATA_PATH=/path/to/data.jsonl \
MODEL_PATH=/path/to/model \
num_nodes=1 \
MAX_INFERENCE_SAMPLES=2000 \
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh
```

With Megatron-Bridge overrides:

```bash
INFERENCE_BACKEND=megatron_bridge \
INFERENCE_CONFIG=avlm/inference/megatron-bridge/configs/default_sft_fsdp_local.yaml \
INFERENCE_NAME=mbridge_sft_eval \
DATA_PATH=/path/to/data.jsonl \
MODEL_PATH=/path/to/model \
num_nodes=2 \
MAX_INFERENCE_SAMPLES=2000 \
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh
```

Choose stages:

```bash
STAGES=mcq,judge,judge_eval \
INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
INFERENCE_NAME=sft_eval_epoch_0_step_849 \
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh
```

or:

```bash
STAGES=mcq,judge,judge_eval \
INFERENCE_DIR=avlm/inference/outputs/sft_eval_epoch_0_step_849 \
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh
```

Stop a running pipeline:

```bash
bash avlm/inference/common/scripts/run_inference_eval_pipeline.sh stop avlm/inference/logs/sft_eval_epoch_0_step_849_20260628_120000
```

`stop <run-dir>` and the pipeline's internal `--run <run-dir>` form are
structural arguments. Configuration arguments are not accepted.

## Inference Eval Suites

Suites run the inference + eval pipeline for multiple inference settings from
one YAML file. Outputs are written under
`avlm/inference/outputs/suites/<suite_name>/...` and logs under
`avlm/inference/logs/suites/<suite_name>/...`.

Copy and edit the suite template before your first run:

```bash
cp avlm/inference/common/suites/test_inference_eval_suite.yaml \
   avlm/inference/common/suites/test_inference_eval_suite_local.yaml
```

```bash
SUITE_CONFIG=avlm/inference/common/suites/test_inference_eval_suite_local.yaml \
bash avlm/inference/common/scripts/run_inference_eval_suite.sh
```

With overrides:

```bash
SUITE_CONFIG=avlm/inference/common/suites/test_inference_eval_suite_local.yaml \
MAX_INFERENCE_SAMPLES=2000 \
num_nodes=4 \
bash avlm/inference/common/scripts/run_inference_eval_suite.sh
```

To resume, comment out completed runs and set `RESUME: 1` on the failed run:

```yaml
runs:
  - INFERENCE_NAME: test_suite_automodel_sft
    INFERENCE_CONFIG: avlm/inference/automodel/configs/default_sft_local.yaml
    INFERENCE_BACKEND: automodel
    RESUME: 1
```

Or pass `RESUME=1` with the suite command:

```bash
RESUME=1 \
num_nodes=2 \
SUITE_CONFIG=avlm/inference/common/suites/test_inference_eval_suite_local.yaml \
bash avlm/inference/common/scripts/run_inference_eval_suite.sh
```

Important note: use the same environment overrides when resuming as the original run.
