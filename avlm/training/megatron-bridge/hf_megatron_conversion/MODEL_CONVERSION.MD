# Megatron-Bridge model conversion

This runbook converts full-model checkpoints between Hugging Face and the checkpoint formats used by Megatron-Bridge training. It supports the standard Megatron distributed-optimizer format (`torch_dist`) and Megatron FSDP (`fsdp_dtensor`).

Run all commands from the **repo root**.

> **Path note:** It is safest to resolve absolute host paths with `realpath` before adding them to a conversion or launch configuration. On some systems, paths contain symlinks—for example, `/home/...` may resolve to a storage path such as `/lustre/...`. Using the resolved path helps ensure it is accessible inside the container.
>
> ```bash
> realpath /path/to/file-or-directory
> ```
>
> Paths intentionally relative to `CACHE_DIR` should remain relative.

## Checkpoint formats and conversion directions

The standard Megatron distributed-optimizer and Megatron FSDP checkpoint formats are not interchangeable. Choose the conversion direction that matches the training recipe:

| `CONVERSION_DIRECTION` | Input | Output | Checkpoint format |
|------------------------|-------|--------|-------------------|
| `HF_to_MEG` | `HF_MODEL_ID` | `MEGATRON_CKPT_PATH` | Standard `torch_dist` |
| `MEG_to_HF` | `MEGATRON_CKPT_PATH` | `HF_EXPORT_PATH` | Standard `torch_dist` → HF |
| `HF_to_MEG_FSDP` | `HF_MODEL_ID` | `MEGATRON_CKPT_PATH` | `fsdp_dtensor` |
| `MEG_FSDP_to_HF` | `MEGATRON_CKPT_PATH` | `HF_EXPORT_PATH` | `fsdp_dtensor` → HF |

The provided MBridge [SFT](../sft/configs/nemotron_omni_mbridge_sft_singlenode.yaml) and [LoRA](../lora/configs/nemotron_omni_mbridge_lora_singlenode.yaml) recipes require `HF_to_MEG_FSDP` output in `fsdp_dtensor` format.

See the NVIDIA [checkpointing documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html) and [Megatron FSDP overview](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/megatron-fsdp.html) for format details.

## Setup

1. **Copy and edit the launch template.** Copying the template resets the tracked local launch configuration before you edit it for your environment.

   ```bash
   cp avlm/training/megatron-bridge/hf_megatron_conversion/launch.yaml \
      avlm/training/megatron-bridge/hf_megatron_conversion/launch_local.yaml
   ```

2. **Copy and edit the conversion template.** Copying the template resets the tracked local conversion configuration. Conversion reads `conversion_local.yaml`, not `conversion.yaml`.

   ```bash
   cp avlm/training/megatron-bridge/hf_megatron_conversion/conversion.yaml \
      avlm/training/megatron-bridge/hf_megatron_conversion/conversion_local.yaml
   ```

3. **Replace environment-specific settings.** Search both local files for `REPLACE_ME`. At minimum, update:

   - In `launch_local.yaml`: `GPUS_PER_NODE`, `CACHE_DIR`, `CONTAINER_IMAGE`, `SINGLE_NODE_PARTITION`, and `SLURM_ACCOUNTS`; update `CONTAINER_MOUNT_HOST` if needed.
   - In `conversion_local.yaml`: `CONVERSION_DIRECTION`, `MEGATRON_CKPT_PATH`.
   - Update `HF_EXPORT_PATH` for an HF export or a standard parity check.
   - Update `HF_MODEL_ID` when converting a different model or a local HF checkpoint.

`HF_MODEL_ID` may be a Hugging Face Hub id (`org/model`) or an absolute path to a local HF checkpoint.

## Run conversion

Launch the interactive conversion container:

```bash
bash avlm/training/megatron-bridge/hf_megatron_conversion/launch_interactive_session.sh
```

Inside the container, run the conversion configured in `conversion_local.yaml`:

```bash
bash avlm/training/megatron-bridge/hf_megatron_conversion/convert_interactive.sh
```

Prefix environment variables or trailing `KEY=VALUE` arguments override `conversion_local.yaml`. For example:

```bash
CONVERSION_DIRECTION=HF_to_MEG_FSDP \
MEGATRON_CKPT_PATH=megatron_bridge_workspace/models/nemotron-omni-megatron-fsdp-dtensor \
  bash avlm/training/megatron-bridge/hf_megatron_conversion/convert_interactive.sh
```

If the Hugging Face model is not cached, allow the conversion process to download it:

```bash
HF_HUB_OFFLINE=0 \
  bash avlm/training/megatron-bridge/hf_megatron_conversion/convert_interactive.sh
```

## Standard Megatron conversion (`torch_dist`)

Standard import and export use a single conversion process. TP, EP, ETP, and PP do not change the converted checkpoint; those settings apply only to the optional parity check.

**Hugging Face → standard Megatron:**

```yaml
HF_MODEL_ID: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
CONVERSION_DIRECTION: HF_to_MEG
MEGATRON_CKPT_PATH: megatron_bridge_workspace/models/nemotron-omni-megatron
```

The resulting `torch_dist` checkpoint can initialize a standard distributed-optimizer SFT or LoRA recipe.

**Standard Megatron → Hugging Face:**

```yaml
HF_MODEL_ID: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
CONVERSION_DIRECTION: MEG_to_HF
MEGATRON_CKPT_PATH: /path/to/torch_dist_checkpoint
HF_EXPORT_PATH: megatron_bridge_workspace/models/nemotron-omni-hf-export
```

For export, `HF_MODEL_ID` supplies the model architecture, configuration, tokenizer, and processor. The trained weights come from `MEGATRON_CKPT_PATH`.

## Megatron FSDP conversion (`fsdp_dtensor`)

Megatron FSDP conversion uses the distributed FSDP converter and writes or reads `fsdp_dtensor` checkpoints.

**Hugging Face → Megatron FSDP:**

```yaml
HF_MODEL_ID: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
CONVERSION_DIRECTION: HF_to_MEG_FSDP
MEGATRON_CKPT_PATH: megatron_bridge_workspace/models/nemotron-omni-megatron-fsdp-dtensor
```

The resulting checkpoint can initialize a Megatron FSDP SFT or LoRA recipe.

**Megatron FSDP → Hugging Face:**

```yaml
HF_MODEL_ID: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
CONVERSION_DIRECTION: MEG_FSDP_to_HF
MEGATRON_CKPT_PATH: /path/to/fsdp_dtensor_checkpoint
HF_EXPORT_PATH: megatron_bridge_workspace/models/nemotron-omni-fsdp-hf-export
```

## Paths and workspace cleanup

Relative conversion paths are resolved under `CACHE_DIR`; absolute paths are used unchanged.

For HF → Megatron conversion, `CLEAN_WORKSPACE=1` clears `WORKSPACE` before conversion to avoid reusing stale files. Use a dedicated workspace that contains no source checkpoints or unrelated files. Set `CLEAN_WORKSPACE=0` to keep its existing contents.

Megatron → HF conversion does not clear `WORKSPACE`.

## GPU requirements

| Operation | Processes and GPUs |
|-----------|--------------------|
| Standard import/export | One conversion process; allocate at least one GPU with enough memory for the model. |
| FSDP import/export | `NPROC_PER_NODE` processes; defaults to all `GPUS_PER_NODE` GPUs allocated by the interactive launcher. |
| Standard parity check | `NPROC` processes and GPUs; template default is four. |

For standard parity, `NPROC` must be compatible with the selected TP/EP/ETP/PP layout. With the template defaults (`NPROC=4`, `TP=2`, `EP=2`, `ETP=1`, `PP=1`), the check uses four GPUs.

## Optional standard parity check

Set `RUN_PARITY_CHECK=1` to run an HF → Megatron → HF in-memory round trip after a standard conversion. It compares the converted tensors and validates the architecture conversion path.

On success, the round-trip HF model is written to `HF_EXPORT_PATH`; an existing directory at that path is replaced. `MEGATRON_CKPT_PATH` is not overwritten. `CHECK_OUTPUT_DIR` changes only the temporary staging parent.

`NPROC`, `TP`, `EP`, `ETP`, and `PP` control only this check. They do not affect standard import/export checkpoints and are ignored by FSDP conversion.

## Common overrides

| Variable | Typical use |
|----------|-------------|
| `CONVERSION_DIRECTION` | Select one of the four conversion directions. |
| `HF_MODEL_ID` | Hub id or local HF model path. |
| `MEGATRON_CKPT_PATH` | Megatron input or output checkpoint. |
| `HF_EXPORT_PATH` | HF export or standard-parity output. |
| `CLEAN_WORKSPACE` | `1` clears the conversion workspace before imports; `0` retains it. |
| `RUN_PARITY_CHECK` | `1` enables the standard-only round-trip check. |
| `NPROC_PER_NODE` | FSDP conversion process count; defaults to `GPUS_PER_NODE`. |
| `NPROC`, `TP`, `EP`, `ETP`, `PP` | Standard parity-check layout only. |

Prefix assignments and trailing `KEY=VALUE` arguments take precedence over `conversion_local.yaml`.

## Notes

- For training setup, see the [Megatron-Bridge SFT guide](../sft/SFT_GUIDE.MD) or [Megatron-Bridge LoRA guide](../lora/LORA_GUIDE.MD).
- This workflow converts full-model checkpoints. For LoRA adapter conversion, see [Convert adapters for AutoModel](../lora/LORA_GUIDE.MD#convert-adapters-for-automodel).
