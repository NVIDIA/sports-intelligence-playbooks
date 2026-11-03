# Megatron-Bridge: conversion prerequisite and SFT/LoRA differences

Agent checklist for the Megatron-Bridge stack. Full detail: [MODEL_CONVERSION.MD](../../../avlm/training/megatron-bridge/hf_megatron_conversion/MODEL_CONVERSION.MD), [SFT_GUIDE](../../../avlm/training/megatron-bridge/sft/SFT_GUIDE.MD), [LORA_GUIDE](../../../avlm/training/megatron-bridge/lora/LORA_GUIDE.MD).

## 1. Base checkpoint format

The provided recipes set `distributed.use_megatron_fsdp: true` and load `checkpoint.pretrained_checkpoint` in `fsdp_dtensor` format. Only `CONVERSION_DIRECTION: HF_to_MEG_FSDP` produces that. A `torch_dist` checkpoint (`HF_to_MEG`) or raw Hugging Face weights do not load into these recipes.

If the user already has a converted `fsdp_dtensor` directory, skip to step 3.

## 2. Convert HF to Megatron FSDP (one 8-GPU node, container `nemo:26.06.00`)

```bash
C=./avlm/training/megatron-bridge/hf_megatron_conversion
cp $C/launch.yaml $C/launch_local.yaml           # CACHE_DIR, CONTAINER_IMAGE, SINGLE_NODE_PARTITION, SLURM_ACCOUNTS
cp $C/conversion.yaml $C/conversion_local.yaml   # read by convert_interactive.sh
grep -nE '^[^#]*(/path/to|your-[a-z-]+)' $C/launch_local.yaml
```

In `conversion_local.yaml` change `CONVERSION_DIRECTION` to `HF_to_MEG_FSDP` (the template default `HF_to_MEG` writes `torch_dist`, which the recipes cannot load), and set a dedicated `WORKSPACE` (for example `mbridge_conversion`) with `MEGATRON_CKPT_PATH` inside it. Do not use `megatron_bridge_workspace` (the default on some branches): training and inference stage their Megatron-Bridge overlay there, and `CLEAN_WORKSPACE: 1` empties `WORKSPACE` first. Relative paths resolve under `CACHE_DIR`.

Conversion has no batch launcher; it runs inside the interactive container. With a terminal:

```bash
bash $C/launch_interactive_session.sh       # login node: allocates 1 node, opens a shell in the container
bash $C/convert_interactive.sh              # inside the container, from the repo root
```

Without a terminal (agent shell), pipe the command into the session. `srun` reports `Not using a pseudo-terminal, disregarding --pty option`, runs the piped command, and releases the node when input ends:

```bash
echo "bash ${C#./}/convert_interactive.sh; echo CONVERT_EXIT=\$?" \
  | bash $C/launch_interactive_session.sh 2>&1 | tee conversion.log
```

Success ends with `info: conversion complete` and `CONVERT_EXIT=0`. On 8x A100 the import took about 4 minutes and wrote about 62 GB. The output directory is `CACHE_DIR/<MEGATRON_CKPT_PATH>` unless the path is absolute; put that absolute path in the recipe. Hugging Face weights download into `CACHE_DIR` on first use unless `HF_HUB_OFFLINE=1` is set.

## 3. Recipe fields specific to Megatron-Bridge

- `checkpoint.pretrained_checkpoint`: absolute path of the `fsdp_dtensor` directory from step 2. A `PRETRAINED_CHECKPOINT=<dir>` command prefix overrides it for one run.
- `validation_dataset` is always unpacked and has no `video_metadata_path`; only `dataset.video_metadata_path` is required.
- Token limit override is `SEQ_LENGTH` (per sample when unpacked, whole pack when packed); packed runs keep `micro_batch_size: 1`.
- The provided recipes use TP=1, EP=8, so the node must expose 8 GPUs (`world_size=... < required=8` otherwise).
- Framework code comes from the container (`/opt/Megatron-Bridge`); git bootstrap is only for pinning another upstream commit.

## 4. SFT versus LoRA

| | SFT | LoRA |
|---|---|---|
| Launcher dir | `./avlm/training/megatron-bridge/sft` | `./avlm/training/megatron-bridge/lora` |
| Trains | Language model (encoders frozen by recipe `freeze_*`) | Adapters; `LORA_DIM`, `LORA_ALPHA`, `LORA_TARGET_MODULES` override the recipe |
| Resume | Same `MODEL_NAME` and topology; loads the latest checkpoint automatically | Same `MODEL_NAME` and topology, plus `RESUME_CHECKPOINT=1`; a fresh LoRA run ignores existing adapters |
| Checkpoints | `$D/slurm/outputs/<MODEL_NAME>/checkpoints_n<N>_pp1_tp<TP>_ep<EP>/iter_<NNNNNNN>/`, about 180 GB each with optimizer state | Same layout; adapters only, about 0.3 GB |
| After training | `MEG_FSDP_to_HF` export via MODEL_CONVERSION.MD | Adapter export to Hugging Face PEFT: [Convert adapters for AutoModel](../../../avlm/training/megatron-bridge/lora/LORA_GUIDE.MD#convert-adapters-for-automodel) |

Optimizer state resumes only when `checkpoint.save_optim: true`.

## 5. After training

The `serve` and `quantize` skills need a full Hugging Face checkpoint. Each job below uses a GPU: show the command and wait for the user's approval, as for training.

- **SFT:** convert with `MEG_FSDP_to_HF` ([MODEL_CONVERSION.MD](../../../avlm/training/megatron-bridge/hf_megatron_conversion/MODEL_CONVERSION.MD)). In a fresh `conversion_local.yaml` (section 2) set `CONVERSION_DIRECTION: MEG_FSDP_to_HF`, `MEGATRON_CKPT_PATH` (the run's `iter_<NNNNNNN>`) and a new `HF_EXPORT_PATH`, then run it as in section 2:

  ```bash
  C=./avlm/training/megatron-bridge/hf_megatron_conversion
  echo "bash ${C#./}/convert_interactive.sh; echo CONVERT_EXIT=\$?" | bash $C/launch_interactive_session.sh 2>&1 | tee conversion.log
  ```
- **LoRA, merged into a full checkpoint:** the provided recipes write `fsdp_dtensor`, so use [`convert_mbridge_lora_to_automodel_full_fsdp.sh`](../../../avlm/training/megatron-bridge/lora/examples/convert_mbridge_lora_to_automodel_full_fsdp.sh) (one GPU by default); `convert_mbridge_lora_to_automodel_full.sh` is for `torch_dist` checkpoints. Copy the script to a `_local.sh` name and set `FSDP_BASE_CHECKPOINT` (the run's `checkpoint.pretrained_checkpoint`), `LORA_CHECKPOINT` (the run's `iter_<NNNNNNN>`) and a new `HF_OUTPUT`; `HF_MODEL_PATH` (default the reasoning BF16 Hub id) must be the base the run trained from. Run it from the repo root inside the LoRA interactive container:

  ```bash
  E=avlm/training/megatron-bridge/lora/examples
  cp $E/convert_mbridge_lora_to_automodel_full_fsdp.sh $E/convert_mbridge_lora_to_automodel_full_fsdp_local.sh   # then set the paths
  bash avlm/training/megatron-bridge/lora/slurm/interactive/launch_interactive_session.sh
  # inside the container, from the repo root:
  bash $E/convert_mbridge_lora_to_automodel_full_fsdp_local.sh
  ```

   It stops with `error: output already exists` instead of overwriting `HF_OUTPUT`.
- **LoRA, adapter only:** [Convert adapters for AutoModel](../../../avlm/training/megatron-bridge/lora/LORA_GUIDE.MD#convert-adapters-for-automodel) writes a PEFT adapter for batch inference. An adapter cannot be served or quantized.
