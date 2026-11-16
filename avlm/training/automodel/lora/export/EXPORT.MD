# Export an AutoModel LoRA for serving

vLLM serves Hugging Face checkpoints only, so an AutoModel LoRA has to be merged into the
base weights before it can be served (see [vLLM inference](../../../../deployment/vLLM/README.md)).
The two scripts here do that without Megatron-Bridge: one merges the adapter into the BF16
Hub checkpoint, the other carries the merged weights into the published ModelOpt FP8
checkpoint. Both are plain PyTorch + `safetensors` scripts; they run on CPU and need host
RAM for one shard at a time.

Run commands from the repo root.

## Inputs

- **Base snapshot** — the Hub checkpoint the recipe trained on, as a local directory
  (`huggingface-cli download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --local-dir ...`,
  or the snapshot under `HF_HOME/hub/models--.../snapshots/<hash>`).
- **Adapter** — the consolidated adapter directory written by training:
  `<OUTPUT>/checkpoints_<layout>/<epoch_x_step_y>/model/`, holding `adapter_config.json`
  and `adapter_model.safetensors`. `<OUTPUT>/checkpoints_<layout>/LATEST` points at the
  newest step.

## Merge into BF16

```bash
python avlm/training/automodel/lora/export/merge_lora_adapter.py \
  --base /path/to/base_snapshot \
  --adapter /path/to/checkpoints_n1_tp1_pp1_ep1/epoch_4_step_99/model \
  --out /path/to/merged_bf16
```

Every adapted weight becomes `W + (alpha / r) * B @ A` in the base dtype; shards keep their
names and the index, config and processor files are copied, so `merged_bf16` is a complete
checkpoint. Point `model_path` in `avlm/deployment/vLLM/configs/vllm_server_params*.yaml` at it.

The script fails if an adapter key has no matching base weight, if a shape does not match,
or if the number of merged weights differs from the number of adapter pairs.

## FP8 without re-calibration

The ModelOpt FP8 checkpoint quantizes each linear weight per tensor (e4m3 weight plus a
`<weight>_scale`) and keeps a calibrated `input_scale` per layer. A LoRA touches a few
hundred of those weights, so instead of running PTQ again the script copies the FP8
checkpoint and re-quantizes only the touched weights from the merged BF16 checkpoint.
`input_scale`s and every untouched tensor stay as published.

```bash
python avlm/training/automodel/lora/export/requantize_fp8_merged.py \
  --fp8-base /path/to/fp8_snapshot \
  --merged /path/to/merged_bf16 \
  --adapter /path/to/checkpoints_n1_tp1_pp1_ep1/epoch_4_step_99/model \
  --out /path/to/merged_fp8
```

Serve `merged_fp8` with the FP8 server parameters (`vllm_server_params_fp8.yaml`). For a
calibrated quantization of the merged checkpoint, run the
[ModelOpt PTQ recipe](../../../../quantization/ptq/nemotron_3_nano_omni/modelopt/README.md)
on `merged_bf16` instead.

The script fails if a touched weight is not FP8-quantized in the base checkpoint, since
such a layer would need the full PTQ path.
