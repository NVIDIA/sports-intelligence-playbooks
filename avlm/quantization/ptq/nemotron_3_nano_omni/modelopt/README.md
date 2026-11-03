# Model Optimizer PTQ — Nemotron-3-Nano-Omni

Run these steps from the **repo root**, inside the NeMo container. `quantize.sh` uses **1 GPU**. It reads `configs/quantization_hf_tennis.yaml` unless `QUANTIZATION_PARAMS` points at another recipe. The public reasoning recipe is `configs/quantization_hf.yaml`.

## 1. Start the container

The quantization job relies on the **NeMo** image `nvcr.io/nvidia/nemo:26.06.00`. PyTorch and the Omni stack come from that image. `quantize.sh` installs Model Optimizer on top of it.

Pull the image and open a shell. Run `docker run` from the repo root so the repo is available at `/workspace`:

```bash
docker pull nvcr.io/nvidia/nemo:26.06.00
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/nemo:26.06.00 bash
```

The container sees only the repo. Add `-v <dir>:<dir>` for each checkpoint or data directory outside it, so absolute paths in the recipe resolve inside the container.

On a Slurm cluster, import that same image to squashfs and let `slurm/launch_interactive_session.sh` load it:

```bash
enroot import -o nemo_26_06_00.sqsh \
  docker://nvcr.io/nvidia/nemo:26.06.00
```

```bash
MODEL_OPT=avlm/quantization/ptq/nemotron_3_nano_omni/modelopt
cp "${MODEL_OPT}/slurm/launch.yaml" "${MODEL_OPT}/slurm/launch_local.yaml"  # first time only
```

Set `CONTAINER_IMAGE` in `slurm/launch_local.yaml` to the absolute path of `nemo_26_06_00.sqsh`, set the Slurm account, then start the session:

```bash
bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/slurm/launch_interactive_session.sh
```

`quantize.sh` does not read `launch_local.yaml`. The session may request several GPUs; the quant job still uses one.

## 2. Log in to the Hugging Face Hub

Inside the container:

```bash
read -rs HF_TOKEN && export HF_TOKEN   # paste the token; it is not echoed
```

`quantize.sh` moves `HF_HOME` under the workspace, so a token saved by `hf auth login` is not found; export `HF_TOKEN` in the shell that runs it. This lets the job download the checkpoint and the public calibration set `nemotron_vlm_dataset_v2`.

## 3. Edit the recipe

The default recipe is `configs/quantization_hf_tennis.yaml`. To keep the tracked file unchanged, copy it to `configs/quantization_hf_tennis_local.yaml`, edit the copy, and pass it with `QUANTIZATION_PARAMS`. Pass `configs/quantization_hf.yaml` with `QUANTIZATION_PARAMS` to quantize the public reasoning checkpoint instead. Both files use the same fields.

| Field | Meaning |
| --- | --- |
| `HF_MODEL_ID` | Hub id or a local consolidated checkpoint directory. A Hub id is downloaded to `${WORKSPACE}/base/<repo-name-lowercase>/`. |
| `QUANT_FORMAT` | `fp8` (preset `fp8_fp8_kv`) or `nvfp4` (preset `nvfp4_fp8_mixed_fp8_kv`). A command-line `QUANT_FORMAT=` overrides this. Set `QUANTIZE_CONFIG` only to replace the preset name. |
| `NUM_SAMPLES` | Calibration forwards. MoE needs at least 32. The recipes use 512. Each forward has batch size 1. |
| `QUANT_RUN_LABEL` | Names the export: `${WORKSPACE}/models/${QUANT_RUN_LABEL}-${QUANT_FORMAT}-hf`. |
| `VIDEO_PRUNING_RATE` | EVS video-token pruning during calibration and in the exported config. `0` keeps every token. |
| `WORKSPACE` | Run root. `"@repo/..."` is under the repo. A relative path is under `CACHE_DIR`. |
| `CACHE_DIR` | Scratch directory for downloads. |
| `QUANT_USE_CONTAINER_PKGS` | `1` uses the container Python and installs a small overlay. Leave this at `1` in the NeMo container. |
| `MODELOPT_ROOT` | Model Optimizer checkout. With `MODELOPT_GIT_BOOTSTRAP: 1`, `quantize.sh` clones `MODELOPT_GIT_REF` there when the checkout is missing. |

`QUANT_BACKEND` stays `hf`.

## 4. Calibration data

Calibration runs the model forward and records value ranges so Model Optimizer can choose scales. It does not update the weights.

The recipes calibrate on `NUM_SAMPLES` rows from `nemotron_vlm_dataset_v2`. Those shards download to `${WORKSPACE}/datasets/hub/`.

If you have video from the task you will serve, uncomment the JSONL block in your recipe copy (step 3) and point it at that data. Scales chosen on samples closer to that distribution can fit those clips better.

| Field | Meaning |
| --- | --- |
| `CALIB_DATA_ROOT` | Directory that holds the source JSONL files. Expanded inside `CALIB_JSONL_SOURCES`. |
| `CALIB_JSONL_SOURCES` | JSONL files to sample from (for example train and validation). |
| `VIDEO_ROOT` | Directory that the video paths in those rows are relative to. |
| `CALIB_JSONL` | Where the sampled subset is written. |
| `CALIB_VIDEO_FPS` | Frames per second when the quant job reads a clip. Default 2. |
| `CALIB_MAX_VIDEO_FRAMES` | Maximum frames taken from each clip. Default 128. |
| `QUANT_RUN_LABEL` | Set a label for this run. The commented example is `nemotron-3-nano-omni-sft`. |

Build the subset once. This writes `CALIB_JSONL` and does not quantize:

```bash
python avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/lib/vlm_calib.py \
  --quant-params avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs/quantization_hf_tennis_local.yaml
```

Without `--quant-params` it reads the tracked `configs/quantization_hf_tennis.yaml`.

`quantize.sh` reads that file for both FP8 and NVFP4. Rebuild it only if the sources or `NUM_SAMPLES` change. `--sources` and `--video-root` override the YAML.

## 5. Quantize

Inside the container:

```bash
# Default recipe, FP8
bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh

# Same recipe, NVFP4
bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh hf QUANT_FORMAT=nvfp4

# Public reasoning checkpoint
QUANTIZATION_PARAMS=avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs/quantization_hf.yaml \
  bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh
```

The exported checkpoint is `${WORKSPACE}/models/${QUANT_RUN_LABEL}-${QUANT_FORMAT}-hf`. The log line `info: hf_quant_export_path=<path>` prints that directory.

## 6. Serve

Point vLLM at the export directory (log line `info: hf_quant_export_path=<path>`). Use `quantization=modelopt` for FP8 and `modelopt_fp4` for NVFP4, and `--kv-cache-dtype fp8` for these checkpoints. Server examples are under `avlm/deployment/vLLM/`.
