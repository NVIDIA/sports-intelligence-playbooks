# Cosmos3 LoRA guide (transformers + PEFT)

[Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano) is a compact vision-language model that takes video or image frames and text. This recipe adapts it to a sports task with a LoRA on its language model, using transformers and PEFT rather than NeMo AutoModel: the AutoModel recipes in this repository are built around Nemotron Omni's dataset and model classes, and Cosmos3 is a different architecture. The trainer is one Python file; the dataset format is the same conversation JSONL the other recipes use.

Clips are fed as **frames, not video**. `extract_frames.py` writes a fixed number of JPEG frames per clip once, and `train_lora.py` passes them to the chat template as that many image items followed by the question. Serving clients that sample frames themselves (every OpenAI-compatible client that sends `image_url` items) then see exactly the training input. Cosmos3 also accepts video input; this recipe does not use it.

Validated on `nvidia/Cosmos3-Nano` and `nvidia/Cosmos3-Edge` on one 96 GB GPU with a four-way multiple-choice task over short baseball clips (32 frames per clip, 150 steps, learning rate 2e-4, rank 64; on Edge, 8.3k tokens per sample, 18.7 GB of GPU memory and 27 minutes). Run commands from the repo root.

## 1. Build the container

The trainer needs the transformers release that ships Cosmos3; the `nemo-automodel:26.06` image carries an older one. The Dockerfile installs it with `--no-deps` on top of that image so torch, PEFT and CUDA stay as shipped.

```bash
docker build -t cosmos3-lora avlm/training/hf_peft/cosmos3/
```

## 2. Dataset

The conversation JSONL from the [data prep examples](../../../data_prep_example/ONBOARD_NEW_DATASET.md): one user turn with a `video` item (path relative to the video root) and a `text` question, one assistant turn with the answer. Only those three fields are read; `id`, `class` and the category fields pass through untouched.

```json
{"id": "...", "conversation": [
  {"role": "user", "content": [{"type": "video", "path": "game_01/clip_0042.mp4"}, {"type": "text", "text": "What happened in this clip?\n(A) ...\n(B) ..."}]},
  {"role": "assistant", "content": [{"type": "text", "text": "(A) ..."}]}]}
```

## 3. Extract frames

Once per split. Defaults match the Nemotron Omni recipes: decode at 2 fps, resize to 672×384, keep 32 frames sampled uniformly from first to last. Clips that decode to fewer frames keep all of them and are listed at the end. Needs `ffmpeg` on the host (or run it inside the container).

```bash
python avlm/training/hf_peft/cosmos3/extract_frames.py \
  --jsonl /path/to/train.jsonl --video-root /path/to/videos --out /path/to/frames
python avlm/training/hf_peft/cosmos3/extract_frames.py \
  --jsonl /path/to/validation.jsonl --video-root /path/to/videos --out /path/to/frames
```

Frames land in `<out>/<video path without extension>/f00.jpg ...`. To train on the frames a specific client sends at inference, extract them the way that client does (its frame rate, resolution and JPEG settings) into the same layout.

## 4. Train

```bash
docker run --rm --gpus all --ipc=host --shm-size 16g \
  -v "$PWD":/workspace -v /path/to/data:/data -v /path/to/hf_cache:/hf \
  -w /workspace -e HF_HOME=/hf cosmos3-lora \
  env TRAIN_JSONL=/data/train.jsonl VAL_JSONL=/data/validation.jsonl \
      FRAMES_ROOT=/data/frames OUTPUT_DIR=/data/runs/cosmos3_lora \
  bash avlm/training/hf_peft/cosmos3/train_lora.sh
```

`train_lora.sh` wraps `train_lora.py` with the defaults below; set `EXTRA_ARGS` for the other flags, or call the Python script directly. Add `EXTRA_ARGS=--smoke` to build one sample and run one forward pass before a full run.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `nvidia/Cosmos3-Nano` | Hub id or local checkpoint |
| `STEPS` | `150` | Optimizer steps |
| `GLOBAL_BATCH_SIZE` | `8` | Samples per step, as gradient accumulation on one GPU |
| `LR` | `2e-4` | Peak learning rate; cosine decay to `1e-5` |
| `WARMUP` | `10` | Linear warmup steps |
| `RANK` | `64` | LoRA rank; alpha is `2 × rank` |

What the trainer does: LoRA on the language model's attention and MLP projections (`q/k/v/o` and the MLP linears: `gate/up/down` on Cosmos3-Nano, `fc1/fc2` on Cosmos3-Edge) with the vision tower and projector frozen; AdamW with weight decay 0.05; gradient clipping at 1.0; gradient checkpointing; loss on the answer tokens only. Validation scores the whole validation JSONL every 10 steps and at the end. Each step prints and appends a JSON record (loss, gradient norm, learning rate, peak memory, elapsed time, and `val_loss` when computed) to `<OUTPUT_DIR>/train_log.jsonl`.

Learning rate matters more than step count on small datasets: with a few hundred clips, 2e-4 for 150–200 steps learned the task where 5e-4 for 100 steps only learned the answer format.

## 5. Evaluate

`eval_mcq.py` scores a multiple-choice split without a server: for each row it scores every `(X) option` line of the question as the full answer given the frames, predicts the best-scoring letter and compares it with the row's `label`. Run it once without `--adapter` for the base model and once with it; the same container works.

```bash
python avlm/training/hf_peft/cosmos3/eval_mcq.py \
  --jsonl /data/test.jsonl --frames /data/frames --out /data/runs/cosmos3_lora/test_scores.jsonl \
  --model nvidia/Cosmos3-Nano --adapter /data/runs/cosmos3_lora/adapter
```

## 6. Outputs and serving

`<OUTPUT_DIR>/adapter/` is a standard PEFT adapter (`adapter_config.json`, `adapter_model.safetensors`). Load it with `peft.PeftModel.from_pretrained` for transformers inference (as `eval_mcq.py` does). A Cosmos3-Nano adapter can also be served with vLLM next to the base model: flatten the key names first and start the server with `--enable-lora`, as described in [Serve a Cosmos3 LoRA adapter](../../../deployment/vLLM/README.md#serve-a-cosmos3-lora-adapter). vLLM's Cosmos3-Edge model class does not implement LoRA (checked up to main as of November 2026), so an Edge adapter stays on the transformers path.

The adapter is kept as a LoRA rather than merged for serving: vLLM's Cosmos3 loader reads the Hub checkpoint layout, which a transformers `save_pretrained()` does not produce. A merged checkpoint does reload with transformers and reproduces the adapter's scores (verified on Edge), so merging is an option for transformers-only deployments.
