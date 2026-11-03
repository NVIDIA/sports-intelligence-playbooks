# vLLM inference

[vLLM](https://github.com/vllm-project/vllm) is an inference engine for serving large models on GPUs. It keeps the model loaded and handles many requests at once by batching them and paging the key-value cache, instead of running one full forward pass per caller. This directory starts that server and sends it chat requests, including video. The server speaks the OpenAI HTTP API, so any client that can call `/v1/chat/completions` can use it.

Run commands from the repo root.

## Container

Use the [NGC vLLM image `26.09-py3`](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/vllm). vLLM is already installed in that image.

```bash
docker pull nvcr.io/nvidia/vllm:26.09-py3
docker run --gpus all -it --rm \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/vllm:26.09-py3 bash
```

On clusters that use enroot, convert the image to a squashfs file with the following command for faster loading.

```bash
enroot import -o vllm_26.09-py3.sqsh \
  docker://nvcr.io/nvidia/vllm:26.09-py3
```

## Start the server

Run these inside the container from the section above. vLLM is already installed at `/usr/bin/python3`. The image cannot decode H.264 video without a Decord-backed decoder; `start` handles that automatically (default `VLLM_INSTALL_EXTRA=1`): it installs into `cache_dir/vllm_extra_venv` when the image or that venv cannot import the default extra set — **decord** (x86_64: `decord==0.6.0`; aarch64: `decord2`), then **`pip install -r client/requirements.txt`**, into `cache_dir/vllm_extra_venv` so you can run the HTTP clients on the same node without a separate venv. Restarts skip pip when the stamp (decord spec + requirements file hash) and imports still match; changing `client/requirements.txt` triggers a refresh on the next `start`.

```bash
bash avlm/deployment/vLLM/vllm_server.sh start
```

Set `VLLM_INSTALL_EXTRA=0` only if you do not need video decode or you manage `PYTHON` yourself.

Once that command reports the server is ready, run the client in the next section. `status` checks whether it is still running. `stop` tears down the tracked `serve.py` process (and its children when possible), any leftover `VLLM::Worker` / `VLLM::Engine` processes, and listeners on `vllm_port` through `vllm_port + 5` (override span with `VLLM_STOP_PORT_SPAN`). On a shared node where other jobs use vLLM, set `VLLM_STOP_KILL_ALL_WORKERS=0` so `stop` does not `pkill` global worker names. Port cleanup tries `fuser`, then `lsof` / `ss`, then a `/proc`-based fallback (works in minimal NGC images without `fuser`).

```bash
bash avlm/deployment/vLLM/vllm_server.sh status
bash avlm/deployment/vLLM/vllm_server.sh stop
```

The server binds `0.0.0.0`. On the same machine the API is `http://127.0.0.1:<vllm_port>/v1`. From another machine use `http://<server-ip>:<vllm_port>/v1` and set `VLLM_BASE_URL` to that address for the client. Logs and the process id are under `.cache/vllm_server/`.

## Client

With `vllm_server.sh start`, client dependencies are also installed in `cache_dir/vllm_extra_venv`. **Activate it before running clients on the same node** — bare container `python3` lacks ffmpeg for AAC.

```bash
source "${CACHE_DIR:-${PWD}/.cache}/vllm_extra_venv/bin/activate"
```

On another host, or without `VLLM_INSTALL_EXTRA`, use a separate environment and install the same file there:

```bash
python3 -m venv .venv-vllm-client && source .venv-vllm-client/bin/activate
pip install -U pip && pip install -r avlm/deployment/vLLM/client/requirements.txt
```

```bash
bash avlm/deployment/vLLM/client/chat_video_completion.sh
```

```bash
VLLM_CHAT_MESSAGE='Custom question' bash avlm/deployment/vLLM/client/chat_video_completion.sh
```

Video with audio (MP4s with AAC or other container audio): extract a 16 kHz WAV with ffmpeg and send `video_url` + `audio_url`. Do not rely on `use_audio_in_video: true` for AAC-in-MP4 on this vLLM build.

```bash
python3 avlm/deployment/vLLM/client/chat_video_with_audio_completion.py \
  --video /path/to/point_with_audio.mp4 \
  --question 'Describe what you hear and see in this tennis point.'
```

One video question over HTTP. Run it from the repo root so `$PWD` is that directory. `<video>` is concatenated with the question with no extra newline.

```bash
curl -sS http://127.0.0.1:12500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF
{
  "model": "nemotron_3_nano_omni",
  "max_tokens": 512,
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "chat_template_kwargs": {"enable_thinking": false},
  "mm_processor_kwargs": {"use_audio_in_video": false},
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video_url",
          "video_url": {
            "url": "file://${PWD}/assets/tennis_demo_video/full_tennis_point.mp4"
          }
        },
        {
          "type": "text",
          "text": "<video>Who served and who won this tennis point?\n(A) Mustermann served and Mustermann won\n(B) Major served and Mustermann won\n(C) Mustermann served and Major won\n(D) Major served and Major won"
        }
      ]
    }
  ]
}
EOF
```

## Differences from Hugging Face

Even when the weights are kept the same, vLLM and Hugging Face inference run different kernels for attention, layer norms, and mixture-of-experts sums. Low-precision math rounds those sums differently, and one rounding step is enough for greedy decoding to pick another token when two candidates are close. With temperature above zero, each stack also draws from its own random stream, and `top_k` and `top_p` break ties differently.

## Server parameters

Edit `configs/vllm_server_params.yaml` (tennis Hub vs consolidated `model_path` — one line commented). Default when `SERVE_CONFIG` is unset. Quant compare: `configs/vllm_server_params_{bf16,fp8,nvfp4}.yaml` — FP8/NVFP4 document **A100 (Marlin)** vs **Blackwell (auto / cutlass MoE)** in comments; enable one block. `vllm_enforce_eager: false` on all three. Override with `SERVE_CONFIG` or `MODEL_PATH`.

`model_path` is either a consolidated checkpoint directory or a Hugging Face model name. `${REPO_ROOT}` expands when the file is loaded. The file you copy starts from `nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis`.

For a LoRA (AutoModel or Megatron-Bridge), merge the adapter into the base weights first, then set `model_path` to that consolidated directory (HF format; vLLM serves Hugging Face format only). For Megatron-Bridge, see the `lora/examples/convert_mbridge_lora_to_automodel_full*.sh` scripts.

| Parameter | Purpose |
| --- | --- |
| `cache_dir` | Directory for the extra-package venv. `${REPO_ROOT}` expands when the file is loaded |
| `served_model_name` | Name clients send in the `model` field |
| `vllm_port` | HTTP port |
| `tensor_parallel_size` | GPUs used by one model copy |
| `data_parallel_size` | Number of full copies. Requests stay on `vllm_port` and vLLM load-balances them. Visible GPUs must equal this times `tensor_parallel_size` |
| `cuda_visible_devices` | GPU indices assigned to that process |
| `max_model_len` | Maximum context length |
| `max_num_seqs` | Maximum concurrent sequences on each copy |
| `max_num_queued_reqs` | Maximum unfinished requests (waiting + running) before HTTP 503 |
| `num_frames` | Number of frames sampled from each video |
| `fps` | Frame sampling rate |
| `video_backend` | How frames are selected and decoded |
| `video_pruning_rate` | Fraction of video tokens dropped before the model |
| `vllm_dtype` | Weight dtype |
| `vllm_kv_cache_memory` | Fixed KV cache size in bytes. Use the same value on BF16 and FP8 YAMLs to compare quantization on weights; omit to let vLLM auto-size KV |
| `vllm_kv_cache_dtype` | KV storage dtype (e.g. `fp8` on ModelOpt checkpoints) |
| `vllm_quantization` | ModelOpt quant mode (e.g. `modelopt`, `modelopt_fp4`) |
| `vllm_linear_backend` | FP4/FP8 linear kernel (`marlin` vs `auto`); arch-dependent |
| `vllm_moe_backend` | MoE override when required. A100 NVFP4: `marlin`. Blackwell NVFP4: `cutlass`. NvFP4: not `triton` |
| `vllm_enforce_eager` | When true, disables CUDA graphs |
| `enable_reasoning` | When true, starts the server with a reasoning parser |
| `max_tokens` | Default completion length for clients |
| `stream` | When true, clients receive a token stream |

`data_parallel_size` times `tensor_parallel_size` is the number of GPUs the server needs. With `pin_serve_gpus: true`, the server uses only `cuda_visible_devices`. An environment variable set before `vllm_server.sh` overrides the YAML for `MODEL_PATH`, the port, the tensor-parallel size, the data-parallel size, and `VLLM_MAX_NUM_QUEUED_REQS`. Restart after editing the YAML.
