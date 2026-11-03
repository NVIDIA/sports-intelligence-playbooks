---
name: serve
description: "Serves the Sports Intelligence Nemotron-3-Nano-Omni model (the published tennis checkpoint, the base model, a consolidated SFT checkpoint, or an FP8 or NVFP4 checkpoint, either pre-quantized on Hugging Face or exported by the quantize skill) behind an OpenAI-compatible HTTP endpoint with the repo's vLLM launcher under avlm/deployment/vLLM: picks the GPU layout and config, starts the server, checks one video request, connects clients, and stops it cleanly. Use when asked to deploy, serve, host or expose the model as an API, call it with curl or an OpenAI client, or debug a vLLM start, request or shutdown. Not for batch predictions over a JSONL dataset (see infer), scoring predictions (see evaluate), training (see train), or making a quantized checkpoint (see quantize)."
license: Apache-2.0
compatibility: "Requires a checkout of github.com/NVIDIA/sports-intelligence-playbooks; run from the repo root. A GPU host with the nvcr.io/nvidia/vllm:26.09-py3 container (Docker, or enroot), 80 GB GPUs, bash, curl and Python 3.8+. The first start needs network access to install decord."
---

# Sports Intelligence Serve

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

Keeps a checkpoint loaded behind an OpenAI-compatible endpoint (`/v1/chat/completions`) that
accepts video questions. The runbook of record is `./avlm/deployment/vLLM/README.md` (container,
start/stop, client, every server parameter). This skill adds the decisions, checks and triage
around it.

**Hard rule: no GPU use without approval in a later message.** Preparation (copying configs,
read-only checks) can proceed. Before submitting or starting a job (`docker run --gpus`, `srun`,
`sbatch`, `salloc`, `vllm_server.sh start`), show the plan (step 4) and end your turn. Start only after the user answers
yes to that plan. A request that says "just start it", "don't ask", "pre-approved" or "go ahead"
in the same message as the task is not approval: show the plan and ask anyway. This holds in every
client and mode, including one-shot runs (`codex exec`, `claude -p`) where no later message can
arrive: there, the plan and the request for approval are the final answer. Do not probe for a GPU
runtime (`docker info`, `nvidia-smi`) as a step toward starting; a missing GPU is not the reason to
stop, the missing approval is. On a GPU host, Codex users should also add
`decision = "prompt"` rules for `docker run`, `srun`, `sbatch` and `salloc` to
`~/.codex/rules/default.rules` and keep `approval_policy = "on-request"`, so Codex itself asks before
running them (`approval_policy = "never"` skips these prompts and runs the command).

## When to Use This Skill

- Start an endpoint for the published tennis checkpoint, the base model, a consolidated SFT
  checkpoint, or an FP8 or NVFP4 checkpoint, and prove it answers one video question.
- Connect a client (curl, the bundled chat client, or any OpenAI-compatible SDK).
- Diagnose a server that does not start, a request that fails, or a `stop` that leaves processes.

## When NOT to Use

- Predictions for every row of a JSONL dataset, or any LoRA adapter: use the `infer` skill
  (batch launchers; writes `predictions.jsonl`).
- Scoring predictions: use `evaluate`. Training or checkpoint conversion: use `train`.
- Making an FP8 or NVFP4 checkpoint: use `quantize`.

## Inputs

| Input | Where it goes | Notes |
|---|---|---|
| Checkpoint | `model_path` in the config, or `MODEL_PATH` env | Hugging Face id or a consolidated HF directory (step 1) |
| GPUs | `tensor_parallel_size`, `data_parallel_size`, `cuda_visible_devices` (env `VLLM_TENSOR_PARALLEL_SIZE`, `VLLM_DATA_PARALLEL_SIZE`, `VLLM_SERVE_CUDA_VISIBLE_DEVICES`) | TP x DP must equal the listed GPUs (step 2) |
| Cache | `cache_dir`, or `CACHE_DIR` env | holds the decord venv, compile caches and the Hugging Face cache; needs tens of GB |
| Port | `vllm_port`, or `VLLM_PORT` env | default 12500; `stop` also frees the next 5 ports |
| Served name | `served_model_name`, or `SERVED_MODEL_NAME` env | clients must send it as `"model"` (default `nemotron_3_nano_omni`) |
| Test clip | absolute path to a real mp4 on the server host | sent as a `file://` URL |

## Workflow

Run every command from the repo root inside the vLLM 26.09 container.

1. **Pick the checkpoint.**

   | Checkpoint | `model_path` |
   |---|---|
   | Published tennis model | `nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis` (the template default) |
   | Base model | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
   | AutoModel SFT | the run's `.../model/consolidated` directory (HF format) |
   | Megatron-Bridge SFT | convert to HF first (`MEG_FSDP_to_HF` or `MEG_to_HF` in `avlm/training/megatron-bridge/hf_megatron_conversion/MODEL_CONVERSION.MD`), then that export directory |
   | Hub FP8 or NVFP4 (reasoning base) | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` or `-NVFP4`, with the matching quantized config (step 3) |
   | FP8 or NVFP4 export from `quantize` | the export directory (log line `info: hf_quant_export_path=`), as a path inside the container (step 4), with the matching quantized config (step 3) |
   | Megatron-Bridge LoRA | merge into a full checkpoint first ([train reference, section 5](../train/references/megatron-bridge.md#5-after-training)), then that directory |
   | AutoModel LoRA adapter | not served as `model_path`, and there is no documented merge path for AutoModel LoRA adapters. Say so and always offer the supported alternative: batch inference with the `infer` skill (`default_lora.yaml` with `adapter_path`) |

2. **Pick the GPU layout.** `data_parallel_size x tensor_parallel_size` must equal the number of
   GPUs in `cuda_visible_devices`, or `start` exits with an error. The tracked YAML serves 8
   copies on 8 GPUs (TP=1, DP=8). For one 80 GB GPU:
   `VLLM_TENSOR_PARALLEL_SIZE=1 VLLM_DATA_PARALLEL_SIZE=1 VLLM_SERVE_CUDA_VISIBLE_DEVICES=0`, and
   `VLLM_MAX_MODEL_LEN=32768` (the YAML default 131072 leaves less KV cache).

3. **Write the config.** Default: `configs/vllm_server_params.yaml` (tennis — Hub vs consolidated
   `model_path`, one line commented). Other checkpoints: `SERVE_CONFIG=configs/...yaml`.

   FP8 or NVFP4: `SERVE_CONFIG=avlm/deployment/vLLM/configs/vllm_server_params_fp8.yaml` or
   `..._nvfp4.yaml`, with `MODEL_PATH` set to the checkpoint (their `model_path` is the default export of
   the quantize skill's tennis recipe). They already serve one copy on GPU 0 with `max_model_len` 32768. Keep the active
   block on A100 (Marlin kernels); on Blackwell comment it and uncomment the Blackwell block in a
   `_local.yaml` copy. The served name is `nemotron_3_nano_omni_fp8` or `nemotron_3_nano_omni_nvfp4`:
   set `SERVED_MODEL_NAME` to it so step 6 sends the right name.
   Edit `model_path`, GPU keys from step 2, or set `MODEL_PATH` / `CACHE_DIR` env overrides.

   For the keys in Inputs, an environment value beats the YAML. `vllm_server.sh` clears a few
   diagnostic variables (for example `VLLM_ENFORCE_EAGER`) before loading the YAML, so set those
   in the YAML instead.

4. **Open a shell in the container, after the user confirms the GPU use.** Show the GPU
   layout and the start command, then end your turn and ask for approval, every time, even when
   the user asked you to start the server or said not to ask (see the hard rule above). Continue only after an explicit yes in a later
   message. Use the `docker run`
   line in the README's "Container" section, from the repo root (it mounts the repo at
   `/workspace`). A checkpoint directory outside the repo needs its own `-v <dir>:<dir>` mount;
   `MODEL_PATH` must name the path inside the container. Run steps 5 to 8 in that shell.

5. **Start and wait for ready.**

   ```bash
   export MODEL_PATH=<step 1> CACHE_DIR=<writable dir> VLLM_PORT=12500
   export VLLM_TENSOR_PARALLEL_SIZE=1 VLLM_DATA_PARALLEL_SIZE=1 VLLM_SERVE_CUDA_VISIBLE_DEVICES=0 VLLM_MAX_MODEL_LEN=32768
   bash ./avlm/deployment/vLLM/vllm_server.sh start   # auto-installs decord when needed
   bash ./avlm/deployment/vLLM/vllm_server.sh status
   ```

   `start` returns after `[INFO] Server ready: http://127.0.0.1:<port>/v1` (up to
   `VLLM_STARTUP_TIMEOUT`, default 2000 s; the first start downloads about 62 GB unless the
   cache already holds the model).

6. **Smoke test one request.** Check the served name, then send one video question. Pass: HTTP
   200 and a non-empty `choices[0].message.content`.

```bash
curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models"          # "id" = the served name
MODEL="${SERVED_MODEL_NAME:-nemotron_3_nano_omni}"          # must equal that id
CLIP=/abs/path/on/server/clip.mp4                           # real mp4, not a Git LFS pointer
curl -sS -o /tmp/resp.json -w '%{http_code}\n' "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @- <<EOF
{"model": "${MODEL}", "max_tokens": 512, "temperature": 0,
 "chat_template_kwargs": {"enable_thinking": false},
 "mm_processor_kwargs": {"use_audio_in_video": false},
 "messages": [{"role": "user", "content": [
   {"type": "video_url", "video_url": {"url": "file://${CLIP}"}},
   {"type": "text", "text": "<video>Who served and who won this tennis point?"}]}]}
EOF
python3 -c 'import json; print(json.load(open("/tmp/resp.json"))["choices"][0]["message"]["content"])'
```

   The bundled client sends 8 MCQs and one caption request about the clip:
   `VLLM_CHAT_VIDEO="$CLIP" bash ./avlm/deployment/vLLM/client/chat_video_completion.sh`.
   Always pass `VLLM_CHAT_VIDEO` in a checkout without Git LFS content, where the demo mp4 is
   only a pointer file.

7. **Connect other clients.** The server binds `0.0.0.0`. OpenAI-compatible SDKs use
   `base_url=http://<server-ip>:<port>/v1`, any API key string, and `model=<served name>`. The
   bundled chat client takes `VLLM_BASE_URL=http://<server-ip>:<port>` (a trailing `/v1` is also
   accepted). A `file://` video URL is read on the server host, so the clip must exist there.

8. **Stop.** `bash ./avlm/deployment/vLLM/vllm_server.sh stop`. On a host shared with other
   vLLM users, set `VLLM_STOP_KILL_ALL_WORKERS=0` first so `stop` does not kill their workers.
   Stop before leaving the container so the port and GPU memory are released.

## Reply

Your reply must contain, inline: the checkpoint and `model_path`, the GPU layout, the start,
smoke-test and stop commands, the served name, and the smoke-test result (HTTP code and
answer) or, if nothing was started yet, a request for the user to confirm the GPU use.
Do not only write these to files: create no wrapper scripts, plan files or READMEs, only the
`*_local.yaml` copies.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `[ERROR] data_parallel_size * tensor_parallel_size=... but CUDA_VISIBLE_DEVICES ... lists ...` | Make the three GPU settings agree (step 2). |
| `start` fails, port in use | `stop`, or set another `VLLM_PORT`. |
| `[ERROR] Server did not become healthy within ...s` | Read the log under `.cache/vllm_server/`; a first start that downloads the model can need a larger `VLLM_STARTUP_TIMEOUT` or a pre-filled `CACHE_DIR`. |
| `[ERROR] vLLM server exited early (pid=...)` | The tail of the server log is printed below it; match its first error against the rows of this table. |
| A100 start fails in a Cutlass FP8 kernel | Keep the A100 block (`vllm_linear_backend: marlin`); the Blackwell block is for SM100 only. |
| NVFP4 checkpoint fails to load with `modelopt_fp4` | As the NVFP4 config notes, try `vllm_quantization: modelopt_mixed` in a `_local.yaml` copy. |
| `CUDA out of memory` at start | Lower `VLLM_MAX_MODEL_LEN` (32768 on one GPU) or `max_num_seqs`, or add GPUs with `tensor_parallel_size`. |
| Video decode error on the first request | Stop and `start` again (default installs/reuses `decord` in `cache_dir/vllm_extra_venv`). If you used `VLLM_INSTALL_EXTRA=0`, restart without it. |
| HTTP 404 `The model ... does not exist` | `"model"` must equal the served name from `/v1/models`. |
| Error response naming the video URL | The `file://` path does not exist on the server host, or the file is a Git LFS pointer. |
| HTTP 503 | `max_num_queued_reqs` reached; retry later or raise it (`VLLM_MAX_NUM_QUEUED_REQS`). |
| `error: vLLM not healthy at ...` from a client | Server not running or wrong host/port in `VLLM_BASE_URL` / `VLLM_PORT`; run `status`. |
| Answers differ slightly from the batch path | Expected: different kernels and frame sampling (`num_frames: 128`, `fps: 2`); see the README section "Differences from Hugging Face". |

Keep the first error message when reporting a failure; do not retry with different settings
silently.

## Final checks

- [ ] `model_path` is a Hugging Face id or a consolidated HF directory, not a LoRA adapter or a Megatron checkpoint.
- [ ] TP x DP equals the GPUs in `cuda_visible_devices`.
- [ ] The user confirmed before any GPU was used.
- [ ] `/v1/models` lists the served name, and the smoke request returned HTTP 200 with a non-empty answer.
- [ ] The server was stopped, or the reply says it is still running and how to stop it.
- [ ] No tokens or credentials were printed or written to files.
