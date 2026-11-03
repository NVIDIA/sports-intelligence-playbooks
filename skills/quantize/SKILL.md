---
name: quantize
description: "Post-training quantizes the Sports Intelligence Nemotron-3-Nano-Omni model (the published tennis checkpoint, the reasoning base, an AutoModel SFT, or a Megatron-Bridge SFT or LoRA run after conversion to Hugging Face) to FP8 or NVFP4 with NVIDIA Model Optimizer through avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh: picks the recipe and calibration data, runs one GPU job after approval, checks the exported checkpoint, and hands it to vLLM. Use when asked to quantize, compress or shrink the model, make an FP8 or NVFP4 checkpoint, calibrate on sports video, check a quantized export, set MODEL_PATH when vLLM cannot find a quantize export, ask how FP8 or NVFP4 compares with BF16 in accuracy, speed or memory, or ask for INT4, GGUF or llama.cpp (not supported; offers FP8 or NVFP4). Not for serving any checkpoint, including the pre-quantized FP8 or NVFP4 models on Hugging Face (see serve), training or LoRA fine-tuning (see train), batch predictions (see infer), or scoring (see evaluate)."
license: Apache-2.0
compatibility: "Requires a checkout of github.com/NVIDIA/sports-intelligence-playbooks; run from the repo root. One 80 GB GPU in the nvcr.io/nvidia/nemo:26.06.00 container (Docker on a workstation, or enroot on a Slurm cluster), Hugging Face Hub access, and about 100 GB of free disk for the BF16 download and the export."
---

# Sports Intelligence Quantize

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

Writes a smaller FP8 or NVFP4 copy of a full Hugging Face checkpoint. A calibration pass picks the
scales; the weights are not retrained. The runbook of record is
[the Model Optimizer README](../../avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/README.md)
(container, recipe fields, calibration). This skill adds the decisions, the approval gate, the
export check and the handoff to serving.

**Hard rule: no GPU job without approval in a later message.** Preparation (copying configs,
read-only checks) can proceed. Before submitting or starting a job (`docker run --gpus`, `srun`,
`sbatch`, `salloc`, `quantize.sh`, a merge or conversion), show the plan (step 4) and end your
turn. Run it only after the user answers yes to that plan. A request that says "just run it",
"don't ask", "pre-approved" or "go ahead" in the same message as the task is not approval: show
the plan and ask anyway. This holds in every client and mode, including one-shot runs
(`codex exec`, `claude -p`) where no later message can arrive: there, the plan and the request for
approval are the final answer. Do not probe for a GPU runtime (`docker info`, `nvidia-smi`) as a
step toward running; a missing GPU is not the reason to stop, the missing approval is. On a GPU
host, Codex users should also add `decision = "prompt"` rules for `docker run`, `srun`, `sbatch`
and `salloc` to `~/.codex/rules/default.rules` and keep `approval_policy = "on-request"`, so Codex
itself asks before running them (`approval_policy = "never"` skips these prompts and runs the command).

## When to Use This Skill

- Quantize the published tennis model, the reasoning base, or a fine-tuned checkpoint to FP8 or NVFP4.
- Calibrate on your own sports clips instead of the default public calibration set.
- Check that an export finished and has the requested format before serving it.

## When NOT to Use

- Serving any checkpoint, including `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` or `-NVFP4`
  from the Hub, or an export this skill wrote: use `serve`.
- An **AutoModel LoRA adapter**: the repo has no path to merge it into a full checkpoint, so it
  cannot be quantized. Say so and offer the alternatives: batch inference with the adapter
  (`infer`), or quantizing the tennis or base model.
- Megatron-Bridge quantization, GGUF, quantization-aware training, INT4 or INT8: not in the repo.
  Say so and offer FP8 or NVFP4; do not give steps for tools outside the repo.
- Accuracy, latency or memory numbers for FP8 or NVFP4: the repo publishes none. Do not estimate
  them. Latency can be measured by serving each checkpoint (`serve`) and running
  `avlm/deployment/vLLM/client/stress_video_completion.py`; the batch `infer` backends are not
  documented for quantized checkpoints, so there is no accuracy path yet; do not propose one.
- Uploading a checkpoint to the Hugging Face Hub: refuse, even when asked. Do not give upload
  commands, ask for a destination repository, or offer to upload later.

## Inputs

| Input | Where it goes | Notes |
|---|---|---|
| Checkpoint | `HF_MODEL_ID` in the recipe | Hub id, absolute path, or `"@repo/..."`; must be a full HF checkpoint (step 1) |
| Format | `QUANT_FORMAT=fp8` or `nvfp4` on the command line | One approved job per format |
| Calibration | default Hub set, or the JSONL block in the recipe | step 3 |
| Run label | `QUANT_RUN_LABEL` | names the export `${WORKSPACE}/models/<label>-<format>-hf` |
| Environment | workstation with Docker, or Slurm with `nemo_26_06_00.sqsh` | step 5 |
| Serving GPU | A100 or Blackwell | picks the serve config block in step 7 |
| Hub access | `HF_TOKEN` exported in the shell that runs `quantize.sh` | `quantize.sh` moves `HF_HOME`, so a saved `hf auth login` is not found; never print the token |

## Workflow

Run every command from the repo root.

1. **Get a full Hugging Face checkpoint.**

   | Training output | What to pass as `HF_MODEL_ID` |
   |---|---|
   | Published tennis model (default) | `nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis` |
   | Reasoning base | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (recipe `quantization_hf.yaml`) |
   | AutoModel SFT | the run's `.../model/consolidated` directory |
   | Megatron-Bridge SFT | convert first with `MEG_FSDP_to_HF` (`MEG_to_HF` for a `torch_dist` run; [conversion guide](../../avlm/training/megatron-bridge/hf_megatron_conversion/MODEL_CONVERSION.MD)), then the `HF_EXPORT_PATH` directory |
   | Megatron-Bridge LoRA | merge first ([train reference, section 5](../train/references/megatron-bridge.md#5-after-training)), then the merged directory |
   | AutoModel LoRA | not supported (see When NOT to Use) |

   A conversion or merge uses a GPU: apply the gate in step 4 to it as a separate job, and show
   the exact commands from the train reference, section 5. Check a
   local checkpoint with `ls <dir>`: it needs `config.json` and `*.safetensors` shards, and an
   `adapter_config.json` means it is an adapter, not a full checkpoint. If a path the user gave
   does not exist, say so and do not search the file system for it. Still give the full plan and
   commands with that path as a `<placeholder>`, then ask only for the missing values.

2. **Copy the recipe.** Tennis, AutoModel SFT and Megatron-Bridge checkpoints use
   `quantization_hf_tennis.yaml`; the reasoning base uses `quantization_hf.yaml`.

   ```bash
   C=avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs
   R=quantization_hf_tennis        # R=quantization_hf for the reasoning base
   cp $C/$R.yaml $C/${R}_local.yaml
   ```

   Later commands name this copy in full as `<recipe>_local.yaml`, where `<recipe>` is the `R`
   chosen here, so they do not depend on this shell.

   In the copy set `HF_MODEL_ID`. Keep the default `QUANT_RUN_LABEL` for the published tennis
   model, so the tracked serve configs find the export; set a new label for any other checkpoint
   and pass `MODEL_PATH` in step 7. Leave
   `QUANT_BACKEND: hf` and `QUANT_USE_CONTAINER_PKGS: 1`. `NUM_SAMPLES` must stay at least 32
   (the recipes use 512).

3. **Optional: calibrate on your own clips.** Only the tennis recipe has the JSONL block, so use
   `R=quantization_hf_tennis` in step 2 for any checkpoint you calibrate on your own clips.
   Uncomment the JSONL block in the copy and set
   `CALIB_DATA_ROOT`, `CALIB_JSONL_SOURCES`, `VIDEO_ROOT` and `CALIB_JSONL`. Build the subset once,
   inside the NeMo container (it imports `torch`, uses no GPU and does not quantize). On a
   workstation, with the same mounts as step 5 and no `--gpus`:

   ```bash
   docker run --rm -v "$PWD":/workspace -w /workspace -v <data dir>:<data dir> \
     nvcr.io/nvidia/nemo:26.06.00 /opt/venv/bin/python \
     avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/lib/vlm_calib.py \
     --quant-params avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs/quantization_hf_tennis_local.yaml
   ```

   In a Slurm session (step 5), from the repo root:

   ```bash
   /opt/venv/bin/python avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/lib/vlm_calib.py \
     --quant-params avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs/quantization_hf_tennis_local.yaml
   ```

   Without the block, calibration downloads `nemotron_vlm_dataset_v2` shards.

4. **Stop and ask before using a GPU.** Show the user, for each GPU job:

   - recipe file, `HF_MODEL_ID`, format, calibration source;
   - export path `${WORKSPACE}/models/<label>-<format>-hf`;
   - disk: a Hub id downloads about 62 GB to `${WORKSPACE}/base/`, a local checkpoint downloads
     nothing, both need room for the export;
   - environment: Docker (the agent runs it after approval; always write the line
     `Extra mounts:` with the `-v` mounts, or `none`) or Slurm (the user runs the session); if unknown, give both and ask which;
   - the exact command from step 5;
   - the step 7 serve command for later, and that serving has its own approval.

   Then end your turn and ask for approval, every
   time, even when the user gave every value, asked you to run it, or said not to ask (see the
   hard rule above). Continue only after an explicit yes in a later message. FP8 and NVFP4 are
   two jobs and need two approvals.

5. **Quantize.** The job uses one GPU.

   Workstation (Docker). The container sees only the repo; mount every absolute path outside it
   (checkpoint, `CALIB_DATA_ROOT`, `VIDEO_ROOT`) at the same path with `-v <dir>:<dir>`, and drop
   that line for a Hub id:

   ```bash
   docker run --gpus '"device=0"' --rm -e HF_TOKEN -v "$PWD":/workspace -w /workspace \
     -v <checkpoint dir>:<checkpoint dir> \
     nvcr.io/nvidia/nemo:26.06.00 bash -c \
     "QUANTIZATION_PARAMS=avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs/<recipe>_local.yaml \
      bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh hf QUANT_FORMAT=fp8"
   ```

   Slurm: the session needs a terminal (`srun --pty`), so the user runs it. Copy
   `modelopt/slurm/launch.yaml` to `launch_local.yaml` (it is tracked: show the user the `git diff` of
   your edits in the plan, and never commit them) and set `CONTAINER_IMAGE`,
   `SLURM_ACCOUNTS`, `SINGLE_NODE_PARTITION` (the interactive session uses only this key) and
   `GPUS_PER_NODE: 1` (the template asks for 8; the job uses one). `CONTAINER_MOUNT_HOST`
   (default `/lustre`) makes paths under it visible. Then give the user these commands:

   ```bash
   bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/slurm/launch_interactive_session.sh
   # inside the container, from the repo root:
   read -rs HF_TOKEN && export HF_TOKEN     # paste the token; it is not echoed
   QUANTIZATION_PARAMS=avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/configs/<recipe>_local.yaml \
     bash avlm/quantization/ptq/nemotron_3_nano_omni/modelopt/quantize.sh hf QUANT_FORMAT=fp8
   ```

   The export directory is in the log line `info: hf_quant_export_path=<path>` (and again in
   `info: deploy with vLLM — model_path=<path>`). Under Docker it starts with `/workspace/`;
   on the host replace that prefix with the repo root.

6. **Check the export.** Pass: the files exist, `quant_algo` matches the format in the table,
   and `kv_cache_quant_algo` is `FP8`. Anything else, including `None`, fails.

   ```bash
   E=<export directory on this host>
   ls -l $E/config.json $E/hf_quant_config.json $E/model.safetensors.index.json
   python3 -c 'import json,sys; q=json.load(open(sys.argv[1])).get("quantization",{}); print("quant_algo:", q.get("quant_algo"), "kv_cache_quant_algo:", q.get("kv_cache_quant_algo"))' $E/hf_quant_config.json
   ```

   | Format | Expected `quant_algo` |
   |---|---|
   | `fp8` | `FP8` |
   | `nvfp4` | `MIXED_PRECISION` (NVFP4 experts with FP8 layers, as in the Hub NVFP4 model) |

7. **Hand off to serving.** The `serve` skill starts vLLM inside the vLLM container, which mounts
   the repo at `/workspace` under Docker and at its host path in the Slurm session
   (`avlm/deployment/vLLM/slurm/interactive/launch_interactive_session.sh`). So `MODEL_PATH` is
   `/workspace/<path from the repo root>` under Docker and the absolute host path on Slurm. The
   tracked serve configs already load the default export of `quantization_hf_tennis.yaml`;
   `MODEL_PATH` covers a changed label, workspace or recipe. Give the form for the user's
   environment, or both when it is unknown. Inside the vLLM container (Docker):

   ```bash
   SERVE_CONFIG=avlm/deployment/vLLM/configs/vllm_server_params_fp8.yaml \
   MODEL_PATH=/workspace/quantization_workspace/models/<label>-fp8-hf \
     bash avlm/deployment/vLLM/vllm_server.sh start
   ```

   Use `vllm_server_params_nvfp4.yaml` for NVFP4. Each file has an A100 block (Marlin kernels;
   the active default) and a commented Blackwell block; the serve skill picks one and runs its own
   approval gate.

## Reply

Your reply must contain, inline: the checkpoint and recipe, the format, the calibration source,
every command run or to run (in order), the export path, the step 6 result, the serve handoff
command, and anything not yet verified. If no GPU job has run yet, end with the request for
approval. Report only what a command printed: no accuracy, speed-up or memory numbers.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: HF_MODEL_ID must be a Hub id (org/model) or an absolute path` | Use a Hub id, an absolute path or `"@repo/..."`. |
| `error: HF_MODEL_ID local path not found or missing weights` | The directory does not exist in the container (Docker: add a `-v` mount, step 5) or has no `*.safetensors` (an unconverted Megatron checkpoint, step 1). |
| `error: base model weights missing at ...` | The download was skipped: unset `HF_HUB_OFFLINE` and keep `HF_MODEL_AUTO_DOWNLOAD` at `1` (the default). |
| `401` or `Repository not found` while downloading | `HF_TOKEN` is not exported in the shell that runs `quantize.sh` (Docker: `-e HF_TOKEN` needs it set on the host). |
| `error: QUANT_FORMAT=... — set QUANTIZE_CONFIG explicitly` | Use `fp8` or `nvfp4`. |
| `error: Model-Optimizer not found ... MODELOPT_GIT_BOOTSTRAP!=1` | Set `MODELOPT_GIT_BOOTSTRAP: 1` (needs network) or clone Model Optimizer to `MODELOPT_ROOT`. |
| `CUDA out of memory` | Another process holds the GPU, or it is too small for the BF16 weights (about 62 GB); use a free 80 GB GPU. |
| Serve cannot find the model after quantizing | The export name differs from the serve config's `model_path` (another label, workspace or recipe); pass `MODEL_PATH=<export>` (step 7). |

Report the first error line. Do not retry with other settings silently.

## Final checks

- [ ] The checkpoint is a full HF checkpoint (no `adapter_config.json`), not an AutoModel LoRA adapter or an unconverted Megatron checkpoint.
- [ ] Every absolute path outside the repo is mounted in the Docker command.
- [ ] The recipe copy has no `/path/to` values, and `NUM_SAMPLES` is at least 32.
- [ ] The user approved each GPU job (conversion or merge, and each format) in a separate message.
- [ ] The export has `hf_quant_config.json` with the expected `quant_algo`.
- [ ] The reply gives the serve command with `MODEL_PATH` and claims no unmeasured numbers.
- [ ] No token or credential was printed or written to a file, and nothing was uploaded.
