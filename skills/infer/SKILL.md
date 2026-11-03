---
name: infer
description: "Runs batch video inference for the Sports Intelligence video model (Nemotron-3-Nano-Omni) from a base, SFT or LoRA checkpoint with the repo's AutoModel or Megatron-Bridge Slurm launchers under avlm/inference, writes predictions.jsonl and checks it. Use when asked to generate predictions or answers for a JSONL of sports video questions, smoke-test a trained checkpoint on a few clips, pick the inference backend, resume an interrupted inference job, or debug an inference launch. Not for scoring predictions (MCQ accuracy, LLM judge; see evaluate), for training or fine-tuning (see train), for serving the model behind an HTTP endpoint with vLLM (see serve), for quantizing a checkpoint (see quantize), or for building datasets."
license: Apache-2.0
compatibility: "Requires a checkout of github.com/NVIDIA/sports-intelligence-playbooks with Git LFS; run from the repo root. Slurm cluster with pyxis/enroot containers (nvcr.io/nvidia/nemo-automodel:26.06.00 for AutoModel, nvcr.io/nvidia/nemo:26.06.00 for Megatron-Bridge), 80 GB GPUs, bash, Python 3.8+ with PyYAML on the login node."
---

# Sports Intelligence Batch Inference

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

Turn an existing checkpoint plus a JSONL of video questions into
`avlm/inference/outputs/<INFERENCE_NAME>/predictions.jsonl`. The runbooks of record are
`./avlm/inference/README.md` (every launch variant) and `./avlm/inference/INFERENCE_ENV_VARS.MD`
(every override). This skill adds the decisions, checks and triage around them.

## When to Use This Skill

- Generate predictions with the base Nemotron-3-Nano-Omni model or a trained SFT/LoRA checkpoint.
- Run a bounded smoke test (for example 16 samples on 1 GPU) before a full run.
- Choose between the `automodel` and `megatron_bridge` backends for a checkpoint.
- Resume an interrupted inference run, or diagnose a launch that failed or produced no output.

## When NOT to Use

- Scoring existing predictions (MCQ accuracy, LLM judge, judge summaries): use the
  `evaluate` skill. This skill stops at a verified `predictions.jsonl`.
- Training, LoRA fine-tuning or checkpoint conversion: use `train`.
- Serving a checkpoint behind an OpenAI-compatible endpoint with vLLM (start, smoke test, load
  test, stop): use `serve`.
- Building or converting datasets.
- Quantizing a checkpoint to FP8 or NVFP4: use `quantize`.

## Inputs

| Input | Where it goes | Notes |
|---|---|---|
| Checkpoint type | picks backend and template (step 2) | HF base, AutoModel SFT, AutoModel LoRA adapter, Megatron-Bridge SFT/LoRA |
| `model_path` / `adapter_path` | config, or `MODEL_PATH` / `ADAPTER_PATH` env | LoRA: `model_path` is the base model, never a consolidated SFT dir |
| `data_path` | config, or `DATA_PATH` env | JSONL, one question per row (format below) |
| `video_root` | config only (no env override) | prefix for relative video paths |
| `inference_name` | config, or `INFERENCE_NAME` env | output subdirectory; keep unique per run |
| Sample bound | `MAX_INFERENCE_SAMPLES` (+ `MAX_INFERENCE_SAMPLES_SEED`) | unset = whole file; seed `-1` = first N. For smoke tests set a seed (for example `0`): the first rows of a split often share one clip |
| GPUs | `num_nodes`, `GPUS_PER_NODE`, `use_exclusive`, `SLURM_MEM` | fixed by backend and template (step 5) |

Row format for `./avlm/inference/common/run_inference.py` (HF conversation, one mp4 video per row):

`{"id", "conversation": [{"role": "user", "content": [{"type": "video", "path": "<rel or abs>.mp4"}, {"type": "text", "text": "<question>"}]}, {"role": "assistant", "content": [{"type": "text", "text": "<answer>"}]}]}`

The assistant turn is copied to `ground_truth`; it may be empty for unlabeled data.

## Workflow

Run every command from the repo root. Inference overrides are environment-only: put
`KEY=VALUE` before `bash <launcher>`. Trailing `KEY=VALUE` or positional arguments are rejected
with exit 2.

1. **Read the runbook section** for the chosen backend in `./avlm/inference/README.md`; look up any
   variable not covered here in `./avlm/inference/INFERENCE_ENV_VARS.MD`.

2. **Choose the backend and config template.** Only two backend names exist: `automodel` and
   `megatron_bridge` (the value of `INFERENCE_BACKEND`). The directories are
   `./avlm/inference/automodel/` and `./avlm/inference/megatron-bridge/` (hyphen).

   | Checkpoint | Backend | Template under `configs/` | Notes |
   |---|---|---|---|
   | HF base model (`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`) | automodel | `default_sft.yaml` | set `model_path` to the HF id or a local snapshot |
   | AutoModel SFT checkpoint | automodel | `default_sft.yaml` | `<model_path>/model/consolidated` is picked up automatically |
   | AutoModel LoRA adapter | automodel | `default_lora.yaml` | `model_path` = base; `adapter_path` = dir with `adapter_config.json` (or its `model/` subdir) |
   | Megatron-Bridge FSDP SFT / LoRA | megatron_bridge | `default_sft_fsdp.yaml` / `default_lora_fsdp.yaml` | `is_fsdp: true`, TP=1 |
   | Megatron-Bridge non-FSDP (`torch_dist`) SFT / LoRA | megatron_bridge | `default_sft.yaml` / `default_lora.yaml` | `is_fsdp: false`, TP=2, so at least 2 GPUs (step 5) |

   `hf_model_path` stays the HF base id in every template; it supplies processor and model code.

3. **Write a run config.** Copy the template to its `*_local.yaml` name in the same directory
   and fill `data_path`, `video_root`, `model_path`, `adapter_path` (LoRA only) and
   `inference_name`. The tracked `default_*_local.yaml` copies can hold a maintainer's values on the
   development branch; overwriting them from the templates is expected.

   ```bash
   cp avlm/inference/automodel/configs/default_sft.yaml avlm/inference/automodel/configs/default_sft_local.yaml
   # edit: inference_name, data_path, video_root, model_path
   ```

   Frame policy comes from the config, not the code: every template sets `video_fps: 2.0` and
   `max_video_frames: 128` (both keys are required). Frames are sampled evenly at
   `int(video_fps)` frames per second of clip, capped at `max_video_frames`, and every frame is
   used when the clip is shorter. Fractional fps is truncated (`0.5` becomes `0`, which falls back
   to 8 frames). Audio from the same mp4 is included. `max_new_tokens: 256` and greedy decoding
   are the defaults; reasoning (`<think>`) is off.

4. **Preflight on the login node (no GPU).** Confirm before submitting, because a missing video
   aborts the whole job mid-run:

   ```bash
   test -f "$DATA" && wc -l "$DATA"                       # dataset exists, row count
   head -c 400 "$DATA"                                    # row format and first video path
   ls -l "$VIDEO_ROOT/<video path from that row>"         # clip resolves under video_root
   ls "<CACHE_DIR>/huggingface/hub" | grep -i nemotron    # HF base cached (jobs run offline)
   ```

   `<CACHE_DIR>` is the `CACHE_DIR` value in your `launch_local.yaml` (next paragraph), not a
   shell variable.

   Cluster settings come from `<backend>/slurm/launch_local.yaml`. The tracked copy can hold
   a maintainer's values: overwrite it from `launch.yaml` and set `CACHE_DIR`, `CONTAINER_IMAGE`,
   `SLURM_ACCOUNTS` and the partition list there, or point `CLUSTER_PARAMS` at your own YAML.
   Set `CACHE_DIR` in that YAML, not in the environment. A `CACHE_DIR` that the launching shell
   inherited from its parent process loses to the YAML value, and `CACHE_DIR="$CACHE_DIR"` before
   `bash` repeats the same inherited value, so the launcher ignores it too. `partition=` and
   `SLURM_ACCOUNTS=` with explicit values before `bash` work per run.

5. **Build the command, stop and ask, then submit.** Show the command, then end your turn and
   ask for approval, every time, even when the user gave every value or asked you to run it.
   Submit only after an explicit yes in a later message. With Codex on a cluster, also add a
   `decision = "prompt"` rule for `sbatch` to `~/.codex/rules/default.rules` and keep
   `approval_policy = "on-request"` (not `"never"`), so Codex itself asks before submitting.
   Submitting consumes a GPU
   allocation. The GPU request is fixed by backend and template (TP comes from the templates
   under `configs/`; both `sbatch_starter.sh` files default to `use_exclusive=1` and `--mem=0`,
   a whole node):

   | Backend, template | TP | GPUs | GPU prefixes |
   |---|---|---|---|
   | `automodel`, any template | none (one full BF16 copy, about 62 GB, per GPU) | 1 or more; 1 x 80 GB is enough | 1 GPU: `num_nodes=1 GPUS_PER_NODE=1 use_exclusive=0 SLURM_MEM=200G` |
   | `megatron_bridge`, `*_fsdp.yaml` | 1 | 1 or more (validated on 1 node x 8) | full node: `num_nodes=1 GPUS_PER_NODE=8` |
   | `megatron_bridge`, non-FSDP `default_sft.yaml` / `default_lora.yaml` | 2 | at least 2, a multiple of 2 | smallest: `num_nodes=1 GPUS_PER_NODE=2 use_exclusive=0 SLURM_MEM=200G` |

   - Keep `tensor_model_parallel_size` at the template value; it has no env override. Never set
     it to 1 in a non-FSDP config and never request 1 GPU for one. If the user asks for one GPU
     with a non-FSDP checkpoint, say it needs 2 GPUs and give the 2-GPU command; the only 1-GPU
     Megatron-Bridge path is an `fsdp_dtensor` checkpoint with a `*_fsdp.yaml` template.
   - Any request below a full node needs `use_exclusive=0` and a bounded `SLURM_MEM`.

   Put these before `bash` even when the config or `launch_local.yaml` already holds the same
   value (the tracked `launch.yaml` defaults to `num_nodes: 2`, `GPUS_PER_NODE: 8`):

   - Required: `INFERENCE_CONFIG` (your `*_local.yaml`), `INFERENCE_NAME`, `num_nodes`,
     `GPUS_PER_NODE`, plus `use_exclusive=0 SLURM_MEM=<bounded>` below a full node.
   - Optional: `MAX_INFERENCE_SAMPLES` (+ `MAX_INFERENCE_SAMPLES_SEED`; unset for the whole
     file), `DATA_PATH`, `MODEL_PATH`, `ADAPTER_PATH`, `partition`, `SLURM_ACCOUNTS`,
     `SLURM_TIME_LIMIT`, `RESUME=1`.

   Example that was run end to end (base model, 16 rows, 1 GPU, about 5 minutes including model
   load):

   ```bash
   INFERENCE_CONFIG=avlm/inference/automodel/configs/default_sft_local.yaml \
   INFERENCE_NAME=base_val16_1gpu \
   MAX_INFERENCE_SAMPLES=16 MAX_INFERENCE_SAMPLES_SEED=0 \
   num_nodes=1 GPUS_PER_NODE=1 use_exclusive=0 SLURM_MEM=200G \
   partition="$PARTITION" SLURM_ACCOUNTS="$SLURM_ACCOUNT" SLURM_TIME_LIMIT=0:45:00 \
   bash ./avlm/inference/automodel/slurm/sbatch/sbatch_starter.sh
   ```

   The inference launchers have no `DRY_RUN`: `DRY_RUN=1` is ignored and the job is submitted, so
   steps 3-4 are the only preflight.
   A short `SLURM_TIME_LIMIT` schedules sooner for smoke runs; scale it with the row count.
   `partition` accepts a comma-separated list; some clusters refuse to mix backfill and
   non-backfill partitions in one submission.

   - LoRA: same command with the `default_lora.yaml` copy and `ADAPTER_PATH=`.
   - Megatron-Bridge: `bash ./avlm/inference/megatron-bridge/slurm/sbatch/sbatch_starter.sh`,
     with the GPU prefixes from the table above.
   - Interactive node: `launch_interactive_session.sh`, then `infer_interactive.sh` inside it
     (README "Interactive").
   - Through the pipeline (inference only): `STAGES=inference INFERENCE_BACKEND=automodel
     INFERENCE_CONFIG=... bash ./avlm/inference/common/scripts/run_inference_eval_pipeline.sh`.
     It detaches; errors appear only in the printed `pipeline.log`.

6. **Monitor.** The launcher prints `Submitted batch job <id>`. Poll, do not busy-wait:

   ```bash
   squeue -j <id> -h -o '%T %M %R'          # PENDING reason / RUNNING time
   ls avlm/inference/logs/<INFERENCE_NAME>_<timestamp>/  # slurm_<id>.out + <job-name>_<id>_<ts>.log
   tail -5 avlm/inference/logs/<INFERENCE_NAME>_*/avlm-infer-*.log   # "[k/N] <id>: <prediction>" lines
   sacct -j <id> -X -o State,ExitCode,Elapsed
   ```

7. **Verify the output** with the bundled checker (`scripts/check_predictions.py`, relative to
   this skill). Exit 0 means every line parses, required keys are present, `index` runs 0..N-1
   and the count matches:

   ```bash
   python3 skills/infer/scripts/check_predictions.py \
     avlm/inference/outputs/<INFERENCE_NAME> --expected 16
   ```

   Then read two or three `prediction` values: they should be answers in the question's format
   (for MCQ, an option such as `(A) ...`), not empty strings or echoed prompts.

8. **Resume** an interrupted run with `RESUME=1` and the identical config, name, data, model,
   adapter, sample limit, seed and GPU count. Completed rows in `rank_results/` are skipped.

9. **Report** job id, final Slurm state, GPU count, backend, checkpoint, rows written,
   empty-prediction count and the `predictions.jsonl` path. Hand scoring to the evaluate skill.

## Output

```text
avlm/inference/outputs/<INFERENCE_NAME>/predictions.jsonl   # one record per input row, input order
avlm/inference/outputs/<INFERENCE_NAME>/predictions.json    # same records as a JSON array
avlm/inference/outputs/<INFERENCE_NAME>/rank_results/       # per-rank partials + inference_state.json (resume)
avlm/inference/logs/<INFERENCE_NAME>_<YYYYmmdd_HHMMSS>/     # Slurm .out and torchrun .log
```

Record keys: `index`, `id`, `class`, `super_category`, `fine_category`, `evaluation_type`,
`video-sound`, `start_time`, `end_time`, `ground_truth`, `prediction`, `inference_status`,
`conversations`, `reasoning_trace`, `metadata`. `predictions.jsonl` is written only after every
row succeeded; a crash leaves `rank_results/` without it.

Your reply must contain, inline, the config edits (file and keys), the exact launch command(s),
the verification command (`check_predictions.py ... --expected <N>`) and the
`predictions.jsonl` path. Do not only write them to files: create no wrapper scripts, plan
files or READMEs, only `*_local.yaml` copies.
If nothing was submitted yet, end by asking the user to confirm the submission.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: inference overrides are environment-only; place 'X=Y' before 'bash <script>'` (exit 2) | Move every `KEY=VALUE` in front of `bash`. |
| `INFERENCE_CONFIG: Set INFERENCE_CONFIG` or `error: INFERENCE_CONFIG not found: ...` | Export a path relative to the repo root or absolute; the file must exist in this checkout. |
| `error: missing .../launch_local.yaml` | `cp <backend>/slurm/launch.yaml <backend>/slurm/launch_local.yaml` and edit, or set `CLUSTER_PARAMS`. |
| Pipeline prints `Started pipeline` but no job appears | Read `pipeline.log`; `error: backend must be one of: automodel, megatron_bridge` means a wrong `INFERENCE_BACKEND`. `Unknown stage` means a wrong `STAGES`. |
| `sbatch: error: Batch job submission failed: Invalid partition name specified` | A partition in `launch_local.yaml` or `partition=` does not exist on this cluster; nothing was queued. Pick names from `sinfo` and set `partition=`. |
| 1-GPU job stays `PENDING` while nodes have idle GPUs | `SLURM_MEM` unset asks for a whole node's memory: set `use_exclusive=0` and a bounded `SLURM_MEM` (step 5). |
| `LocalEntryNotFoundError` / cannot reach huggingface.co | Jobs run offline (`HF_HUB_OFFLINE=1`). Point `CACHE_DIR` at a cache whose `huggingface/hub` holds the model, or allow one download with `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 LOCAL_FILES_ONLY=0` (about 62 GB). |
| `PermissionError` under `CACHE_DIR` | `CACHE_DIR` in `launch_local.yaml` is not writable for you. Overwrite the tracked copy from `launch.yaml` and set a writable `CACHE_DIR` in it (step 4). |
| `soundfile.LibsndfileError: Error opening '<clip>.mp4': Format not recognised` on the first sample | librosa 1.x is loaded (it cannot read mp4 audio). The launcher installs the pinned `librosa==0.11.0` into the extra-packages dir when another version is found and logs `[vlm-deps] librosa ... reinstalling`; without that line the dependency step was skipped (`SKIP_VLM_INFERENCE_DEPS=1`). Unset it and resubmit. |
| `FileNotFoundError: <video>` or `no video path in row id=...` | The run aborts at the first bad row. Fix `video_root` or the row, then resubmit with `RESUME=1`. |
| `LoRA should used based model, not: .../consolidated` | LoRA `model_path` must be the base model, not a consolidated SFT checkpoint. |
| `Resume state mismatch` | `RESUME=1` needs the same GPU count and `data_path` as the first attempt; otherwise use a new `INFERENCE_NAME`. |
| `Missing N predictions` at the merge | A rank died; read the per-job `.log`, then rerun with `RESUME=1`. |
| `CUDA out of memory` | Use 80 GB GPUs; lower `max_video_frames` in the config; keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| `[vlm-deps] error: pip install ... failed` | Compute node cannot reach PyPI. Pre-populate `<CACHE_DIR>/automodel_container_pkgs` from a node with network, or set `SKIP_VLM_INFERENCE_DEPS=1` if the container already has `decord` and `librosa`. |

Keep the first error message when reporting a failure; do not retry with different settings
silently.

## Final checks

- [ ] Backend is `automodel` or `megatron_bridge` and the template matches the checkpoint type.
- [ ] GPU count follows the step 5 table (non-FSDP Megatron-Bridge: at least 2 GPUs, TP left at 2).
- [ ] Overrides were placed before `bash`, including `INFERENCE_NAME`, `num_nodes` and `GPUS_PER_NODE`; `CACHE_DIR`, account and partitions come from your `launch_local.yaml` copy.
- [ ] The reply shows the config edits, launch command, `check_predictions.py` command and output path inline (Output section), and asks for confirmation before any submission.
- [ ] `sacct` shows `COMPLETED` and `check_predictions.py` exited 0 with the expected row count.
- [ ] Spot-checked predictions are non-empty answers.
- [ ] Report includes job id, checkpoint, backend, GPU count and the `predictions.jsonl` path.
- [ ] No tokens or credentials were printed or written to files.
