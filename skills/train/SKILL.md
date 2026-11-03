---
name: train
description: "Configures, launches, monitors, and resumes Nemotron-3-Nano-Omni (AVLM) SFT or LoRA fine-tuning on Slurm with the repo's NeMo AutoModel or Megatron-Bridge launchers: copies launch and recipe templates to _local.yaml, resolves REPLACE_ME placeholders, runs the DRY_RUN=1 preflight, submits batch or interactive jobs, verifies finite loss and written checkpoints, triages launch failures, and summarizes step time and tokens/s/GPU from a training log. Use when asked to fine-tune, train, smoke-test, resume, or measure training throughput of the sports AVLM on a Slurm GPU cluster, including the HF-to-Megatron checkpoint conversion that Megatron-Bridge training needs. Not for single-GPU or workstation training, dataset generation or annotation, inference, evaluation scoring, quantization, or vLLM serving."
license: Apache-2.0
compatibility: "Requires a checkout of github.com/NVIDIA/sports-intelligence-playbooks with Git LFS; run from the repo root. Slurm login node with sbatch, squeue, sacct, and pyxis/enroot containers; nodes with 8 GPUs; bash and Python 3.8+ with PyYAML on the login node."
---

# AVLM Sports Intelligence Train

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

Launches fine-tuning of the sports AVLM (`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`) through the repo's own Slurm launchers. The launchers take no flags: they read `launch_local.yaml` plus a recipe YAML and accept `KEY=value` command-prefix overrides. This skill is the agent checklist; the per-stack guides remain the source of truth and are linked, not copied.

Run every command from the repository root on the Slurm login node. Never train on the login node itself.

## When to Use This Skill

- Fine-tune (SFT or LoRA) the AVLM on Slurm with AutoModel or Megatron-Bridge.
- Run a short smoke training to prove a new cluster, container, or dataset works.
- Resume an interrupted run, or decide whether a finished run succeeded.
- Diagnose a job stuck in `PENDING`, a quiet job that may be hung, or a launch failure.
- Convert the HF base checkpoint to Megatron FSDP format before Megatron-Bridge training.
- Extract step time and tokens/s/GPU from a finished training log for the performance dashboard.

## When NOT to Use

- Single-GPU or workstation training: every provided recipe needs one full node of 8 GPUs (`REQUIRE_CUDA_DEVICES=8`, expert parallel size 8).
- Building QA/MCQ data or video metadata: see [the tennis data-prep README](../../avlm/data_prep_example/tennis/README.md).
- Generating answers from a checkpoint: see [the inference guide](../../avlm/inference/README.md).
- Scoring checkpoints (MCQ accuracy, LLM judge): see [the evaluation guide](../../avlm/evals/README.MD).
- Serving with vLLM: see [the vLLM README](../../avlm/deployment/vLLM/README.md).
- Quantizing a checkpoint to FP8 or NVFP4: use the `quantize` skill.

## Inputs to collect

Ask the user for any missing value. Never invent cluster values. The tracked `*_local.yaml` files can hold a maintainer's cluster values on the development branch; overwriting them from the templates (step 2) is expected, so do not reuse their values.

| Input | Where it goes |
|---|---|
| Stack and method | Selects `D` in the table below (default: AutoModel LoRA) |
| Slurm account and partitions | `SLURM_ACCOUNTS`, `SINGLE_NODE_PARTITION`, `FEW_NODES_PARTITIONS`, `PARTITIONS` in `launch_local.yaml` |
| Container `.sqsh` path | `CONTAINER_IMAGE`: `nemo-automodel:26.06.00` for AutoModel, `nemo:26.06.00` for Megatron-Bridge |
| Writable cache directory | `CACHE_DIR`: Hugging Face weights (`CACHE_DIR/huggingface`), Triton, runtime pip packages |
| Train/validation JSONL, video root, video-metadata JSON | `dataset.*` and `validation_dataset.*` in the recipe |
| Megatron FSDP base checkpoint (Megatron-Bridge only) | `checkpoint.pretrained_checkpoint` in the recipe |
| Run name, node count, steps, W&B mode | `MODEL_NAME`, `num_nodes`, `MAX_STEPS`, `WANDB_MODE` prefixes |

## Pick stack and method

| Stack / method | `D` (launcher dir) | Recipe base name in `$D/configs/` | Runbook | Base weights |
|---|---|---|---|---|
| AutoModel LoRA (default) | `./avlm/training/automodel/lora` | `nemotron_omni_automodel_lora_singlenode` | [LORA_GUIDE](../../avlm/training/automodel/lora/LORA_GUIDE.MD) | HF id, read from `CACHE_DIR` |
| AutoModel SFT | `./avlm/training/automodel/sft` | `nemotron_omni_automodel_sft_singlenode` | [SFT_GUIDE](../../avlm/training/automodel/sft/SFT_GUIDE.MD) | HF id |
| Megatron-Bridge LoRA | `./avlm/training/megatron-bridge/lora` | `nemotron_omni_mbridge_lora_singlenode` | [LORA_GUIDE](../../avlm/training/megatron-bridge/lora/LORA_GUIDE.MD) | `fsdp_dtensor` checkpoint |
| Megatron-Bridge SFT | `./avlm/training/megatron-bridge/sft` | `nemotron_omni_mbridge_sft_singlenode` | [SFT_GUIDE](../../avlm/training/megatron-bridge/sft/SFT_GUIDE.MD) | `fsdp_dtensor` checkpoint |

- AutoModel trains directly from Hugging Face weights; choose it unless the user asks for Megatron-Bridge or needs its parallelism. Megatron-Bridge also needs [references/megatron-bridge.md](references/megatron-bridge.md).
- LoRA trains adapters only (checkpoint about 0.3 GB); SFT updates the language model (each checkpoint 180-240 GB with optimizer state, and even a 30-step smoke writes one). Checkpoints land in `$D/slurm/outputs/<MODEL_NAME>/` inside the checkout by default. For SFT, check free space first (`df -h <dir>` plus your site's quota command), prefix `OUTPUT_BASE=<dir with room>`, and confirm `CHECKPOINT_DIR` in the `DRY_RUN=1` output. Megatron-Bridge `SAVE_OPTIM=false` writes weights only (no optimizer resume); for a pipeline-only test, LoRA avoids the large checkpoint.
- `*_singlenode` recipes (32 frames, 8,192 tokens, no packing) are for smoke runs; swap `_singlenode` for `_multinode` (128 frames, 21,504-token packing) for real training.
- Every override variable is listed in `$D/TRAINING_ENV_VARS.MD`.

## Workflow

1. **Check prerequisites.** Confirm the account, partitions, and container image exist. AutoModel recipes use `dispatcher: deepep`; on A100 the launcher installs the pre-Hopper wheel from `wheels/deepep/`, which must be real files, not Git LFS pointers:

   ```bash
   sacctmgr -nP show assoc user=$USER format=account,partition,qos | sort -u
   sinfo -h -o '%P %a %D %G' | sort -u
   ls -l <container.sqsh>
   head -c 40 wheels/deepep/*.whl     # "version https://git-lfs..." means pointers: git lfs pull --include 'wheels/deepep/*.whl'
   ```

2. **Copy templates to `_local.yaml`.** Start from the template every time, and point the launch config at the local recipe:

   ```bash
   D=./avlm/training/automodel/lora                  # from the table above
   RECIPE=nemotron_omni_automodel_lora_singlenode     # from the table above
   cp $D/slurm/launch.yaml $D/slurm/launch_local.yaml
   cp $D/configs/$RECIPE.yaml $D/configs/${RECIPE}_local.yaml
   sed -i "s#^CONFIG_YAML_REL: .*#CONFIG_YAML_REL: ${D#./}/configs/${RECIPE}_local.yaml#" $D/slurm/launch_local.yaml
   ```

3. **Resolve placeholders.** In `$D/slurm/launch_local.yaml` set `CACHE_DIR`, `CONTAINER_IMAGE`, `CONTAINER_MOUNT_HOST`, `SLURM_ACCOUNTS` and `SLURM_ACCOUNTS_RACE` (same account; keep `SLURM_ACCOUNT_RACE: 0`), and all three partition keys. In the local recipe set the `dataset` and `validation_dataset` paths (Megatron-Bridge also `checkpoint.pretrained_checkpoint`). Use `realpath` for host paths so they resolve inside the container. `REPLACE_ME` also marks tunable knobs such as frame counts, so check for placeholder values rather than markers:

   ```bash
   grep -nE '^[^#]*(/path/to|your-[a-z-]+)' $D/slurm/launch_local.yaml $D/configs/${RECIPE}_local.yaml
   ls -l <train.jsonl> <validation.jsonl> <train_video_metadata.json> <video_root>
   ```

   Acceptable leftovers: `wandb.entity/project/name` when W&B is disabled and `inference_validation.*` while `enabled: false`. The partition keys hold example names (marked `REPLACE_ME`) that the grep does not flag; replace them with names that `sinfo` lists. A 1-node batch job is offered to `SINGLE_NODE_PARTITION`, `FEW_NODES_PARTITIONS`, and `PARTITIONS` together; 2-3 nodes use the last two; more nodes use `PARTITIONS`. The partition can decide the QoS: a list that includes a backfill partition may run the whole job under a preemptible low-priority QoS and wait in `PENDING (Priority)`. Remove every backfill partition from all three keys, even when the user names one, and tell them why; or pin non-backfill partitions with the `partition=<a>,<b>` prefix. If the video-metadata JSON is missing, build it first (`python avlm/utils/build_video_metadata.py --help`; needs `decord`).

4. **Megatron-Bridge only: provide the base checkpoint.** The provided recipes load an `fsdp_dtensor` checkpoint produced by `HF_to_MEG_FSDP` conversion. Follow [references/megatron-bridge.md](references/megatron-bridge.md), then set `checkpoint.pretrained_checkpoint`.

5. **Preflight with `DRY_RUN=1`.** Runs on the login node without containers or GPUs and prints the resolved configuration. Check `CONFIG_YAML` (the `_local` recipe), `partition`, `MAX_STEPS`, and `CHECKPOINT_DIR`. It does not validate paths, placeholders, Hugging Face cache, or DeepEP wheels, so steps 1 and 3 still matter.

   ```bash
   DRY_RUN=1 num_nodes=1 MAX_STEPS=30 VAL_EVERY_STEPS=15 VAL_MAX_BATCHES=8 WANDB_MODE=disabled MODEL_NAME=avlm_lora_smoke \
     bash $D/slurm/sbatch/sbatch_starter.sh
   ```

   If `$D/slurm/outputs/<MODEL_NAME>/` (or `<OUTPUT_BASE>/<MODEL_NAME>/`) already exists, AutoModel and Megatron-Bridge SFT resume from its checkpoint under the same `MODEL_NAME`, so a finished run ends at once. For a new run pick a new `MODEL_NAME`; reuse one only to resume (step 10).

   These are the smoke knobs verified on GPU. AutoModel always validates and checkpoints on the last step, and `VAL_MAX_BATCHES` bounds validation. For Megatron-Bridge replace `VAL_MAX_BATCHES=8` with `EVAL_ITERS=2 CKPT_EVERY_STEPS=30` so a checkpoint is written at the last step.

6. **Stop and ask before submitting.** Show the user the resolved account, partition, nodes, steps, and `MODEL_NAME`, then end your turn and ask for approval. Do this every time, even when the user gave every value or asked you to run the job: submission consumes shared GPU allocation. Submit only after an explicit yes in a later message. With Codex on a cluster, also add a `decision = "prompt"` rule for `sbatch` to `~/.codex/rules/default.rules` and keep `approval_policy = "on-request"` (not `"never"`), so Codex itself asks before submitting.

7. **Submit, after the approval in step 6.** Prefer batch: it is non-blocking and writes logs to files. Repeat the exact preflight command without `DRY_RUN=1`; it prints `job <id>` under `[submit]`.

   ```bash
   num_nodes=1 MAX_STEPS=30 VAL_EVERY_STEPS=15 VAL_MAX_BATCHES=8 WANDB_MODE=disabled MODEL_NAME=avlm_lora_smoke \
     bash $D/slurm/sbatch/sbatch_starter.sh
   ```

   Interactive mode needs a terminal (`srun --pty`), so use it only when the user drives the shell: `bash $D/slurm/interactive/launch_interactive_session.sh`, then inside the container `MAX_STEPS=30 bash $D/slurm/interactive/train_interactive.sh`. If the Hugging Face weights are not yet under `CACHE_DIR/huggingface`, prefix the first run with `HF_HUB_OFFLINE=0`.

8. **Monitor.** Logs go to `$D/slurm/logs/<MODEL_NAME>_<timestamp>/`: `slurm_<jobid>.out` (job header, container errors) and `<MODEL_NAME>-<account>_<jobid>_<timestamp>.log` (training output). On 8x A100 expect about 5 minutes of model loading and several more before the first step is logged (6-15 minutes after start on the reference runs); silence in that window is not a hang. Do not cancel a `RUNNING` job for silence alone in its first 20 minutes; cancel only on an error line.

   ```bash
   squeue -j <jobid> -o '%.10i %.8q %.12P %.8T %.10M %R'
   L=$(ls -td $D/slurm/logs/avlm_lora_smoke_*/ | head -1)
   grep -hE '\| step [0-9]+ \||iteration +[0-9]+/' $L/*_<jobid>_*.log | tail -n 3
   grep -nE -m5 'Traceback|error:|Error:' $L/slurm_<jobid>.out $L/*_<jobid>_*.log
   ```

   `PENDING (Priority)` with idle nodes (`sinfo -t idle`) is usually the QoS: the launchers pass no `--qos`, so the account's default QoS applies, and a backfill partition in the list can put the whole job on a preemptible low-priority QoS. If the `%q` column shows such a QoS, move only your own job: `scontrol update jobid=<jobid> partition=<regular partition>` (add `qos=<regular QoS>` if your site needs it), or `scancel <jobid>` and resubmit the same command with a `partition=<regular partition>` prefix (plus `SBATCH_QOS=<qos>` if your site requires a QoS). Then remove the backfill partition from `launch_local.yaml` (step 3).

9. **Verify success: always all three checks.** For "did my run finish or succeed?" and for every perf row, run or give these three commands, in order. The log name `<MODEL_NAME>-<account>_<jobid>_<timestamp>.log` gives `<MODEL_NAME>` and `<jobid>`. The run passed only if all three pass. A log line saying a checkpoint was saved is not check 3. If a command cannot run where you are (no Slurm, or not on the cluster filesystem), print it verbatim and report that result as not yet checked.

   ```bash
   # 1. Slurm: State COMPLETED, ExitCode 0:0 (TIMEOUT, FAILED, CANCELLED, OUT_OF_MEMORY = not passed)
   sacct -j <jobid> -X --format=JobID,State,ExitCode,Elapsed
   # 2. Log: finite losses, last step reached, step time
   python3 skills/train/scripts/summarize_train_log.py <log> --method <lora|sft>
   # 3. Checkpoint on disk (under <OUTPUT_BASE>/ instead of $D/slurm/outputs/ if OUTPUT_BASE was set)
   ls -l $D/slurm/outputs/<MODEL_NAME>/checkpoints_*/
   # 4. AutoModel LoRA only: every adapter was trained
   python3 skills/train/scripts/check_lora_adapters.py $D/slurm/outputs/<MODEL_NAME>/checkpoints_*/LATEST
   ```

   The script exits `0` only for a passing run and prints one JSON row with `Run status`, `failures`, `final_loss`, `Step time (s)`, and `Tokens/s/GPU`. Report `Step time (s)` as the script prints it, never a hand-computed mean. Megatron-Bridge logs carry no token counts, so its `Tokens/s` and `Tokens/s/GPU` are `null`: report `null`, never an estimate. The checkpoint directory is `checkpoints_n<N>_tp1_pp1_ep8/epoch_<e>_step_<s>/` plus a `LATEST` link for AutoModel, and `checkpoints_n<N>_pp1_tp1_ep8/iter_<NNNNNNN>/` plus `latest_checkpointed_iteration.txt` for Megatron-Bridge. On failure, read [references/troubleshooting.md](references/troubleshooting.md) and report the first actionable error line, not the last.

   Finite losses do not prove every adapter learned. Check 4 reads the saved adapter: a `lora_B` that is still all zero means that module never got a gradient. Exit `1` lists `inactive_groups`; report the run as passed with inactive adapters, name the groups, and give the fix from troubleshooting. If the recipe sets explicit `target_modules`, pass them as `--expect in_proj,q_proj,...` so a target with no adapter also fails (`expect_missing`). `config_targets_not_in_checkpoint` is a note only. Exit `2` means no LoRA tensors in safetensors format (Megatron-Bridge): report check 4 as not run.

10. **Resume, in this order.** Give the full relaunch command even when you cannot see the checkpoint from where you are.
    1. Confirm the original job has ended. It must not be `RUNNING` or `PENDING`; `TIMEOUT`, `FAILED`, `CANCELLED`, or `COMPLETED` are fine. The job name is `<MODEL_NAME>-<account>`.

       ```bash
       squeue -u $USER -o '%.10i %.40j %.8T %.10M'
       sacct -j <old jobid> -X --format=JobID,JobName%40,State,ExitCode,Elapsed
       ```

    2. Confirm the checkpoint to resume from: `ls -l $D/slurm/outputs/<MODEL_NAME>/checkpoints_*/`.
    3. Relaunch with the same `MODEL_NAME`, `num_nodes`, and recipe parallelism, and repeat the original overrides (`MAX_STEPS` is the total step count, not the steps left). Megatron-Bridge LoRA also needs `RESUME_CHECKPOINT=1`; the other stacks resume automatically. Preflight first, then submit the same line without `DRY_RUN=1` after the user approves:

       ```bash
       # RESUME_CHECKPOINT=1 is read by Megatron-Bridge LoRA only
       DRY_RUN=1 RESUME_CHECKPOINT=1 MODEL_NAME=<same name> num_nodes=<same count> MAX_STEPS=<same total> \
         bash $D/slurm/sbatch/sbatch_starter.sh
       ```

    A different topology needs a new `MODEL_NAME`.

## Override precedence

Highest first: command-prefix assignment (`MAX_STEPS=2 bash ...`), then `launch_local.yaml` and the recipe YAML, then variables already exported in the login shell. Put every override on the command line; an inherited `export` loses to YAML. If both are set, `CONFIG_YAML` (absolute path) beats `CONFIG_YAML_REL`. The Megatron-Bridge `DRY_RUN` output labels each knob `(CLI override)` or `(recipe YAML)`.

| Variable | Use |
|---|---|
| `MODEL_NAME` | Run id; names the log and checkpoint directories |
| `num_nodes`, `partition`, `SLURM_TIME_LIMIT` | Node count; pin one partition instead of the YAML list; wall time |
| `MAX_STEPS`, `GLOBAL_BATCH_SIZE` | Run length; optimizer batch |
| `VAL_EVERY_STEPS`, `CKPT_EVERY_STEPS` | Validation and checkpoint intervals |
| `WANDB_MODE` | `disabled`, `offline`, or `online` (`online` needs `WANDB_API_KEY` exported; never print it) |
| `PACK_SIZE` (AutoModel), `SEQ_LENGTH` (Megatron-Bridge) | Packed or per-sample token limit |
| `DRY_RUN=1` | Print the resolved batch configuration and exit |
| `HF_HUB_OFFLINE=0` | Allow the first download of Hugging Face weights |

## Measure training throughput

The [performance page](https://nvidia.github.io/sports-intelligence-playbooks/latest/multimodal/perf_benchmarking.html) reports time per global step and tokens/s/GPU on one 8-GPU node over 13 steps, with the first 3 excluded as warm-up, global batch 128, and 21,504-token packing. To produce a comparable row, prepare `${RECIPE/_singlenode/_multinode}_local.yaml` with steps 2-3, then:

```bash
CONFIG_YAML_REL=${D#./}/configs/${RECIPE/_singlenode/_multinode}_local.yaml \
  num_nodes=1 MAX_STEPS=13 GLOBAL_BATCH_SIZE=128 SKIP_VALIDATION=1 WANDB_MODE=disabled MODEL_NAME=avlm_perf \
  bash $D/slurm/sbatch/sbatch_starter.sh
python3 skills/train/scripts/summarize_train_log.py <log> \
  --warmup-steps 3 --method lora --hardware A100 --output perf_rows.jsonl
```

`SKIP_VALIDATION=1` is AutoModel-only; for Megatron-Bridge set `VAL_EVERY_STEPS` above `MAX_STEPS` instead. The JSON keys match the published [performance dashboard](https://nvidia.github.io/sports-intelligence-playbooks/latest/multimodal/perf_dash/avlm_dashboard.html) data (`Step time (s)`, `Tokens/s`, `Tokens/s/GPU`, `Measured steps`, `Global batch size`, `Run status`). AutoModel `Tokens/s` is the mean of the logged per-step `tps`; Megatron-Bridge logs carry no token counts, so its token fields stay `null`. Report only measured values together with the job id, hardware, container, and recipe; never fill in a number the log does not contain. Dashboard rows also vary loader and activation-checkpointing settings, so a default-recipe run is comparable but not identical.

## Troubleshooting

Match the first error in `slurm_<jobid>.out` or the training log against [references/troubleshooting.md](references/troubleshooting.md). Most common:

- `error: CUDA devices N < 8`: fewer than 8 visible GPUs. Never use `SKIP_CUDA_DEVICE_CHECK=1` for real runs: it only bypasses the check, and the 8-way expert-parallel recipes still fail. Find a partition whose nodes have 8 GPUs (`sinfo -h -o '%P %G'`; the GPU count in the GRES column is 8), then resubmit the same command on it: `GPUS_PER_NODE=8 use_exclusive=1 partition=<8-GPU partition> <same overrides> bash $D/slurm/sbatch/sbatch_starter.sh`.
- `named symbol not found` on the first MoE step: DeepEP built for another GPU generation; see [DEEPEP.MD](../../wheels/deepep/DEEPEP.MD).
- A `.whl` of about 130 bytes starting with `version https://git-lfs`: Git LFS pointer, not a wheel; run `git lfs pull`.
- `/opt/Automodel` or `megatron.bridge` import errors: wrong container for the stack.

## Final answer must contain

- **Commands inline:** every command you ran or the user must run, in order, in code blocks, with placeholders such as `<jobid>` marked. Name the scripts you ran (for example `summarize_train_log.py`).
- **Verification results:** the observed result of each check (`sacct` State and ExitCode, script `Run status` and `Step time (s)`, checkpoint listing). If a check could not run here, give its command and say the result is not yet known.
- **Measured values only:** job id, step time from the script, and `null` token fields for Megatron-Bridge. Never invent numbers.
- **What to check next:** the next command or decision for the user, such as approving submission, the resume command, or a check you could not run. After a passing Megatron-Bridge run that the user wants to serve or quantize, point to [references/megatron-bridge.md](references/megatron-bridge.md#5-after-training) section 5.

## Final checks

- [ ] Both `_local.yaml` files were copied from templates and hold no placeholder values except disabled W&B and inference-validation fields.
- [ ] `DRY_RUN=1` output shows the intended recipe, partition, steps, and run name, and the user approved submission.
- [ ] The reported job id, log path, and checkpoint path exist.
- [ ] `summarize_train_log.py` returned `Run status: Passed`, or the first actionable error was reported.
- [ ] For AutoModel LoRA, `check_lora_adapters.py` was run and any inactive or missing adapters were reported.
- [ ] No secrets (W&B, Hugging Face, or API keys) were printed or written to files, and no other user's jobs were cancelled.
