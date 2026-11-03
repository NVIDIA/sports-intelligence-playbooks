# Training failure triage

Find the first actionable error, not the last line: later ranks usually die with `SIGABRT` or `Terminated` after the root cause. torchrun prints `Root Cause (first observed failure)` with the failing rank.

```bash
L=<run log dir>   # $D/slurm/logs/<MODEL_NAME>_<timestamp>/
grep -nE -m5 'error|Error|Traceback' $L/slurm_<jobid>.out
grep -nE -m10 'Traceback|error:|Error:|out of memory|named symbol not found|NCCL|Root Cause' $L/*_<jobid>_*.log
sacct -j <jobid> -X -o JobID,State,ExitCode,Reason
```

## Before the job starts

| First error | Cause | Fix |
|---|---|---|
| `error: set CONFIG_YAML or CONFIG_YAML_REL`, `error: missing ...launch_local.yaml` | Launch template not copied, or `CONFIG_YAML_REL` points at a missing file | Repeat workflow step 2; `CONFIG_YAML_REL` is repo-relative without `./` |
| `sbatch: error: invalid partition specified` | Template example partition names left in `launch_local.yaml` | Replace all three partition keys with names from `sinfo -h -o %P`, or prefix `partition=<name>` |
| `Invalid account or account/partition combination` | Placeholder or wrong account | `sacctmgr -nP show assoc user=$USER format=account,partition` |
| Job stays `PENDING (Priority)` and `squeue -o %q` shows a low-priority or preemptible QoS | The partition list includes a backfill or preemptible partition, which sets that QoS for the whole job | Cancel only that job and resubmit with `partition=<regular partitions>`; if the job is already queued, `scontrol update jobid=<id> partition=<regular> qos=<regular QoS>` also works |
| Job stays `PENDING` (`Resources`, `Nodes required ... reserved`) | No free 8-GPU node in those partitions | Wait, or add partitions with idle nodes (`sinfo -t idle`); never cancel other users' jobs |
| Interactive or 1-node job never starts on an interactive partition | That partition may allow one job per user | Let the other job finish or use a batch partition |
| `DRY_RUN` shows the template recipe (`..._singlenode.yaml`, not `_local`) | `CONFIG_YAML_REL` not updated, or an exported `CONFIG_YAML` overrides it | Fix `launch_local.yaml`; `unset CONFIG_YAML` |

## Container and environment

| First error | Cause | Fix |
|---|---|---|
| pyxis or enroot cannot open the image | `CONTAINER_IMAGE` path wrong or not visible on compute nodes | Use a `.sqsh` on the shared filesystem under `CONTAINER_MOUNT_HOST` |
| `/opt/Automodel` missing or `No module named 'nemo_automodel'` | AutoModel launcher in the Megatron-Bridge image | AutoModel needs `nemo-automodel:26.06.00` |
| `No module named 'megatron.bridge'` | Megatron-Bridge launcher in the AutoModel image | Megatron-Bridge and conversion need `nemo:26.06.00` |
| `error: CUDA devices N < 8. Use SKIP_CUDA_DEVICE_CHECK=1 to skip.` | Fewer GPUs visible than `REQUIRE_CUDA_DEVICES` (default 8) | Request a full 8-GPU node (`GPUS_PER_NODE: 8`, `use_exclusive=1`). Skipping the check does not help with the provided recipes, which need 8-way expert parallelism |
| `error: EP=8 must divide non_pp=...` (AutoModel), `world_size=... < required=8` (Megatron-Bridge) | Fewer than 8 GPUs in the job | Same as above |
| `OfflineModeIsEnabled`, `LocalEntryNotFoundError`, or "Cannot find the requested files in the local cache" | `HF_HUB_OFFLINE=1` (AutoModel default) and weights not yet in `CACHE_DIR/huggingface` | Prefix `HF_HUB_OFFLINE=0` once; the compute node needs internet access. A git clone of the model whose `*.safetensors` are about 135 bytes is LFS pointers, not weights |
| `KeyError` or missing-clip errors naming a video path | Video-metadata JSON was built from a different JSONL than the one configured | Rebuild it with `avlm/utils/build_video_metadata.py` from the configured JSONL |
| `FileNotFoundError` naming `/path/to/...` or a data path | Placeholder left, or path outside `CONTAINER_MOUNT_HOST` | Workflow step 3; use `realpath` |

## DeepEP and Git LFS (AutoModel recipes with `dispatcher: deepep`)

| First error | Cause | Fix |
|---|---|---|
| `RuntimeError: ... layout.cu:128 'named symbol not found'` | Container DeepEP is built for Hopper and newer; the GPU is A100 (SM 8.x) | Let the launcher install the pre-Hopper wheel (do not set `SKIP_DEEPEP_INSTALL=1`); force with `DEEPEP_WHEEL_PROFILE=pre-hopper-sm80-sm89`. See [DEEPEP.MD](../../../wheels/deepep/DEEPEP.MD) |
| `[deepep-install] error`, `BadZipFile`, or pip reports an invalid wheel | `wheels/deepep/*.whl` are Git LFS pointer files (about 130 bytes, start with `version https://git-lfs.github.com/spec/v1`) | `git lfs install --local && git lfs pull --include 'wheels/deepep/*.whl'`, then relaunch |
| `warn: EP=... > 8 needs DeepEP NVSHMEM (Hopper+)` | Expert parallel above 8 on A100 | Keep EP at 8 or below on A100 |

## During training

| Symptom | Cause | Fix |
|---|---|---|
| `check_lora_adapters.py` lists `*.mixer.out_proj` in `inactive_groups` (AutoModel LoRA) | LoRA was attached to Mamba `out_proj`, but the fused kernel still consumed the base weight, so the adapter never received a gradient | Shipped recipes exclude `*mixer.out_proj*` by default. To train it, remove that exclude; the current recipe redirects the projection through LoRA and logs `Enabled Mamba out_proj LoRA path.` If that message is absent, put the exclude back |
| `check_lora_adapters.py` lists `vision_projector.linear1/linear2` under `config_targets_not_in_checkpoint` (AutoModel LoRA) | The saved adapter config names the vision projector, but the recipe excludes it (`*mlp1*`), so no adapter was created | Nothing to fix; the note does not fail the check |
| `torch.OutOfMemoryError: CUDA out of memory` | Sequence or frame budget too large | Lower `PACK_SIZE` / `SEQ_LENGTH` or `MAX_VIDEO_FRAMES`; start from the `*_singlenode.yaml` recipe |
| Host OOM (`oom-kill`, `Killed`) during data loading or packing | Packing and validation are CPU-heavy | Lower `GLOBAL_BATCH_SIZE`, `VAL_SAMPLE_RATIO`, or set `SKIP_VALIDATION=1` for smoke runs (AutoModel) |
| NCCL timeout or `Watchdog caught collective operation timeout` | Hung rank or bad node | Resubmit (resume picks up the last checkpoint); exclude the node if it repeats |
| `summarize_train_log.py` reports `non-finite loss` | Divergence or bad samples | Stop; report the step; check learning rate and the data batch |
| Megatron-Bridge fails loading `pretrained_checkpoint` | Checkpoint is `torch_dist` or Hugging Face, not `fsdp_dtensor` | Re-convert with `HF_to_MEG_FSDP` ([megatron-bridge.md](megatron-bridge.md)) |
| Resume fails on dataloader or sharding state | `num_nodes` or parallelism changed since the checkpoint | Use a new `MODEL_NAME`, or restore the original topology |
| Megatron-Bridge LoRA restarts from step 0 | `RESUME_CHECKPOINT=1` missing | Relaunch with the prefix and the same `MODEL_NAME` |

Report the job id, the first error line with its file and line number, and the fix applied. Change one thing per resubmission.
