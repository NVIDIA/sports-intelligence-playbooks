#!/usr/bin/env bash
set -euo pipefail

BRIDGE_ROOT="/opt/Megatron-Bridge"
HF_MODEL_PATH="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
FSDP_BASE_CHECKPOINT="/path/to/megatron_fsdp_base_checkpoint/iter_NNNNNNN"  # REPLACE_ME with the fsdp_dtensor base checkpoint iter_* directory containing .metadata.
LORA_CHECKPOINT="/path/to/mbridge_fsdp_lora_checkpoint/iter_NNNNNNN"  # REPLACE_ME with the fsdp_dtensor LoRA checkpoint.
HF_OUTPUT="/path/to/automodel_full_model"  # REPLACE_ME with a new output directory.
NPROC_PER_NODE=1
TRUST_REMOTE_CODE=1

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
PATCH_DIR="${REPO_ROOT}/avlm/training/megatron-bridge/training_data_processing/bridge_integration"

if [[ "${1:-}" == "__convert_worker" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  exec python -c '
import os
from dataclasses import replace
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge
from megatron.bridge.peft.utils import create_peft
from megatron.bridge.training.utils.checkpoint_utils import read_run_config
from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import uneven_dtensor_to_full_tensor

from megatron_fsdp_buffer_index import apply_megatron_fsdp_buffer_index_patch
from megatron_fsdp_mamba_checkpoint import apply_megatron_fsdp_mamba_checkpoint_patch
from nemotron_omni_model_config import apply_nemotron_omni_model_config_override
from peft_fsdp_pretrained_load import apply_peft_fsdp_pretrained_load_patch
from avlm.inference.common.models.nemotron_omni_mbridge import _load_fsdp_model

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
apply_nemotron_omni_model_config_override()
apply_megatron_fsdp_buffer_index_patch()
apply_megatron_fsdp_mamba_checkpoint_patch()
apply_peft_fsdp_pretrained_load_patch()

original_materialize_adapter_weights = MegatronPeftBridge.materialize_adapter_weights


@wraps(original_materialize_adapter_weights)
def materialize_adapter_weights(self, adapter_tasks):
    adapter_weights = original_materialize_adapter_weights(self, adapter_tasks)

    def materialize(weight):
        tensor = weight.weight
        if isinstance(tensor, DTensor):
            tensor = uneven_dtensor_to_full_tensor(tensor)
        return weight._replace(weight=tensor)

    return [
        replace(
            adapter_weight,
            linear_in_weight=materialize(adapter_weight.linear_in_weight),
            linear_out_weight=materialize(adapter_weight.linear_out_weight),
        )
        for adapter_weight in adapter_weights
    ]


MegatronPeftBridge.materialize_adapter_weights = materialize_adapter_weights

hf_model = os.environ["HF_MODEL_PATH"]
adapter_path = os.environ["LORA_CHECKPOINT"]
run_config = read_run_config(str(Path(adapter_path) / "run_config.yaml"))
peft = create_peft(run_config["peft"])
if peft is None:
    raise ValueError("LoRA is not enabled in the checkpoint run_config.yaml")

processor = AutoProcessor.from_pretrained(
    hf_model,
    trust_remote_code=os.environ["TRUST_REMOTE_CODE"] == "1",
)
bridge = AutoBridge.from_hf_pretrained(
    hf_model,
    trust_remote_code=os.environ["TRUST_REMOTE_CODE"] == "1",
)
provider = bridge.to_megatron_provider(load_weights=False)
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.context_parallel_size = 1
provider.expert_model_parallel_size = 1
provider.expert_tensor_parallel_size = 1
provider.pipeline_dtype = torch.bfloat16
provider.dynamic_resolution = True
provider.temporal_patch_dim = processor.video_temporal_patch_dim
provider.separate_video_embedder = True
provider.temporal_ckpt_compat = True
provider.vision_class_token_len = 10
provider.radio_interpolate_only_cpe = False

try:
    model = _load_fsdp_model(
        provider,
        os.environ["FSDP_BASE_CHECKPOINT"],
        "no_shard",
        peft=peft,
        adapter_path=adapter_path,
    )
    model = [model_chunk.cuda().eval() for model_chunk in model]
    bridge.save_hf_pretrained(
        model,
        os.environ["HF_OUTPUT"],
        source_path=hf_model,
        strict=False,
        merge_adapter_weights=True,
    )
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
'
fi

test -d "${BRIDGE_ROOT}"
test -f "${FSDP_BASE_CHECKPOINT}/.metadata"
test -f "${LORA_CHECKPOINT}/.metadata"
test -f "${LORA_CHECKPOINT}/run_config.yaml"
test -f "${PATCH_DIR}/megatron_fsdp_buffer_index.py"
test -f "${PATCH_DIR}/megatron_fsdp_mamba_checkpoint.py"
test -f "${PATCH_DIR}/nemotron_omni_model_config.py"
test -f "${PATCH_DIR}/peft_fsdp_pretrained_load.py"

case "${HF_MODEL_PATH}" in
  /*|./*|../*) test -d "${HF_MODEL_PATH}" ;;
esac

if [[ -e "${HF_OUTPUT}" ]]; then
  echo "error: output already exists: ${HF_OUTPUT}" >&2
  echo "set HF_OUTPUT to a new path or move the existing output first" >&2
  exit 1
fi

mkdir -p "$(dirname "${HF_OUTPUT}")"
export HF_MODEL_PATH FSDP_BASE_CHECKPOINT LORA_CHECKPOINT HF_OUTPUT TRUST_REMOTE_CODE

cd "${BRIDGE_ROOT}"

uv run python -m torch.distributed.run \
  --standalone \
  "--nproc_per_node=${NPROC_PER_NODE}" \
  --no-python \
  "${SCRIPT_PATH}" \
  __convert_worker
