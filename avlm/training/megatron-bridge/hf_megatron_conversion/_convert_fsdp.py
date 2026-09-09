#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NVIDIA's FSDP converter with AVLM's checkpoint compatibility patches."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


integration_dir = Path(__file__).resolve().parents[1] / "training_data_processing" / "bridge_integration"
sys.path.insert(0, str(integration_dir))

from megatron_fsdp_buffer_index import apply_megatron_fsdp_buffer_index_patch  # noqa: E402
from megatron_fsdp_mamba_checkpoint import apply_megatron_fsdp_mamba_checkpoint_patch  # noqa: E402


apply_megatron_fsdp_buffer_index_patch()
apply_megatron_fsdp_mamba_checkpoint_patch()

converter = (
    Path(os.environ["MEGATRON_BRIDGE_ROOT"])
    / "examples"
    / "conversion"
    / "mfsdp"
    / "convert_checkpoints_fsdp.py"
)
spec = importlib.util.spec_from_file_location("_convert_checkpoints_fsdp", converter)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load converter: {converter}")

converter_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter_module
spec.loader.exec_module(converter_module)
original_ddp_config = converter_module.DistributedDataParallelConfig


def conversion_ddp_config(*args, **kwargs):
    # Match AVLM FSDP training and avoid a separate FP32 main-parameter buffer.
    kwargs["megatron_fsdp_main_params_dtype"] = None
    return original_ddp_config(*args, **kwargs)


converter_module.DistributedDataParallelConfig = conversion_ddp_config
converter_module.main()
