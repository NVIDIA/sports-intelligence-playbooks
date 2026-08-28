# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os


overlay_root = os.environ["MB_BRIDGE_OVERLAY"]
multimodal_overlay = os.path.join(
    overlay_root,
    "megatron/core/models/multimodal",
)
multimodal = importlib.import_module("megatron.core.models.multimodal")
multimodal.__path__.insert(0, multimodal_overlay)

recipe_overlay = os.path.join(
    overlay_root,
    "megatron/bridge/recipes/nemotron_omni",
)
recipes = importlib.import_module("megatron.bridge.recipes.nemotron_omni")
recipes.__path__.insert(0, recipe_overlay)
audio_mask_override = importlib.import_module(
    "megatron.bridge.recipes.nemotron_omni.bridge_integration.nemotron_omni_audio_mask",
)
squared_relu_override = importlib.import_module(
    "megatron.bridge.recipes.nemotron_omni.bridge_integration.nemotron_omni_squared_relu",
)
audio_mask_override.apply_nemotron_omni_audio_mask_override()
squared_relu_override.apply_nemotron_omni_squared_relu_override()

importlib.import_module("avlm.inference.common.run_inference").main()
