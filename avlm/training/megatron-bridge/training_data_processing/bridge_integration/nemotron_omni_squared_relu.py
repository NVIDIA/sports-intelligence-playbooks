# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from megatron.core.activations import squared_relu

from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider


def apply_nemotron_omni_squared_relu_override() -> None:
    original = NemotronOmniModelProvider._build_vision_projection_config

    def _build_vision_projection_config(self, language_cfg):
        config = original(self, language_cfg)
        config.activation_func = squared_relu
        return config

    NemotronOmniModelProvider._build_vision_projection_config = _build_vision_projection_config
