# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceDistribution:
    data_rank: int
    data_world_size: int
    writes_results: bool = True
    requires_lockstep: bool = False


class InferenceModel():
    @abstractmethod
    def predict(self, question, video_path, **kwargs):
        raise NotImplementedError

    def inference_distribution(self, rank, world_size):
        return InferenceDistribution(rank, world_size)


def get_model(model_name, **kwargs):
    if model_name == "nemotron_omni_hf":
        from avlm.inference.common.models.nemotron_omni_hf import NemotronOmniHF
        return NemotronOmniHF(**kwargs)
    elif model_name == "nemotron_omni_mbridge":
        from avlm.inference.common.models.nemotron_omni_mbridge import NemotronOmniMBridge
        return NemotronOmniMBridge(**kwargs)
        
    raise ValueError(f"unknown model: {model_name}")
