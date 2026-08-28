# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder


def apply_nemotron_omni_audio_mask_override() -> None:
    if getattr(BridgeSoundEncoder, "_avlm_audio_mask_override", False):
        return

    def forward(self, sound_clips, sound_length):
        max_frames = sound_clips.size(1)
        attention_length = torch.clamp(sound_length - 1, min=0)
        attention_mask = (
            torch.arange(max_frames, device=sound_clips.device)[None, :]
            < attention_length[:, None]
        )
        output = self.encoder(
            input_features=sound_clips,
            attention_mask=attention_mask,
        )
        embedding_lengths = self.encoder._get_subsampling_output_length(sound_length)
        return output.last_hidden_state, embedding_lengths

    BridgeSoundEncoder.forward = forward
    BridgeSoundEncoder._avlm_audio_mask_override = True
