# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Hugging Face preprocessing adapted to Megatron Core's Omni input contract."""

import logging
import os

import numpy as np
import torch

from megatron.bridge.data.energon.nemotron_omni_task_encoder import (
    NemotronOmniTaskEncoder,
    NemotronOmniTaskSample,
)
from megatron.bridge.data.energon.task_encoder_utils import (
    IGNORE_INDEX,
    cook_chatml_sample,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import compute_mel_features
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens
from transformers.video_utils import VideoMetadata

from ..mcore_encoding_utils import (
    build_nemotron_moe_assistant_loss_mask,
    resolve_nemotron_moe_loss_mask_ids,
)


logger = logging.getLogger(__name__)


def _patchify_video(frames, patch_dim):
    """Convert HF-normalized frames to MCore's packed RADIO patch layout."""
    num_frames, channels, height, width = frames.shape
    patches_y = height // patch_dim
    patches_x = width // patch_dim
    patches = (
        frames.contiguous()
        .reshape(num_frames, channels, patches_y, patch_dim, patches_x, patch_dim)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(1, num_frames * patches_y * patches_x, channels * patch_dim * patch_dim)
        .contiguous()
    )
    sizes = torch.tensor([[height, width]] * num_frames, dtype=torch.long)
    return patches, sizes


class HfMcoreNemotronOmniTaskEncoder(NemotronOmniTaskEncoder):
    """Run the model's HF processor first, then adapt its result for MCore."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assistant_header_ids, self._im_end_id = resolve_nemotron_moe_loss_mask_ids(self._tokenizer)

    @property
    def _pad_token_id(self):
        pad_id = self._tokenizer.pad_token_id
        if pad_id is not None and int(pad_id) == int(self._im_end_id):
            return 0
        return int(pad_id) if pad_id is not None else 0

    def _processor_conversation(self, conversation):
        video_token = getattr(self.processor, "video_token", "<video>")
        audio_token = getattr(self.processor, "audio_token", "<so_embedding>")
        video_count = 0
        converted = []

        for turn in conversation:
            parts = []
            for item in turn["content"]:
                item_type = item.get("type")
                if item_type == "video":
                    video_count += 1
                    parts.append(video_token)
                    # parts.append(audio_token)
                    parts.append(f"\n{audio_token}\n")
                elif item_type == "text":
                    parts.append(str(item.get("text", "")))
            converted.append({"role": turn["role"], "content": "".join(parts)})

        if video_count != 1:
            raise ValueError(f"Expected exactly one video placeholder, found {video_count}")
        return converted

    def encode_sample(self, sample):
        video_frames = sample.videos
        if os.environ.get("AVLM_HF_RESIZE", "").lower() in ("1", "true"):
            video_frames = [
                frame.convert("RGB").resize((512, 512))
                for frame in video_frames
            ]
        waveform = sample.audio.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

        conversation = cook_chatml_sample(sample.conversation)
        text_conversation = self._processor_conversation(conversation)
        prompt = self._tokenizer.apply_chat_template(
            text_conversation,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        subflavors = sample.__subflavors__ or {}
        source_fps = float(subflavors["video_source_fps"])
        frame_indices = subflavors["video_frame_indices"]
        metadata = VideoMetadata(
            total_num_frames=len(video_frames),
            fps=source_fps,
            frames_indices=frame_indices,
        )

        processor_output = self.processor(
            text=[prompt],
            videos=video_frames,
            audio=[waveform],
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
            do_sample_frames=False,
        )

        input_ids = processor_output["input_ids"]
        input_ids_np = input_ids[0].detach().cpu().numpy()

        loss_mask_np = build_nemotron_moe_assistant_loss_mask(
            input_ids_np,
            assistant_header_ids=self._assistant_header_ids,
            im_end_id=self._im_end_id,
        )
        labels_np = np.full(len(input_ids_np), IGNORE_INDEX, dtype=np.int64)
        labels_np[:-1] = input_ids_np[1:]
        shifted_loss = np.zeros_like(loss_mask_np)
        shifted_loss[:-1] = loss_mask_np[1:]
        labels_np[shifted_loss == 0.0] = IGNORE_INDEX

        img_start_id = self._tokenizer.convert_tokens_to_ids("<img>")
        img_end_id = self._tokenizer.convert_tokens_to_ids("</img>")
        visual_groups = int((input_ids == img_start_id).sum().item())
        converted = {
            "input_ids": torch.from_numpy(input_ids_np).unsqueeze(0),
            "labels": torch.from_numpy(labels_np).unsqueeze(0),
            "loss_mask": torch.from_numpy(shifted_loss).unsqueeze(0),
        }

        converted = adjust_image_tokens(
            converted,
            torch.ones(visual_groups, dtype=torch.long),
            img_start_id,
            img_end_id,
        )

        processed_pixels = processor_output["pixel_values_videos"]

        packed_pixels, imgs_sizes = _patchify_video(
            processed_pixels,
            self.patch_dim,
        )
        visual_tensors = {"pixel_values": packed_pixels}
        num_frames = torch.tensor([len(processed_pixels)], dtype=torch.long)

        tokens_per_group = (
            (int(imgs_sizes[0, 0]) // self.patch_dim)
            * (int(imgs_sizes[0, 1]) // self.patch_dim)
            // 4
        )
        num_image_tiles = torch.full(
            (visual_groups,),
            tokens_per_group,
            dtype=torch.int,
        )

        final_length = int(converted["input_ids"].shape[-1]) - visual_groups
        final_length += visual_groups * tokens_per_group
        if final_length > self.seq_length:
            logger.warning(
                "Sample %s is estimated at %d tokens after visual expansion; seq_length is %d",
                sample.__key__,
                final_length,
                self.seq_length,
            )

        sound_clips = compute_mel_features(
            waveform,
            sampling_rate=16000,
            num_mel_bins=self.num_mel_bins,
        )
        sound_length = torch.tensor(sound_clips.shape[0], dtype=torch.long)

        return NemotronOmniTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__ or {},
            input_ids=converted["input_ids"].squeeze(0).clone(),
            labels=converted["labels"].squeeze(0).clone(),
            loss_mask=converted["loss_mask"].squeeze(0).clone(),
            visual_tensors=visual_tensors,
            num_patches=None,
            sound_clips=sound_clips,
            sound_length=sound_length,
            imgs_sizes=imgs_sizes,
            num_frames=num_frames,
            num_image_tiles=num_image_tiles,
        )
