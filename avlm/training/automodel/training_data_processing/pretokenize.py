# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pretokenization for video-sound VLM samples with neat packing."""

from __future__ import annotations

import copy
import inspect
import logging
import random
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from nemo_automodel.components.datasets.vlm.collate_fns import build_labels_from_template
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper
from nemo_automodel.components.datasets.vlm.fake_image import (
    _conversation_has_media,
    inject_fake_image_into_conversation,
    mask_fake_vision_tokens_single,
)
from nemo_automodel.components.datasets.vlm.utils import (
    _preload_media,
)
from avlm.training.automodel.training_data_processing.video_sound_utils import (
    inject_sound_token_into_example,
    load_audio_from_video,
    video_path_from_content_item,
)
from avlm.utils.batch_memory import release_decode_intermediates, release_vlm_batch

logger = logging.getLogger(__name__)


def _init_pretokenized_parent(instance, dataset, processor, **kwargs: Any) -> None:
    """Call AutoModel's wrapper init while tolerating version-specific kwargs."""
    signature = inspect.signature(PreTokenizedDatasetWrapper.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not accepts_kwargs:
        kwargs = {key: val for key, val in kwargs.items() if key in signature.parameters}
    PreTokenizedDatasetWrapper.__init__(instance, dataset, processor, **kwargs)


class SampleTooLongError(Exception):
    """Raised when a pretokenized sample exceeds ``max_length`` (packing should skip it)."""

    def __init__(self, idx: int, seq_len: int, max_length: int):
        self.idx = idx
        self.seq_len = seq_len
        self.max_length = max_length
        super().__init__(f"sample {idx}: {seq_len} tokens > max_length {max_length}")


DEFAULT_VIDEO_SAMPLE_FPS = 2.0


def _uniform_frame_indices(total_frames: int, desired_frames: int) -> list[int]:
    """Sample frame indices with the same linspace rule used by inference/VLMEval."""
    desired_frames = max(1, int(desired_frames))
    if desired_frames >= total_frames:
        return list(range(total_frames))
    if desired_frames == 1:
        return [0]

    raw_indices = np.linspace(0, total_frames - 1, desired_frames)
    return [int(i) for i in np.unique(np.round(raw_indices).astype(int))]


def compute_video_frame_indices(
    total_frames: int,
    video_fps: float,
    *,
    max_video_frames: int,
    video_sample_fps: float | None = DEFAULT_VIDEO_SAMPLE_FPS,
    temporal_patch_size: int = 2,
) -> list[int]:
    """Choose frame indices for video decoding.

    This mirrors VLMEval/vllm/huggingface sampling logic:
    compute the desired frame count from ``duration * video_sample_fps``, cap it
    at ``max_video_frames``, then uniformly linspace that many indices across the
    full clip. ``temporal_patch_size`` is kept for API compatibility but does not
    alter the sampled indices on this path.
    """
    if total_frames <= 0:
        raise ValueError(f"Video has no frames (total_frames={total_frames})")

    if video_sample_fps is not None and video_sample_fps > 0 and video_fps > 0:
        total_duration = total_frames / max(1e-6, video_fps)
        desired_frames = max(1, int(total_duration * video_sample_fps))
    else:
        desired_frames = max(1, int(max_video_frames))

    if max_video_frames > 0 and desired_frames > max_video_frames:
        desired_frames = max_video_frames

    return _uniform_frame_indices(total_frames, desired_frames)


def _temporal_patch_size(processor) -> int:
    if processor is not None and hasattr(processor, "video_processor"):
        return getattr(processor.video_processor, "temporal_patch_size", 2)
    return 2


def attach_video_metadata_to_examples(examples: Sequence[dict[str, Any]]) -> Sequence[dict[str, Any]]:
    """Build ``VideoMetadata`` on pre-decoded frames so the processor token count matches vision."""
    from transformers.video_utils import VideoMetadata

    for example in examples:
        for message in example.get("conversation", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != "video":
                    continue
                vid = item.get("video")
                if not isinstance(vid, list) or item.get("metadata") is not None:
                    continue
                if item.get("_video_fps") is None:
                    continue
                indices = item.get("_frame_indices") or []
                item["metadata"] = VideoMetadata(
                    total_num_frames=len(vid),
                    fps=item["_video_fps"],
                    frames_indices=[int(i) for i in indices],
                )
    return examples


def _decode_videos_in_example(
    example: dict[str, Any],
    max_video_frames: int,
    *,
    video_sample_fps: float | None = DEFAULT_VIDEO_SAMPLE_FPS,
    processor=None,
) -> dict[str, Any]:
    """Sample and decode string video paths in one decord pass."""
    import decord

    temporal_patch_size = _temporal_patch_size(processor)
    for message in example.get("conversation", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "video":
                continue
            vid = video_path_from_content_item(item)
            if vid is None:
                continue
            vr = decord.VideoReader(vid)
            total = len(vr)
            video_fps = vr.get_avg_fps()
            indices = compute_video_frame_indices(
                total,
                video_fps,
                max_video_frames=max_video_frames,
                video_sample_fps=video_sample_fps,
                temporal_patch_size=temporal_patch_size,
            )
            frames = vr.get_batch(indices).asnumpy()
            del vr
            item["video"] = [Image.fromarray(f).convert("RGB") for f in frames]
            item["_video_fps"] = video_fps
            item["_frame_indices"] = [int(i) for i in indices]
    return example


def _conversation_to_processor_inputs(
    conversation: list[dict[str, Any]],
    processor,
    audio_waveform: np.ndarray | None,
) -> tuple[str, list[Any], tuple[list[Any], VideoMetadata | None] | None, list[np.ndarray] | None]:
    """Build processor kwargs like ``nemotron_omni_collate_fn`` (single sample)."""
    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(processor, "image_token", "<image>")
    video_token = getattr(processor, "video_token", "<video>")

    conv_images: list[Any] = []
    conv_video: tuple[list[Any], VideoMetadata | None] | None = None
    text_conversation: list[dict[str, Any]] = []

    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "image":
                    img = item.get("image")
                    if img is not None:
                        conv_images.append(img)
                        text_parts.append(image_token)
                elif t == "video":
                    vid = item.get("video")
                    if vid is None:
                        continue
                    if isinstance(vid, str):
                        raise ValueError(
                            "Video path was not pre-decoded; call _decode_videos_in_example first",
                        )
                    if isinstance(vid, list):
                        frames = vid
                        metadata = item.get("metadata")
                        if metadata is None and item.get("_video_fps") is not None:
                            indices = item.get("_frame_indices") or []
                            metadata = VideoMetadata(
                                total_num_frames=len(frames),
                                fps=item["_video_fps"],
                                frames_indices=[int(i) for i in indices],
                            )
                    else:
                        raise ValueError(f"Unsupported video type: {type(vid)}")
                    if conv_video is not None:
                        raise NotImplementedError("Only 1 video per sample is supported")
                    conv_video = (frames, metadata)
                    text_parts.append(video_token)
                elif t == "text":
                    text_parts.append(item.get("text", ""))
            text_conversation.append({"role": message["role"], "content": "".join(text_parts)})
        else:
            text_conversation.append(message)

    text = tokenizer.apply_chat_template(text_conversation, tokenize=False)
    audios = [audio_waveform] if audio_waveform is not None else None
    return text, conv_images, conv_video, audios


class VideoSoundPreTokenizedDatasetWrapper(PreTokenizedDatasetWrapper):
    """Pretokenize video-sound samples with ffmpeg audio (for neat packing)."""

    def __init__(
        self,
        dataset,
        processor,
        max_length=None,
        max_retries=10,
        truncate=False,
        post_tokenize_hook=None,
        max_video_frames: int = 8,
        video_sample_fps: float | None = DEFAULT_VIDEO_SAMPLE_FPS,
        **pretokenize_kwargs: Any,
    ):
        _init_pretokenized_parent(
            self,
            dataset,
            processor,
            max_length=max_length,
            max_retries=max_retries,
            truncate=truncate,
            post_tokenize_hook=post_tokenize_hook,
            **pretokenize_kwargs,
        )
        self.max_video_frames = max_video_frames
        self.video_sample_fps = video_sample_fps

    def __getitem__(self, idx):
        for attempt in range(self.max_retries):
            example = None
            conversation = None
            processor_kwargs = None
            result = None
            try:
                example = copy.deepcopy(self.dataset[idx])
                example = _decode_videos_in_example(
                    example,
                    self.max_video_frames,
                    video_sample_fps=self.video_sample_fps,
                    processor=self.processor,
                )
                example = _preload_media(example, self.processor, preserve_video_metadata=True)
                conversation = example["conversation"]

                injected_fake = not _conversation_has_media(conversation)
                if injected_fake:
                    conversation = inject_fake_image_into_conversation(conversation)
                    example["conversation"] = conversation

                for message in conversation:
                    content = message.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get("image"), Image.Image):
                                item["image"] = item["image"].convert("RGB")

                audio_waveform = None
                if path := example.get("audio_video_path"):
                    target_sr = getattr(self.processor, "audio_sampling_rate", 16000)
                    audio_waveform = load_audio_from_video(path, target_sr=target_sr)
                    example["audio"] = audio_waveform

                inject_sound_token_into_example(example, self.processor)
                conversation = example["conversation"]

                text, images, video, audios = _conversation_to_processor_inputs(
                    conversation,
                    self.processor,
                    audio_waveform,
                )

                processor_kwargs = {
                    "text": [text],
                    "return_tensors": "pt",
                    "do_sample_frames": False,
                }
                if images:
                    processor_kwargs["images"] = images
                if video is not None:
                    frames, metadata = video
                    processor_kwargs["videos"] = frames
                    if metadata is not None:
                        processor_kwargs["videos_kwargs"] = {"video_metadata": metadata}
                if audios:
                    processor_kwargs["audio"] = audios

                result = self.processor(**processor_kwargs)
                had_audio = audio_waveform is not None
                release_decode_intermediates(processor_kwargs, images, video, audios, audio_waveform)
                processor_kwargs = None
                images = video = audios = audio_waveform = None

                if self.post_tokenize_hook is not None:
                    result = self.post_tokenize_hook(result, self.processor)

                input_ids = result["input_ids"][0]
                seq_len = input_ids.shape[0]

                if self.max_length is not None and seq_len > self.max_length:
                    if not self.truncate:
                        raise SampleTooLongError(idx, int(seq_len), int(self.max_length))
                    if had_audio or video is not None or images:
                        raise ValueError(
                            "truncate=True is not supported for video-sound samples "
                            "(token truncation desynchronizes media tensors)",
                        )

                labels = build_labels_from_template(
                    result["input_ids"],
                    [conversation],
                    self.processor,
                )[0]

                if self.truncate and self.max_length is not None and seq_len > self.max_length:
                    ml = self.max_length
                    input_ids = input_ids[:ml]
                    labels = labels[:ml]
                    result = {
                        k: (
                            v[:, :ml]
                            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[1] == seq_len
                            else v[:ml]
                            if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == seq_len
                            else v
                        )
                        for k, v in result.items()
                    }
                    seq_len = ml

                output: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": result["attention_mask"][0],
                    "labels": labels,
                }

                for key in (
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "second_per_grid_ts",
                    "image_position_ids",
                    "mm_token_type_ids",
                    "sound_clips",
                ):
                    if key in result and result[key] is not None:
                        output[key] = result[key]

                if had_audio:
                    if "sound_clips" not in output or output.get("sound_clips") is None:
                        raise RuntimeError(
                            "Processor returned no sound_clips despite audio input; "
                            "check processor audio support and audio_video_path",
                        )
                    target_sr = getattr(self.processor, "audio_sampling_rate", 16000)
                    fe = getattr(self.processor, "_sound_feature_extractor", None)
                    if fe is None:
                        from transformers import ParakeetFeatureExtractor

                        fe = ParakeetFeatureExtractor(sampling_rate=target_sr, feature_size=128)
                        try:
                            self.processor._sound_feature_extractor = fe
                        except Exception:
                            pass
                    clips = output.pop("sound_clips")
                    wave = clips[0] if isinstance(clips, list) else clips
                    if isinstance(wave, torch.Tensor):
                        wave = wave.cpu().numpy()
                    wave = np.asarray(wave, dtype=np.float32).squeeze()
                    audio_inputs = fe([wave], sampling_rate=target_sr, return_tensors="pt")
                    output["sound_features"] = audio_inputs["input_features"].to(torch.float32)
                    if "attention_mask" in audio_inputs:
                        output["sound_attention_mask"] = audio_inputs["attention_mask"].to(torch.long)
                    del clips, wave, audio_inputs

                if injected_fake:
                    mask_fake_vision_tokens_single(output, self.processor)

                return output

            except SampleTooLongError:
                raise
            except Exception as exc:
                logger.warning(
                    "Error processing sample %d (attempt %d/%d): %s",
                    idx,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                idx = random.randint(0, len(self.dataset) - 1)
            finally:
                # Only release locals; example/conversation must not be cleared — they can
                # alias nested structures still referenced by the cached dataset row.
                release_vlm_batch(processor_kwargs)
                release_vlm_batch(result)
                example = conversation = processor_kwargs = result = None

        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} retries")
