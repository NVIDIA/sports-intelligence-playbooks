# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate wrapper for video-sound VLM samples."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from nemo_automodel.components.datasets.vlm.collate_fns import nemotron_omni_collate_fn
from nemo_automodel.components.datasets.vlm.utils import _preload_media

from avlm.training.automodel.training_data_processing.pretokenize import (
    DEFAULT_VIDEO_SAMPLE_FPS,
    SampleTooLongError,
    _decode_videos_in_example,
    attach_video_metadata_to_examples,
)
from avlm.training.automodel.training_data_processing.video_sound_utils import (
    inject_sound_token_after_video,
    load_audio_from_video,
)

logger = logging.getLogger(__name__)


def _patch_vlm_template_labels() -> None:
    """Use chat-template markers for labels instead of fragile BPE pattern matching.

    Stock ``nemotron_omni_collate_fn`` calls legacy ``build_labels``, which re-tokenizes
    assistant text and searches for it in the multimodal ``input_ids``. That often fails
    for video+audio samples and logs full token tensors on every miss. SFT pretokenize
    already uses ``build_labels_from_template``; apply the same strategy for LoRA collate.
    """
    import nemo_automodel.components.datasets.vlm.collate_fns as collate_mod

    if getattr(collate_mod, "_avlm_vlm_template_labels_patched", False):
        return

    _legacy_build_labels = collate_mod.build_labels
    _collate_logger = logging.getLogger(collate_mod.__name__)

    def _legacy_build_labels_quiet(*args, **kwargs):
        prev_level = _collate_logger.level
        _collate_logger.setLevel(logging.ERROR)
        try:
            return _legacy_build_labels(*args, **kwargs)
        finally:
            _collate_logger.setLevel(prev_level)

    def _build_labels_via_template(input_ids_batch, conversations, processor):
        collate_mod.build_labels = _legacy_build_labels_quiet
        try:
            return collate_mod.build_labels_from_template(input_ids_batch, conversations, processor)
        finally:
            collate_mod.build_labels = _build_labels_via_template

    collate_mod.build_labels = _build_labels_via_template
    collate_mod._avlm_vlm_template_labels_patched = True


_patch_vlm_template_labels()


def _preload_sampled_videos(
    examples: Sequence[dict[str, Any]],
    processor,
    *,
    max_video_frames: int,
    video_sample_fps: float | None = DEFAULT_VIDEO_SAMPLE_FPS,
) -> list[dict[str, Any]]:
    """Sample frames (fps and/or cap) and decode before stock collate re-samples."""
    loaded: list[dict[str, Any]] = []
    for example in examples:
        ex = dict(example)
        ex = _decode_videos_in_example(
            ex,
            max_video_frames,
            video_sample_fps=video_sample_fps,
            processor=processor,
        )
        loaded.append(_preload_media(ex, processor, preserve_video_metadata=True))
    return loaded


def video_sound_collate_fn(
    examples: Sequence[dict[str, Any]],
    processor,
    max_length: Optional[int] = None,
    max_video_frames: int = 8,
    video_sample_fps: float | None = DEFAULT_VIDEO_SAMPLE_FPS,
) -> dict[str, Any]:
    """Load video-sound audio tracks, then delegate to ``nemotron_omni_collate_fn``."""
    target_sr = getattr(processor, "audio_sampling_rate", 16000)
    prepared: list[dict[str, Any]] = []
    for example in examples:
        ex = dict(example)
        audio_path = ex.pop("audio_video_path", None)
        if audio_path and ex.get("audio") is None:
            try:
                ex["audio"] = load_audio_from_video(audio_path, target_sr=target_sr)
            except Exception:
                logger.exception("Failed to load audio from %s", audio_path)
                raise
        prepared.append(ex)
    prepared = inject_sound_token_after_video(prepared, processor)
    prepared = _preload_sampled_videos(
        prepared,
        processor,
        max_video_frames=max_video_frames,
        video_sample_fps=video_sample_fps,
    )
    prepared = attach_video_metadata_to_examples(prepared)
    try:
        batch = nemotron_omni_collate_fn(
            prepared,
            processor,
            max_length=None,
            max_video_frames=max_video_frames,
        )
        if max_length is not None:
            seq_len = int(batch["input_ids"].shape[-1])
            if seq_len > max_length:
                raise SampleTooLongError(-1, seq_len, max_length)
        return batch
    finally:
        prepared.clear()
