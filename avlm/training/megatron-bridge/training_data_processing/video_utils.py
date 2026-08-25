# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# JSONL video-sound collate prep: frame sampling and audio via Megatron-Bridge utils.

from __future__ import annotations

import io
import os
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image

# Bridge ``load_audio`` uses librosa/audioread for these; ffmpeg is faster.
_FFMPEG_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg", ".m4a"})


def _ffmpeg_exe() -> str | None:
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _load_audio_from_path(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load mono audio at ``target_sr``; video containers via ffmpeg, else Bridge ``load_audio``."""
    if os.path.splitext(path)[1].lower() in _FFMPEG_SUFFIXES and (ffmpeg := _ffmpeg_exe()):
        try:
            import soundfile as sf

            out = subprocess.check_output(
                [
                    ffmpeg, "-nostdin", "-loglevel", "error", "-i", path,
                    "-vn", "-ac", "1", "-ar", str(target_sr), "-f", "wav", "pipe:1",
                ],
            )
            wav, _ = sf.read(io.BytesIO(out), dtype="float32", always_2d=False)
            return np.asarray(wav, dtype=np.float32)
        except (OSError, subprocess.CalledProcessError):
            pass

    from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import load_audio

    return load_audio(path, target_sr=target_sr)


def sample_video_frames_uniform(video_path, *, max_video_frames, video_sample_fps):
    import decord

    vr = decord.VideoReader(video_path, num_threads=1)
    source_frame_count = len(vr)
    if source_frame_count == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    source_fps = float(vr.get_avg_fps()) or 30.0
    duration = source_frame_count / max(source_fps, 1e-6)
    sample_count = max(1, int(max(0.0, duration) * video_sample_fps))
    if max_video_frames > 0:
        sample_count = min(sample_count, max_video_frames)
    sample_count = min(sample_count, source_frame_count)

    if sample_count >= source_frame_count:
        indices = list(range(source_frame_count))
    elif sample_count == 1:
        indices = [0]
    else:
        raw = np.linspace(0, source_frame_count - 1, sample_count)
        indices = [int(i) for i in np.unique(np.round(raw).astype(int))]

    arrays = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(array).convert("RGB") for array in arrays]
    return frames, source_fps, indices


def decode_video_path_in_example(
    example: dict[str, Any],
    *,
    max_video_frames: int,
    video_sample_fps: float,
) -> dict[str, Any]:
    for message in example.get("conversation", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "video":
                continue
            video_path = item.get("path") or item.get("video")
            if not isinstance(video_path, str):
                continue
            frames, source_fps, frame_indices = sample_video_frames_uniform(
                video_path,
                max_video_frames=max_video_frames,
                video_sample_fps=video_sample_fps,
            )
            example["_avlm_video_frames"] = frames
            example["_avlm_video_source_fps"] = source_fps
            example["_avlm_video_frame_indices"] = frame_indices
    return example


def prepare_collate_example(
    example: dict[str, Any],
    *,
    max_video_frames: int,
    video_sample_fps: float,
) -> dict[str, Any]:
    ex = dict(example)
    audio_path = ex.pop("audio_video_path", None)
    decode_video_path_in_example(
        ex,
        max_video_frames=max_video_frames,
        video_sample_fps=video_sample_fps,
    )
    if audio_path and ex.get("audio") is None and ex.get("audio_path") is None:
        target_sr = 16000
        ex["audio"] = (_load_audio_from_path(audio_path, target_sr=target_sr), target_sr)
    return ex
