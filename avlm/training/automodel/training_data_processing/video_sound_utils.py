# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared audio decode and conversation helpers for video-sound training."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_FFMPEG_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg", ".m4a"})


def _ffmpeg_exe() -> str | None:
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def video_path_from_content_item(item: dict[str, Any]) -> str | None:
    """Return a string video path from HF ``content`` items (``video`` or ``path`` key)."""
    val = item.get("video")
    if isinstance(val, str) and val.strip():
        return val.strip()
    # Decoded frames live under ``video`` as a list; do not fall back to relative ``path``.
    if val is not None:
        return None
    val = item.get("path")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def video_has_audio_stream(video_path: str) -> bool:
    """Return True when ``video_path`` contains at least one audio stream."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return bool(proc.stdout.strip())
    ffmpeg = _ffmpeg_exe()
    if not ffmpeg:
        return False
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", video_path],
        capture_output=True,
        text=True,
    )
    stderr = probe.stderr or ""
    return any(line.strip().startswith("Stream") and " Audio:" in line for line in stderr.splitlines())


def load_audio_from_video(video_path: str, target_sr: int = 16000) -> np.ndarray | None:
    """Decode mono float32 audio from a video container, or ``None`` if there is no audio track."""
    if not video_has_audio_stream(video_path):
        return None
    if os.path.splitext(video_path)[1].lower() in _FFMPEG_SUFFIXES and (ffmpeg := _ffmpeg_exe()):
        try:
            proc = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    str(target_sr),
                    "-",
                ],
                capture_output=True,
                check=True,
            )
            if not proc.stdout:
                return None
            pcm = np.frombuffer(proc.stdout, dtype=np.int16)
            return (pcm.astype(np.float32) / 32768.0).copy()
        except (OSError, subprocess.CalledProcessError):
            pass

    import librosa

    wav, _ = librosa.load(video_path, sr=target_sr, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    return wav if wav.size else None


def inject_sound_token_into_example(example: dict[str, Any], processor) -> dict[str, Any]:
    """Insert ``<sound>`` after each video block when the example carries audio."""
    sound_token = getattr(processor, "audio_token", None)
    if not sound_token or example.get("audio") is None:
        return example

    new_conversation: list[dict[str, Any]] = []
    for message in example.get("conversation", []):
        msg = dict(message)
        content = msg.get("content")
        if not isinstance(content, list):
            new_conversation.append(msg)
            continue

        new_content: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue
            new_content.append(dict(item))
            if item.get("type") == "video" and item.get("video") is not None:
                # new_content.append({"type": "text", "text": sound_token})
                new_content.append({"type": "text", "text": f"\n{sound_token}\n"})
        msg["content"] = new_content
        new_conversation.append(msg)

    example["conversation"] = new_conversation
    return example


def inject_sound_token_after_video(
    examples: Sequence[dict[str, Any]],
    processor,
) -> list[dict[str, Any]]:
    """Batch wrapper around :func:`inject_sound_token_into_example`."""
    return [inject_sound_token_into_example(dict(ex), processor) for ex in examples]
