# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hub checkpoint video_io.py parity for vLLM (--media-io-kwargs video_backend=hub_checkpoint).

Matches Hub snapshot video_io:
  - int(duration * fps) frame budget (then cap num_frames)
  - np.unique(np.round(np.linspace(...))) indices
  - VideoMetadata-style dict: total_num_frames = len(sampled), do_sample_frames = False
  - Frame decode via decord when importable (same as Hub); else OpenCV fallback.

Decord is imported from the serve interpreter (the container Python, or the slim extra
venv when VLLM_EXTRA_PIP_SPEC is set). If it is missing, decode falls back to OpenCV.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)

logger = init_logger(__name__)


def _decord_available() -> bool:
    try:
        import decord  # noqa: F401

        return True
    except ImportError:
        return False


@VIDEO_LOADER_REGISTRY.register("hub_checkpoint")
class HubCheckpointVideoBackend(VideoBackend):
    """Hub video_io index math + decord decode (Hub inference path)."""

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> list[int]:
        total_frames = source.total_frames_num
        total_duration = source.duration
        target_fps = target.fps
        nframe_max = target.num_frames

        if target_fps > 0:
            required_frames = int(total_duration * target_fps)
            desired_frames = max(1, required_frames)
            if nframe_max > 0 and desired_frames > nframe_max:
                desired_frames = nframe_max
            if desired_frames >= total_frames:
                indices = list(range(total_frames))
            elif desired_frames == 1:
                indices = [0]
            else:
                raw_indices = np.linspace(0, total_frames - 1, desired_frames)
                indices = list(np.unique(np.round(raw_indices).astype(int)))
            return [int(i) for i in indices]

        return super().compute_frames_index_to_sample(source, target, **kwargs)

    @classmethod
    def create_hf_metadata(
        cls,
        source: VideoSourceMetadata,
        valid_frame_indices: list[int],
        video_backend: str,
    ) -> dict[str, Any]:
        sampled = len(valid_frame_indices)
        return {
            "total_num_frames": sampled,
            "fps": source.original_fps,
            "duration": source.duration,
            "video_backend": video_backend,
            "frames_indices": [int(i) for i in valid_frame_indices],
            "do_sample_frames": False,
        }

    @classmethod
    def _source_from_decord(cls, video_path: str) -> VideoSourceMetadata:
        import decord

        vr = decord.VideoReader(video_path)
        total = len(vr)
        fps = float(vr.get_avg_fps())
        duration = total / fps if fps > 0 else 0.0
        return VideoSourceMetadata(
            total_frames_num=total,
            original_fps=fps,
            duration=duration,
        )

    @classmethod
    def _load_bytes_decord(
        cls,
        data: bytes,
        num_frames: int,
        fps: int,
        max_duration: int,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        import decord

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            source = cls._source_from_decord(tmp_path)
            target = VideoTargetMetadata(
                num_frames=num_frames, fps=fps, max_duration=max_duration
            )
            frame_idx = cls.compute_frames_index_to_sample(
                source=source, target=target, **kwargs
            )
            vr = decord.VideoReader(tmp_path)
            frames_list = [vr[i].asnumpy() for i in frame_idx]
            if not frames_list:
                raise ValueError("decord produced no frames")
            frames = np.stack(frames_list, axis=0)
            valid = [int(i) for i in frame_idx]
            return frames, cls.create_hf_metadata(
                source=source,
                video_backend="decord",
                valid_frame_indices=valid,
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
        frame_recovery: bool = False,
        *,
        backend: Literal["opencv", "pyav"] = "opencv",
        decode: str | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """decode: 'decord' (default when available), 'opencv', or 'auto'."""
        mode = (decode or os.environ.get("VLLM_HUB_VIDEO_DECODE", "auto")).lower()
        use_decord = mode in {"decord", "auto"} and _decord_available()
        if mode == "decord" and not use_decord:
            logger.warning(
                "hub_checkpoint: decode=decord requested but decord not importable; "
                "install it with VLLM_EXTRA_PIP_SPEC or use the image copy"
            )

        if use_decord:
            try:
                return cls._load_bytes_decord(
                    data,
                    num_frames=num_frames,
                    fps=fps,
                    max_duration=max_duration,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "hub_checkpoint: decord load failed (%s); falling back to opencv",
                    exc,
                )

        frames, metadata = super().load_bytes(
            data,
            num_frames=num_frames,
            fps=fps,
            max_duration=max_duration,
            frame_recovery=frame_recovery,
            backend=backend,
            **kwargs,
        )
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata["video_backend"] = f"opencv{cls._sampling_suffix}"
        return frames, metadata
