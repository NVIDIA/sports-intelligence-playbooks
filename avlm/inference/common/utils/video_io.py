# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Gotten from 
# https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning/blob/f8f1bb79ab6fd77d3c34046bd93dc16b0c7bb996/video_io.py

import os
import base64
import mimetypes
import tempfile
from contextlib import contextmanager
from PIL import Image
import io
from transformers.video_utils import VideoMetadata


def encode_pil_to_jpeg_data_url(pil_image):
    from io import BytesIO
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sample_video_frames_to_data_urls(video_path_local, fps=1, nframe=0, nframe_max=-1,
                                     include_metadata=True):
    """
    Sample frames from a video and return base64-encoded data URLs along with metadata.
    
    Args:
        video_path_local: Path to the video file
        fps: Target frames per second for sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample (used if fps <= 0)
        nframe_max: Maximum number of frames to sample
    
    Returns:
        tuple: (frame_data_urls, metadata)
        - frame_data_urls: List of base64-encoded frame images
        - metadata: VideoMetadata dataclass containing info about the sampled frames:
            - total_num_frames: Number of sampled frames
            - fps: Effective frame rate of the sampled frames
            - duration: Duration covered by the sampled frames (in seconds)
            - video_backend: Backend used for video processing ('decord')
    """
    import numpy as np
    from PIL import Image
    import decord

    vid = decord.VideoReader(video_path_local)
    total_frames = len(vid)
    video_fps = vid.get_avg_fps()
    total_duration = total_frames / max(1e-6, video_fps)

    if fps > 0:
        required_frames = int(total_duration * fps)
        desired_frames = max(1, required_frames)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))
    else:
        desired_frames = max(1, int(nframe) if nframe and nframe > 0 else 8)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))

    images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
    frame_urls = [encode_pil_to_jpeg_data_url(im) for im in images]
    
    # Calculate timestamps for each sampled frame
    timestamps = [float(idx) / video_fps for idx in indices]
    
    # Calculate metadata for the sampled frames
    sampled_num_frames = len(indices)
    
    # Duration is the time span from first to last frame
    if len(timestamps) > 1:
        sampled_duration = timestamps[-1] - timestamps[0]
        sampled_fps = (sampled_num_frames - 1) / sampled_duration if sampled_duration > 0 else 1.0
    else:
        # Single frame case
        sampled_duration = None
        sampled_fps = None
    
    # metadata = VideoMetadata(
    #     total_num_frames=sampled_num_frames,
    #     fps=sampled_fps,
    #     duration=sampled_duration,
    #     video_backend=None,
    # )

    if not include_metadata:
        return frame_urls, None

    metadata = VideoMetadata(
        total_num_frames=len(indices),
        fps=video_fps,
        frames_indices=[int(i) for i in indices],
    )
    
    return frame_urls, metadata


@contextmanager
def local_video_path(path_or_url):
    """Yield a filesystem path for local ``.mp4`` files or ``data:video/mp4`` URLs.

    HTTP(S) URLs and other non-video inputs yield ``None``. Temporary files are removed
    when the context exits. Batch JSONL inference only uses local paths under ``video_root``.
    """
    val = str(path_or_url or "")
    low = val.lower()

    if low.startswith("data:"):
        if not low.startswith("data:video/mp4"):
            yield None
            return
        _, _, b64part = val.partition(",")
        if not b64part:
            yield None
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            tmp.write(base64.b64decode(b64part))
            tmp.flush()
            tmp.close()
            yield tmp.name
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        return

    if os.path.exists(val):
        mime, _ = mimetypes.guess_type(val)
        if mime == "video/mp4" or (mime is None and val.lower().endswith(".mp4")):
            yield val
            return

    yield None


def maybe_path_or_url_to_data_urls(path_or_url, fps=1, nframe=0, nframe_max=-1,
                                   include_metadata=True):
    """
    Convert a local path or ``data:`` URL to frame data URLs (images or sampled video).

    HTTP(S) video URLs are not downloaded; the input is returned unchanged with no metadata.
    
    Args:
        path_or_url: Local filesystem path or data URL
        fps: Target frames per second for video sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample from video (used if fps <= 0)
        nframe_max: Maximum number of frames to sample
    
    Returns:
        tuple: (data_urls, metadata)
        - data_urls: List of base64-encoded data URLs
        - metadata: VideoMetadata dataclass with video metadata or None for images
    """
    val = str(path_or_url or "")
    low = val.lower()

    if low.startswith("http://") or low.startswith("https://"):
        raise ValueError(
            "Remote HTTP(S) media URLs are not supported for batch inference. "
            "Use a local .mp4 path (JSONL paths are resolved under video_root)."
        )

    if low.startswith("data:") and not low.startswith("data:video/mp4"):
        return [val], None

    with local_video_path(path_or_url) as local:
        if local is not None:
            return sample_video_frames_to_data_urls(
                local,
                fps=fps,
                nframe=nframe,
                nframe_max=nframe_max,
                include_metadata=include_metadata,
            )

    if os.path.exists(val):
        mime, _ = mimetypes.guess_type(val)
        if mime and mime.startswith("image/"):
            with open(val, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return [f"data:{mime};base64,{b64}"], None
        with open(val, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return [f"data:image/jpeg;base64,{b64}"], None

    return [val], None


def pil_image_from_base64(b64_str: str) -> Image.Image:
    # Handle data URLs like "data:image/png;base64,...."
    if b64_str.startswith('data:'):
        b64_str = b64_str.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))
