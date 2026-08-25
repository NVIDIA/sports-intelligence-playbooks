# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load HF conversation JSONL for video-sound VLM fine-tuning."""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

from nemo_automodel.components.datasets.vlm.datasets import (
    _load_json_or_jsonl,
    _load_jsonl_for_rank,
)

from avlm.training.automodel.training_data_processing.video_sound_utils import (
    video_path_from_content_item,
)

logger = logging.getLogger(__name__)


def _resolve_media_path(path: str, media_root: str) -> str:
    if os.path.isabs(path) or path.startswith(("http:", "https:", "file:", "data:")):
        return path
    return os.path.join(media_root, path)


def _load_video_metadata(path: str | None) -> dict[str, tuple[float, int, float]]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as metadata_file:
            raw = json.load(metadata_file)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Could not load video metadata from %s: %s", path, exc)
        return {}

    metadata: dict[str, tuple[float, int, float]] = {}
    for video_key, values in raw.items():
        try:
            duration, total_frames, fps = values
            metadata[str(video_key)] = (float(duration), int(total_frames), float(fps))
        except (TypeError, ValueError):
            logger.warning("Skipping invalid video metadata entry for %s", video_key)
    return metadata


def _copy_row_identity_fields(row: dict[str, Any], example: dict[str, Any]) -> None:
    for key in ("id", "sample_id", "key", "class"):
        if row.get(key) is not None:
            example[key] = row[key]


def row_to_example(
    row: dict[str, Any],
    video_root: str,
    video_metadata: dict[str, tuple[float, int, float]],
) -> dict[str, Any] | None:
    """Convert one HF conversation JSONL row to an Automodel training example."""
    conversation = row.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return None

    conversation = copy.deepcopy(conversation)
    first_video_abs: str | None = None
    first_video_key: str | None = None

    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue
            rel_path = video_path_from_content_item(item)
            if rel_path is None:
                continue
            abs_path = _resolve_media_path(rel_path, video_root)
            if not os.path.isfile(abs_path):
                logger.debug("Skipping missing video: %s", abs_path)
                return None
            item["video"] = abs_path
            if first_video_abs is None:
                first_video_abs = abs_path
                first_video_key = rel_path

    if first_video_abs is None:
        return None

    example: dict[str, Any] = {
        "conversation": conversation,
        # Audio is decoded from the same MP4 at collate/pretokenize time (ffmpeg).
        "audio_video_path": first_video_abs,
    }
    if first_video_key and first_video_key in video_metadata:
        example["_video_metadata"] = video_metadata[first_video_key]
    _copy_row_identity_fields(row, example)
    return example


def make_video_sound_jsonl_dataset(
    path_or_dataset: str,
    video_root: str,
    video_metadata_path: str | None = None,
    split: str = "train",
    sample_ratio: float = 1.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Load HF conversation JSONL for Automodel training.

    Each row must have a top-level ``conversation`` list. Video paths in
    ``content[].path`` (or ``content[].video``) are joined with ``video_root``.
    Every loaded example gets ``audio_video_path`` pointing at the same file so
    downstream collate/pretokenize can extract audio via ffmpeg.

    Args:
        path_or_dataset: Path to ``.jsonl`` (set under ``dataset:`` in training YAML).
        video_root: Directory prefix for relative video paths.
        video_metadata_path: Optional JSON mapping relative paths to
            ``[duration, total_frames, fps]`` for packing length estimates.
        split: ``train`` or ``validation`` (logging label).
        sample_ratio: Fraction of lines to load (deterministic, seed 42).
        **kwargs: Unused; kept for YAML ``_target_`` compatibility.

    Returns:
        List of dicts with ``conversation`` and ``audio_video_path``.
    """
    del kwargs

    jsonl_path = path_or_dataset
    if not jsonl_path:
        raise ValueError("path_or_dataset is required (set under dataset: in training YAML)")
    if not video_root:
        raise ValueError("video_root is required (set under dataset: in training YAML)")
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if not os.path.isdir(video_root):
        raise FileNotFoundError(f"Video root not found: {video_root}")

    if jsonl_path.endswith(".jsonl") and sample_ratio != 1.0:
        raw_rows, total = _load_jsonl_for_rank(jsonl_path, sample_ratio, None, None)
    else:
        raw_rows = _load_json_or_jsonl(jsonl_path)
        total = len(raw_rows)

    video_metadata = _load_video_metadata(video_metadata_path)
    examples: list[dict[str, Any]] = []
    skipped_missing = 0
    for row in raw_rows:
        converted = row_to_example(row, video_root, video_metadata)
        if converted is None:
            skipped_missing += 1
            continue
        examples.append(converted)

    if skipped_missing:
        logger.warning(
            "HF JSONL (%s): skipped %d/%d rows (missing video path or file).",
            jsonl_path,
            skipped_missing,
            len(raw_rows),
        )
    logger.info(
        "HF JSONL (%s): %d examples from %d lines (split=%s).",
        jsonl_path,
        len(examples),
        total,
        split,
    )
    return examples
