# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import copy
import heapq
import json
import logging
import os
import re
from typing import Any, Literal

from megatron.bridge.data.vlm_datasets.preloaded_provider import (
    _load_preloaded_examples,
    _normalize_paths,
)

logger = logging.getLogger(__name__)

JsonlFormat = Literal["llava", "hf"]
_VIDEO_SOUND_TAG = re.compile(r"<video-sound>", re.IGNORECASE)
_TOKENS_PER_VIDEO_FRAME = 135
_SOUND_TOKEN_BUDGET = 256


def normalize_jsonl_format(fmt: str) -> JsonlFormat:
    key = (fmt or "llava").strip().lower()
    if key in ("llava", "sharegpt", "llava_sharegpt"):
        return "llava"
    if key in ("hf", "bridge", "conversation", "hf_conversation"):
        return "hf"
    raise ValueError(f"Unsupported jsonl_format={fmt!r}; use llava or hf")


def _resolve_media_path(path: str, media_root: str) -> str:
    if os.path.isabs(path) or path.startswith(("http:", "https:", "file:", "data:")):
        return path
    return _normalize_paths([path], media_root)[0]


def _copy_row_identity_fields(row: dict[str, Any], example: dict[str, Any]) -> None:
    """Preserve JSONL row ids so collate logging can key on QA/MCQ, not video path."""
    for key in ("id", "sample_id", "key", "class"):
        if row.get(key) is not None:
            example[key] = row[key]
    rel = row.get("video-sound") or row.get("video")
    if rel is not None:
        example["video-sound"] = rel


def row_to_llava_example(row: dict[str, Any], video_root: str) -> dict[str, Any] | None:
    """LLaVA / ShareGPT rows: conversations + video-sound (or video) column."""
    rel = row.get("video-sound") or row.get("video")
    if not rel:
        return None
    video_path = _resolve_media_path(rel, video_root)
    if not os.path.isfile(video_path):
        return None

    conversation: list[dict[str, Any]] = []
    video_inserted = False
    for turn in row.get("conversations", []):
        role = "user" if turn.get("from", "human").lower() in ("human", "user") else "assistant"
        text = turn.get("value", "")
        has_video_marker = False
        if isinstance(text, str):
            has_video_marker = _VIDEO_SOUND_TAG.search(text) is not None
            text = _VIDEO_SOUND_TAG.sub("", text).strip()
        content: list[dict[str, Any]] = []
        if role == "user" and has_video_marker and not video_inserted:
            content.append({"type": "video", "path": video_path})
            video_inserted = True
        if text:
            content.append({"type": "text", "text": text})
        conversation.append({"role": role, "content": content})

    example: dict[str, Any] = {"conversation": conversation}
    if row.get("video-sound"):
        example["audio_video_path"] = video_path
    _copy_row_identity_fields(row, example)
    return example


def row_to_hf_example(row: dict[str, Any], media_root: str) -> dict[str, Any] | None:
    """HF / Bridge rows: top-level conversation list (processor chat-template schema)."""
    conversation = row.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return None

    conversation = copy.deepcopy(conversation)
    example: dict[str, Any] = {"conversation": conversation}
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("video", "image"):
                for key in ("path", "video", "image", "url"):
                    val = item.get(key)
                    if isinstance(val, str) and not val.startswith("data:"):
                        item[key] = _resolve_media_path(val, media_root)
                        if item.get("type") == "video":
                            example["audio_video_path"] = item[key]

    for key in ("audio_path", "audio_video_path"):
        if key not in row:
            continue
        val = row[key]
        if key.endswith("_path") and isinstance(val, str):
            val = _resolve_media_path(val, media_root)
            if not os.path.isfile(val):
                return None
        example[key] = val
    _copy_row_identity_fields(row, example)
    return example


def row_to_example(row: dict[str, Any], media_root: str, jsonl_format: JsonlFormat) -> dict[str, Any] | None:
    if jsonl_format == "llava":
        return row_to_llava_example(row, media_root)
    return row_to_hf_example(row, media_root)


def load_jsonl_examples(
    jsonl_path: str,
    media_root: str,
    *,
    jsonl_format: JsonlFormat = "llava",
    max_samples: int | None = None,
    sample_ratio: float | None = None,
    rng_seed: int = 42,
    video_metadata_path: str | None = None,
) -> list[dict[str, Any]]:
    fmt = normalize_jsonl_format(jsonl_format)
    video_metadata: dict[str, tuple[float, int, float]] = {}
    if video_metadata_path:
        try:
            with open(video_metadata_path, "r", encoding="utf-8") as metadata_file:
                raw_metadata = json.load(metadata_file)
            for video_key, values in raw_metadata.items():
                duration, total_frames, fps = values
                video_metadata[os.path.normpath(str(video_key))] = (
                    float(duration),
                    int(total_frames),
                    float(fps),
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not load video metadata from %s: %s", video_metadata_path, exc)

    examples: list[dict[str, Any]] = []
    for row in _load_preloaded_examples(jsonl_path):
        ex = row_to_example(row, media_root, fmt)
        if ex is not None:
            if video_metadata:
                video_key = row.get("video-sound") or row.get("video")
                if not isinstance(video_key, str):
                    for message in row.get("conversation", []):
                        for item in message.get("content", []) if isinstance(message.get("content"), list) else []:
                            if isinstance(item, dict) and item.get("type") == "video":
                                video_key = item.get("path") or item.get("video") or item.get("url")
                                if isinstance(video_key, str):
                                    break
                        if isinstance(video_key, str):
                            break
                if isinstance(video_key, str):
                    normalized = os.path.normpath(video_key)
                    if os.path.isabs(normalized):
                        normalized = os.path.relpath(normalized, media_root)
                    if normalized in video_metadata:
                        ex["_video_metadata"] = video_metadata[normalized]
            examples.append(ex)
        if max_samples is not None and len(examples) >= max_samples:
            break
    if sample_ratio is not None and 0 < sample_ratio < 1 and examples:
        import random

        keep = max(1, int(len(examples) * sample_ratio))
        examples = random.Random(rng_seed).sample(examples, min(keep, len(examples)))
    if not examples:
        raise ValueError(f"No usable examples loaded from {jsonl_path} (jsonl_format={fmt})")
    logger.info("Loaded %d %s JSONL examples from %s", len(examples), fmt, jsonl_path)
    return examples


def plan_sequence_packs(
    examples: list[dict[str, Any]],
    *,
    seq_length: int,
    packing_ratio: float,
    max_video_frames: int,
    video_sample_fps: float,
) -> list[list[dict[str, Any]]]:
    """Deterministically group examples into variable-cardinality token-budget packs."""
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive for packing, got {seq_length}")
    if not 0 < packing_ratio <= 1:
        raise ValueError(f"packing_ratio must be in (0, 1], got {packing_ratio}")
    capacity = max(1, int(seq_length * packing_ratio))
    estimated_lengths: list[int] = []

    for example in examples:
        text_tokens = example.get("_text_tokens")
        if text_tokens is None:
            total_chars = 0
            for message in example.get("conversation", []):
                content = message.get("content", [])
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    total_chars += sum(
                        len(str(item.get("text", "")))
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
            text_tokens = total_chars // 3

        metadata = example.get("_video_metadata")
        total_frames = 0
        source_fps = 0.0
        if metadata is not None:
            _, total_frames, source_fps = metadata
        else:
            video_path = None
            for message in example.get("conversation", []):
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        video_path = item.get("path") or item.get("video") or item.get("url")
                        if isinstance(video_path, str):
                            break
                if isinstance(video_path, str):
                    break
            if isinstance(video_path, str):
                try:
                    import decord

                    reader = decord.VideoReader(video_path, num_threads=1)
                    total_frames = len(reader)
                    source_fps = float(reader.get_avg_fps()) or 30.0
                except Exception as exc:
                    logger.warning("Could not read video metadata for packing estimate %s: %s", video_path, exc)

        if total_frames > 0:
            if video_sample_fps > 0 and source_fps > 0:
                sampled_frames = max(1, int((total_frames / source_fps) * video_sample_fps))
            else:
                sampled_frames = total_frames
            if max_video_frames > 0:
                sampled_frames = min(sampled_frames, max_video_frames)
            sampled_frames = min(sampled_frames, total_frames)
        else:
            sampled_frames = max(1, max_video_frames)

        estimate = int(text_tokens) + sampled_frames * _TOKENS_PER_VIDEO_FRAME
        if example.get("audio_video_path"):
            estimate += _SOUND_TOKEN_BUDGET
        estimated_lengths.append(max(1, min(estimate, capacity)))

    sorted_indices = sorted(range(len(examples)), key=lambda idx: (-estimated_lengths[idx], idx))
    heap: list[tuple[int, int]] = []
    bins: list[list[int]] = []
    for idx in sorted_indices:
        length = estimated_lengths[idx]
        if heap and heap[0][0] + length <= capacity:
            fill, bin_idx = heapq.heappop(heap)
            bins[bin_idx].append(idx)
            heapq.heappush(heap, (fill + length, bin_idx))
        else:
            bin_idx = len(bins)
            bins.append([idx])
            heapq.heappush(heap, (length, bin_idx))

    total_tokens = sum(estimated_lengths)
    utilization = total_tokens / (len(bins) * capacity) if bins else 0.0
    logger.info(
        "Planned %d examples into %d packs (capacity=%d, estimated utilization=%.1f%%)",
        len(examples),
        len(bins),
        capacity,
        utilization * 100,
    )
    return [[examples[idx] for idx in bin_indices] for bin_indices in bins]
