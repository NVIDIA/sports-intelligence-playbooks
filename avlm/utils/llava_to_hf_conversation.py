#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert LLaVA / ShareGPT video-sound JSONL to pure HF conversation JSONL.

Input (LLaVA):
  {
    "id": "...",
    "video-sound": "relative/path.mp4",
    "conversations": [
      {"from": "human", "value": "<video-sound>Question ..."},
      {"from": "gpt", "value": "Answer"}
    ]
  }

Output (HF conversation + optional ``id`` and ``class``):
  {
    "id": "...",
    "class": "...",
    "conversation": [
      {
        "role": "user",
        "content": [
          {"type": "video", "path": "relative/path.mp4"},
          {"type": "text", "text": "Question ..."}
        ]
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Answer"}]
      }
    ]
  }

Default split paths live in ``avlm/data/llava_to_hf_conversation.yaml`` (JSON configs
with the same schema are also supported). Video paths stay relative; training YAMLs
provide ``video_root``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_VIDEO_SOUND_TAG = re.compile(r"<video-sound>", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return repo_root() / "data" / "llava_to_hf_conversation.yaml"


@dataclass(frozen=True)
class ConversionStats:
    input_rows: int = 0
    output_rows: int = 0
    skipped_no_video: int = 0
    skipped_missing_file: int = 0
    skipped_empty_conversation: int = 0


@dataclass(frozen=True)
class ConversionConfig:
    video_root: str | None
    manifest: Path | None
    splits: dict[str, dict[str, str]]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    root = base or Path.cwd()
    return (root / candidate).resolve()


def load_conversion_config(config_path: str | Path, *, base: Path | None = None) -> ConversionConfig:
    """Load split input/output paths from YAML or JSON."""
    path = resolve_path(config_path, base=base)
    if not path.is_file():
        raise FileNotFoundError(f"conversion config not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"PyYAML is required to read {path}; install pyyaml or use a .json config",
            ) from exc
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config root must be a mapping")

    splits_raw = payload.get("splits")
    if not isinstance(splits_raw, dict) or not splits_raw:
        raise ValueError(f"{path}: config must define a non-empty 'splits' mapping")

    repo = repo_root()
    splits: dict[str, dict[str, str]] = {}
    for split_name, spec in splits_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: split {split_name!r} must be a mapping")
        input_path = spec.get("input")
        output_path = spec.get("output")
        if not input_path or not output_path:
            raise ValueError(f"{path}: split {split_name!r} requires 'input' and 'output'")
        splits[str(split_name)] = {
            "input": str(resolve_path(str(input_path), base=repo)),
            "output": str(resolve_path(str(output_path), base=repo)),
        }

    video_root = payload.get("video_root")
    manifest = payload.get("manifest")
    manifest_path = resolve_path(str(manifest), base=repo) if manifest else None

    return ConversionConfig(
        video_root=str(video_root) if video_root else None,
        manifest=manifest_path,
        splits=splits,
    )


def _resolve_video_rel(row: dict[str, Any]) -> str | None:
    rel = row.get("video-sound") or row.get("video")
    if isinstance(rel, str) and rel.strip():
        return rel.strip()
    return None


def _llava_role(from_role: str) -> str:
    return "user" if from_role.strip().lower() in ("human", "user") else "assistant"


def llava_row_to_hf_conversation(
    row: dict[str, Any],
    *,
    verify_video_path: str | None = None,
) -> dict[str, Any] | None:
    """Convert one LLaVA row to pure HF conversation schema."""
    rel_video = _resolve_video_rel(row)
    if rel_video is None:
        return None
    if verify_video_path is not None and not os.path.isfile(verify_video_path):
        return None

    conversation: list[dict[str, Any]] = []
    video_inserted = False
    for turn in row.get("conversations") or []:
        if not isinstance(turn, dict):
            continue
        role = _llava_role(str(turn.get("from", "human")))
        text = turn.get("value", "")
        has_video_marker = False
        if isinstance(text, str):
            has_video_marker = _VIDEO_SOUND_TAG.search(text) is not None
            text = _VIDEO_SOUND_TAG.sub("", text).strip()
        else:
            text = str(text).strip()

        content: list[dict[str, Any]] = []
        if role == "user" and has_video_marker and not video_inserted:
            content.append({"type": "video", "path": rel_video})
            video_inserted = True
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            continue
        conversation.append({"role": role, "content": content})

    if not conversation:
        return None

    example: dict[str, Any] = {"conversation": conversation}
    row_id = row.get("id")
    if row_id is not None:
        example["id"] = row_id
    row_class = row.get("class")
    if row_class is not None:
        example["class"] = row_class
    return example


def iter_jsonl_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object per line")
            yield row


def convert_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    video_root: str | None = None,
    verify_videos: bool = False,
) -> ConversionStats:
    """Convert one LLaVA JSONL file to pure HF conversation JSONL."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_rows = 0
    output_rows = 0
    skipped_no_video = 0
    skipped_missing_file = 0
    skipped_empty_conversation = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_handle:
        for row in iter_jsonl_rows(input_path):
            input_rows += 1

            rel_video = _resolve_video_rel(row)
            if rel_video is None:
                skipped_no_video += 1
                continue

            verify_path = None
            if verify_videos and video_root:
                verify_path = rel_video if os.path.isabs(rel_video) else os.path.join(video_root, rel_video)

            example = llava_row_to_hf_conversation(row, verify_video_path=verify_path)
            if example is None:
                if verify_videos and video_root:
                    skipped_missing_file += 1
                else:
                    skipped_empty_conversation += 1
                continue

            out_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            output_rows += 1

    return ConversionStats(
        input_rows=input_rows,
        output_rows=output_rows,
        skipped_no_video=skipped_no_video,
        skipped_missing_file=skipped_missing_file,
        skipped_empty_conversation=skipped_empty_conversation,
    )


def _log_stats(split: str, stats: ConversionStats, output_path: str) -> None:
    logger.info(
        "%s: wrote %d/%d rows → %s (skipped: no_video=%d missing_file=%d empty=%d)",
        split,
        stats.output_rows,
        stats.input_rows,
        output_path,
        stats.skipped_no_video,
        stats.skipped_missing_file,
        stats.skipped_empty_conversation,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML/JSON file with split input/output paths (default: avlm/data/llava_to_hf_conversation.yaml)",
    )
    parser.add_argument("--input", type=Path, help="Single-file mode: LLaVA JSONL input")
    parser.add_argument("--output", type=Path, help="Single-file mode: HF JSONL output")
    parser.add_argument(
        "--video-root",
        default=os.environ.get("VIDEO_ROOT"),
        help="Root for relative video paths (overrides config; used with --verify-videos)",
    )
    parser.add_argument(
        "--verify-videos",
        action="store_true",
        help="Skip rows whose video file is missing under --video-root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Convert only these split names from --config",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write conversion stats JSON (default: manifest path from config, if set)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    manifest: dict[str, Any] = {}
    config_manifest_path: Path | None = None

    if args.input is not None or args.output is not None:
        if args.input is None or args.output is None:
            logger.error("single-file mode requires both --input and --output")
            return 2
        video_root = args.video_root
        stats = convert_jsonl_file(
            args.input,
            args.output,
            video_root=video_root,
            verify_videos=args.verify_videos,
        )
        _log_stats("custom", stats, str(args.output))
        manifest["custom"] = {
            "input": str(args.input),
            "output": str(args.output),
            **stats.__dict__,
        }
    else:
        config_path = args.config or default_config_path()
        try:
            config = load_conversion_config(config_path, base=repo_root())
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.error("%s", exc)
            return 1

        video_root = args.video_root or config.video_root
        selected = args.splits or list(config.splits.keys())
        unknown = [name for name in selected if name not in config.splits]
        if unknown:
            logger.error("unknown split(s) in --splits: %s", ", ".join(unknown))
            return 2

        for split in selected:
            spec = config.splits[split]
            if not os.path.isfile(spec["input"]):
                logger.error("missing input for %s: %s", split, spec["input"])
                return 1
            stats = convert_jsonl_file(
                spec["input"],
                spec["output"],
                video_root=video_root,
                verify_videos=args.verify_videos,
            )
            _log_stats(split, stats, spec["output"])
            manifest[split] = {
                "input": spec["input"],
                "output": spec["output"],
                **stats.__dict__,
            }
        config_manifest_path = config.manifest

    manifest_path = args.manifest or config_manifest_path
    if manifest_path:
        manifest_path = resolve_path(manifest_path, base=repo_root())
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote manifest → %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
