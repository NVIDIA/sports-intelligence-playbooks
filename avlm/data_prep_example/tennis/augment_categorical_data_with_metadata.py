#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augment categorical MCQ/QA JSON samples with point-level annotation metadata.

Reads HuggingFace-style conversation records from a categorical data directory
and enriches each sample with a ``metadata`` block compatible with
``prepare_opencomment_eval_data.py`` (from ai4g-vila-internal), using raw
aggregated annotation JSON as the metadata source.

Original input files are never modified unless ``--in-place`` is passed. In that
case, each source file is copied to a timestamped backup directory under the
input directory before being overwritten. After a successful run the backup is
kept by default; pass ``--discard-backup`` (or ``keep_backup=False``) to remove
it — useful when a pre-metadata snapshot already exists elsewhere (e.g. the
pipeline's ``generated/`` stage).

Usage:
    python augment_categorical_data_with_metadata.py \\
        --input-dir ../../data/categorical_data \\
        --metadata-dir aggregated_data_complete

    python augment_categorical_data_with_metadata.py \\
        --input-dir ../../data/categorical_data \\
        --metadata-dir aggregated_data_complete \\
        --output-dir ../../data/categorical_data_with_metadata

    python augment_categorical_data_with_metadata.py \\
        --input-dir ../../data/categorical_data \\
        --metadata-dir aggregated_data_complete \\
        --in-place --discard-backup
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Annotation QC / audit workflow fields — excluded from enriched output.
EXCLUDED_POINT_FIELDS = {
    "qc_feedback",
    "qc_pass_fail",
    "audit_feedback",
    "audit_pass_fail",
    "comments_audit",
    "rework_counter",
    "timestamp_correct",
}

REFERENCE_POINT_FIELDS = {
    "id",
    "point_id",
    "start_time",
    "end_time",
    "server",
    "receiver",
    "winner",
    "how_point_ended",
    "point_end_type",
    "how_point_ended_description",
    "point_caption",
    "audio_cues",
    "video",
    "score_before",
    "score_after",
    "current_score",
    "serving_score",
    "receiving_score",
    "num_serving_attempts_until_successful",
    "num_shots_exchanged",
    "point_start_in_segment",
    "point_concluded_in_segment",
}

SCORE_PLAYER_RE = re.compile(
    r'^\s*[\'"]?(?P<p1>.+?)[\'"]?\s+vs\.?\s+[\'"]?(?P<p2>.+?)[\'"]?\s*[:，,]',
    re.IGNORECASE,
)


@dataclass
class VideoContext:
    video_id: str
    file_path: str
    player_names: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetadataCache:
    points: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    videos: Dict[str, VideoContext] = field(default_factory=dict)


def instance_value(data: dict[str, Any], class_name: str) -> str:
    for instance in data.get("instances", []):
        if instance.get("className") != class_name:
            continue
        attributes = instance.get("attributes", [])
        if not attributes:
            continue
        value = str(attributes[0].get("name", "") or "").strip()
        if value:
            return value
    return ""


def player_names_from_scores(data: dict[str, Any]) -> Tuple[str, str]:
    for instance in data.get("instances", []):
        if instance.get("className") != "table":
            continue
        attributes = instance.get("attributes", [])
        if not attributes:
            continue
        rows = attributes[0].get("name", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            for key in ("score_before", "score_after"):
                match = SCORE_PLAYER_RE.match(str(row.get(key) or ""))
                if match:
                    return match.group("p1").strip(), match.group("p2").strip()
    return "", ""


def extract_player_names(data: dict[str, Any]) -> Dict[str, str]:
    player1 = instance_value(data, "player_1_name")
    player2 = instance_value(data, "player_2_name")
    if not player1 or not player2:
        score_p1, score_p2 = player_names_from_scores(data)
        player1 = player1 or score_p1 or "Player 1"
        player2 = player2 or score_p2 or "Player 2"
    return {
        "player1_name": player1 or "Player 1",
        "player2_name": player2 or "Player 2",
    }


def load_metadata_cache(metadata_dir: Path, metadata_label: str) -> MetadataCache:
    cache = MetadataCache()
    json_files = sorted(metadata_dir.glob("*.json"))
    print(f"Loading metadata from {metadata_dir} ({len(json_files)} files)")

    for json_file in json_files:
        video_id = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"  Warning: failed to load {json_file.name}: {exc}")
            continue

        player_names = extract_player_names(data)
        cache.videos[video_id] = VideoContext(
            video_id=video_id,
            file_path=f"{metadata_label}/{json_file.name}",
            player_names=player_names,
        )

        for instance in data.get("instances", []):
            if instance.get("className") != "table":
                continue
            for attr in instance.get("attributes", []):
                if not isinstance(attr, dict):
                    continue
                if attr.get("groupName") not in (None, "value_table"):
                    continue
                points_list = attr.get("name", [])
                if not isinstance(points_list, list):
                    continue
                for point in points_list:
                    if not isinstance(point, dict):
                        continue
                    point_id = point.get("id")
                    if not point_id:
                        continue
                    point_copy = dict(point)
                    point_copy["_video_id"] = video_id
                    cache.points[str(point_id)] = point_copy

    print(f"  Cached points: {len(cache.points):,}")
    print(f"  Cached videos: {len(cache.videos):,}")
    return cache


def extract_point_id(entry_id: str) -> Optional[str]:
    parts = entry_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return None


def extract_video_path(entry: dict[str, Any]) -> Optional[str]:
    for turn in entry.get("conversation", []):
        if turn.get("role") != "user":
            continue
        for item in turn.get("content", []):
            if item.get("type") == "video" and item.get("path"):
                return str(item["path"])
    return None


def extract_video_id(entry: dict[str, Any]) -> Optional[str]:
    video_path = extract_video_path(entry)
    if not video_path:
        return None
    return video_path.split("/", 1)[0]


def extract_field_value(entry: dict[str, Any]) -> str:
    for turn in entry.get("conversation", []):
        if turn.get("role") != "assistant":
            continue
        for item in turn.get("content", []):
            if item.get("type") == "text":
                return str(item.get("text", "") or "")
    return ""


def enrich_point_data(
    point_data: dict[str, Any],
    reference_only: bool,
) -> dict[str, Any]:
    enriched = {
        k: v
        for k, v in point_data.items()
        if k not in EXCLUDED_POINT_FIELDS and k != "_video_id"
    }

    if "id" in enriched:
        enriched["point_id"] = enriched["id"]
    if "how_point_ended" in enriched:
        enriched["point_end_type"] = enriched["how_point_ended"]

    score_after = enriched.get("score_after", "")
    if score_after:
        match = re.search(r"\{([^}]+)\}", str(score_after))
        if match:
            simple_score = match.group(1)
            enriched["current_score"] = simple_score
            enriched["serving_score"] = simple_score
            enriched["receiving_score"] = simple_score

    if reference_only:
        enriched = {k: v for k, v in enriched.items() if k in REFERENCE_POINT_FIELDS}
    return enriched


def build_metadata(
    entry: dict[str, Any],
    cache: MetadataCache,
    metadata_label: str,
    reference_only: bool,
) -> Optional[dict[str, Any]]:
    entry_id = str(entry.get("id", "") or "")
    point_id = extract_point_id(entry_id)
    if not point_id or point_id not in cache.points:
        return None

    point_raw = cache.points[point_id]
    video_id = point_raw.get("_video_id") or extract_video_id(entry)
    if not video_id:
        return None

    video_ctx = cache.videos.get(str(video_id))
    file_path = video_ctx.file_path if video_ctx else f"{metadata_label}/{video_id}.json"
    player_names = video_ctx.player_names if video_ctx else {}

    metadata: dict[str, Any] = {
        "entry_id": file_path,
        "question_id": entry_id,
        "question_type": entry.get("class", ""),
        "field_value": extract_field_value(entry),
        "file_path": file_path,
        "point_data": enrich_point_data(point_raw, reference_only=reference_only),
    }
    if player_names:
        metadata["player_names"] = player_names

    # Useful extras while preserving the core metadata schema above.
    if video_id:
        metadata["video_id"] = video_id
    if entry.get("super_category"):
        metadata["super_category"] = entry["super_category"]
    if entry.get("fine_category"):
        metadata["fine_category"] = entry["fine_category"]

    return metadata


def augment_entry(
    entry: dict[str, Any],
    cache: MetadataCache,
    metadata_label: str,
    reference_only: bool,
    open_ended_only: bool = False,
) -> dict[str, Any]:
    augmented = dict(entry)
    if open_ended_only and entry.get("evaluation_type") != "open_ended":
        # Also remove stale metadata when reprocessing a previously enriched file.
        augmented.pop("metadata", None)
        return augmented

    metadata = build_metadata(entry, cache, metadata_label, reference_only)
    if metadata is not None:
        augmented["metadata"] = metadata
    return augmented


def load_records(path: Path) -> List[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: List[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "data", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2, ensure_ascii=False)


def discover_input_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("**/*.json", "**/*.jsonl"):
        for path in sorted(input_dir.glob(pattern)):
            # Skip leftover in-place backup trees (e.g. .backup_YYYYMMDD_HHMMSS/).
            if any(part.startswith(".backup_") for part in path.relative_to(input_dir).parts):
                continue
            files.append(path)
    return files


def relative_output_path(input_dir: Path, input_file: Path, output_dir: Path) -> Path:
    rel = input_file.relative_to(input_dir)
    return output_dir / rel


def backup_file(source: Path, backup_root: Path, input_dir: Path) -> Path:
    dest = backup_root / source.relative_to(input_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def process_file(
    input_file: Path,
    output_file: Path,
    cache: MetadataCache,
    metadata_label: str,
    reference_only: bool,
    open_ended_only: bool = False,
) -> Tuple[int, int, int]:
    records = load_records(input_file)
    augmented: List[dict[str, Any]] = []
    eligible = 0
    matched = 0
    for record in records:
        if not open_ended_only or record.get("evaluation_type") == "open_ended":
            eligible += 1
        enriched = augment_entry(
            record,
            cache,
            metadata_label,
            reference_only,
            open_ended_only,
        )
        if "metadata" in enriched:
            matched += 1
        augmented.append(enriched)
    write_records(output_file, augmented)
    return len(records), eligible, matched


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "../../data/categorical_data"
    default_metadata = script_dir / "aggregated_data_complete"
    default_output = script_dir / "../../data/categorical_data_with_metadata"

    parser = argparse.ArgumentParser(
        description="Augment categorical JSON/JSONL samples with annotation metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input.resolve(),
        help="Directory containing categorical JSON/JSONL files.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=default_metadata.resolve(),
        help="Directory containing aggregated annotation JSON files (<videoId>.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for augmented outputs. Defaults to <input-dir>_with_metadata.",
    )
    parser.add_argument(
        "--metadata-label",
        default="aggregated_data_complete",
        help="Label used in metadata entry_id/file_path fields.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Keep only the minimal point_data fields used by reference-format eval data.",
    )
    parser.add_argument(
        "--open-ended-only",
        action="store_true",
        help="Add metadata only to records with evaluation_type=open_ended.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write augmented files back into --input-dir after creating a backup copy.",
    )
    parser.add_argument(
        "--discard-backup",
        action="store_true",
        help=(
            "With --in-place, delete the temporary backup directory after a "
            "successful run (e.g. when a pre-metadata snapshot already exists)."
        ),
    )
    return parser.parse_args()


def run_augmentation(
    input_dir: Path | str,
    metadata_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
    metadata_label: str | None = None,
    reference_only: bool = False,
    open_ended_only: bool = False,
    in_place: bool = False,
    keep_backup: bool = True,
) -> dict[str, int | float]:
    """Augment all JSON/JSONL files under ``input_dir`` and return summary stats.

    When ``in_place=True``, files are backed up under a ``.backup_*`` directory
    first. Set ``keep_backup=False`` to remove that backup after success.
    """
    input_dir = Path(input_dir).resolve()
    metadata_dir = Path(metadata_dir).resolve()
    if metadata_label is None:
        metadata_label = metadata_dir.name

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    if in_place:
        output_path = input_dir
        backup_root = input_dir / f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_path = Path(output_dir or input_dir.with_name(input_dir.name + "_with_metadata")).resolve()
        backup_root = None

    cache = load_metadata_cache(metadata_dir, metadata_label)
    input_files = discover_input_files(input_dir)
    if not input_files:
        raise ValueError(f"No JSON/JSONL files found under {input_dir}")

    total_records = 0
    total_eligible = 0
    total_matched = 0
    try:
        for input_file in input_files:
            if backup_root is not None:
                backup_file(input_file, backup_root, input_dir)

            output_file = (
                input_file
                if in_place
                else relative_output_path(input_dir, input_file, output_path)
            )
            count, eligible, matched = process_file(
                input_file,
                output_file,
                cache,
                metadata_label,
                reference_only,
                open_ended_only,
            )
            total_records += count
            total_eligible += eligible
            total_matched += matched
            coverage = (matched / eligible * 100) if eligible else 0.0
            rel = input_file.relative_to(input_dir)
            print(
                f"  {rel}: {matched:,}/{eligible:,} eligible records with metadata "
                f"({coverage:.1f}%; {count:,} total)"
            )
    except Exception:
        # Keep the backup so a failed in-place run can be recovered.
        if backup_root is not None and backup_root.exists():
            print(f"  ERROR: keeping backup at {backup_root}")
        raise

    overall = (total_matched / total_eligible * 100) if total_eligible else 0.0
    print(
        f"  Metadata augmentation: {total_matched:,}/{total_eligible:,} eligible "
        f"records ({overall:.1f}% coverage; {total_records:,} total)"
    )
    if backup_root is not None:
        if keep_backup:
            print(f"  Backups written to: {backup_root}")
        else:
            shutil.rmtree(backup_root)
            print(f"  Discarded temporary backup: {backup_root.name}")

    return {
        "files": len(input_files),
        "records": total_records,
        "eligible": total_eligible,
        "matched": total_matched,
        "coverage_pct": overall,
        "output_dir": str(output_path),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    metadata_dir = args.metadata_dir.resolve()

    print(f"\nInput dir:   {input_dir}")
    print(f"Metadata:    {metadata_dir}")
    print(f"Mode:        {'reference-only' if args.reference_only else 'enriched'}")
    print(f"Scope:       {'open-ended only' if args.open_ended_only else 'all records'}")
    print("=" * 80)

    stats = run_augmentation(
        input_dir,
        metadata_dir,
        output_dir=args.output_dir,
        metadata_label=args.metadata_label,
        reference_only=args.reference_only,
        open_ended_only=args.open_ended_only,
        in_place=args.in_place,
        keep_backup=not args.discard_backup,
    )
    print("=" * 80)
    print(
        f"Done. {stats['matched']:,}/{stats['eligible']:,} eligible records "
        f"augmented ({stats['coverage_pct']:.1f}% coverage; "
        f"{stats['records']:,} total records)."
    )
    print(f"Output dir:  {stats['output_dir']}")


if __name__ == "__main__":
    main()
