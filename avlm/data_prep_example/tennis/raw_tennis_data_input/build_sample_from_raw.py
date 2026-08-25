#!/usr/bin/env python3
"""Build per-video annotation JSON from the Data Factory tennis export.

Aggregates segment files into one JSON per game, assigns relative clip paths
using the same rules as ``aggregate_json.py``, and optionally extracts
point-level training clips with ffmpeg.

Output:
    <output-dir>/<videoId>.json
    extracted_segments/<videoId>/...   (when ``--extract-clips`` is set)

Usage:
    python raw_tennis_data_input/build_sample_from_raw.py \\
        --raw-root /path/to/Tennis_Summarization_Consolidated \\
        --games <video_id> [<video_id> ...]
    python raw_tennis_data_input/build_sample_from_raw.py \\
        --raw-root /path/to/Tennis_Summarization_Consolidated \\
        --all-games \\
        --points-per-game 0 \\
        --output-dir ../aggregated_data_complete
    python raw_tennis_data_input/build_sample_from_raw.py \\
        --raw-root /path/to/Tennis_Summarization_Consolidated \\
        --games <video_id> \\
        --extract-clips \\
        --video-base-dir /path/to/data_factory_videos \\
        --output-clips-dir ../extracted_segments
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

POINTS_PER_GAME = 10
TIME_OFFSET = 3.0
DEFAULT_MAX_WORKERS = 4

KEEP_CLASSNAMES = {
    "video_id",
    "video_duration",
    "original_segment_start",
    "original_segment_end",
    "player_1_name",
    "player_1_description",
    "player_2_name",
    "player_2_description",
    "general_context",
}
KEEP_METADATA_FIELDS = {"height", "width", "name", "status"}
DROP_INSTANCE_KEYS = {"createdBy", "updatedBy"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENNIS_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_CLIPS_DIR = os.path.join(TENNIS_DIR, "extracted_segments")


# ---------------------------------------------------------------------------
# Sample trimming helpers
# ---------------------------------------------------------------------------

def sanitize_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if k in KEEP_METADATA_FIELDS}


def sanitize_instance(instance: dict) -> dict:
    clean = json.loads(json.dumps(instance))
    for key in DROP_INSTANCE_KEYS:
        clean.pop(key, None)
    return clean


def segment_sort_key(path: str) -> int:
    match = re.search(r"_seg_(\d+)\.json\.json$", os.path.basename(path))
    return int(match.group(1)) if match else 0


def is_complete_point(point: dict) -> bool:
    return (
        point.get("point_start_in_segment", "Yes") == "Yes"
        and point.get("point_concluded_in_segment", "Yes") == "Yes"
        and point.get("end_time") is not None
    )


def table_points(instance: dict) -> list:
    attrs = instance.get("attributes", [])
    if not attrs:
        return []
    name = attrs[0].get("name", [])
    return name if isinstance(name, list) else []


MATCH_CONTEXT_CLASSNAMES = (
    "player_1_name",
    "player_1_description",
    "player_2_name",
    "player_2_description",
)


def instance_attribute_text(instance: dict) -> str:
    attrs = instance.get("attributes", [])
    if not attrs:
        return ""
    return str(attrs[0].get("name", "") or "").strip()


def collect_match_player_context(instances: list) -> dict:
    """Return the first non-empty player name/description values."""
    context = {name: "" for name in MATCH_CONTEXT_CLASSNAMES}
    for instance in instances:
        class_name = instance.get("className")
        if class_name not in context or context[class_name]:
            continue
        value = instance_attribute_text(instance)
        if value:
            context[class_name] = value
    return context


def update_match_player_context(context: dict, instances: list) -> None:
    """Fill empty match-context slots from ``instances`` in place."""
    for class_name, value in collect_match_player_context(instances).items():
        if value and not context.get(class_name):
            context[class_name] = value


def apply_match_player_context(instances: list, context: dict) -> None:
    """Write carried player name/description values into empty instances."""
    for instance in instances:
        class_name = instance.get("className")
        if class_name not in context:
            continue
        carried = context.get(class_name) or ""
        if not carried:
            continue
        attrs = instance.setdefault("attributes", [{}])
        if not attrs:
            attrs.append({})
            instance["attributes"] = attrs
        if not str(attrs[0].get("name", "") or "").strip():
            attrs[0]["name"] = carried


def list_game_ids(raw_root: str) -> List[str]:
    """Return video IDs under the raw export that contain segment JSON files."""
    if not os.path.isdir(raw_root):
        return []
    game_ids = []
    for name in sorted(os.listdir(raw_root)):
        game_dir = os.path.join(raw_root, name)
        if not os.path.isdir(game_dir):
            continue
        if glob.glob(os.path.join(game_dir, "*.json.json")):
            game_ids.append(name)
    return game_ids


def write_trimmed_segments(
    video_id: str,
    seg_dir: str,
    raw_root: str,
    points_per_game: int = POINTS_PER_GAME,
) -> int:
    """Write sanitized per-segment JSON files to seg_dir; return raw point count.

    ``points_per_game <= 0`` keeps every annotated point (including incomplete
    cross-segment fragments so later merging can reconstruct full points).
    Positive values keep the sample-trim behavior: only complete in-segment
    points, capped at ``points_per_game``.

    Player names/descriptions are annotated only on the first segment of a
    match. Capture them whenever present and rewrite them onto later segments
    so the whole-game JSON keeps a stable match context.
    """
    raw_dir = os.path.join(raw_root, video_id)
    seg_files = sorted(glob.glob(os.path.join(raw_dir, "*.json.json")), key=segment_sort_key)
    if not seg_files:
        print(f"  ! no raw files for {video_id} under {raw_dir}")
        return 0

    os.makedirs(seg_dir, exist_ok=True)
    keep_all = points_per_game <= 0
    remaining = None if keep_all else points_per_game
    written_points = 0
    match_context = {name: "" for name in MATCH_CONTEXT_CLASSNAMES}

    # First pass: capture match-level player context from any segment.
    for seg_file in seg_files:
        with open(seg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        update_match_player_context(match_context, data.get("instances", []))

    for seg_file in seg_files:
        if remaining is not None and remaining <= 0:
            break

        with open(seg_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        instances = data.get("instances", [])

        selected_points = []
        for inst in instances:
            if inst.get("className") != "table":
                continue
            for point in table_points(inst):
                if remaining is not None and remaining <= 0:
                    break
                if keep_all or is_complete_point(point):
                    selected_points.append(point)
                    if remaining is not None:
                        remaining -= 1

        if not selected_points:
            continue

        kept_instances = [
            sanitize_instance(inst)
            for inst in instances
            if inst.get("className") in KEEP_CLASSNAMES
        ]
        apply_match_player_context(kept_instances, match_context)
        table_instance = next(
            (inst for inst in instances if inst.get("className") == "table"), None
        )
        if table_instance is None:
            continue
        trimmed_table = sanitize_instance(table_instance)
        trimmed_table["attributes"][0]["name"] = selected_points
        kept_instances.append(trimmed_table)

        trimmed = {
            "metadata": sanitize_metadata(data.get("metadata", {})),
            "instances": kept_instances,
        }
        out_path = os.path.join(seg_dir, os.path.basename(seg_file))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)

        written_points += len(selected_points)
        mode = "kept" if keep_all else "trimmed"
        print(f"  {video_id}: {mode} {len(selected_points)} points -> {os.path.basename(seg_file)}")

    return written_points


# ---------------------------------------------------------------------------
# Video extraction (aligned with aggregate_json.py)
# ---------------------------------------------------------------------------

@dataclass
class VideoExtractionTask:
    segment_video_path: str
    start_time: float
    end_time: float
    output_dir: str
    output_filename: str
    source_video: str
    last_15_seconds: bool = False

    def __hash__(self) -> int:
        return hash(
            (self.segment_video_path, self.start_time, self.end_time, self.output_filename, self.last_15_seconds)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VideoExtractionTask):
            return False
        return (
            self.segment_video_path == other.segment_video_path
            and self.start_time == other.start_time
            and self.end_time == other.end_time
            and self.output_filename == other.output_filename
            and self.last_15_seconds == other.last_15_seconds
        )


@dataclass
class VideoConcatTask:
    segment_tasks: List[VideoExtractionTask]
    output_dir: str
    output_filename: str
    cross_segment_info: Dict[str, Any]
    temp_dir: Optional[str] = None

    def __hash__(self) -> int:
        return hash((tuple(self.segment_tasks), self.output_filename))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VideoConcatTask):
            return False
        return self.segment_tasks == other.segment_tasks and self.output_filename == other.output_filename


def extract_video_clip_task(
    task: VideoExtractionTask,
    max_retries: int = 2,
) -> Tuple[VideoExtractionTask, Optional[str]]:
    os.makedirs(task.output_dir, exist_ok=True)
    output_path = os.path.join(task.output_dir, f"{task.output_filename}.mp4")

    if os.path.exists(output_path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", output_path],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return task, output_path
            os.remove(output_path)
        except OSError:
            if os.path.exists(output_path):
                os.remove(output_path)

    if task.last_15_seconds:
        duration = 15.0
        start_time = max(0.0, task.end_time - 15.0)
    else:
        duration = task.end_time - task.start_time
        start_time = task.start_time

    adaptive_timeout = min(300 + int(duration * 20), 1200)
    last_exception = None

    for attempt in range(max_retries):
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start_time),
            "-i", task.segment_video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "ultrafast",
            "-crf", "23",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=adaptive_timeout,
                           stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"    ok extracted {task.output_filename}.mp4 (attempt {attempt + 1}/{max_retries})")
                return task, output_path
        except subprocess.TimeoutExpired:
            last_exception = f"Timeout after {adaptive_timeout}s (duration: {duration:.1f}s)"
            if attempt < max_retries - 1:
                print(f"    ! timeout on {task.output_filename}, retrying...")
        except subprocess.CalledProcessError as exc:
            last_exception = exc.stderr.decode() if exc.stderr else str(exc)
            if attempt < max_retries - 1:
                print(f"    ! ffmpeg error on {task.output_filename}, retrying...")
        except Exception as exc:
            last_exception = str(exc)
            break
        if os.path.exists(output_path):
            os.remove(output_path)

    print(f"    x failed to extract {task.output_filename}: {last_exception}")
    return task, None


def concat_video_clips_task(
    task: VideoConcatTask,
    max_retries: int = 2,
) -> Tuple[VideoConcatTask, Optional[str]]:
    output_path = os.path.join(task.output_dir, f"{task.output_filename}.mp4")
    os.makedirs(task.output_dir, exist_ok=True)

    if os.path.exists(output_path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", output_path],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return task, output_path
            os.remove(output_path)
        except OSError:
            if os.path.exists(output_path):
                os.remove(output_path)

    last_exception = None
    for attempt in range(max_retries):
        with tempfile.TemporaryDirectory() as temp_dir:
            task.temp_dir = temp_dir
            segment_files = []
            total_duration = 0.0
            print(
                f"    -> extracting {len(task.segment_tasks)} segments for concatenation "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            for index, segment_task in enumerate(task.segment_tasks):
                temp_output = os.path.join(temp_dir, f"segment_{index:02d}_{segment_task.output_filename}.mp4")
                duration = segment_task.end_time - segment_task.start_time
                start_time = segment_task.start_time
                total_duration += duration
                segment_timeout = min(300 + int(duration * 20), 1200)
                cmd = [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", str(start_time),
                    "-i", segment_task.segment_video_path,
                    "-t", str(duration),
                    "-c:v", "libx264", "-c:a", "aac",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    temp_output,
                ]
                try:
                    subprocess.run(cmd, check=True, timeout=segment_timeout,
                                   stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                        segment_files.append(temp_output)
                        print(f"      ok segment {index + 1}/{len(task.segment_tasks)}")
                    else:
                        last_exception = f"Segment {index + 1} extraction produced empty file"
                        break
                except subprocess.TimeoutExpired:
                    last_exception = f"Segment {index + 1} timeout after {segment_timeout}s"
                    break
                except Exception as exc:
                    last_exception = f"Segment {index + 1} extraction failed: {exc}"
                    break

            if len(segment_files) != len(task.segment_tasks):
                if attempt < max_retries - 1:
                    print("    ! incomplete segments, retrying...")
                    continue
                print("    x no segments extracted for concatenation")
                return task, None

            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, "w", encoding="utf-8") as handle:
                for segment_file in segment_files:
                    handle.write(f"file '{segment_file}'\n")

            concat_timeout = min(300 + int(total_duration * 10), 1200)
            concat_cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c:v", "libx264", "-c:a", "aac",
                "-preset", "ultrafast",
                "-crf", "23",
                output_path,
            ]
            try:
                subprocess.run(concat_cmd, check=True, timeout=concat_timeout,
                               stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"    ok concatenated {task.output_filename}.mp4")
                    return task, output_path
                last_exception = "Concatenation produced empty file"
                if os.path.exists(output_path):
                    os.remove(output_path)
            except subprocess.TimeoutExpired:
                last_exception = f"Concatenation timeout after {concat_timeout}s"
                if os.path.exists(output_path):
                    os.remove(output_path)
                if attempt < max_retries - 1:
                    print("    ! concatenation timeout, retrying...")
            except Exception as exc:
                last_exception = f"Concatenation failed: {exc}"
                if os.path.exists(output_path):
                    os.remove(output_path)
                if attempt < max_retries - 1:
                    print("    ! concatenation error, retrying...")

    print(f"    x failed to concatenate segments: {last_exception}")
    return task, None


def verify_video_file(video_path: str, timeout: int = 10) -> bool:
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip()) > 0
        return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def shuffle_tasks_by_video(tasks: List[VideoExtractionTask]) -> List[VideoExtractionTask]:
    if not tasks:
        return tasks

    video_groups: Dict[str, List[VideoExtractionTask]] = {}
    for task in tasks:
        video_groups.setdefault(task.source_video, []).append(task)

    for group in video_groups.values():
        random.shuffle(group)

    shuffled: List[VideoExtractionTask] = []
    group_items = list(video_groups.items())
    random.shuffle(group_items)
    max_tasks = max(len(group) for group in video_groups.values())

    for index in range(max_tasks):
        for _, group in group_items:
            if index < len(group):
                shuffled.append(group[index])

    print(f"    ok shuffled {len(tasks)} tasks across {len(video_groups)} videos")
    return shuffled


def process_video_tasks_concurrently(
    tasks: List[VideoExtractionTask],
    max_workers: int,
) -> Dict[VideoExtractionTask, Optional[str]]:
    if not tasks:
        return {}

    print(f"    processing {len(tasks)} video extraction tasks with {max_workers} workers...")
    unique_tasks = list(set(tasks))
    if len(unique_tasks) != len(tasks):
        print(f"    ok removed {len(tasks) - len(unique_tasks)} duplicate tasks")

    valid_tasks = [task for task in unique_tasks if verify_video_file(task.segment_video_path)]
    invalid_count = len(unique_tasks) - len(valid_tasks)
    if invalid_count:
        print(f"    ! skipped {invalid_count} tasks due to invalid source videos")
    if not valid_tasks:
        print("    x no valid extraction tasks to process")
        return {}

    shuffled = shuffle_tasks_by_video(valid_tasks)
    results: Dict[VideoExtractionTask, Optional[str]] = {}
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(extract_video_clip_task, task): task for task in shuffled}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                _, output_path = future.result()
            except Exception as exc:
                print(f"    x task execution failed for {task.output_filename}: {exc}")
                output_path = None
            results[task] = output_path
            if output_path:
                success += 1
            else:
                failed += 1

    print(f"    ok completed: {success} successful, {failed} failed")
    return results


def process_concat_tasks_concurrently(
    tasks: List[VideoConcatTask],
    max_workers: int,
) -> Dict[VideoConcatTask, Optional[str]]:
    if not tasks:
        return {}

    print(f"    processing {len(tasks)} concatenation tasks with {max_workers} workers...")
    unique_tasks = list(set(tasks))
    valid_tasks = []
    invalid_count = 0
    for task in unique_tasks:
        if all(verify_video_file(segment_task.segment_video_path) for segment_task in task.segment_tasks):
            valid_tasks.append(task)
        else:
            invalid_count += 1
            print(f"    ! skipping concat task due to invalid source segment(s): {task.output_filename}")

    if invalid_count:
        print(f"    ! skipped {invalid_count} concatenation tasks due to invalid source videos")
    if not valid_tasks:
        print("    x no valid concatenation tasks to process")
        return {}

    results: Dict[VideoConcatTask, Optional[str]] = {}
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(concat_video_clips_task, task): task for task in valid_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                _, output_path = future.result()
            except Exception as exc:
                print(f"    x concatenation task failed for {task.output_filename}: {exc}")
                output_path = None
            results[task] = output_path
            if output_path:
                success += 1
            else:
                failed += 1

    print(f"    ok completed: {success} successful, {failed} failed concatenations")
    return results


@dataclass
class CrossSegmentPoint:
    point_data: Dict[str, Any]
    segment_name: str
    segment_index: int
    point_start_in_segment: str
    point_concluded_in_segment: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def is_start_point(self) -> bool:
        return self.point_start_in_segment == "Yes" and self.point_concluded_in_segment == "No"

    def is_end_point(self) -> bool:
        return self.point_start_in_segment == "No" and self.point_concluded_in_segment == "Yes"

    def is_middle_point(self) -> bool:
        return self.point_start_in_segment == "No" and self.point_concluded_in_segment == "No"


def segment_number_from_name(segment_name: str, fallback: int) -> int:
    """Prefer the `_seg_N` number from the filename for consecutive matching."""
    match = re.search(r"_seg_(\d+)$", segment_name)
    return int(match.group(1)) if match else fallback


class CrossSegmentPointManager:
    def __init__(self) -> None:
        self.pending_start_points: List[CrossSegmentPoint] = []
        self.middle_points: List[CrossSegmentPoint] = []
        self.completed_points: List[Dict[str, Any]] = []

    def add_point(
        self,
        point_data: Dict[str, Any],
        segment_name: str,
        segment_index: int,
    ) -> Optional[Dict[str, Any]]:
        cross_point = CrossSegmentPoint(
            point_data=point_data.copy(),
            segment_name=segment_name,
            segment_index=segment_number_from_name(segment_name, segment_index),
            point_start_in_segment=point_data.get("point_start_in_segment", "Yes"),
            point_concluded_in_segment=point_data.get("point_concluded_in_segment", "Yes"),
            start_time=point_data.get("start_time"),
            end_time=point_data.get("end_time"),
        )

        if cross_point.is_start_point():
            self.pending_start_points.append(cross_point)
            return None
        if cross_point.is_end_point():
            return self._match_end_point(cross_point)
        if cross_point.is_middle_point():
            self.middle_points.append(cross_point)
            return None
        return point_data

    def _match_end_point(self, end_point: CrossSegmentPoint) -> Optional[Dict[str, Any]]:
        """Match an end fragment only to a consecutive start/middle chain.

        Cross-segment points must cover adjacent segment indices with no gaps:
        start at i, optional middles at i+1..j-1, end at j.
        Non-consecutive jumps (for example seg_09 -> seg_12 with no middles) are
        treated as incomplete and skipped.
        """
        matching_start = None
        start_index = -1
        matching_middles: List[CrossSegmentPoint] = []

        for index, start_point in enumerate(self.pending_start_points):
            if start_point.segment_index >= end_point.segment_index:
                continue
            expected_middle_indices = set(
                range(start_point.segment_index + 1, end_point.segment_index)
            )
            candidate_middles = [
                middle_point
                for middle_point in self.middle_points
                if middle_point.segment_index in expected_middle_indices
            ]
            covered = {middle_point.segment_index for middle_point in candidate_middles}
            if covered != expected_middle_indices:
                continue
            # Prefer the closest valid start (largest segment_index).
            if matching_start is None or start_point.segment_index > matching_start.segment_index:
                matching_start = start_point
                start_index = index
                matching_middles = sorted(
                    candidate_middles, key=lambda point: point.segment_index
                )

        if matching_start is None:
            print(
                f"    ! skipping non-consecutive/incomplete end point in "
                f"{end_point.segment_name}"
            )
            return None

        concatenated_point = self._concatenate_points(
            matching_start, matching_middles, end_point
        )
        # Incomplete reconstructed points (missing timing) are not usable.
        if concatenated_point.get("start_time") is None or concatenated_point.get("end_time") is None:
            print(
                f"    ! skipping incomplete cross-segment point "
                f"{matching_start.segment_name} -> {end_point.segment_name} "
                f"(missing start/end time)"
            )
            self.pending_start_points.pop(start_index)
            used_middle_indices = {middle.segment_index for middle in matching_middles}
            self.middle_points = [
                middle_point
                for middle_point in self.middle_points
                if middle_point.segment_index not in used_middle_indices
            ]
            return None

        self.pending_start_points.pop(start_index)
        used_middle_indices = {middle.segment_index for middle in matching_middles}
        self.middle_points = [
            middle_point
            for middle_point in self.middle_points
            if middle_point.segment_index not in used_middle_indices
        ]
        self.completed_points.append(concatenated_point)
        print(
            f"    ok concatenated point across segments "
            f"{matching_start.segment_name} -> {end_point.segment_name}"
        )
        return concatenated_point

    def _concatenate_points(
        self,
        start_point: CrossSegmentPoint,
        middle_points: List[CrossSegmentPoint],
        end_point: CrossSegmentPoint,
    ) -> Dict[str, Any]:
        def safe_int(value: Any) -> int:
            if value is None or value == "":
                return 0
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0

        result = start_point.point_data.copy()
        total_shots = safe_int(start_point.point_data.get("num_shots_exchanged"))
        for middle_point in middle_points:
            total_shots += safe_int(middle_point.point_data.get("num_shots_exchanged"))
        total_shots += safe_int(end_point.point_data.get("num_shots_exchanged"))

        score_before = start_point.point_data.get("score_before") or result.get("score_before")
        server = start_point.point_data.get("server") or result.get("server")
        receiver = start_point.point_data.get("receiver") or result.get("receiver")
        num_serve_attempts = safe_int(start_point.point_data.get("num_serving_attempts_until_successful"))
        if num_serve_attempts == 0:
            num_serve_attempts = safe_int(result.get("num_serving_attempts_until_successful", 1))
        if num_serve_attempts == 0:
            num_serve_attempts = 1

        score_after = end_point.point_data.get("score_after") or result.get("score_after")
        winner = end_point.point_data.get("winner") or result.get("winner")
        how_ended = end_point.point_data.get("how_point_ended") or result.get("how_point_ended")
        how_ended_desc = end_point.point_data.get("how_point_ended_description") or result.get("how_point_ended_description")

        result.update({
            "start_time": start_point.start_time,
            "end_time": end_point.end_time,
            "point_concluded_in_segment": "Yes",
            "score_before": score_before if score_before else "",
            "server": server if server else "",
            "receiver": receiver if receiver else "",
            "num_serving_attempts_until_successful": num_serve_attempts,
            "how_point_ended": how_ended if how_ended else "",
            "how_point_ended_description": how_ended_desc if how_ended_desc else "",
            "score_after": score_after if score_after else "",
            "winner": winner if winner else "",
            "num_shots_exchanged": total_shots,
        })

        captions = []
        descriptions = []
        audio_cues = []
        for segment in [start_point, *middle_points, end_point]:
            if segment.point_data.get("point_caption"):
                captions.append(f"[{segment.segment_name}] {segment.point_data['point_caption']}")
            if segment.point_data.get("how_point_ended_description"):
                descriptions.append(
                    f"[{segment.segment_name}] {segment.point_data['how_point_ended_description']}"
                )
            if segment.point_data.get("audio_cues"):
                audio_cues.append(segment.point_data["audio_cues"])

        if captions:
            result["point_caption"] = " ".join(captions)
        if descriptions:
            result["how_point_ended_description"] = " ".join(descriptions)
        if audio_cues:
            result["audio_cues"] = ", ".join(audio_cues)

        result["_cross_segment_info"] = {
            "start_segment": start_point.segment_name,
            "end_segment": end_point.segment_name,
            "middle_segments": [middle_point.segment_name for middle_point in middle_points],
            "total_segments": 2 + len(middle_points),
            "video_segments": [],
        }

        for segment in [start_point, *middle_points, end_point]:
            result["_cross_segment_info"]["video_segments"].append({
                "segment_name": segment.segment_name,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "point_start_in_segment": segment.point_start_in_segment,
                "point_concluded_in_segment": segment.point_concluded_in_segment,
            })

        return result

    def get_pending_count(self) -> Dict[str, int]:
        return {
            "pending_starts": len(self.pending_start_points),
            "middle_points": len(self.middle_points),
            "completed": len(self.completed_points),
        }


def merge_with_previous_context(point: Dict[str, Any], previous_context: Dict[str, Any]) -> Dict[str, Any]:
    merged_point = point.copy()
    if "score_after" in previous_context and not merged_point.get("score_before"):
        merged_point["score_before"] = previous_context["score_after"]
    for field in ("server", "receiver", "num_serving_attempts_until_successful"):
        if field in previous_context and not merged_point.get(field):
            merged_point[field] = previous_context[field]
    return merged_point


def update_previous_context(instance: Dict[str, Any], previous_context: Dict[str, Any]) -> None:
    attributes = instance.get("attributes", [])
    if not attributes:
        return
    table_data = attributes[0].get("name", [])
    if not isinstance(table_data, list) or not table_data:
        return
    last_point = table_data[-1]
    for field in ("score_after", "server", "receiver", "num_serving_attempts_until_successful"):
        if field in last_point:
            previous_context[field] = last_point[field]


def video_subdir_from_segment(segment_name: str) -> str:
    return segment_name.rsplit("_seg_", 1)[0] if "_seg_" in segment_name else segment_name


def resolve_segment_video(video_base_dir: str, segment_name: str) -> str:
    video_subdir = video_subdir_from_segment(segment_name)
    return os.path.join(video_base_dir, video_subdir, f"{segment_name}.mp4")


def process_single_point(
    point: Dict[str, Any],
    segment_video_path: Optional[str],
    extracted_segments_dir: Optional[str],
    segment_name: Optional[str],
    time_offset: float,
    video_tasks: Optional[List[VideoExtractionTask]],
    skip_count: Dict[str, int],
    video_base_path: Optional[str],
    concat_tasks: Optional[List[VideoConcatTask]],
    extract_clips: bool,
) -> Optional[Dict[str, Any]]:
    adjusted_point = point.copy()

    if adjusted_point.get("start_time") is None:
        adjusted_point["start_time"] = 0

    is_cross_segment = "_cross_segment_info" in adjusted_point
    end_time = point.get("end_time")
    start_time = adjusted_point.get("start_time")

    if not is_cross_segment:
        if end_time is None:
            skip_count["total"] += 1
            skip_count["missing_time"] += 1
            return None

        duration_seconds = end_time - start_time
        how_texts = [
            str(adjusted_point.get("how_point_ended", "")).lower(),
            str(adjusted_point.get("how_point_ended_description", "")).lower(),
            str(adjusted_point.get("point_caption", "")).lower(),
        ]
        is_ace = any(re.search(r"\bace\b", text) for text in how_texts)
        if duration_seconds < 3 and not is_ace:
            skip_count["total"] += 1
            skip_count["short_duration"] += 1
            return None

    if "_cross_segment_info" in adjusted_point:
        cross_info = adjusted_point["_cross_segment_info"]
        first_seg_start = cross_info["video_segments"][0].get("start_time")
        last_seg_end = cross_info["video_segments"][-1].get("end_time")
        if first_seg_start is None or last_seg_end is None:
            skip_count["total"] += 1
            skip_count["incomplete_cross_segment"] = (
                skip_count.get("incomplete_cross_segment", 0) + 1
            )
            return None
        clip_filename = (
            f"cross_segment_{cross_info['start_segment']}_to_{cross_info['end_segment']}_"
            f"{int(first_seg_start)}_{int(last_seg_end)}_timeoffset_end_{int(time_offset)}"
        )
        first_seg_name = cross_info["video_segments"][0]["segment_name"]
        video_subdir = video_subdir_from_segment(first_seg_name)
        adjusted_point["video"] = f"{video_subdir}/{clip_filename}.mp4"

        if extract_clips and video_base_path and concat_tasks is not None and extracted_segments_dir:
            segment_tasks: List[VideoExtractionTask] = []
            for segment_info in cross_info["video_segments"]:
                seg_name = segment_info["segment_name"]
                seg_video_path = resolve_segment_video(video_base_path, seg_name)
                if not os.path.exists(seg_video_path):
                    print(f"    ! source segment not found for cross-segment point: {seg_video_path}")
                    continue

                seg_start = segment_info.get("start_time")
                seg_end = segment_info.get("end_time")
                if seg_start is None or seg_end is None:
                    try:
                        result = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", seg_video_path],
                            capture_output=True, text=True, timeout=10, check=False,
                        )
                        video_duration = float(result.stdout.strip())
                        if seg_start is None:
                            seg_start = 0.0
                        if seg_end is None:
                            seg_end = video_duration
                    except Exception as exc:
                        print(f"    ! could not determine video duration for {seg_name}: {exc}")
                        continue

                if segment_info["point_concluded_in_segment"] == "Yes":
                    seg_end += time_offset

                segment_info["start_time"] = seg_start
                segment_info["end_time"] = seg_end
                segment_tasks.append(VideoExtractionTask(
                    segment_video_path=seg_video_path,
                    start_time=seg_start,
                    end_time=seg_end,
                    output_dir=extracted_segments_dir,
                    output_filename=f"{seg_name}_{int(seg_start)}_{int(seg_end)}",
                    source_video=seg_name,
                    last_15_seconds=False,
                ))

            if segment_tasks:
                concat_task = VideoConcatTask(
                    segment_tasks=segment_tasks,
                    output_dir=extracted_segments_dir,
                    output_filename=clip_filename,
                    cross_segment_info=cross_info,
                )
                concat_tasks.append(concat_task)
                adjusted_point["_video_concat_task"] = concat_task
                adjusted_point["_is_cross_segment_video"] = True
            else:
                adjusted_point["video"] = None

        return adjusted_point

    start_time = adjusted_point.get("start_time", 0)
    end_time = adjusted_point.get("end_time")
    if start_time is None or end_time is None or not segment_name:
        adjusted_point["video"] = None
        return adjusted_point

    safe_start_time = start_time if start_time is not None else 0
    safe_end_time = end_time if end_time is not None else 0
    clip_filename = (
        f"{segment_name}_{int(safe_start_time)}_{int(safe_end_time)}_timeoffset_end_{int(time_offset)}"
    )
    video_subdir = video_subdir_from_segment(segment_name)
    adjusted_point["video"] = f"{video_subdir}/{clip_filename}.mp4"

    if extract_clips and video_tasks is not None and extracted_segments_dir:
        if not (segment_video_path and os.path.exists(segment_video_path)):
            adjusted_point["video"] = None
            return adjusted_point

        task = VideoExtractionTask(
            segment_video_path=segment_video_path,
            start_time=start_time,
            end_time=end_time + time_offset,
            output_dir=extracted_segments_dir,
            output_filename=clip_filename,
            source_video=os.path.basename(segment_video_path),
            last_15_seconds=False,
        )
        video_tasks.append(task)
        adjusted_point["_video_task"] = task

    return adjusted_point


def process_table_instance(
    instance: Dict[str, Any],
    previous_context: Dict[str, Any],
    skip_count: Dict[str, int],
    segment_video_path: Optional[str],
    extracted_segments_dir: Optional[str],
    segment_name: Optional[str],
    time_offset: float,
    video_tasks: Optional[List[VideoExtractionTask]],
    cross_segment_manager: CrossSegmentPointManager,
    segment_index: int,
    concat_tasks: Optional[List[VideoConcatTask]],
    video_base_path: Optional[str],
    extract_clips: bool,
) -> Dict[str, Any]:
    adjusted_instance = instance.copy()
    attributes = instance.get("attributes", [])
    if not attributes:
        return adjusted_instance

    table_data = attributes[0].get("name", [])
    if not isinstance(table_data, list):
        return adjusted_instance

    adjusted_points = []
    for point in table_data:
        point_start_in_segment = point.get("point_start_in_segment", "Yes")
        point_concluded_in_segment = point.get("point_concluded_in_segment", "Yes")
        is_cross_segment = point_start_in_segment == "No" or point_concluded_in_segment == "No"

        if is_cross_segment:
            point_with_context = merge_with_previous_context(point, previous_context)
            completed_point = cross_segment_manager.add_point(
                point_with_context, segment_name or "", segment_index
            )
            if completed_point is not None:
                processed_point = process_single_point(
                    completed_point, segment_video_path, extracted_segments_dir,
                    segment_name, time_offset, video_tasks, skip_count,
                    video_base_path, concat_tasks, extract_clips,
                )
                if processed_point is not None:
                    adjusted_points.append(processed_point)
            continue

        processed_point = process_single_point(
            point, segment_video_path, extracted_segments_dir,
            segment_name, time_offset, video_tasks, skip_count,
            video_base_path, concat_tasks, extract_clips,
        )
        if processed_point is not None:
            adjusted_points.append(processed_point)

    adjusted_instance["attributes"][0]["name"] = adjusted_points
    update_previous_context(adjusted_instance, previous_context)
    return adjusted_instance


def strip_internal_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    clean = {key: value for key, value in data.items() if not key.startswith("_")}
    for instance in clean.get("instances", []):
        if instance.get("className") != "table":
            continue
        attrs = instance.get("attributes", [])
        if not attrs:
            continue
        points = attrs[0].get("name", [])
        if not isinstance(points, list):
            continue
        attrs[0]["name"] = [
            {key: value for key, value in point.items() if not key.startswith("_")}
            for point in points
        ]
    return clean


def preprocess_tennis_json_files(
    directory_path: str,
    output_file: str,
    video_base_dir: Optional[str] = None,
    extracted_segments_dir: Optional[str] = None,
    time_offset: float = TIME_OFFSET,
    max_workers: int = DEFAULT_MAX_WORKERS,
    extract_clips: bool = False,
) -> Dict[str, Any]:
    json_files = sorted(glob.glob(os.path.join(directory_path, "*.json.json")), key=segment_sort_key)
    if not json_files:
        raise ValueError(f"No JSON files found in {directory_path}")

    merged_data: Dict[str, Any] = {"metadata": {}, "instances": []}
    previous_context: Dict[str, Any] = {}
    skip_count = {
        "total": 0,
        "short_duration": 0,
        "missing_time": 0,
        "incomplete_cross_segment": 0,
        "cross_segment_no_manager": 0,
        "other": 0,
    }
    video_tasks: List[VideoExtractionTask] = []
    concat_tasks: List[VideoConcatTask] = []
    cross_segment_manager = CrossSegmentPointManager()

    print("  Phase 1: processing segment JSON files and assigning clip paths...")
    for segment_index, json_file in enumerate(json_files):
        filename = os.path.basename(json_file)
        print(f"    {filename} ({segment_index + 1}/{len(json_files)})")

        with open(json_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not merged_data["metadata"]:
            merged_data["metadata"] = data.get("metadata", {})

        segment_name = filename.replace(".json.json", "")
        segment_video_path = None
        if video_base_dir:
            segment_video_path = resolve_segment_video(video_base_dir, segment_name)
            if not os.path.exists(segment_video_path):
                segment_video_path = None

        for instance in data.get("instances", []):
            if instance.get("className") == "table":
                adjusted_instance = process_table_instance(
                    instance, previous_context, skip_count,
                    segment_video_path, extracted_segments_dir, segment_name,
                    time_offset,
                    video_tasks if extract_clips else None,
                    cross_segment_manager, segment_index, concat_tasks if extract_clips else None,
                    video_base_dir, extract_clips,
                )
                merged_data["instances"].append(adjusted_instance)
            else:
                merged_data["instances"].append(instance)

    pending_counts = cross_segment_manager.get_pending_count()
    if pending_counts["pending_starts"] or pending_counts["middle_points"]:
        print(
            f"  ! warning: {pending_counts['pending_starts']} unmatched start points and "
            f"{pending_counts['middle_points']} middle points remain"
        )
    if pending_counts["completed"]:
        print(f"  ok concatenated {pending_counts['completed']} cross-segment points")

    # Player names/descriptions live on the first segment; rewrite empties so the
    # whole-game JSON exposes a stable match context to downstream generators.
    match_context = collect_match_player_context(merged_data.get("instances", []))
    apply_match_player_context(merged_data.get("instances", []), match_context)

    if extract_clips and (video_tasks or concat_tasks) and extracted_segments_dir:
        print(f"\n  Phase 2: extracting {len(video_tasks)} clips and concatenating {len(concat_tasks)} cross-segment clips...")
        task_results = process_video_tasks_concurrently(video_tasks, max_workers) if video_tasks else {}
        concat_results = process_concat_tasks_concurrently(concat_tasks, max_workers) if concat_tasks else {}

        print("  Phase 3: updating video fields based on extraction results...")
        for instance in merged_data["instances"]:
            if instance.get("className") != "table":
                continue
            attributes = instance.get("attributes", [])
            if not attributes:
                continue
            table_data = attributes[0].get("name", [])
            if not isinstance(table_data, list):
                continue
            for point in table_data:
                if "_video_task" in point:
                    task = point["_video_task"]
                    if not task_results.get(task):
                        point["video"] = None
                    del point["_video_task"]
                elif "_video_concat_task" in point:
                    task = point["_video_concat_task"]
                    if not concat_results.get(task):
                        point["video"] = None
                    del point["_video_concat_task"]
                    point.pop("_is_cross_segment_video", None)

    clean_data = strip_internal_fields(merged_data)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(clean_data, handle, indent=2, ensure_ascii=False)
    print(f"  saved {output_file}")

    print("  Skip statistics:")
    print(f"    total skipped: {skip_count['total']}")
    print(f"    short duration (<3s, not ace): {skip_count['short_duration']}")
    print(f"    missing time data: {skip_count['missing_time']}")
    print(f"    incomplete cross-segment: {skip_count['incomplete_cross_segment']}")

    return merged_data


def build_for_game(
    video_id: str,
    raw_root: str,
    extract_clips: bool,
    video_base_dir: str | None,
    output_clips_dir: str,
    max_workers: int,
    output_dir: str,
    points_per_game: int,
) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        seg_dir = os.path.join(tmp, video_id)
        raw_point_count = write_trimmed_segments(
            video_id, seg_dir, raw_root, points_per_game=points_per_game
        )
        if raw_point_count == 0:
            return 0

        out_path = os.path.join(output_dir, f"{video_id}.json")
        extracted_segments_dir = os.path.join(output_clips_dir, video_id) if extract_clips else None
        if extract_clips and extracted_segments_dir:
            os.makedirs(extracted_segments_dir, exist_ok=True)

        preprocess_tennis_json_files(
            seg_dir,
            out_path,
            video_base_dir=video_base_dir if extract_clips else None,
            extracted_segments_dir=extracted_segments_dir,
            time_offset=TIME_OFFSET,
            max_workers=max_workers,
            extract_clips=extract_clips,
        )

        with open(out_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        kept_points = sum(
            len(table_points(instance))
            for instance in data.get("instances", [])
            if instance.get("className") == "table"
        )
        print(f"  {video_id}: kept {kept_points} points -> {os.path.basename(out_path)}")
        return kept_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Path to the Data Factory tennis export (Tennis_Summarization_Consolidated)",
    )
    parser.add_argument(
        "--extract-clips",
        action="store_true",
        help="Extract point-level mp4 clips with ffmpeg after building annotation JSON",
    )
    parser.add_argument(
        "--video-base-dir",
        default=None,
        help="Base directory containing pre-segmented source videos (required with --extract-clips)",
    )
    parser.add_argument(
        "--output-clips-dir",
        default=DEFAULT_OUTPUT_CLIPS_DIR,
        help="Directory to write extracted point clips",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write per-video annotation JSON files",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Process every game under the Data Factory tennis export",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=None,
        help="Optional explicit game/video IDs to process",
    )
    parser.add_argument(
        "--points-per-game",
        type=int,
        default=POINTS_PER_GAME,
        help=(
            "Maximum complete points to keep per game in sample mode. "
            "Use 0 to keep all annotated points (needed for full aggregation)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Concurrent ffmpeg workers when --extract-clips is set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_root = os.path.abspath(args.raw_root)
    if not os.path.isdir(raw_root):
        raise SystemExit(f"Raw export directory not found: {raw_root}")

    if args.extract_clips:
        if not args.video_base_dir:
            raise SystemExit("--extract-clips requires --video-base-dir")
        if not os.path.isdir(args.video_base_dir):
            raise SystemExit(f"Video base directory not found: {args.video_base_dir}")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.games:
        games = list(args.games)
    elif args.all_games:
        games = list_game_ids(raw_root)
    else:
        raise SystemExit(
            "Specify --games <video_id> ... or --all-games to process games under --raw-root."
        )
    if not games:
        raise SystemExit(f"No games found under {raw_root}")

    print(f"Building per-video annotation JSON under {output_dir} ...")
    print(f"Games: {len(games)}" + (" (all)" if args.all_games else " (sample set)"))
    if args.points_per_game <= 0:
        print("Point cap: none (keep all annotated points)")
    else:
        print(f"Point cap: {args.points_per_game} complete points per game")
    if args.extract_clips:
        print(f"Point-level clip extraction enabled -> {args.output_clips_dir}")
    else:
        print("Point-level clip extraction disabled (paths only)")

    # Legacy sample builds wrote temporary per-game subdirs under the output root.
    for video_id in games:
        old_subdir = os.path.join(output_dir, video_id)
        if os.path.isdir(old_subdir):
            shutil.rmtree(old_subdir)

    total = 0
    succeeded = 0
    failed = 0
    for index, video_id in enumerate(games, 1):
        print(f"\n[{index}/{len(games)}] {video_id}")
        try:
            kept = build_for_game(
                video_id,
                raw_root=raw_root,
                extract_clips=args.extract_clips,
                video_base_dir=args.video_base_dir,
                output_clips_dir=args.output_clips_dir,
                max_workers=args.max_workers,
                output_dir=output_dir,
                points_per_game=args.points_per_game,
            )
            if kept:
                succeeded += 1
                total += kept
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"  x failed on {video_id}: {exc}")

    print(
        f"\nDone. {total} points across {succeeded} games "
        f"({failed} empty/failed) -> {output_dir}"
    )


if __name__ == "__main__":
    main()
