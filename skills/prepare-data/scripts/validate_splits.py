#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict, read-only check of HF conversation JSONL splits before training.

Checks (any failure exits 1):
  * each requested split file <prefix><split>.jsonl exists and is non-empty
  * each line is a JSON object with a unique string "id" (unique across all
    splits) and a "conversation" holding a user turn with a video item
    ("path" or "video") plus text, and an assistant turn with non-empty text
  * train/eval contamination: no clip path in train appears in any other split
    (except --train-pool-splits, sampled from the train clips by design), and no
    source video (parent directory of a clip path) of the test_unseen* split
    appears in any other split
  * with --video-root: every referenced clip exists under that root (the
    training loaders silently drop rows whose clip is missing)

Prints a JSON summary to stdout and errors to stderr.
Exit codes: 0 passed, 1 validation failed, 2 bad arguments or missing directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

DEFAULT_SPLITS = "train,validation,test_seen_videos,test_unseen_videos"
MAX_REPORTED = 10


def check_record(rec: object) -> tuple[list[str], list[str]]:
    """Return (clip paths, problems) for one JSONL record."""
    if not isinstance(rec, dict):
        return [], ["record is not a JSON object"]
    problems = []
    if not isinstance(rec.get("id"), str) or not rec["id"]:
        problems.append("missing non-empty string 'id'")
    conversation = rec.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return [], problems + ["missing non-empty 'conversation' list"]
    clips: list[str] = []
    user_turn_ok = assistant_text = False  # need one user turn with video + text, and an assistant answer
    for turn in conversation:
        content = turn.get("content") if isinstance(turn, dict) else None
        if not isinstance(content, list):
            problems.append("conversation turn without a 'content' list")
            continue
        role = turn.get("role")
        has_video = has_text = False
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "video":
                path = item.get("path") or item.get("video")
                if isinstance(path, str) and path:
                    clips.append(path)
                    has_video = True
                else:
                    problems.append("video item without a 'path'")
            elif item.get("type") == "text" and str(item.get("text") or "").strip():
                has_text = True
        if role == "user" and has_video and has_text:
            user_turn_ok = True
        elif role == "assistant" and has_text:
            assistant_text = True
        if role != "user" and has_video:
            problems.append(f"video item in a '{role}' turn (expected in the user turn)")
    if not clips:
        problems.append("no video item in the conversation")
    elif not user_turn_ok:
        problems.append("no user turn with both the video and the question text")
    if not assistant_text:
        problems.append("no assistant text (answer)")
    return clips, problems


def scan_split(path: Path, seen_ids: dict[str, str], split: str, errors: list[str]) -> dict:
    """Validate one split file; return its stats and clip set."""
    stats = {"records": 0, "clips": set(), "classes": set()}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            where = f"{path.name}:{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON ({exc.msg})")
                continue
            clips, problems = check_record(rec)
            errors.extend(f"{where}: {p}" for p in problems)
            stats["records"] += 1
            stats["clips"].update(clips)
            if isinstance(rec, dict):
                if isinstance(rec.get("class"), str):
                    stats["classes"].add(rec["class"])
                rid = rec.get("id")
                if isinstance(rid, str) and rid:
                    if rid in seen_ids:
                        errors.append(f"{where}: duplicate id {rid!r} (first seen in {seen_ids[rid]})")
                    else:
                        seen_ids[rid] = split
    if stats["records"] == 0:
        errors.append(f"{path.name}: split is empty")
    return stats


def overlap_error(label: str, items: set[str]) -> str:
    sample = ", ".join(sorted(items)[:3])
    return f"contamination: {len(items)} {label} (e.g. {sample})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits-dir", required=True, type=Path,
                        help="directory holding the split JSONL files")
    parser.add_argument("--prefix", default="tennis_mcq_qa_",
                        help="split file name prefix (default: %(default)s)")
    parser.add_argument("--splits", default=DEFAULT_SPLITS,
                        help="comma-separated split names; the first is the training split "
                             "(default: %(default)s)")
    parser.add_argument("--train-pool-splits", default="",
                        help="comma-separated splits sampled from the train clips by design "
                             "(basketball: val); skipped in the train-clip overlap check")
    parser.add_argument("--video-root", default="",
                        help="also require every referenced clip to exist under this directory")
    args = parser.parse_args()

    if not args.splits_dir.is_dir():
        print(f"error: --splits-dir {args.splits_dir} is not a directory", file=sys.stderr)
        return 2
    args.video_root = Path(args.video_root) if args.video_root.strip() else None  # "" = skip the check
    if args.video_root is not None and not args.video_root.is_dir():
        print(f"error: --video-root {args.video_root} is not a directory", file=sys.stderr)
        return 2
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("error: --splits is empty", file=sys.stderr)
        return 2
    train_split = splits[0]
    train_pool = {s.strip() for s in args.train_pool_splits.split(",") if s.strip()}
    if train_pool - set(splits):
        print(f"error: --train-pool-splits {', '.join(sorted(train_pool - set(splits)))} "
              "not in --splits", file=sys.stderr)
        return 2
    unseen_split = next((s for s in splits if s.startswith("test_unseen")), None)

    errors: list[str] = []
    seen_ids: dict[str, str] = {}
    stats: dict[str, dict] = {}
    for split in splits:
        path = args.splits_dir / f"{args.prefix}{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing split file {path}")
            continue
        stats[split] = scan_split(path, seen_ids, split, errors)

    contamination = {"train_eval_clip_overlap": 0, "unseen_source_video_overlap": 0}
    if train_split in stats:
        for split, st in stats.items():
            if split == train_split or split in train_pool:
                continue
            shared = stats[train_split]["clips"] & st["clips"]
            contamination["train_eval_clip_overlap"] += len(shared)
            if shared:
                errors.append(overlap_error(f"clips in both {train_split} and {split}", shared))
    if unseen_split in stats:
        def sources(clips: set[str]) -> set[str]:
            return {str(PurePosixPath(c).parent) for c in clips} - {"."}

        unseen = sources(stats[unseen_split]["clips"])
        for split, st in stats.items():
            if split == unseen_split:
                continue
            shared = unseen & sources(st["clips"])
            contamination["unseen_source_video_overlap"] += len(shared)
            if shared:
                errors.append(overlap_error(
                    f"{unseen_split} source videos also in {split}", shared))

    missing_clips = None
    if args.video_root is not None:
        all_clips = set().union(*(st["clips"] for st in stats.values()))
        root = args.video_root.resolve()
        missing, escaped = set(), set()
        for c in all_clips:
            candidate = (root / c).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:  # absolute path or '..' leaving --video-root
                escaped.add(c)
                continue
            if not candidate.is_file():
                missing.add(c)
        missing_clips = len(missing)
        if escaped:
            errors.append(f"{len(escaped)} clips escape --video-root {args.video_root} "
                          f"(e.g. {', '.join(sorted(escaped)[:3])})")
        if missing:
            errors.append(f"{len(missing)} of {len(all_clips)} clips missing under "
                          f"{args.video_root} (e.g. {', '.join(sorted(missing)[:3])})")

    summary = {
        "passed": not errors,
        "splits_dir": os.path.abspath(args.splits_dir),
        "splits": {
            split: {"records": st["records"], "clips": len(st["clips"]),
                    "classes": len(st["classes"])}
            for split, st in stats.items()
        },
        "total_records": sum(st["records"] for st in stats.values()),
        "contamination": contamination,
        "missing_clips": "not_checked" if missing_clips is None else missing_clips,
        "error_count": len(errors),
    }
    print(json.dumps(summary, indent=2))
    for err in errors[:MAX_REPORTED]:
        print(f"error: {err}", file=sys.stderr)
    if len(errors) > MAX_REPORTED:
        print(f"error: ... {len(errors) - MAX_REPORTED} more", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
