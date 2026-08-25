import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import decord


def _video_key(row):
    video = row.get("video-sound") or row.get("video")
    if isinstance(video, str):
        return video

    for message in row.get("conversation") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue
            for key in ("video", "path"):
                video = item.get(key)
                if isinstance(video, str):
                    return video

    return None


def _read_video_metadata(args):
    video_key, video_root = args
    video_path = video_key if os.path.isabs(video_key) else os.path.join(video_root, video_key)

    video_reader = decord.VideoReader(video_path, num_threads=1)
    try:
        total_frames = len(video_reader)
        fps = float(video_reader.get_avg_fps())
        duration = total_frames / max(fps, 1e-6)
    finally:
        del video_reader
    return video_key, [duration, total_frames, fps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    source_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else source_path.parent
    output_path = output_dir / f"{source_path.stem}_video_metadata.json"

    video_keys = []
    seen = set()
    with source_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            row = json.loads(line)
            video_key = _video_key(row)
            if video_key is None or video_key in seen:
                continue
            seen.add(video_key)
            video_keys.append(video_key)

    work = [(video_key, args.video_root) for video_key in video_keys]
    if args.workers is None:
        items = [_read_video_metadata(item) for item in work]
    else:
        with Pool(args.workers) as pool:
            items = pool.map(_read_video_metadata, work)

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(dict(items), metadata_file)
    print(f"Wrote {len(items)} videos to {output_path}")


if __name__ == "__main__":
    main()
