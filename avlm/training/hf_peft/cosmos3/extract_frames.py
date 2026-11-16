# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract a fixed number of JPEG frames per clip for frame-based LoRA training.

For every distinct `video` path in a conversation JSONL, decode the clip at `--fps`,
resize with the ffmpeg `--scale` expression, and keep `--frames` frames sampled
uniformly from first to last decoded frame (all of them when the clip decodes to
fewer). Frames land in `<out>/<video path without extension>/f00.jpg ...`, which is the
layout train_lora.py reads. A clip whose frames already exist is skipped.

usage:
  python avlm/training/hf_peft/cosmos3/extract_frames.py \
    --jsonl /path/to/train.jsonl --video-root /path/to/videos --out /path/to/frames
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def video_paths(jsonl: str) -> list[str]:
    paths: dict[str, None] = {}
    with open(jsonl) as f:
        for line in f:
            row = json.loads(line)
            for item in row["conversation"][0]["content"]:
                if item.get("type") == "video":
                    paths[item["path"]] = None
    return list(paths)


def clip_dir(out: str, path: str) -> str:
    return os.path.join(out, os.path.splitext(path)[0])


def extract(video_root: str, out: str, path: str, frames: int, fps: float, scale: str, quality: int) -> tuple[str, int]:
    target = clip_dir(out, path)
    if os.path.exists(os.path.join(target, f"f{frames - 1:02d}.jpg")):
        return path, frames
    os.makedirs(target, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", os.path.join(video_root, path),
         "-vf", f"fps={fps},scale={scale}", "-q:v", str(quality), os.path.join(target, "tmp_%04d.jpg")],
        check=True,
    )
    decoded = sorted(f for f in os.listdir(target) if f.startswith("tmp_"))
    if not decoded:
        raise RuntimeError(f"{path}: ffmpeg decoded no frames")
    n = len(decoded)
    picks = [round(i * (n - 1) / (frames - 1)) for i in range(frames)] if n >= frames else list(range(n))
    for j, i in enumerate(picks):
        os.replace(os.path.join(target, decoded[i]), os.path.join(target, f"f{j:02d}.jpg"))
    for f in os.listdir(target):
        if f.startswith("tmp_"):
            os.remove(os.path.join(target, f))
    return path, len(picks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", required=True, help="conversation JSONL (train, validation or test split)")
    parser.add_argument("--video-root", required=True, help="directory the JSONL video paths are relative to")
    parser.add_argument("--out", required=True, help="frames root")
    parser.add_argument("--frames", type=int, default=32, help="frames kept per clip")
    parser.add_argument("--fps", type=float, default=2.0, help="decode rate before uniform sampling")
    parser.add_argument("--scale", default="672:384", help="ffmpeg scale expression (W:H; -2 keeps aspect)")
    parser.add_argument("--quality", type=int, default=3, help="ffmpeg JPEG -q:v (2 best .. 31 worst)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    paths = video_paths(args.jsonl)
    short: list[tuple[str, int]] = []
    with ThreadPoolExecutor(args.workers) as pool:
        for path, n in pool.map(lambda p: extract(args.video_root, args.out, p, args.frames, args.fps, args.scale, args.quality), paths):
            if n != args.frames:
                short.append((path, n))
    print(f"{len(paths)} clips under {args.out}; {len(short)} with fewer than {args.frames} frames: {short[:10]}")


if __name__ == "__main__":
    main()
