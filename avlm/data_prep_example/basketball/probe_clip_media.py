#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe every labelled BARD clip's duration, frame count and fps into one media table.

``generate_categorical_mcq_qa.py`` reads clip lengths from this table (for the duration-only
baseline and the per-split ``*_meta_input_video_metadata.json`` files that sequence packing
needs) instead of decoding 14,676 videos itself. Only clips listed in ``dataset_paths.csv``
are probed: the BARD tree holds 24,692 mp4 files but only 14,676 carry labels.

Needs ``decord`` (shipped in the NeMo containers). The output format is the one
``avlm/utils/build_video_metadata.py`` produces:
``{"<game>/<n>.mp4": [duration_s, frames, fps]}``.

Usage (from this directory):
    python probe_clip_media.py --dataset-root raw_basketball_data_input/BARD \\
        --output training_basketball_data_output/clip_media.json --workers 48
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool

import decord


def probe_clip(args: tuple[str, str]) -> tuple[str, list[float]]:
    dataset_root, relative_path = args
    reader = decord.VideoReader(os.path.join(dataset_root, relative_path), num_threads=1)
    num_frames = len(reader)
    fps = float(reader.get_avg_fps())
    del reader
    return relative_path, [num_frames / max(fps, 1e-6), num_frames, fps]


def labelled_clips(dataset_root: str) -> list[str]:
    """Relative clip paths in dataset_paths.csv order (the table keeps that order)."""
    csv_path = os.path.join(dataset_root, "dataset_paths.csv")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader)  # header: urls;actions;numerosity
        return [row[0] for row in reader]


def probe_clips(dataset_root: str, output: str, workers: int) -> None:
    """Probe every labelled clip with ``workers`` decoder processes and write the table."""
    paths = labelled_clips(dataset_root)
    with Pool(workers) as pool:
        items = pool.map(probe_clip, [(dataset_root, p) for p in paths], chunksize=32)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(dict(items), handle)
    print(f"Wrote {len(items)} clips to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-root", required=True, help="the BARD tree (dataset_paths.csv, <game>/<n>.mp4)"
    )
    parser.add_argument("--output", required=True, help="where to write the media table JSON")
    parser.add_argument("--workers", type=int, default=48, help="decoder processes")
    args = parser.parse_args()
    probe_clips(args.dataset_root, args.output, args.workers)


if __name__ == "__main__":
    main()
