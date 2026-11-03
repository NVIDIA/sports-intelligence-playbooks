#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end basketball data-prep example (BARD clip annotations -> HF train/eval data).

Runs the full pipeline and produces the categorical MCQ/QA splits in **HuggingFace
conversation format**::

    <dataset-root>/dataset_paths.csv + captions/caption.json + <game>/<n>.mp4
        │  (1) probe_clip_media.py            (decord; skipped when the media table exists)
        ▼
    <out-dir>/clip_media.json                  {"<game>/<n>.mp4": [duration_s, frames, fps]}
        │  (2) generate_categorical_mcq_qa.py  (16 templates, quality gates, game-level split)
        ▼
    <out-dir>/<prefix>_{train,val,test_seen,test_unseen}.jsonl
    <out-dir>/<prefix>_<split>_meta_input.jsonl + _meta_input_video_metadata.json
    <out-dir>/split_stats.json

Usage (from this directory, with BARD cloned to raw_basketball_data_input/BARD):
    python prepare_training_data_pipeline.py
    # BARD elsewhere, and/or reuse an already probed media table (no decord needed):
    python prepare_training_data_pipeline.py --dataset-root /path/to/BARD \\
        --media-json /path/to/clip_media.json
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from generate_categorical_mcq_qa import DEFAULT_PREFIX, DEFAULT_SEED, generate_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_ROOT = os.path.join(BASE_DIR, "raw_basketball_data_input", "BARD")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "training_basketball_data_output")


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def stage_probe_media(dataset_root: str, media_json: str, workers: int) -> None:
    print_banner("STAGE 1/2: Probe clip media (duration, frames, fps)")
    if os.path.exists(media_json):
        print(f"  Media table exists, skipping: {media_json}")
        return
    # Imported here so the pipeline runs without decord when a media table is supplied.
    from probe_clip_media import probe_clips

    probe_clips(dataset_root, media_json, workers)


def stage_generate(
    dataset_root: str, out_dir: str, media_json: str, seed: int, prefix: str
) -> None:
    print_banner("STAGE 2/2: Generate categorical MCQ + QA splits")
    generate_dataset(Path(dataset_root), Path(out_dir), Path(media_json), seed=seed, prefix=prefix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-root", default=DEFAULT_DATASET_ROOT,
        help="the BARD tree (dataset_paths.csv, captions/, <game>/); "
        "default: raw_basketball_data_input/BARD next to this script",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUTPUT_DIR,
        help="where the splits, metadata and split_stats.json are written; "
        "default: training_basketball_data_output/ next to this script",
    )
    parser.add_argument(
        "--media-json", default=None,
        help="probed media table; defaults to <out-dir>/clip_media.json and is probed "
        "with decord when missing",
    )
    parser.add_argument("--workers", type=int, default=48, help="decoder processes for the probe")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="42 reproduces the catalog's split"
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="output file prefix")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.dataset_root, "dataset_paths.csv")):
        raise SystemExit(
            f"BARD not found at {args.dataset_root} (no dataset_paths.csv): "
            "clone it there or pass --dataset-root"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    media_json = args.media_json or os.path.join(args.out_dir, "clip_media.json")

    stage_probe_media(args.dataset_root, media_json, args.workers)
    stage_generate(args.dataset_root, args.out_dir, media_json, args.seed, args.prefix)
    print(f"\nDone. Splits are in {args.out_dir}")


if __name__ == "__main__":
    main()
