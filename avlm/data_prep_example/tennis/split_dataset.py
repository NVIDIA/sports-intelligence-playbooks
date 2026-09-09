#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Split the MCQ/QA dataset into train/validation/test sets (HF conversation format).

The per-video input files are already HuggingFace conversation records (produced
by ``generate_mcq_qa.py``); this script only splits them and writes JSONL.

Split properties:
1. All MCQ and QA questions for the same tennis point stay together
2. Test sets cover both:
   - completely unseen videos (held out entirely), and
   - seen videos but unseen points
3. Splitting is done at the point level to prevent data leakage

Usage:
    python split_dataset.py --input_dir training_tennis_data_output/mcq_qa_per_video --output_dir training_tennis_data_output/data_splits_hf
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import glob


def load_all_conversations(mcq_qa_dir: str) -> Dict[str, List[dict]]:
    """
    Load all conversation files from the per-video directory.
    
    Args:
        mcq_qa_dir: Directory containing per-video MCQ/QA JSON files
        
    Returns:
        Dictionary mapping video_id to list of conversations
    """
    video_conversations = {}
    
    json_files = sorted(glob.glob(os.path.join(mcq_qa_dir, "*_derived_mcq_qa.json")))
    
    print(f"Found {len(json_files)} video files")
    
    for json_file in json_files:
        video_id = os.path.basename(json_file).replace("_derived_mcq_qa.json", "")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
                video_conversations[video_id] = conversations
                print(f"  Loaded {len(conversations)} conversations from {video_id}")
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
    
    return video_conversations


def group_by_points(conversations: List[dict]) -> Dict[str, List[dict]]:
    """
    Group conversations by point ID.
    
    All MCQ and QA questions for the same point will be grouped together.
    Point ID is extracted from the conversation ID.
    
    Args:
        conversations: List of conversation dictionaries
        
    Returns:
        Dictionary mapping point_id to list of conversations for that point
    """
    point_groups = defaultdict(list)
    
    for conv in conversations:
        conv_id = conv.get('id', '')
        
        # Extract point ID from conversation ID
        # Format: videoId_TaskName_pointId
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2:
            point_id = parts[1]  # Last part is the point UUID
            point_groups[point_id].append(conv)
        else:
            # Fallback: use entire ID as point ID
            point_groups[conv_id].append(conv)
    
    return dict(point_groups)


def split_dataset(video_conversations: Dict[str, List[dict]], 
                 train_ratio: float = 0.85,
                 val_ratio: float = 0.05,
                 test_seen_ratio: float = 0.05,
                 test_unseen_ratio: float = 0.05,
                 random_seed: int = 42) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """
    Split dataset into train/validation/test sets with sophisticated strategy.
    
    Strategy:
    1. First, set aside 5% of VIDEOS (not conversations) COMPLETELY for test_unseen (held-out videos)
    2. From the remaining 95% of videos, split at the point level:
       - Each video contributes ~5% of its points to validation
       - Each video contributes ~5% of its points to test_seen
       - Each video contributes ~90% of its points to training
       - This ensures EVERY video contributes to all three splits
    3. All MCQ and QA for a given point stay together (grouped by point ID)
    
    Note: The final split percentages will be approximately 90%/2.5%/2.5%/5% but may vary
    slightly since we split at the point level (not conversation level).
    
    Args:
        video_conversations: Dictionary mapping video_id to conversations
        train_ratio: Target proportion of data for training (default: 0.85)
        val_ratio: Target proportion of data for validation (default: 0.05)
        test_seen_ratio: Target proportion for test from seen videos (default: 0.05)
        test_unseen_ratio: Proportion of VIDEOS for test from unseen videos (default: 0.05)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_conversations, val_conversations, test_seen_videos, test_unseen_videos)
    """
    random.seed(random_seed)
    
    # Calculate how many conversations we need for each split
    total_conversations = sum(len(convs) for convs in video_conversations.values())
    target_train = int(total_conversations * train_ratio)
    target_val = int(total_conversations * val_ratio)
    target_test_seen = int(total_conversations * test_seen_ratio)
    target_test_unseen = int(total_conversations * test_unseen_ratio)
    
    print(f"\nDataset Split Configuration:")
    print(f"  Total conversations: {total_conversations}")
    print(f"  Total videos: {len(video_conversations)}")
    print(f"  Target train: {target_train} ({train_ratio*100:.1f}%)")
    print(f"  Target validation: {target_val} ({val_ratio*100:.1f}%)")
    print(f"  Target test (seen videos): {target_test_seen} ({test_seen_ratio*100:.1f}%)")
    print(f"  Target test (unseen videos): ~{target_test_unseen} conversations ({test_unseen_ratio*100:.1f}% of videos)")
    print(f"  Strategy: Set aside {test_unseen_ratio*100:.1f}% of videos (not conversations) for test_unseen, then split remaining 95% at point-level")
    
    # Group all videos by size (number of conversations)
    video_ids = list(video_conversations.keys())
    random.shuffle(video_ids)  # Shuffle for randomness
    
    # Calculate 5% of VIDEOS (not conversations)
    total_videos = len(video_ids)
    num_test_unseen_videos = max(1, int(total_videos * test_unseen_ratio))
    
    # Select videos for complete hold-out (TEST UNSEEN - 5% of all videos)
    unseen_test_videos = video_ids[:num_test_unseen_videos]
    seen_videos = video_ids[num_test_unseen_videos:]
    
    # Count conversations in each split
    test_unseen_convs = sum(len(video_conversations[vid]) for vid in unseen_test_videos)
    seen_convs = sum(len(video_conversations[vid]) for vid in seen_videos)
    
    print(f"\nVideo Distribution:")
    print(f"  Held-out videos (test_unseen only): {len(unseen_test_videos)} videos ({len(unseen_test_videos)/total_videos*100:.1f}% of videos)")
    print(f"    - These {len(unseen_test_videos)} videos contain {test_unseen_convs} conversations ({test_unseen_convs/total_conversations*100:.1f}% of data)")
    print(f"  Remaining videos (for train + val + test_seen): {len(seen_videos)} videos ({len(seen_videos)/total_videos*100:.1f}% of videos)")
    print(f"    - These {len(seen_videos)} videos contain {seen_convs} conversations ({seen_convs/total_conversations*100:.1f}% of data)")
    
    # Collect conversations from unseen videos
    val_conversations = []
    test_seen_videos = []  # Test data from seen videos (unseen points)
    test_unseen_videos = []  # Test data from completely held-out videos
    train_conversations = []
    
    # Add all conversations from unseen test videos
    for video_id in unseen_test_videos:
        test_unseen_videos.extend(video_conversations[video_id])
    
    # Now split the REMAINING (95%) videos at the point level for train/val/test_seen
    print(f"\nSplitting remaining videos at point-level:")
    print(f"  Each video will contribute ~5% points to validation, ~5% to test_seen, ~90% to training")
    print(f"  This ensures all videos contribute to all splits")
    print(f"  All MCQ and QA for each point are kept together")
    
    # Process seen videos: split their points
    for video_id in seen_videos:
        convs = video_conversations[video_id]
        
        # Group by points
        point_groups = group_by_points(convs)
        point_ids = list(point_groups.keys())
        random.shuffle(point_ids)
        
        # Split points for this video
        num_points = len(point_ids)
        
        # Allocate val_ratio of points to val, test_seen_ratio to test_seen, rest to train
        # (using floor division). With the default 5% ratios, videos with < 20 points
        # get 0 allocated to val/test; the example uses larger ratios so small sample
        # videos still populate every split.
        num_val_points = int(num_points * val_ratio)
        num_test_points = int(num_points * test_seen_ratio)
        
        # Allocate points
        val_points = point_ids[:num_val_points]
        test_points = point_ids[num_val_points:num_val_points + num_test_points]
        train_points = point_ids[num_val_points + num_test_points:]
        
        # Add conversations for each point group (keeping all MCQ/QA together)
        for point_id in val_points:
            point_convs = point_groups[point_id]
            val_conversations.extend(point_convs)
        
        for point_id in test_points:
            point_convs = point_groups[point_id]
            test_seen_videos.extend(point_convs)
        
        for point_id in train_points:
            train_conversations.extend(point_groups[point_id])
    
    print(f"\nFinal Split Sizes:")
    print(f"  Train: {len(train_conversations)} ({len(train_conversations)/total_conversations*100:.1f}%)")
    print(f"  Validation: {len(val_conversations)} ({len(val_conversations)/total_conversations*100:.1f}%)")
    print(f"  Test (seen videos - unseen points): {len(test_seen_videos)} ({len(test_seen_videos)/total_conversations*100:.1f}%)")
    print(f"  Test (unseen videos): {len(test_unseen_videos)} ({len(test_unseen_videos)/total_conversations*100:.1f}%)")
    print(f"  Total: {len(train_conversations) + len(val_conversations) + len(test_seen_videos) + len(test_unseen_videos)}")
    
    return train_conversations, val_conversations, test_seen_videos, test_unseen_videos


def save_split_jsonl(conversations: List[dict], output_file: str) -> None:
    """Save HF conversation records as JSONL (one record per line)."""
    written = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in conversations:
            if not entry.get("conversation"):
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    print(f"  Saved {written} HF conversation records to {output_file}")


def analyze_split(train: List[dict], val: List[dict], test_seen: List[dict], test_unseen: List[dict]) -> None:
    """Analyze and print statistics about the split."""
    
    def get_video_ids(conversations: List[dict]) -> set:
        """Extract unique video IDs from conversations."""
        video_ids = set()
        for conv in conversations:
            conv_id = conv.get('id', '')
            # Extract video ID (everything before the task name)
            parts = conv_id.split('_')
            if len(parts) >= 2:
                # Video ID is everything except the last two parts (TaskName_pointId)
                video_id = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                video_ids.add(video_id)
        return video_ids
    
    def get_points(conversations: List[dict]) -> Dict[str, set]:
        """Extract points grouped by video."""
        video_points = defaultdict(set)
        for conv in conversations:
            conv_id = conv.get('id', '')
            parts = conv_id.split('_')
            if len(parts) >= 2:
                video_id = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                point_id = parts[-1]
                video_points[video_id].add(point_id)
        return dict(video_points)
    
    # Combine test sets for total analysis
    test_all = test_seen + test_unseen
    total = len(train) + len(val) + len(test_all)
    
    print("\n" + "=" * 60)
    print("SPLIT ANALYSIS")
    print("=" * 60)
    
    print(f"\nConversation Distribution:")
    print(f"  Train: {len(train)} ({len(train)/total*100:.2f}%)")
    print(f"  Validation: {len(val)} ({len(val)/total*100:.2f}%)")
    print(f"  Test (TOTAL): {len(test_all)} ({len(test_all)/total*100:.2f}%)")
    print(f"    - Test (seen videos, unseen points): {len(test_seen)} ({len(test_seen)/total*100:.2f}%)")
    print(f"    - Test (unseen videos): {len(test_unseen)} ({len(test_unseen)/total*100:.2f}%)")
    print(f"  Total: {total}")
    
    # Video analysis
    train_videos = get_video_ids(train)
    val_videos = get_video_ids(val)
    test_seen_videos = get_video_ids(test_seen)
    test_unseen_videos_set = get_video_ids(test_unseen)
    
    print(f"\nVideo Distribution:")
    print(f"  Train unique videos: {len(train_videos)}")
    print(f"  Validation unique videos: {len(val_videos)}")
    print(f"    - ALL from training videos (no held-out videos for validation)")
    print(f"  Test unique videos:")
    print(f"    - Seen videos (unseen points): {len(test_seen_videos)}")
    print(f"    - Unseen videos (held-out): {len(test_unseen_videos_set)}")
    
    # Point analysis
    train_points = get_points(train)
    val_points = get_points(val)
    test_seen_points = get_points(test_seen)
    test_unseen_points = get_points(test_unseen)
    
    total_train_points = sum(len(points) for points in train_points.values())
    total_val_points = sum(len(points) for points in val_points.values())
    total_test_seen_points = sum(len(points) for points in test_seen_points.values())
    total_test_unseen_points = sum(len(points) for points in test_unseen_points.values())
    
    print(f"\nPoint Distribution:")
    print(f"  Train unique points: {total_train_points}")
    print(f"  Validation unique points: {total_val_points}")
    print(f"  Test (seen videos) unique points: {total_test_seen_points}")
    print(f"  Test (unseen videos) unique points: {total_test_unseen_points}")
    
    # Check for point overlap in seen videos
    val_seen_points = 0
    val_unseen_points = 0
    for video_id in val_videos:
        if video_id in train_points and video_id in val_points:
            overlap = train_points[video_id] & val_points[video_id]
            val_seen_points += len(overlap)
            val_unseen_points += len(val_points[video_id] - overlap)
    
    test_seen_point_overlap = 0
    test_seen_point_unseen = 0
    for video_id in test_seen_videos:
        if video_id in train_points and video_id in test_seen_points:
            overlap = train_points[video_id] & test_seen_points[video_id]
            test_seen_point_overlap += len(overlap)
            test_seen_point_unseen += len(test_seen_points[video_id] - overlap)
    
    print(f"\nPoint Overlap Analysis:")
    print(f"  Strategy: 5% videos held-out completely for test_unseen, 95% split at point-level")
    print(f"  Validation:")
    print(f"    - ALL points from training videos (no held-out videos)")
    print(f"    - Unseen points from training videos: {val_unseen_points}")
    print(f"    - Overlapping points (should be 0): {val_seen_points}")
    print(f"  Test (seen videos):")
    print(f"    - Unseen points from training videos: {test_seen_point_unseen}")
    print(f"    - Overlapping points (should be 0): {test_seen_point_overlap}")
    print(f"  Test (unseen videos):")
    print(f"    - Points from held-out videos (~5% of all videos): {total_test_unseen_points}")
    
    # Question type distribution
    def get_question_types(conversations: List[dict]) -> Dict[str, int]:
        """Count question types from conversation IDs (videoId_TaskName_pointId)."""
        types = defaultdict(int)
        for conv in conversations:
            conv_id = conv.get('id', '')
            before_point, _, _ = conv_id.rpartition('_')
            _, _, task_name = before_point.partition('_')
            types[task_name or 'unknown'] += 1
        return dict(types)
    
    print(f"\nQuestion Type Distribution (Train):")
    train_types = get_question_types(train)
    for qtype, count in sorted(train_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {qtype}: {count}")
    
    print("\n" + "=" * 60)


def run_split(mcq_qa_dir: str = "training_tennis_data_output/mcq_qa_per_video",
              output_dir: str = "training_tennis_data_output/data_splits_hf",
              train_ratio: float = 0.85,
              val_ratio: float = 0.05,
              test_seen_ratio: float = 0.05,
              test_unseen_ratio: float = 0.05,
              random_seed: int = 42) -> None:
    """Load per-video HF conversation records, split, and write JSONL files.

    Args:
        mcq_qa_dir: Directory of ``*_derived_mcq_qa.json`` per-video files.
        output_dir: Where to write the split JSONL files.
        train_ratio/val_ratio/test_seen_ratio/test_unseen_ratio: Split targets.
        random_seed: Reproducibility seed.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "tennis_mcq_qa_train.jsonl")
    val_file = os.path.join(output_dir, "tennis_mcq_qa_validation.jsonl")
    test_seen_file = os.path.join(output_dir, "tennis_mcq_qa_test_seen_videos.jsonl")
    test_unseen_file = os.path.join(output_dir, "tennis_mcq_qa_test_unseen_videos.jsonl")

    print("Tennis MCQ/QA Dataset Splitter")
    print("=" * 60)
    print(f"Input directory: {mcq_qa_dir}")
    print(f"Output files:")
    print(f"  - Train: {train_file}")
    print(f"  - Validation: {val_file}")
    print(f"  - Test (seen videos): {test_seen_file}")
    print(f"  - Test (unseen videos): {test_unseen_file}")
    print()

    # Load all conversations
    print("Loading conversations...")
    video_conversations = load_all_conversations(mcq_qa_dir)

    if not video_conversations:
        print("Error: No conversations loaded. Check the input directory.")
        return

    # Split dataset
    print("\nSplitting dataset...")
    train, val, test_seen, test_unseen = split_dataset(
        video_conversations,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_seen_ratio=test_seen_ratio,
        test_unseen_ratio=test_unseen_ratio,
        random_seed=random_seed
    )

    # Save splits
    print("\nSaving splits...")
    save_split_jsonl(train, train_file)
    save_split_jsonl(val, val_file)
    save_split_jsonl(test_seen, test_seen_file)
    save_split_jsonl(test_unseen, test_unseen_file)

    # Analyze and print statistics
    analyze_split(train, val, test_seen, test_unseen)

    print("\nDataset split completed successfully!")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split tennis MCQ/QA into train/val/test (HF conversation JSONL).")
    parser.add_argument("--input_dir", default="training_tennis_data_output/mcq_qa_per_video",
                        help="Directory of *_derived_mcq_qa.json per-video files")
    parser.add_argument("--output_dir", default="training_tennis_data_output/data_splits_hf",
                        help="Directory to write split JSONL files")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_seen_ratio", type=float, default=0.05)
    parser.add_argument("--test_unseen_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    run_split(
        mcq_qa_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_seen_ratio=args.test_seen_ratio,
        test_unseen_ratio=args.test_unseen_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
