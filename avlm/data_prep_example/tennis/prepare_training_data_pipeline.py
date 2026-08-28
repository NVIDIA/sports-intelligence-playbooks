#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tennis data-prep example (annotations -> HF train/eval data).

Runs the full pipeline and produces training / evaluation splits in
**HuggingFace conversation format**.

Legacy mode (default)::

    raw_tennis_data_input/<videoId>.json
        │  (1) generate_mcq_qa.py + tennis_mcq_config.json / tennis_qa_config.json
        ▼
    training_tennis_data_output/mcq_qa_per_video/<videoId>_derived_mcq_qa.json
        │  (2) split_dataset.run_split
        ▼
    training_tennis_data_output/data_splits_hf/*.jsonl
        │  (3) augment_categorical_data_with_metadata.py
        ▼
    enriched per-video JSON + JSONL with ``metadata`` blocks

Categorical mode (``--mode categorical``)::

    <input-dir>/<videoId>.json
        │  (1) generate + split
        ▼
    <output-dir>/generated/          # original HF data (no metadata)
        per_video/ + *.jsonl
        │  (2) copy → with_metadata/, then augment
        ▼
    <output-dir>/with_metadata/      # metadata enriched
        *.jsonl
        │  (3) copy → with_metadata_paraphrased/, then paraphrase
        ▼
    <output-dir>/with_metadata_paraphrased/   # metadata + split-aware paraphrases
        *.jsonl

Usage:
    python prepare_training_data_pipeline.py
    python prepare_training_data_pipeline.py --keep
    python prepare_training_data_pipeline.py --question-types-config configs/generation/question_types.json
    python prepare_training_data_pipeline.py --mode categorical --offline
    python prepare_training_data_pipeline.py \\
        --mode categorical \\
        --input-dir aggregated_data_complete \\
        --output-dir ../../data/categorical_data_v2 \\
        --max-workers 8
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import split_dataset
from generate_mcq_qa import ConfigDrivenMCQGenerator, load_question_types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "raw_tennis_data_input")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "training_tennis_data_output")

LEGACY_PER_VIDEO_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "mcq_qa_per_video")
LEGACY_SPLITS_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "data_splits_hf")
DEFAULT_CATEGORICAL_OUTPUT = os.path.join(
    DEFAULT_OUTPUT_ROOT, "categorical_mcq_qa_per_video"
)

# Categorical stage directory names (under --output-dir).
STAGE_GENERATED = "generated"
STAGE_WITH_METADATA = "with_metadata"
STAGE_WITH_METADATA_PARAPHRASED = "with_metadata_paraphrased"
CATEGORICAL_STAGE_DIRS = (
    STAGE_GENERATED,
    STAGE_WITH_METADATA,
    STAGE_WITH_METADATA_PARAPHRASED,
)

MCQ_CONFIG = os.path.join(BASE_DIR, "configs", "generation", "tennis_mcq_config.json")
QA_CONFIG = os.path.join(BASE_DIR, "configs", "generation", "tennis_qa_config.json")
CATEGORICAL_CONFIG = os.path.join(
    BASE_DIR, "configs", "generation", "tennis_categorical_eval_config.json"
)
SERVE_RULE_PARAPHRASE_BANK = os.path.join(
    BASE_DIR, "configs", "paraphrases", "serve_rule_paraphrase_bank.json"
)
POINT_OUTCOME_PARAPHRASE_BANK = os.path.join(
    BASE_DIR, "configs", "paraphrases", "point_outcome_paraphrase_bank.json"
)
NONE_OPTION_PARAPHRASE_BANK = os.path.join(
    BASE_DIR, "configs", "paraphrases", "none_option_paraphrase_bank.json"
)
RULES_KNOWLEDGE_PARAPHRASE_BANK = os.path.join(
    BASE_DIR, "configs", "paraphrases", "rules_knowledge_paraphrase_bank.json"
)
CATEGORICAL_PARAPHRASE_BANKS = (
    SERVE_RULE_PARAPHRASE_BANK,
    POINT_OUTCOME_PARAPHRASE_BANK,
    NONE_OPTION_PARAPHRASE_BANK,
    RULES_KNOWLEDGE_PARAPHRASE_BANK,
)


def find_input_files(input_dir: str) -> list[str]:
    """Return per-video JSON files in the input directory."""
    return sorted(glob.glob(os.path.join(input_dir, "*.json")))


def list_split_jsonl(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("tennis_mcq_qa_*.jsonl"))


def copy_split_jsonl(src_dir: str | Path, dst_dir: str | Path) -> list[Path]:
    """Copy HF split JSONL files from src_dir into dst_dir (creates dst)."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in list_split_jsonl(src_dir):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    if not copied:
        raise SystemExit(f"No tennis_mcq_qa_*.jsonl files found under {src_dir}")
    return copied


def stage_generate_legacy(
    input_dir: str,
    per_video_dir: str,
    question_types: list[str] | None = None,
) -> None:
    print("\n" + "=" * 70)
    print("STAGE 1/3: Generate MCQ + open-ended QA (legacy configs)")
    print("=" * 70)
    if not find_input_files(input_dir):
        raise SystemExit(
            f"No per-video JSON files found under {input_dir} (expected <videoId>.json)"
        )
    if question_types is None:
        print("  Generating all question types defined in MCQ/QA configs")
    else:
        print(f"  Generating {len(question_types)} question types from config")
    generator = ConfigDrivenMCQGenerator(
        mcq_config_file=MCQ_CONFIG,
        qa_config_file=QA_CONFIG,
        max_workers=4,
        api_rate_limit=0.0,
        enable_progress_bar=False,
        question_types=question_types,
    )
    generator.process_data(input_dir, per_video_dir)


def stage_generate_categorical(
    input_dir: str,
    per_video_dir: str,
    question_types: list[str] | None = None,
    offline: bool = False,
    max_workers: int = 4,
    api_rate_limit: float = 0.01,
    video_root: str | None = None,
) -> None:
    from generate_categorical_mcq_qa import CategoricalMCQGenerator

    print("\n" + "=" * 70)
    print("STAGE 1/4: Generate MCQ + open-ended QA (categorical config)")
    print("=" * 70)
    if not find_input_files(input_dir):
        raise SystemExit(
            f"No per-video JSON files found under {input_dir} (expected <videoId>.json)"
        )
    if question_types is None:
        print("  Generating all question types defined in categorical config")
    else:
        print(f"  Generating {len(question_types)} question types from config")
    if offline:
        print("  Offline mode: skipping llm_gen_distractor templates")
    else:
        print("  Online mode: including llm_gen_distractor templates")
    if video_root:
        print(f"  Video root: {video_root}")
    print(f"  Output: {per_video_dir}")
    generator = CategoricalMCQGenerator(
        categorical_config_file=CATEGORICAL_CONFIG,
        max_workers=max_workers,
        api_rate_limit=api_rate_limit,
        enable_progress_bar=False,
        question_types=question_types,
        offline=offline,
        video_root=video_root,
    )
    generator.process_data(input_dir, per_video_dir)


def stage_split(
    per_video_dir: str,
    splits_dir: str,
    *,
    production_ratios: bool,
    stage_label: str = "STAGE 2/2",
) -> None:
    print("\n" + "=" * 70)
    print(f"{stage_label}: Split into train/val/test in HF conversation format")
    print("=" * 70)
    print(f"  Splits dir: {splits_dir}")
    if production_ratios:
        # Match the production split used for avlm/data legacy JSONL.
        split_dataset.run_split(
            mcq_qa_dir=per_video_dir,
            output_dir=splits_dir,
            train_ratio=0.85,
            val_ratio=0.05,
            test_seen_ratio=0.05,
            test_unseen_ratio=0.05,
            random_seed=42,
        )
    else:
        # Larger val/test ratios than production so the tiny sample populates
        # every split; test_unseen holds out one whole game (unseen video).
        split_dataset.run_split(
            mcq_qa_dir=per_video_dir,
            output_dir=splits_dir,
            val_ratio=0.2,
            test_seen_ratio=0.2,
            test_unseen_ratio=0.1,
            random_seed=42,
        )


def stage_augment_metadata(
    data_dirs: list[str],
    metadata_dir: str,
    *,
    metadata_label: str | None = None,
    reference_only: bool = False,
    stage_label: str = "STAGE 3/4",
) -> None:
    from augment_categorical_data_with_metadata import run_augmentation

    print("\n" + "=" * 70)
    print(f"{stage_label}: Augment samples with point-level annotation metadata")
    print("=" * 70)
    print(f"  Metadata dir:  {metadata_dir}")
    print(f"  Mode:          {'reference-only' if reference_only else 'enriched'}")
    print("  Scope:         open-ended questions only")
    label = metadata_label or os.path.basename(metadata_dir.rstrip(os.sep))
    for data_dir in data_dirs:
        print(f"\n  Enriching: {data_dir}")
        # generated/ already holds the pre-metadata snapshot, so discard the
        # temporary .backup_* tree after a successful in-place augment.
        run_augmentation(
            data_dir,
            metadata_dir,
            metadata_label=label,
            reference_only=reference_only,
            open_ended_only=True,
            in_place=True,
            keep_backup=False,
        )


def stage_paraphrase_mcq_options(
    splits_dir: str,
    *,
    bank_paths: list[str] | tuple[str, ...] = CATEGORICAL_PARAPHRASE_BANKS,
    stage_label: str = "STAGE 4/4",
) -> None:
    from paraphrase_serve_rule_options import run_rewrite

    print("\n" + "=" * 70)
    print(f"{stage_label}: Paraphrase closed-vocab MCQ options (split-aware pools)")
    print("=" * 70)
    print(f"  Splits dir: {splits_dir}")
    for bank_path in bank_paths:
        print(f"\n  Bank: {bank_path}")
        run_rewrite(splits_dir, bank_path, check_overlap=True)


def clean_outputs(mode: str, paths: list[str]) -> None:
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "categorical"),
        default="legacy",
        help="Which config/generator path to run (default: legacy)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Per-video annotation JSON directory "
            f"(default: {DEFAULT_INPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Destination root. Categorical mode writes three stage dirs under "
            f"<output-dir>/{{{STAGE_GENERATED},{STAGE_WITH_METADATA},"
            f"{STAGE_WITH_METADATA_PARAPHRASED}}}. "
            f"Default categorical: {DEFAULT_CATEGORICAL_OUTPUT}"
        ),
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help=(
            "Directory used to resolve relative point clip paths. Points whose "
            "clips are missing/empty are skipped. Pass empty string to disable."
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep existing generated outputs for this mode (default: clean and regenerate)",
    )
    parser.add_argument(
        "--question-types-config",
        default=None,
        help="JSON file listing question types to generate "
        "(default: all types defined for the selected mode)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Categorical mode only: skip llm_gen_distractor templates (no API calls)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Categorical mode only: worker threads for generation",
    )
    parser.add_argument(
        "--api-rate-limit",
        type=float,
        default=0.01,
        help="Categorical mode only: minimum seconds between API calls",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Categorical mode only: LLM model used by templates that generate "
            "distractors (for example azure/openai/gpt-5.6-terra). Overrides "
            "the LLM_MODEL environment variable."
        ),
    )
    parser.add_argument(
        "--skip-metadata-augment",
        action="store_true",
        help="Skip the metadata enrichment stage",
    )
    parser.add_argument(
        "--reference-only-metadata",
        action="store_true",
        help="Keep minimal point_data fields during metadata enrichment",
    )
    parser.add_argument(
        "--skip-serve-rule-paraphrase",
        action="store_true",
        help=(
            "Categorical mode only: skip split-aware MCQ option paraphrasing "
            "(serve-rule + point-outcome + none-option banks) after metadata"
        ),
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir or DEFAULT_INPUT_DIR)
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    question_types = None
    if args.question_types_config:
        question_types = load_question_types(args.question_types_config)

    if args.mode == "legacy":
        per_video_dir = LEGACY_PER_VIDEO_DIR
        splits_dir = LEGACY_SPLITS_DIR
        if args.output_dir:
            output_root = os.path.abspath(args.output_dir)
            per_video_dir = os.path.join(output_root, "per_video")
            splits_dir = output_root
        clean_paths = [per_video_dir, splits_dir]
    else:
        output_root = os.path.abspath(args.output_dir or DEFAULT_CATEGORICAL_OUTPUT)
        # Normalize if caller points at a stage leaf or legacy flat/per_video path.
        leaf = os.path.basename(output_root.rstrip(os.sep))
        if leaf in CATEGORICAL_STAGE_DIRS or leaf == "per_video":
            output_root = os.path.dirname(output_root)
        generated_dir = os.path.join(output_root, STAGE_GENERATED)
        with_metadata_dir = os.path.join(output_root, STAGE_WITH_METADATA)
        paraphrased_dir = os.path.join(
            output_root, STAGE_WITH_METADATA_PARAPHRASED
        )
        per_video_dir = os.path.join(generated_dir, "per_video")
        splits_dir = generated_dir  # initial split lands in generated/
        clean_paths = [generated_dir, with_metadata_dir, paraphrased_dir]

    if not args.keep:
        clean_outputs(args.mode, clean_paths)

    n_inputs = len(find_input_files(input_dir))
    production_ratios = n_inputs >= 20

    if args.mode == "legacy":
        if args.offline:
            print("Note: --offline applies only to --mode categorical; ignored for legacy.")
        if args.skip_serve_rule_paraphrase:
            print(
                "Note: --skip-serve-rule-paraphrase applies only to "
                "--mode categorical; ignored for legacy."
            )
        stage_generate_legacy(
            input_dir=input_dir,
            per_video_dir=per_video_dir,
            question_types=question_types,
        )
        stage_split(
            per_video_dir,
            splits_dir,
            production_ratios=production_ratios,
            stage_label="STAGE 2/3",
        )
        if not args.skip_metadata_augment:
            stage_augment_metadata(
                data_dirs=[per_video_dir, splits_dir],
                metadata_dir=input_dir,
                reference_only=args.reference_only_metadata,
                stage_label="STAGE 3/3",
            )
        else:
            print("\nSkipping metadata augmentation (--skip-metadata-augment).")

        print("\n" + "=" * 70)
        print("DONE. HF conversation train/eval data written to:")
        print(f"  {splits_dir}")
        print(f"Per-video conversations:")
        print(f"  {per_video_dir}")
        print("=" * 70)
        return

    # --- Categorical: three named stage directories ---
    video_root = args.video_root or None
    print("\nCategorical output layout:")
    print(f"  root:         {output_root}")
    print(f"  generated:    {generated_dir}")
    print(f"  metadata:     {with_metadata_dir}")
    print(f"  paraphrased:  {paraphrased_dir}")

    stage_generate_categorical(
        input_dir=input_dir,
        per_video_dir=per_video_dir,
        question_types=question_types,
        offline=args.offline,
        max_workers=args.max_workers,
        api_rate_limit=args.api_rate_limit,
        video_root=video_root,
    )
    stage_split(
        per_video_dir,
        generated_dir,
        production_ratios=production_ratios,
        stage_label="STAGE 2/4",
    )

    metadata_source_dir = generated_dir
    if not args.skip_metadata_augment:
        print("\n" + "-" * 70)
        print(f"Snapshot → {STAGE_WITH_METADATA}/ (copy of generated splits)")
        print("-" * 70)
        if os.path.isdir(with_metadata_dir):
            shutil.rmtree(with_metadata_dir)
        copy_split_jsonl(generated_dir, with_metadata_dir)
        stage_augment_metadata(
            data_dirs=[with_metadata_dir],
            metadata_dir=input_dir,
            reference_only=args.reference_only_metadata,
            stage_label="STAGE 3/4",
        )
        metadata_source_dir = with_metadata_dir
    else:
        print("\nSkipping metadata augmentation (--skip-metadata-augment).")

    if not args.skip_serve_rule_paraphrase:
        print("\n" + "-" * 70)
        print(
            f"Snapshot → {STAGE_WITH_METADATA_PARAPHRASED}/ "
            f"(copy of {os.path.basename(metadata_source_dir)})"
        )
        print("-" * 70)
        if os.path.isdir(paraphrased_dir):
            shutil.rmtree(paraphrased_dir)
        copy_split_jsonl(metadata_source_dir, paraphrased_dir)
        stage_paraphrase_mcq_options(paraphrased_dir)
    else:
        print(
            "\nSkipping MCQ option paraphrasing (--skip-serve-rule-paraphrase)."
        )

    print("\n" + "=" * 70)
    print("DONE. Categorical stage directories:")
    print(f"  {STAGE_GENERATED}:                 {generated_dir}")
    if not args.skip_metadata_augment:
        print(f"  {STAGE_WITH_METADATA}:             {with_metadata_dir}")
    if not args.skip_serve_rule_paraphrase:
        print(f"  {STAGE_WITH_METADATA_PARAPHRASED}: {paraphrased_dir}")
    print(f"Per-video conversations (under generated):")
    print(f"  {per_video_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
