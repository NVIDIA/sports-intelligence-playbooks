---
name: prepare-data
description: Turns raw sports annotations into Hugging Face conversation JSONL train, validation and test splits with the repo's data-prep pipeline for that sport, checks the splits before training, and reports the paths a training recipe needs. Covers tennis (references/tennis.md) and basketball from the public BARD dataset (references/basketball.md). Use when the user asks to generate, regenerate, split or validate sports MCQ/QA training data, choose question templates, run offline or with LLM distractors, or fill dataset paths in a training recipe. Not for launching or monitoring training, running inference, or scoring checkpoints.
license: Apache-2.0
compatibility: "Requires a checkout of github.com/NVIDIA/sports-intelligence-playbooks with Git LFS; run from the repo root. Per-sport pipeline requirements are in the sport guide (references/<sport>.md). scripts/validate_splits.py: Python 3.8+, stdlib only. decord only for clip duration/FPS metadata."
metadata:
  author: NVIDIA
  version: "1.0.0"
---

# Prepare Data

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Produces training-ready split files from sports annotations and checks that they are safe to train on. It never launches training. Run every command from the repository root.

## When to Use This Skill

- Generate MCQ/QA splits from the bundled sample annotations or from new annotations in the same format.
- Choose question templates, or offline vs LLM-distractor generation.
- Validate existing split JSONL (schema, ids, clip paths, train/eval leakage) before a training run.
- Fill the dataset fields of an AutoModel or Megatron-Bridge recipe with the generated paths.

Not for: launching, resuming or monitoring training; running inference; scoring checkpoints; obtaining match video the user has no rights to.

## Sports

| Sport | Guide | Pipeline |
|---|---|---|
| Tennis | [references/tennis.md](references/tennis.md) | `avlm/data_prep_example/tennis/` |
| Basketball (BARD) | [references/basketball.md](references/basketball.md) | `avlm/data_prep_example/basketball/` |

## Workflow

1. **Pick the sport** and open its guide from the table above. If the sport has no guide, stop; to build a pipeline for a new labelled video dataset, follow [ONBOARD_NEW_DATASET.md](../../avlm/data_prep_example/ONBOARD_NEW_DATASET.md) only if the user asks for that.

2. **Generate the splits** by following the sport guide: install, mode, command and expected output. Write to a new directory outside the repo, held in `$DATA_OUT`. Follow the sport guide on whether to use an LLM (for example for MCQ distractors). Use one only when the guide calls for it or the user explicitly asks, and the user agrees to send annotation text to an external, billed API. Never print or commit keys.

3. **Validate** the training-ready split directory (`$SPLITS`) with the bundled checker (`scripts/validate_splits.py`, stdlib only). Use the exact command from the sport guide (prefix, split names and flags):

   ```bash
   python3 skills/prepare-data/scripts/validate_splits.py \
     --splits-dir "$SPLITS" --prefix "<prefix from the sport guide>"
   ```

   Once clips exist, add `--video-root "$VIDEO_ROOT"`. The training loaders silently drop rows whose clip is missing, so this check matters. Exit code 0 means pass. 1 means a validation failure, with up to ten `error:` lines on stderr. 2 means a bad argument or missing directory. It checks:
   - every split file exists and is non-empty
   - every record has a unique `id` across splits, and one user turn holding both the video and the question, plus an assistant answer
   - no train clip appears in an eval split (except splits the guide passes in `--train-pool-splits`), and no source video of the held-out `test_unseen*` split appears elsewhere
   - with `--video-root`, every clip exists under that root

   The stdout JSON gives per-split `records`, `clips` and `classes`. Quote it in the report.

4. **Build clip metadata** (real clips only; needs `decord`), unless the sport guide says its pipeline writes it. Run once for the train file and once for the validation file, then point `video_metadata_path` at the printed `*_video_metadata.json`:

   ```bash
   python3 ./avlm/utils/build_video_metadata.py \
     --dataset_path "$SPLITS/<train file>" \
     --video_root "$VIDEO_ROOT" --workers 8 --output-dir "$DATA_OUT/metadata"
   ```

   Details: "Precompute clip duration and FPS" in the [AutoModel SFT guide](../../avlm/training/automodel/sft/SFT_GUIDE.MD).

5. **Hand off to training.** Do not launch training here. Report these recipe values, as absolute paths resolved with `realpath`. They go in the user's `*_local.yaml` copy of an AutoModel or Megatron-Bridge recipe under `avlm/training/`:

   | Recipe field | Value |
   |---|---|
   | `dataset.path_or_dataset` | `$SPLITS/<train file>` |
   | `dataset.video_root`, `validation_dataset.video_root` | `$VIDEO_ROOT` |
   | `dataset.video_metadata_path` | train `*_video_metadata.json` |
   | `validation_dataset.path_or_dataset`, `inference_validation.data_path` | `$SPLITS/<validation file>` |
   | `validation_dataset.video_metadata_path` (AutoModel recipes only) | validation `*_video_metadata.json` |

   Megatron-Bridge recipes also need `jsonl_format: hf`, which is already set. Keep the test splits for evaluation only. Sample annotations without videos prove only that the pipeline works; they cannot be used for training.

## Output

Report all of the following:
- sport, mode, input dir and training-ready split dir
- the validator JSON (records per split, contamination counts, missing clips)
- skipped points, and any question types that were excluded
- the recipe field values from step 5

## Final Checks

- [ ] Validator exit code is 0, and both `contamination` counts are 0.
- [ ] The sport guide's own checks pass.
- [ ] Output is outside the repo, and `git status` shows no changes to tracked sample outputs.
- [ ] No API key appears in logs, files or the reply.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Validator `contamination: N clips in both train and ...` | Split files from different runs were mixed, or were hand-edited. Regenerate all splits in one run. |
| Validator `N of M clips missing under ...` | `--video-root` does not match the paths in the records. Fix the root, or regenerate with `--video-root` so missing clips are filtered out. |
| Validator `N clips escape --video-root` | A record uses an absolute path or `..`. Regenerate with clip paths relative to the video root. |
| Validator `duplicate id` | Records were concatenated twice, or two runs were merged. Regenerate. |
| Sport-specific errors | See the sport guide's troubleshooting table. |
