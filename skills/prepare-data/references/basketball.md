# Basketball (BARD) data preparation

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Sport guide for the [prepare-data skill](../SKILL.md).
Run from `avlm/data_prep_example/basketball/` in a `sports-intelligence-playbooks` checkout;
the scripts live there, not next to this file. Defaults: BARD under `raw_basketball_data_input/BARD`,
media table `<out-dir>/clip_media.json`, seed 42. Write outputs to `$DATA_OUT`, a new directory
outside the repo (the script's default `training_basketball_data_output/` is inside it).

## 1. Check first

1. `python3 --version` is 3.8 or newer (standard library only).
2. BARD: `<dataset-root>/dataset_paths.csv` has 14,677 lines (header + 14,676 clips) and
   `<dataset-root>/captions/caption.json` exists. Fewer lines means a partial download, which the
   generator rejects.
3. Media table `<out-dir>/clip_media.json` exists; if not, `python3 -c "import decord"` must work.
4. `git lfs version` works (only if BARD must be downloaded).

## 2. Ask only for what is missing

- No BARD: ask for its folder, or offer to download it (~324 GB, git-lfs, must be complete).
  Confirm before downloading.
- No media table: ask for an existing `clip_media.json`, or, if decord works, how many CPU
  workers to probe with (default 48). Without decord, install it with `python3 -m pip install decord`
  (the AutoModel container does not include it; there, add `--no-deps --target <dir>` and put
  `<dir>` on `PYTHONPATH`).
- Do not ask about seed or prefix unless the user raises them.

## 3. Run

```bash
export DATA_OUT="${DATA_OUT:-$(mktemp -d)}"
export BARD_ROOT="${BARD_ROOT:-$PWD/avlm/data_prep_example/basketball/raw_basketball_data_input/BARD}"
cd avlm/data_prep_example/basketball
git lfs install && git clone https://huggingface.co/datasets/GabrieleGiudici/BARD "$BARD_ROOT"   # only if downloading
python3 prepare_training_data_pipeline.py --dataset-root "$BARD_ROOT" --out-dir "$DATA_OUT"  # other flags only for non-defaults: --media-json --workers
cd -
```

The clone takes hours and the probe minutes: run them in the background and poll. Generation
takes under a minute and ends with `Done. Splits are in <out-dir>`.

## 4. Report

Exit code 0 and, on the complete dataset with seed 42, `built 175193 rows from 14676 clips`;
splits train 127,211 / val 6,695 / test_seen 15,086 / test_unseen 26,201 rows. Report the
counts, the output folder and its files: `bard_cat_<split>.jsonl`, `*_meta_input.jsonl`,
`*_meta_input_video_metadata.json`, `split_stats.json`.

## 5. Validate and hand off

From the repository root, with `$DATA_OUT` the directory printed by `Done. Splits are in <dir>` and
`$BARD_ROOT` the dataset root used above:

```bash
python3 skills/prepare-data/scripts/validate_splits.py \
  --splits-dir "$DATA_OUT" --prefix bard_cat_ \
  --splits train,val,test_seen,test_unseen --train-pool-splits val \
  --video-root "$BARD_ROOT"
```

`val` is sampled from the training clips by design, so `--train-pool-splits val` skips it in the
train-clip overlap check; `test_seen` and `test_unseen` are still checked. The pipeline already
writes the clip metadata, so skip the skill's metadata step. Hand-off values: train
`bard_cat_train.jsonl`, validation `bard_cat_val.jsonl`, video root = the dataset root, metadata
`bard_cat_train_meta_input_video_metadata.json` and `bard_cat_val_meta_input_video_metadata.json`.
Keep `test_seen` and `test_unseen` for evaluation only.

## Troubleshooting

- `BARD not found at ...`: wrong `--dataset-root`; the folder must contain `dataset_paths.csv`.
- `ModuleNotFoundError: decord`: pass an existing table with `--media-json`, or install decord (see 2).
- `AssertionError: per-class totals diverged`: incomplete or changed dataset. Stop and tell the
  user; do not edit the generator's expected totals.
- Validator `contamination: N clips in both train and val`: add `--train-pool-splits val`.
