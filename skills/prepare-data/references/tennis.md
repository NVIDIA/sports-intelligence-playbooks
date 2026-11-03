# Prepare Data: Tennis

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Sport guide for the [prepare-data skill](../SKILL.md). It turns per-match tennis annotation JSON into `tennis_mcq_qa_{train,validation,test_seen_videos,test_unseen_videos}.jsonl` with the pipeline in `avlm/data_prep_example/tennis/`. Then return to the skill's workflow to validate and hand off.

## Inputs

| Input | Default / rule |
|---|---|
| Annotation dir | One `<videoId>.json` per match. Sample: `./avlm/data_prep_example/tennis/raw_tennis_data_input` (2 matches, 20 points, no videos). Format: "Input format" in the [tennis README](../../../avlm/data_prep_example/tennis/README.md). |
| Output dir | A new directory outside the repo (`$DATA_OUT`). Every run deletes its outputs first. In legacy mode that means the entire `--output-dir`, so never point it at a directory that holds other files. Legacy without `--output-dir` overwrites tracked files in `training_tennis_data_output/`. |
| Mode | `categorical` unless the user asks for the legacy templates. |
| Video root | Directory with the clips the records reference. Only needed for real training data. |
| LLM key | Only for the legacy `mcq_*` templates, and only if the user explicitly wants LLM distractors (see "LLM distractors" below). |

Clip extraction from source footage is covered only by the tennis README (`build_sample_from_raw.py --extract-clips`, needs ffmpeg).

## Install

Python 3.10+ and `tqdm`, from the tennis `pyproject.toml` and `uv.lock`:

```bash
(cd avlm/data_prep_example/tennis && uv sync)
source avlm/data_prep_example/tennis/.venv/bin/activate
python3 -c "import tqdm; print('deps ok')"
```

Without uv: `python3 -m pip install tqdm openai python-dotenv` (append `--break-system-packages` on an externally managed system Python).

## Pick the mode

| | `categorical` (recommended) | `legacy` |
|---|---|---|
| Templates | `configs/generation/tennis_categorical_eval_config.json` (5 evaluation areas, with `super_category`/`fine_category`) | `tennis_mcq_config.json` + `tennis_qa_config.json` |
| LLM use | None with the current config. `--offline` guarantees zero API calls. | 9 `mcq_*` templates call an LLM for distractors. `--offline` skips them; without a key and without `--offline` the run stops (exit 2). |
| Training-ready dir (`$SPLITS`) | `$DATA_OUT/categorical/with_metadata_paraphrased/` | `$DATA_OUT/legacy/` |
| Sample-data records | 510-518 total (varies); test_unseen_videos 210 in every run | 177 = 53/18/17/89 with `--offline` |

Categorical counts change by a few records between runs: four templates use `generation_probability: 0.15` with an unseeded RNG. Validate invariants, not exact counts.

## Generate

Categorical, offline:

```bash
export DATA_OUT="${DATA_OUT:-$(mktemp -d)}"
python3 ./avlm/data_prep_example/tennis/prepare_training_data_pipeline.py \
  --mode categorical --offline \
  --input-dir ./avlm/data_prep_example/tennis/raw_tennis_data_input \
  --output-dir "$DATA_OUT/categorical" \
  --video-root ""
```

Always pass `--video-root` explicitly: the default differs between the development branch (a maintainer cluster path) and the public release (unset). `--video-root ""` disables clip filtering for annotation-only data such as the sample (legacy ignores it). With real clips pass `--video-root "$VIDEO_ROOT"`: points whose clip is missing or empty are skipped and reported as `Points skipped for missing clips: N`. The run ends with four `Overlap check passed (train option texts ∩ eval option texts = ∅)` lines, one per paraphrase bank. A missing line means the paraphrase stage failed.

Legacy, offline (skips the 9 `mcq_*` templates):

```bash
python3 ./avlm/data_prep_example/tennis/prepare_training_data_pipeline.py \
  --mode legacy --offline \
  --input-dir ./avlm/data_prep_example/tennis/raw_tennis_data_input \
  --output-dir "$DATA_OUT/legacy"
```

LLM distractors: prefer `--offline` for tennis. With the current tennis templates, the same caption annotation produces both a QA item and an MCQ item, so LLM-generated distractors leak between the QA and MCQ variants; the team does not use them for training.

Without `--offline` and without a key, the pipeline stops before it deletes or writes anything, with `error: 9 selected templates generate LLM distractors (llm_gen_distractor) but NVIDIA_API_KEY is not set. ...` (exit 2). If the user explicitly wants LLM distractors: copy `avlm/data_prep_example/tennis/dev.env` to `dev_local.env` (gitignored) and fill in `NVIDIA_API_KEY`, or set `LLM_API_PROVIDER=openai` and `OPENAI_API_KEY`. The default distractor model, set in `dev.env`, is a frontier LLM; it is not the model being fine-tuned. `--llm-model` overrides it. If a key is set but the calls fail, each `mcq_*` record logs `Error generating LLM distractors: ...` and keeps three canned distractors, counted as `Fallback calls`; the run ends with `warning: N mcq_* questions use fixed generic distractors`. Treat that output as unusable. The committed sample splits in `training_tennis_data_output/data_splits_hf/` contain 180 such records out of 357.

Split policy: all questions for a point stay in one split, and whole matches are held out as `test_unseen_videos` (at least one). With 20 or more match files the ratios are train/validation/test_seen/test_unseen = 0.85/0.05/0.05/0.05. With fewer, validation and test_seen get 0.2 each and test_unseen 0.1, so the sample fills every split. The split seed is fixed at 42.

## Validate and hand off

The validator's defaults match tennis (`--prefix tennis_mcq_qa_`, splits `train,validation,test_seen_videos,test_unseen_videos`):

```bash
python3 skills/prepare-data/scripts/validate_splits.py \
  --splits-dir "$DATA_OUT/categorical/with_metadata_paraphrased"
```

For legacy use `--splits-dir "$DATA_OUT/legacy"`. For the recipe hand-off, the train file is `tennis_mcq_qa_train.jsonl` and the validation file is `tennis_mcq_qa_validation.jsonl`. Keep both `test_*` splits for evaluation only.

## Tennis checks

- [ ] Categorical runs printed four `Overlap check passed` lines.
- [ ] Legacy logs contain no `Error generating LLM distractors` lines, and `Fallback calls (fixed generic distractors)` is 0.
- [ ] `git status` shows no changes under `avlm/data_prep_example/tennis/training_tennis_data_output/`.

## Tennis troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No per-video JSON files found under ... (expected <videoId>.json)` (exit 1) | `--input-dir` is wrong. Relative paths resolve from the current directory. Point it at the folder of match JSON files. |
| `error: 9 selected templates generate LLM distractors (llm_gen_distractor) but NVIDIA_API_KEY is not set` (exit 2) | Legacy `mcq_*` templates were selected without a key. Nothing was written. Rerun with `--offline`. |
| `Error generating LLM distractors: ...`, then `warning: N mcq_* questions use fixed generic distractors` | A key is set but the LLM calls failed. Those records have canned distractors, so the output is unusable. Regenerate with `--offline`, or fix the key. |
| `Points skipped for missing clips` equals all points, or splits are empty | `--video-root` does not match the relative clip paths in the annotations. Pass the directory they resolve against, or `--video-root ""` for annotation-only data. |
| Tracked files under `training_tennis_data_output/` changed | Legacy ran without `--output-dir`. Show `git diff --stat` to the user and restore only with their approval. |
| Counts differ from a previous categorical run | Expected, because of the unseeded sampling. Compare validator invariants, not counts. |
