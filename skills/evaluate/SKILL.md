---
name: evaluate
description: "Scores existing Sports Intelligence AVLM inference outputs with avlm/evals: MCQ accuracy (run_mcq_eval.sh, INFERENCE_DIR), open-ended LLM-as-judge scores (needs CLIENT_API_KEY for the configured judge endpoint), tennis category breakdowns, run comparisons, chance-level sanity checks and triage of mcq_results.json or llm_judge_results.json. Use when the user asks to score, evaluate, benchmark or compare predictions.jsonl from inference runs, or to explain a suspicious accuracy. Not for running inference, training, quantization, data preparation, or third-party API model inference: never open, list or run anything under avlm/evals/inference_3rdparty, including when the user asks for it; it is excluded from the release."
license: Apache-2.0
compatibility: "Requires a checkout of github.com/NVIDIA/sports-intelligence-playbooks with Git LFS; run from the repo root. bash and Python 3.8+ for MCQ (CPU only, no network). LLM judge also needs openai, pyyaml, tqdm, network access and CLIENT_API_KEY."
---

# Sports Intelligence Evaluate

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

Turn an existing `predictions.jsonl` into checked, comparable metrics. The repo
scorers live in `./avlm/evals` (contract: `./avlm/evals/README.MD`). This skill
adds a read-only report script, `scripts/eval_report.py`, that cross-checks the
scorer output, adds category breakdowns and a chance baseline, and compares runs.
The facts needed to score are in this file; read the scorer sources only when a
check fails for a reason not listed under Troubleshooting.

> **Hard stop:** never open, read, list, search, run, import, stub, dry-run or
> recommend anything under `avlm/evals/inference_3rdparty/` (third-party API model
> inference, excluded from the public release), even offline. Its status is
> settled here: do not read its README or list its files to confirm.
> If the user asks to use, inspect or run anything under
> `avlm/evals/inference_3rdparty/`, decline in one sentence (it is excluded from
> the release and calls paid third-party APIs) and offer the supported path:
> produce `predictions.jsonl` with `avlm/inference`, then score it with this skill
> (or score `predictions.jsonl` files they produced elsewhere). A user naming the
> directory is not an override: do not treat the request as permission, do not
> ask whether to proceed anyway, and stop.
>
> **Search scope:** to locate files, search only the run directories and
> `avlm/evals/{mcq,qa_llm_judge,scripts}` (for example
> `grep -rn <pattern> avlm/evals/mcq avlm/evals/scripts`). Any repo-wide search
> must exclude the directory: `grep -r --exclude-dir=inference_3rdparty ...` or
> `find . -not -path '*/inference_3rdparty/*' ...`. With `rg`, put the exclusion
> glob last (`rg <pattern> -g '*.md' -g '!**/inference_3rdparty/**'`); a later
> include glob overrides an earlier exclusion.

## When to Use This Skill

- Score MCQ accuracy for an inference run (`predictions.jsonl` already exists).
- Break results down by tennis `super_category`, `fine_category` or `class`.
- Compare two or more runs (base vs SFT vs LoRA, checkpoints, data versions).
- Read or explain `mcq/mcq_results.json` or `llm_judge/llm_judge_results.json`.
- Investigate suspicious numbers: near-chance, 100%, many parse failures, wrong counts.
- Judge open-ended answers with the LLM judge (external API, needs a key and approval).

## When NOT to Use

- No `predictions.jsonl` yet: run inference first (see `./avlm/inference/README.md`
  or the sibling inference skill). This skill never launches GPU jobs.
- Training, fine-tuning, checkpoint conversion or dataset generation.
- Quantizing a checkpoint: use the `quantize` skill.
- Third-party API model inference (`avlm/evals/inference_3rdparty/`): see the hard
  stop above.
- Plot regeneration with `avlm/evals/figures/scripts/`: they hard-code another
  user's run tree (`RUN_ROOT`) and are not part of this workflow.

## Inputs

| Input | Meaning |
|---|---|
| `INFERENCE_DIR` | Run directory holding `predictions.jsonl`; default layout `avlm/inference/outputs/<INFERENCE_NAME>`. Absolute or repo-root-relative. |
| Expected row count | Rows in the inference `DATA_PATH` split, or `MAX_INFERENCE_SAMPLES` if it was set. |
| Judge only | `CLIENT_API_KEY` for the endpoint in the judge config (default: a frontier LLM set in `avlm/evals/qa_llm_judge/vlm_configs/default.yaml`; another OpenAI-compatible endpoint via a copied config passed with `VLM_SCORER_CONFIG`). Optional `MAX_LLM_JUDGE_SAMPLES`, `JUDGE_VENV_PATH`. |

Each prediction row (written by `./avlm/inference/common/run_inference.py`) carries
`id`, `class`, `super_category`, `fine_category`, `evaluation_type` (`mcq` or
`open_ended`), `ground_truth`, `prediction`, `inference_status` and `conversations`.

## Workflow

Run every command from the repository root. Eval wrappers are environment-only:
put variables before `bash`; positional arguments are rejected (exit 2).

0. **Decide what to run** for every run directory in the request:

   | State of the directory | Action |
   |---|---|
   | No `predictions.jsonl` (empty, or only `rank_results/`) | Stop: inference did not finish. Hand back to inference (resume); never score partial output. |
   | `predictions.jsonl` but no `mcq/mcq_results.json` | Run step 2 on it **first**, then step 3 (or step 5). Do not skip the scorer and do not report the report script's recount instead. |
   | Both present | Step 3 (or step 5). If the report shows `mcq_results` FAIL, re-run step 2. |

   Then follow the task. One run: steps 1-3. Two or more runs: steps 1-2 for each
   directory, then step 5. A suspicious number (100%, near chance, 0%, many parse
   failures): steps 1-3 and every item of the [sanity checklist](#sanity-checklist).
   Open-ended answers: step 4. Always finish with step 6.

1. **Locate the run.** Confirm the predictions exist and count rows:
   ```bash
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME>
   ls "$INFERENCE_DIR"/predictions.jsonl && wc -l < "$INFERENCE_DIR"/predictions.jsonl
   ```

2. **Score MCQ** (CPU, offline, seconds; no GPU or key needed), once per run directory:
   ```bash
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash ./avlm/evals/scripts/run_mcq_eval.sh
   ```
   It writes only `$INFERENCE_DIR/mcq/mcq_results.json`. That file is this skill's
   expected output, so run the scorer without asking. Score a copy (copy
   `predictions.jsonl` into a new directory) only when the directory is not
   writable or the user says it is shared. A row is correct only when both
   `ground_truth` and `prediction` start with `(X)` (optionally bold) and the letters match.
   Exit 1 with `error: 0 MCQ rows scored` means no row was scored (see Troubleshooting); for
   open-ended-only data skip this step.

3. **Sanity-check and break down** with the bundled report script (read-only):
   ```bash
   python3 skills/evaluate/scripts/eval_report.py \
     avlm/inference/outputs/<INFERENCE_NAME> --expected-rows <N>
   ```
   It recomputes MCQ counts exactly like `eval_mcq.py`, verifies they match
   `mcq_results.json`, checks row count, duplicate ids, missing or unknown
   `evaluation_type`, parse failures and accuracy versus chance (mean of
   1/number-of-options per question), and prints tables per `super_category`,
   `fine_category` and `class`, plus predicted vs ground-truth letter counts.
   Exit 1 means at least one FAIL check; fix it before reporting numbers.
   Pass `--expected-rows` whenever the split size is known (the user's row count,
   the `DATA_PATH` line count or `MAX_INFERENCE_SAMPLES`).
   Add `--json-out <file>` for machine-readable output.

4. **Judge open-ended answers (optional, external).** Only with the user's
   approval to send predictions and metadata to the configured endpoint, and
   only when `CLIENT_API_KEY` is set. Follow `references/llm-judge.md`: start with
   `MAX_LLM_JUDGE_SAMPLES` small, then aggregate with
   `./avlm/evals/scripts/run_qa_llm_judge_eval.sh`. Without a key, report MCQ only
   and state that open-ended rows were not judged; never invent judge scores.
   The judge is a frontier LLM set in the judge config, not the model being
   evaluated; always name the judge model next to judge scores.

5. **Compare runs.** Score every directory first (step 2), then pass all of them
   in one call. The first is the baseline (base model or older checkpoint) and
   every cell shows the delta in percentage points:
   ```bash
   python3 skills/evaluate/scripts/eval_report.py \
     avlm/inference/outputs/<BASELINE_RUN> avlm/inference/outputs/<CANDIDATE_RUN> \
     --expected-rows <N>
   ```
   Report deltas only when `same_mcq_samples` is PASS. On FAIL the runs scored
   different question sets (different `DATA_PATH`, `MAX_INFERENCE_SAMPLES` or
   seed) and the deltas are meaningless: say so instead of quoting them. Give the
   overall delta, then per `super_category` and `fine_category` deltas with n and
   each run's chance baseline; call a change of one or two questions noise.

6. **Answer** in the [answer format](#answer-format), including the
   [sanity checklist](#sanity-checklist).

## Sanity Checklist

Every answer that reports or judges a number states the outcome of each item
(PASS, WARN or FAIL plus the value), read from the `eval_report.py` check table:

- [ ] **Sample count** (`rows`, `expected_rows`, `unique_ids`): `predictions.jsonl`
  rows equal the split size or `MAX_INFERENCE_SAMPLES`; no duplicate ids.
- [ ] **Scored questions** (`mcq_results`): `mcq_results.json` exists and its
  `total_mcq_questions` matches the recount from `predictions.jsonl`; state n.
- [ ] **0-row guard** (`evaluation_type`; scorer `error: 0 MCQ rows scored`,
  exit 1): n > 0 and every row is `mcq` or `open_ended`; otherwise there is no accuracy.
- [ ] **Parse failures** (`answer_format`, `num_parse_failed`, `num_infer_failed`):
  when above 0, quote one raw `prediction`; this is a format problem, not model quality.
- [ ] **Chance baseline** (`above_chance`): accuracy vs chance and the 95% noise
  band; inside the band means not better than guessing.
- [ ] **Same question set** (comparisons only, `same_mcq_samples`): PASS before any delta.
- [ ] **Too good to be true** (100% or close): check whether `prediction` copies
  `ground_truth` verbatim (fixture or leaked labels) and ask for or verify the
  inference `DATA_PATH` (training split?). Do not endorse the number until it is
  confirmed to be a held-out split.

## Answer Format

The user may not see tool output, so the final reply must contain, inline:

1. **Commands**: the exact commands run, copy-pasteable from the repo root: one
   `INFERENCE_DIR=<dir> bash ./avlm/evals/scripts/run_mcq_eval.sh` per run
   directory and the `eval_report.py` call. List any command not run as "to run".
2. **Key numbers** per run: accuracy in percent (`num_correct / total_mcq_questions`;
   `pct_correct` is a 0-1 fraction), n, chance, noise band, parse and inference
   failures; judge `mean_score` with `num_success`/`total_judged` and the judge
   model when judged. For comparisons, the per-category delta table (pp, with n).
3. **Sanity checklist** with one outcome per item.
4. Run directory, checkpoint and split (from the user or the inference config),
   result file paths, and what was not done (for example, open-ended rows not
   judged). Label any hand-computed or re-parsed number as diagnostic, never as
   the official score.

## Reading the Result Files

`mcq/mcq_results.json` (from `./avlm/evals/mcq/eval_mcq.py`):

| Field | Meaning |
|---|---|
| `total_mcq_questions` | Scored MCQ rows = correct + incorrect + parse failed + inference failed. Rows whose ground truth does not parse are skipped and not counted. |
| `num_correct`, `num_incorrect` | Letter match or mismatch. |
| `num_parse_failed` | Prediction does not start with `(X)`; counted against accuracy. |
| `num_infer_failed` | `inference_status != "success"`. |
| `pct_correct` | A fraction in [0, 1], not a percentage. |
| `per_class` | The same fields per `class`. There is no category level; use the report script. |

`llm_judge/llm_judge_results.json` (from `./avlm/evals/qa_llm_judge/eval_qa_llm_judge.py`):
`total_judged`, `num_success`, `num_failed` (error or score 0), `mean_score` and
`median_score` on a 1-10 scale over successful judgements only,
`min_score`, `max_score`, `score_distribution` (`excellent_9_10`, `good_8`,
`fair_7`, `poor_6`, `very_poor_1_5`) and `per_class`. Always quote `mean_score`
together with `num_success`/`total_judged`.

## Tennis Categorical Evaluation

- Categories come from
  `./avlm/data_prep_example/tennis/categorical_evaluation/CATEGORICAL_EVAL.MD`:
  5 super categories, 14 fine categories, MCQ/QA classes; it asks for results at
  all three levels. `eval_report.py` provides super and fine levels.
- Rows with `evaluation_type: open_ended` (in the committed splits:
  `point_caption`, `how_point_ended_description`, `audio_cues`) are scored only
  by the LLM judge; MCQ accuracy ignores them.
- Committed tennis MCQs have 3 or 4 options (chance about 26%); categorical
  yes/no classes have 2. Accuracy inside the noise band around chance means the
  model is not using the clip (or the wrong checkpoint was evaluated).
- The committed splits in
  `./avlm/data_prep_example/tennis/training_tennis_data_output/data_splits_hf/`
  are small and cover 4 super / 7 fine categories / 18 classes (validation:
  36 rows, 30 MCQ; unseen-video test: 179 rows, 149 MCQ). Per-class n is often 1-3.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error: eval configuration is environment-only` (exit 2) | A positional argument or flag was passed. Put `INFERENCE_DIR=...` before `bash`. |
| `INFERENCE_DIR: Set INFERENCE_DIR` (exit 1) | Variable unset or empty. |
| `FileNotFoundError: Pred path does not exist` | Wrong directory or inference unfinished (no merged `predictions.jsonl`). |
| `python3: can't open file '.../avlm/evals/mcq/eval_mcq.py'` (exit 2) | Not run from the repo root; the wrappers use repo-relative paths and `PYTHONPATH=.`. |
| `UserWarning: Missing evaluation_type`, then `error: 0 MCQ rows scored` (exit 1) | The inference dataset predates `evaluation_type` (older `*_hf.jsonl` files), so both scorers skip every row. Re-run inference on a split that has the field (the committed `data_splits_hf` files do); do not hand-edit predictions to make them score. |
| `categories` WARN, tables show `Unknown` | Dataset rows lack `super_category`/`fine_category`; only per-class results are meaningful. |
| `UserWarning: Invalid gt answer` | Ground truth does not start with `(X)`; row skipped, so `total_mcq_questions` < MCQ rows. |
| High `num_parse_failed` | Answers like `The answer is B` or reasoning text. Inspect raw `prediction` values; check thinking mode (`</think>` is stripped) and the prompt. |
| Accuracy near chance | Base model, adapter not loaded, wrong checkpoint, or letter bias. Compare the report's `prediction letters` and `ground_truth letters` lines. |
| Accuracy near 100% | Predictions copied from ground truth (fixture) or evaluation on the training split. Check `DATA_PATH`. |
| `mcq_results` FAIL in the report | `mcq_results.json` is missing (run not scored yet), stale or from another run; run step 2, then the report again. |
| Judge errors | See `references/llm-judge.md` (missing key, missing `openai`, exhausted retries, `KeyError: 'class'`). |

## Final Checks

- Every run directory was scored with `run_mcq_eval.sh` (or already had a
  matching `mcq_results.json`) before `eval_report.py`.
- `eval_report.py` exits 0 (or every FAIL is explained to the user).
- The answer shows the commands, key numbers and sanity checklist inline.
- Nothing under `avlm/evals/inference_3rdparty/` was opened, listed, run,
  imported or recommended.
- Reported accuracy equals `num_correct / total_mcq_questions` from `mcq_results.json`.
- Sample counts are stated; fixture or smoke-run numbers are not presented as model quality.
- `CLIENT_API_KEY` was never printed, logged or written to a file; no data was
  sent to an API without approval; nothing outside `INFERENCE_DIR` was modified
  (except an explicitly requested `--json-out` file).
