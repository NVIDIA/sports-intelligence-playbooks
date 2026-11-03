# LLM judge for open-ended answers

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

The judge sends every `open_ended` row (question, ground truth, prediction and
point `metadata`) to an OpenAI-compatible chat endpoint and asks for a 1-10
score. It is an external action: get the user's approval for the data transfer
and the spend before running it. Source: `avlm/evals/README.MD`.

## Requirements

- Python packages `openai`, `pyyaml`, `tqdm` in base `python3` or in a virtual
  environment whose root is given by `JUDGE_VENV_PATH` (the wrapper then runs
  `$JUDGE_VENV_PATH/bin/python`).
- `CLIENT_API_KEY` exported in the environment, valid for the configured endpoint.
  Never print, echo, log or write it.
- Endpoint and model from `avlm/evals/qa_llm_judge/vlm_configs/default.yaml`
  (`base_url`, `model`). The default judge is a frontier LLM, not the model being
  evaluated. Use a copied YAML and `VLM_SCORER_CONFIG=<path>` to change them; do
  not edit the default in place. Report which judge model was used.
- Network access from the machine running the judge.

## Steps

Run from the repo root, after MCQ scoring.

1. Check readiness without revealing the key:
   ```bash
   test -n "${CLIENT_API_KEY:-}" && echo "CLIENT_API_KEY is set" || echo "CLIENT_API_KEY missing"
   ```
   If it is missing, stop here and use an offline option below.
2. Judge a small batch first (writes `llm_judge/llm_judge_predictions.json`):
   ```bash
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   MAX_LLM_JUDGE_SAMPLES=6 \
   bash ./avlm/evals/scripts/run_qa_llm_judge.sh
   ```
   Inspect the printed SCORING SUMMARY and the failure counts, then rerun
   without `MAX_LLM_JUDGE_SAMPLES` for the full set. Each run overwrites the file.
3. Aggregate (offline, no key; writes `llm_judge/llm_judge_results.json`):
   ```bash
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash ./avlm/evals/scripts/run_qa_llm_judge_eval.sh
   ```
4. Optional HTML failure-pattern summary (calls the API again; writes
   `llm_judge/llm_judge_summary.html`):
   ```bash
   INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
   bash ./avlm/evals/scripts/run_qa_llm_judge_summary.sh
   ```

## Offline alternatives when no key is available

- Report MCQ results only and say explicitly that open-ended rows were not judged.
- Step 3 needs no key: if `llm_judge_predictions.json` already exists (for
  example produced on another machine), aggregate it locally.
- Self-hosted judge: the client accepts any OpenAI-compatible `base_url`
  (for example a vLLM server on the cluster). Put that `base_url` and `model` in
  a copied config, pass it with `VLM_SCORER_CONFIG`, and set `CLIENT_API_KEY` to
  the server's key (the script requires a non-empty value). Scores from a
  different judge model are not comparable with scores from the default judge.

## Failure messages

| Message | Meaning and fix |
|---|---|
| `[FATAL ERROR] ValueError: No client api key set` (exit 1) | `CLIENT_API_KEY` unset or empty. Export it, or use an offline alternative. An empty `llm_judge/` directory is left behind. |
| `ModuleNotFoundError: No module named 'openai'` (or `tqdm`) | Install `openai pyyaml tqdm`, or set `JUDGE_VENV_PATH` to a venv that has them. |
| `[FATAL] All N retry attempts exhausted` then `Program terminated` | Endpoint unreachable, key lacks access to `model`, or rate limit. Check `base_url`/`model` in the config; do not switch judge models silently. |
| `FileNotFoundError: Config file not found` | `VLM_SCORER_CONFIG` path is wrong; relative paths resolve from the current directory, then from `avlm/evals/qa_llm_judge/`. |
| `FileNotFoundError: Pred path does not exist: .../llm_judge/llm_judge_predictions.json` | The eval or summary step ran before a successful judge step. |
| `KeyError: 'class'` in the eval step | The config's `result_fields` dropped `class`; restore the default fields and re-judge. |
| `total_judged: 0` although the judge ran | The config sets `stats_only: true`, so no per-row results were saved; re-judge with `stats_only: false`. |
| Many `failed_evaluations` / `num_failed` | Empty or unparseable judge responses or timeouts; see `error_counts_by_category` in `llm_judge_predictions.json`. |
