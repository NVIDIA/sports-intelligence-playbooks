#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

judge_venv_path="${JUDGE_VENV_PATH:-}"
args=()

if (($#)); then
  echo "error: eval configuration is environment-only; use INFERENCE_DIR and VLM_SCORER_CONFIG" >&2
  exit 2
fi

if [[ -n "${VLM_SCORER_CONFIG:-}" ]]; then
  config="${VLM_SCORER_CONFIG}"
  if [[ "${config}" != /* && -n "${AVLM_BASE_REPO_ROOT:-}" ]]; then
    config="${AVLM_BASE_REPO_ROOT}/${config}"
  fi
  args+=(--config "${config}")
fi

: "${INFERENCE_DIR:?Set INFERENCE_DIR}"

python_cmd="python3"
if [[ -n "${judge_venv_path}" ]]; then
  python_cmd="${judge_venv_path}/bin/python"
fi

PYTHONPATH="." "$python_cmd" avlm/evals/qa_llm_judge/summarize_qa_llm_judge.py --inference-dir "${INFERENCE_DIR}" "${args[@]}"
