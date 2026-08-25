#!/usr/bin/env bash
set -euo pipefail

if (($#)); then
  echo "error: eval configuration is environment-only; set INFERENCE_DIR before the command" >&2
  exit 2
fi
: "${INFERENCE_DIR:?Set INFERENCE_DIR}"

PYTHONPATH="." python3 avlm/evals/qa_llm_judge/eval_qa_llm_judge.py --inference-dir "${INFERENCE_DIR}"
