# AVLM eval guide

<img src="../../assets/Evaluation_Pipeline.png" alt="AVLM Evaluation Pipeline" width="100%">

The evaluation pipeline routes inference predictions by `evaluation_type`: MCQ
answers use deterministic letter matching, while open-ended answers use an LLM
judge. The resulting scores are aggregated into JSON metrics and an HTML report.

AVLM evals run on inference outputs. Inference must be run first, with predictions
written to:

```text
avlm/inference/outputs/<INFERENCE_NAME>/predictions.jsonl
```

Run all commands from the **repo root**.

Eval wrappers are environment-only. Set `INFERENCE_DIR`,
`VLM_SCORER_CONFIG`, `MAX_LLM_JUDGE_SAMPLES`, and `JUDGE_VENV_PATH` before the
command when needed. Positional inference directories and forwarded Python
flags are rejected.

---

## MCQ Eval

MCQ eval scores multiple-choice question classes from `predictions.jsonl` and writes
results under the inference directory.

```bash
INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
bash avlm/evals/scripts/run_mcq_eval.sh
```

Output:

```text
avlm/inference/outputs/<INFERENCE_NAME>/mcq/mcq_results.json
```

---

## LLM Judge Inference

LLM judge inference scores descriptive question classes from `predictions.jsonl`.

### Python Requirements

LLM judge requires:

```text
openai
pyyaml
tqdm
```

Install them in base Python or in a virtual environment:

```bash
pip install openai pyyaml tqdm
```

Set the API key before running LLM judge:

```bash
export CLIENT_API_KEY=<YOUR_CLIENT_API_KEY>
```

`CLIENT_API_KEY` is the API key used to call the LLM judge API client.

If using a virtual environment, set `JUDGE_VENV_PATH`. `JUDGE_VENV_PATH` is the path to the Python virtual environment root. If unset, the script uses
base `python3`. 

```bash
export JUDGE_VENV_PATH=<PATH_TO_JUDGE_VENV>
```

LLM judge loads its API endpoint, model, and runtime settings from
`avlm/evals/qa_llm_judge/vlm_configs/default.yaml` by default. Set `VLM_SCORER_CONFIG=path/to/config.yaml`
to use a different config.

Then run

```bash
INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
bash avlm/evals/scripts/run_qa_llm_judge.sh
```

With optional sample limit:

```bash
INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
MAX_LLM_JUDGE_SAMPLES=6 \
bash avlm/evals/scripts/run_qa_llm_judge.sh
```

Output:

```text
avlm/inference/outputs/<INFERENCE_NAME>/llm_judge/llm_judge_predictions.json
```

---

## LLM Judge Eval

LLM judge eval summarizes scores from `llm_judge_predictions.json`.

```bash
INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
bash avlm/evals/scripts/run_qa_llm_judge_eval.sh
```

Output:

```text
avlm/inference/outputs/<INFERENCE_NAME>/llm_judge/llm_judge_results.json
```

---

## LLM Judge Summary

LLM judge summary generates an HTML summary from `llm_judge_predictions.json`.

```bash
INFERENCE_DIR=avlm/inference/outputs/<INFERENCE_NAME> \
bash avlm/evals/scripts/run_qa_llm_judge_summary.sh
```

Output:

```text
avlm/inference/outputs/<INFERENCE_NAME>/llm_judge/llm_judge_summary.html
```
