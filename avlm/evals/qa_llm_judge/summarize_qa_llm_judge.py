import argparse
import concurrent.futures
import json
import os
import tempfile
import time
from typing import Any, Optional

from openai import OpenAI

from avlm.evals.qa_llm_judge.run_qa_llm_judge import load_config_file


SUMMARY_SYSTEM_PROMPT = (
    "You are an expert evaluator analyzing completed LLM-judge outputs for "
    "tennis VLM inference. Produce a concise, standalone HTML report of recurring "
    "model failure patterns and judge reliability issues."
)

CHUNK_SYSTEM_PROMPT = (
    "You are an expert evaluator analyzing one portion of completed LLM-judge "
    "outputs for tennis VLM inference. Produce concise plain-text findings for a "
    "later final report."
)

SUMMARY_RESULT_FIELDS = ("id", "class", "score", "gt", "pred", "reasoning")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize LLM judge predictions")
    parser.add_argument(
        "--inference-dir",
        type=str,
        required=True,
        help="Path to inference results directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file (overrides VLM_SCORER_CONFIG and default.yaml)",
    )
    return parser.parse_args()


def read_predictions_file(inference_dir: str) -> tuple[str, dict[str, Any]]:
    llm_judge_dir = os.path.join(inference_dir, "llm_judge")
    pred_path = os.path.join(llm_judge_dir, "llm_judge_predictions.json")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Pred path does not exist: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, dict):
        raise ValueError(f"Pred path must contain a JSON object: {pred_path}")

    results = predictions.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Pred path must contain a results list: {pred_path}")
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"Result at index {index} must be a JSON object")

    return pred_path, predictions


def build_summary_chunks(results: list[dict[str, Any]], max_chars: int) -> list[str]:
    if max_chars < 1:
        raise ValueError("summary_chunk_max_chars must be greater than 0")

    def sort_key(result: dict[str, Any]) -> tuple[str, int, str]:
        try:
            score = int(result.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        return str(result.get("class", "")), score, str(result.get("id", ""))

    chunks: list[str] = []
    current_items: list[str] = []
    current_chars = 2  # JSON list brackets

    for result in sorted(results, key=sort_key):
        selected = {field: result.get(field) for field in SUMMARY_RESULT_FIELDS}
        if "error" in result:
            selected["error"] = result["error"]
        item_text = json.dumps(selected, separators=(",", ":"), ensure_ascii=False)
        item_chars = len(item_text) + 2

        if item_chars > max_chars:
            raise ValueError(
                f"Selected result '{result.get('id', 'unknown')}' is {item_chars} "
                f"characters, exceeding summary_chunk_max_chars={max_chars}"
            )

        added_chars = len(item_text) + (1 if current_items else 0)
        if current_items and current_chars + added_chars > max_chars:
            chunks.append("[" + ",".join(current_items) + "]")
            current_items = []
            current_chars = 2
            added_chars = len(item_text)

        current_items.append(item_text)
        current_chars += added_chars

    if current_items:
        chunks.append("[" + ",".join(current_items) + "]")

    return chunks


def build_summary_prompt(summary_input_text: str) -> str:
    return f"""Analyze the complete LLM judge summary input below.

Every row present in the source results list was included. Only fields relevant to
this report were retained.
Use exact supplied statistics for numeric claims; do not invent or estimate global counts.
Focus on recurring findings across the judged predictions, such as:
- players being mixed up
- server/receiver or winner/loser attribution errors
- forehand/backhand or shot-type errors
- point-ending mismatches, including net vs out/long/wide
- serve sequence or rally-length issues
- cases where GT and pred appear close but the judge may have over-penalized
- cases where GT and pred genuinely do not match

Write a complete standalone HTML document only. Do not use Markdown or code fences.
Include <!DOCTYPE html>, <html>, <head>, embedded CSS, and <body>.
Keep it practical and cite representative entry ids when useful.

<llm_judge_summary_input>
{summary_input_text}
</llm_judge_summary_input>
"""


def build_chunk_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    return f"""Analyze result chunk {chunk_index + 1} of {total_chunks} below.

Do not write HTML or Markdown. Return concise plain text for a later final report.
Identify recurring model failure patterns and possible judge reliability issues.
Distinguish genuine GT/pred mismatches from cases the judge may have over- or
under-penalized. For each important pattern, describe its prevalence within this
chunk and cite a few representative entry ids. Do not claim global prevalence.

<llm_judge_result_chunk>
{chunk_text}
</llm_judge_result_chunk>
"""


def call_summary_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    request_timeout_seconds: float,
    max_retries: int,
    retry_delay: float,
    max_tokens: int,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=request_timeout_seconds,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Empty response from summary API")
            return content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Summary API call failed after {max_retries} attempts: {last_error}"
    ) from last_error


def write_summary_atomic(output_path: str, summary_text: str) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix="llm_judge_summary_", suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(summary_text)
            if not summary_text.endswith("\n"):
                f.write("\n")
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main():
    try:
        args = parse_args()
        inference_dir = args.inference_dir
        output_path = os.path.join(
            inference_dir, "llm_judge", "llm_judge_summary.html"
        )

        pred_path, predictions = read_predictions_file(inference_dir)
        cfg = load_config_file(config_path=args.config)

        base_url = cfg.get("base_url")
        model = cfg.get("model")
        if not base_url:
            raise ValueError("Missing base_url in config")
        if not model:
            raise ValueError("Missing model in config")

        client_api_key = os.getenv("CLIENT_API_KEY")
        if not client_api_key:
            raise ValueError("No client api key set")

        client = OpenAI(api_key=client_api_key, base_url=base_url)
        chunk_max_chars = int(cfg.get("summary_chunk_max_chars", 450000))
        summary_max_workers = int(cfg.get("summary_max_workers", 4))
        request_timeout_seconds = float(
            cfg.get("summary_request_timeout_seconds", 180)
        )
        max_retries = int(cfg.get("max_retries", 3))
        retry_delay = float(cfg.get("retry_delay", 5))

        if summary_max_workers < 1:
            raise ValueError("summary_max_workers must be greater than 0")

        results = predictions["results"]
        chunks = build_summary_chunks(results, chunk_max_chars)
        metadata = predictions.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        report_context = {
            "total_results": len(results),
            "selected_result_fields": list(SUMMARY_RESULT_FIELDS),
            "statistics": predictions.get("statistics", {}),
            "failures": predictions.get("failures", []),
            "error_counts_by_category": predictions.get(
                "error_counts_by_category", {}
            ),
            "metadata": {
                key: metadata[key]
                for key in ("inference_dir", "model_used", "processing_timestamp")
                if key in metadata
            },
        }

        max_chunk_chars = max((len(chunk) for chunk in chunks), default=0)
        print(
            f"Prepared {len(results)} results in {len(chunks)} chunk(s) "
            f"(largest: {max_chunk_chars} characters)"
        )

        if len(chunks) <= 1:
            summary_input_text = (
                '{"report_context":'
                + json.dumps(
                    report_context, separators=(",", ":"), ensure_ascii=False
                )
                + ',"results":'
                + (chunks[0] if chunks else "[]")
                + "}"
            )
        else:
            chunk_summaries = [""] * len(chunks)
            worker_count = min(summary_max_workers, len(chunks))
            print(f"Summarizing chunks with {worker_count} workers...")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count
            ) as executor:
                future_to_index = {
                    executor.submit(
                        call_summary_model,
                        client,
                        model,
                        CHUNK_SYSTEM_PROMPT,
                        build_chunk_prompt(chunk, index, len(chunks)),
                        request_timeout_seconds,
                        max_retries,
                        retry_delay,
                        1500,
                    ): index
                    for index, chunk in enumerate(chunks)
                }
                completed = 0
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    chunk_summaries[index] = future.result()
                    completed += 1
                    print(f"Summarized chunk {completed}/{len(chunks)}")

            summary_input_text = json.dumps(
                {
                    "report_context": report_context,
                    "chunk_summaries": chunk_summaries,
                },
                separators=(",", ":"),
                ensure_ascii=False,
            )

        summary_text = call_summary_model(
            client,
            model,
            SUMMARY_SYSTEM_PROMPT,
            build_summary_prompt(summary_input_text),
            request_timeout_seconds,
            max_retries,
            retry_delay,
            4000,
        )
        write_summary_atomic(output_path, summary_text)

        print(f"Summarized: {pred_path}")
        print(f"Summary saved to: {output_path}")
    except Exception as e:
        print("\n[FATAL ERROR] Program terminated with error:")
        print(f"[FATAL ERROR] {type(e).__name__}: {str(e)}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
