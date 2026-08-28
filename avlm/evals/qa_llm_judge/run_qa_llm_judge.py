# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import yaml
import concurrent.futures
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Callable
import statistics
import threading
import argparse
import warnings

from openai import OpenAI

from avlm.evals.utils.prompt import get_default_system_prompt, get_user_prompt
from avlm.inference.common.utils.json_io import write_json

# ------------------------------
# Helpers
# ------------------------------

class APICallManager:
    """Thread-safe API call manager with simple sliding-window rate limiting and retries."""
    def __init__(self, 
                 client: OpenAI,
                 max_workers: int, 
                 api_rate_limit: float,
                 ):
        
        self.client = client

        self.max_workers = max_workers
        self.api_rate_limit = api_rate_limit
        self._api_call_times: List[float] = []
        self._api_call_lock = threading.Lock()
        # Stats
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retry_attempts = 0
        self.total_api_time = 0.0

    def _wait_for_rate_limit(self) -> float:
        """Return wait time to respect a simplistic per-second rate policy tied to max_workers."""
        current_time = time.time()
        with self._api_call_lock:
            cutoff_time = current_time - 1.0
            self._api_call_times = [t for t in self._api_call_times if t > cutoff_time]
            if len(self._api_call_times) >= self.max_workers:
                wait_time = self.api_rate_limit
            else:
                wait_time = 0.0
            self._api_call_times.append(current_time)
        return wait_time

    def rate_limited_call(
        self,
        api_func: Callable,
        *,
        max_retries: int,
        retry_delay: float,
        entry_id: str,
    ):
        last_exc = None
        for attempt in range(max_retries):
            wait_time = self._wait_for_rate_limit()
            if wait_time > 0:
                time.sleep(wait_time)
            start_time = time.time()
            try:
                with self._api_call_lock:
                    self.total_calls += 1
                result = api_func()
                with self._api_call_lock:
                    self.successful_calls += 1
                    self.total_api_time += time.time() - start_time
                return result
            except Exception as e:
                last_exc = e

                with self._api_call_lock:
                    self.failed_calls += 1
                    if attempt < max_retries - 1:
                        self.retry_attempts += 1
                
                # Only retry silently until the last attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Final attempt failed - print fatal message
                    with self._api_call_lock:
                        self.total_api_time += time.time() - start_time
                    error_type = type(e).__name__
                    print(f"\n{'='*60}")
                    print(f"[FATAL] All {max_retries} retry attempts exhausted for entry '{entry_id}'")
                    print(f"[FATAL] Error type: {error_type}")
                    print(f"[FATAL] Error message: {str(e)}")
                    print(f"{'='*60}\n")
        
        # Raise the exception with clear context after all retries exhausted
        error_msg = f"API call failed after {max_retries} attempts for entry '{entry_id}': {str(last_exc) if last_exc else 'Unknown error'}"
        raise RuntimeError(error_msg) from last_exc

    def get_stats(self) -> Dict[str, Any]:
        avg_time = (self.total_api_time / self.successful_calls) if self.successful_calls else 0.0
        success_rate = (self.successful_calls / max(self.total_calls, 1)) * 100.0
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retry_attempts": self.retry_attempts,
            "success_rate": success_rate,
            "total_api_time": self.total_api_time,
            "average_call_time": avg_time,
        }


def print_api_statistics(stats: Dict[str, Any], max_workers: int, api_rate_limit: float) -> None:
    print("\n" + "=" * 50)
    print("API CALL STATISTICS")
    print("=" * 50)
    print(f"Total API calls: {stats['total_calls']}")
    print(f"Successful calls: {stats['successful_calls']}")
    print(f"Failed calls: {stats['failed_calls']}")
    print(f"Retry attempts: {stats['retry_attempts']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Total API time: {stats['total_api_time']:.2f} seconds")
    print(f"Average call time: {stats['average_call_time']:.3f} seconds")
    if stats['total_calls'] > 0 and stats['average_call_time'] > 0:
        estimated_sequential = stats['total_calls'] * stats['average_call_time']
        speedup = estimated_sequential / max(stats['total_api_time'], 1e-9)
        print(f"Estimated speedup from threading: {speedup:.1f}x")
    print(f"Configured max_workers: {max_workers}")
    print(f"Configured api_rate_limit: {api_rate_limit}s")
    print("=" * 50)

def call_with_timeout(func: Callable[[], Any], timeout_seconds: float) -> Any:
    """Execute func() with a wall-clock timeout. Raises TimeoutError on expiry.
    Note: The underlying task cannot be forcefully killed; we just stop waiting and return control.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
        _future = _executor.submit(func)
        return _future.result(timeout=timeout_seconds)


def load_entries(input_file: str) -> List[Dict[str, Any]]:
    """Load entries from JSON or JSONL."""
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("entries", [data])
        raise ValueError("Input data must be a list or dictionary")
    except json.JSONDecodeError:
        print("Input is not valid JSON; attempting to parse as JSON Lines (JSONL)...")
        entries: List[Dict[str, Any]] = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSONL at line {line_num}: {e}") from e
        return entries


def compose_system_prompt(system_prompt_text: Optional[str]) -> str:
    """Compose the system prompt using a default plus optional inline text."""
    parts: List[str] = [get_default_system_prompt()]
    if system_prompt_text:
        parts.append(system_prompt_text.strip())
    return "\n\n".join([p for p in parts if p])


def normalize_score(raw_score: Any) -> int:
    """Normalize model score to 1-10 integer.
    - Accepts numeric or string
    - If > 10, interpret as 0-100 scale and convert
    - Clamp to [1, 10]
    """
    # Attempt to parse
    if isinstance(raw_score, str):
        try:
            raw_score = float(raw_score.strip())
        except Exception:
            raw_score = 0
    if isinstance(raw_score, bool):
        raw_score = int(raw_score)
    if not isinstance(raw_score, (int, float)):
        raw_score = 0

    # Convert if it looks like a 0-100 score
    if raw_score > 10:
        raw_score = raw_score / 10.0

    # Round and clamp
    score = int(round(raw_score))
    if score < 1:
        score = 1
    if score > 10:
        score = 10
    return score


def calculate_score_distribution(scores: List[int]) -> Dict[str, int]:
    """Bucket scores into intuitive 1-10 ranges."""
    return {
        "excellent_9_10": len([s for s in scores if s >= 9]),
        "good_8": len([s for s in scores if s == 8]),
        "fair_7": len([s for s in scores if s == 7]),
        "poor_6": len([s for s in scores if s == 6]),
        "very_poor_1_5": len([s for s in scores if 1 <= s <= 5]),
    }


def build_metadata_block(metadata: Any, max_chars: int, compact: bool) -> Optional[str]:
    """Render metadata to JSON string, optionally compacted and truncated. Returns None if empty."""
    if metadata is None or metadata == {} or metadata == []:
        return None
    try:
        if compact:
            rendered = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        else:
            rendered = json.dumps(metadata, indent=2, ensure_ascii=False)
    except Exception:
        rendered = str(metadata)
    if max_chars and len(rendered) > max_chars:
        truncated = rendered[:max_chars] + "\n... [truncated]"
        return truncated
    return rendered

def process_single_scoring_entry(
    entry: Dict[str, Any],
    model: str,
    system_prompt: Optional[str],
    metadata_mode: str,
    metadata_max_chars: int,
    metadata_compact: bool,
    api_manager: Optional[APICallManager],
    max_retries: int,
    retry_delay: float,
    request_timeout_seconds: float,
    per_entry_timeout_seconds: float,
    result_fields: Optional[List[str]] = None,
    user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single entry for scoring prediction against ground truth.
    
    Args:
        entry: Dictionary containing 'metadata', 'gt', and 'pred' keys
        model: GPT model to use for evaluation
        system_prompt: Optional system role content to guide evaluation
        metadata_mode: One of ['user','system','omit','auto'] determining where metadata goes
        metadata_max_chars: Maximum characters for metadata block
        metadata_compact: Whether to compact JSON (no pretty printing)
        api_manager: Shared API call manager for rate limiting and stats
        max_retries: Max retries per API call
        retry_delay: Delay between retries
        request_timeout_seconds: Request-level timeout for the model API call
        per_entry_timeout_seconds: Overall watchdog timeout per entry (encompassing retries)
        result_fields: Optional list of keys to include in the final result; if it contains
            'raw_response_text', that field will be included (when available)
        
    Returns:
        Dictionary containing scoring results
    """
    try:
        # required fields
        gt = entry.get('ground_truth')
        entry_id = entry.get('id')
        index = entry.get("index")
        pred = entry.get('prediction')
        conversation_json = entry.get('conversations')
        conversation = json.dumps(conversation_json)
        metadata_source = entry.get("metadata")
        inference_status = entry.get("inference_status")
        
        # Validate required fields

        if inference_status != "success":
            baseline = {
                "index": index,
                "id": entry_id,
                "error": f"Inference failed: {inference_status if inference_status else 'none'}",
                "score": 0,
                "gt": gt,
                "pred": pred,
                "conversation": conversation_json,
                "inference_status": inference_status
            }
            if result_fields:
                filtered: Dict[str, Any] = {}
                for k in result_fields:
                    filtered[k] = baseline.get(k)
                filtered["error"] = baseline["error"]
                return filtered
            return baseline
        
        if not gt:
            baseline = {
                "index": index,
                "id": entry_id,
                "error": "Missing ground truth (gt) field",
                "score": 0,
                "gt": gt,
                "pred": pred,
                "conversation": conversation_json,
                "inference_status": inference_status
            }
            if result_fields:
                filtered: Dict[str, Any] = {}
                for k in result_fields:
                    filtered[k] = baseline.get(k)
                filtered["error"] = baseline["error"]
                return filtered
            return baseline
        
        if not pred:
            baseline = {
                "index": index,
                "id": entry_id,
                "error": "Missing prediction (pred) field", 
                "score": 0,
                "gt": gt,
                "pred": pred,
                "conversation": conversation_json,
                "inference_status": inference_status
            }
            if result_fields:
                filtered: Dict[str, Any] = {}
                for k in result_fields:
                    filtered[k] = baseline.get(k)
                filtered["error"] = baseline["error"]
                return filtered
            return baseline
        
        # System prompt composition (fallback to a reasonable default)
        system_content = system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else get_default_system_prompt()

        # Determine where to place metadata
        selected_mode = metadata_mode
        metadata_block = build_metadata_block(metadata_source, metadata_max_chars, metadata_compact)
        if metadata_mode == "auto":
            if metadata_block and len(metadata_block) > 1500:
                selected_mode = "system"
            else:
                selected_mode = "user"

        system_content_with_meta = system_content
        user_metadata_block: Optional[str] = None
        if selected_mode == "system" and metadata_block:
            system_content_with_meta = f"{system_content}\n\n[Context]\n{metadata_block}"
        elif selected_mode == "user":
            user_metadata_block = metadata_block
        elif selected_mode == "omit":
            user_metadata_block = None
        
        # Create scoring prompt
        prompt = get_user_prompt(
            gt, conversation, pred, user_metadata_block, user_prompt
        )
        
        # Call GPT-4o for evaluation with rate limiting and timeouts
        def _chat_call():
            resp = api_manager.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content_with_meta},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.0,
                timeout=request_timeout_seconds,
            )

            return resp


        def _wrapped_call():
            if api_manager is not None:
                return api_manager.rate_limited_call(
                    _chat_call,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    entry_id=entry_id,
                )

            return _chat_call()

        # Retry up to 5 times on empty or unparseable API response (no retry on timeout)
        max_empty_or_parse_retries = 5
        empty_or_parse_retry_delay = 2.0
        last_response_content = ""
        last_parse_error: Optional[str] = None

        for attempt in range(max_empty_or_parse_retries):
            try:
                response = call_with_timeout(_wrapped_call, timeout_seconds=per_entry_timeout_seconds)
            except concurrent.futures.TimeoutError:
                baseline = {
                    "index": index,
                    "id": entry_id,
                    "error": f"Timeout after {per_entry_timeout_seconds}s",
                    "score": 0,
                    "gt": gt,
                    "pred": pred,
                    "conversation": conversation_json,
                    "inference_status": inference_status
                    }
                if result_fields:
                    filtered: Dict[str, Any] = {}
                    for k in result_fields:
                        filtered[k] = baseline.get(k)
                    filtered["error"] = baseline["error"]
                    return filtered
                return baseline

            raw_content = response.choices[0].message.content
            response_content = (raw_content or "").strip()
            last_response_content = response_content

            if not response_content:
                last_parse_error = (
                    "Empty response from scoring API (no content in choices[0].message.content). "
                    "Possible causes: transient API issue, rate limit, or content filter."
                )
                if attempt < max_empty_or_parse_retries - 1:
                    time.sleep(empty_or_parse_retry_delay)
                    continue
                break

            try:
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")
                json_str = response_content[start_idx:end_idx]
                result = json.loads(json_str)

                result_score = normalize_score(result.get("score", 0))
                result["score"] = result_score
                result.setdefault("reasoning", "")
                result.setdefault("strengths", [])
                result.setdefault("weaknesses", [])
                result.setdefault("metadata_alignment", "")
                result.update({
                    "index": index,
                    "id": entry_id,
                    "gt": gt,
                    "pred": pred,
                    "conversation": conversation_json,
                    "metadata": metadata_source,
                    "model_used": model,
                    "metadata_mode_used": selected_mode,
                    'class': entry.get('class')
                })

                if result_fields:
                    filtered_inner: Dict[str, Any] = {}
                    extracted_reasoning = ""
                    try:
                        js_start = response_content.find('{')
                        js_end = response_content.rfind('}') + 1
                        if js_start != -1 and js_end != 0:
                            resp_json = json.loads(response_content[js_start:js_end])
                            extracted_reasoning = resp_json.get("reasoning", "")
                    except Exception:
                        pass
                    for key in result_fields:
                        if key == "raw_response_text":
                            filtered_inner[key] = response_content
                        elif key == "reasoning":
                            filtered_inner[key] = extracted_reasoning or result.get("reasoning", "")
                        else:
                            filtered_inner[key] = result.get(key)
                    if "reasoning" not in filtered_inner or not filtered_inner["reasoning"]:
                        filtered_inner["reasoning"] = extracted_reasoning or result.get("reasoning", "")
                    return filtered_inner

                result["reasoning"] = result.get("reasoning", "")
                return result

            except (json.JSONDecodeError, ValueError) as e:
                last_parse_error = f"Failed to parse GPT response as JSON: {str(e)}"
                if attempt < max_empty_or_parse_retries - 1:
                    time.sleep(empty_or_parse_retry_delay)
                    continue
                break

        # All retries exhausted; report error once (stored in result, printed in summary)
        response_content = last_response_content
        error_payload = {
            "id": entry_id,
            "error": last_parse_error or "Empty or unparseable response after 5 retries",
            "score": 0,
            "gt": gt,
            "pred": pred
        }
        if result_fields:
            filtered_fail: Dict[str, Any] = {}
            for key in result_fields:
                if key == "raw_response_text":
                    filtered_fail[key] = response_content
                else:
                    filtered_fail[key] = error_payload.get(key)
            filtered_fail["error"] = error_payload["error"]
            return filtered_fail
        return error_payload
            
    except RuntimeError as e:
        # Re-raise RuntimeError from API failures to trigger program exit
        if "API call failed after" in str(e):
            raise
        # Other RuntimeErrors get converted to error results
        baseline = {
            "index": entry.get('index'),
            "id": entry.get('id'),
            "error": f"Error processing entry: {str(e)}",
            "score": 0,
            "gt": gt,
            "pred": pred,
            "conversation": conversation_json,
            "inference_status": inference_status
        }
        if result_fields:
            filtered: Dict[str, Any] = {}
            for k in result_fields:
                filtered[k] = baseline.get(k)
            filtered["error"] = baseline["error"]
            return filtered
        return baseline
    except Exception as e:
        baseline = {
            "index": entry.get('index'),
            "id": entry.get('id'),
            "error": f"Error processing entry: {str(e)}",
            "score": 0,
            "gt": gt,
            "pred": pred,
            "conversation": conversation_json,
            "inference_status": inference_status
        }
        if result_fields:
            filtered: Dict[str, Any] = {}
            for k in result_fields:
                filtered[k] = baseline.get(k)
            filtered["error"] = baseline["error"]
            return filtered
        return baseline


def process_entries_parallel_scoring(
    entries: List[Dict[str, Any]],
    model: str,
    max_workers: int,
    system_prompt: Optional[str],
    metadata_mode: str,
    metadata_max_chars: int,
    metadata_compact: bool,
    api_manager: Optional[APICallManager],
    max_retries: int,
    retry_delay: float,
    request_timeout_seconds: float,
    per_entry_timeout_seconds: float,
    result_fields: Optional[List[str]] = None,
    user_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process entries in parallel for scoring using ThreadPoolExecutor.
    
    Args:
        entries: List of dictionaries containing metadata, gt, and pred
        model: GPT model to use for evaluation
        max_workers: Maximum number of parallel workers
        system_prompt: Optional system role content to guide evaluation
        metadata_mode: Where to place metadata ['user','system','omit','auto']
        metadata_max_chars: Maximum characters for metadata block
        metadata_compact: Whether to compact JSON (no pretty printing)
        api_manager: Shared API call manager
        max_retries: Max retries per API call
        retry_delay: Delay between retries
        request_timeout_seconds: Request-level timeout for the model API call
        per_entry_timeout_seconds: Overall watchdog timeout per entry
        result_fields: Optional list of keys to include in final results. If it includes
            'raw_response_text', that field will be populated from the raw model text
            when available
        
    Returns:
        List of scoring results
    """
    results = []
    
    print(f"Processing {len(entries)} entries for scoring with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {}
        for entry in entries:
            entry_id = entry.get('id', entry.get('entry_id', f'entry_{len(future_to_id)}'))
            future = executor.submit(
                process_single_scoring_entry,
                entry,
                model,
                system_prompt,
                metadata_mode,
                metadata_max_chars,
                metadata_compact,
                api_manager,
                max_retries,
                retry_delay,
                request_timeout_seconds,
                per_entry_timeout_seconds,
                result_fields,
                user_prompt,
            )
            future_to_id[future] = entry_id
        
        # Process results with progress bar
        with tqdm(total=len(entries), desc="Scoring entries") as pbar:
            for future in concurrent.futures.as_completed(future_to_id):
                entry_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                except RuntimeError as e:
                    # RuntimeError from API call failures - exit immediately
                    print(f"\n[FATAL] Stopping execution due to API call failure")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(f"Program terminated: {str(e)}") from e
                except Exception as e:
                    # Other unexpected errors - also exit to be safe
                    print(f"\n[FATAL] Unexpected error processing entry {entry_id}: {str(e)}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(f"Program terminated due to unexpected error for entry {entry_id}: {str(e)}") from e
                
                pbar.update(1)
    
    return results


def categorize_error(error_message: str) -> str:
    """Map an error message to a short category label for reporting."""
    if not error_message:
        return "unknown"
    msg = (error_message or "").strip()
    if "Empty response from scoring API" in msg:
        return "empty_response"
    if "Timeout after" in msg:
        return "timeout"
    if "Failed to parse GPT response as JSON" in msg:
        return "parse_error"
    if "Empty or unparseable response after" in msg:
        return "empty_or_parse_after_retries"
    if "Missing ground truth" in msg:
        return "missing_gt"
    if "Missing prediction" in msg:
        return "missing_pred"
    if "Error processing entry" in msg:
        return "processing_error"
    if "API call failed after" in msg:
        return "api_failed"
    return "other"


def count_errors_by_category(failures: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return a dict of category -> count for the given failures list."""
    counts: Dict[str, int] = {}
    for f in failures:
        err = f.get("error", "")
        cat = categorize_error(err)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from scoring results.
    
    Args:
        results: List of scoring result dictionaries
        
    Returns:
        Dictionary containing statistical summary
    """
    explicit_errors = 0
    invalid_responses = 0
    successful_scores: List[int] = []
    
    for result in results:
        if "error" in result:
            explicit_errors += 1
        else:
            score = int(result.get("score", 0))
            if score > 0:  # Only count non-zero scores as successful
                successful_scores.append(score)
            else:
                # score=0 without explicit error = unparseable/invalid response
                invalid_responses += 1
    
    total_failures = explicit_errors + invalid_responses
    
    stats = {
        "total_entries": len(results),
        "successful_evaluations": len(successful_scores),
        "failed_evaluations": total_failures,
        "explicit_errors": explicit_errors,
        "invalid_responses": invalid_responses,
        "success_rate": len(successful_scores) / len(results) * 100 if results else 0
    }
    
    if successful_scores:
        stats.update({
            "mean_score": statistics.mean(successful_scores),
            "median_score": statistics.median(successful_scores),
            "std_dev": statistics.stdev(successful_scores) if len(successful_scores) > 1 else 0,
            "min_score": min(successful_scores),
            "max_score": max(successful_scores),
            "score_distribution": calculate_score_distribution(successful_scores)
        })
    else:
        stats.update({
            "mean_score": 0,
            "median_score": 0,
            "std_dev": 0,
            "min_score": 0,
            "max_score": 0,
            "score_distribution": {
                "excellent_9_10": 0,
                "good_8": 0,
                "fair_7": 0,
                "poor_6": 0,
                "very_poor_1_5": 0
            }
        })
    
    return stats


def load_config_file(config_path: Optional[str]) -> Dict[str, Any]:
    """Load config from YAML/JSON.
    Precedence:
    1) CLI --config
    2) Env var VLM_SCORER_CONFIG
    3) 'vlm_configs/default.yaml'
    The path is resolved from CWD first, then relative to this script's directory.
    """
    env_path = os.getenv("VLM_SCORER_CONFIG")
    chosen_path = (config_path or env_path or "vlm_configs/default.yaml")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: List[str] = [chosen_path]
    if not os.path.isabs(chosen_path):
        candidates.append(os.path.join(script_dir, chosen_path))

    resolved_path: Optional[str] = next((p for p in candidates if os.path.exists(p)), None)
    if not resolved_path:
        raise FileNotFoundError(f"Config file not found. Tried: {', '.join(candidates)}")

    print(f"Loading config from: {resolved_path}")

    with open(resolved_path, 'r', encoding='utf-8') as f:
        if resolved_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        if resolved_path.endswith('.json'):
            return json.load(f)
        # Default to YAML
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Scorer")
    parser.add_argument(
        "--inference-dir",
        type=str,
        required=True,
        help="Path to inference results directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config file (overrides VLM_SCORER_CONFIG and default.yaml)",
    )
    parser.add_argument(
        "--max-llm-judge-samples",
        type=int,
        default=None,
        metavar="N",
        help="Score only the first N samples (overrides config max_entries if set)",
    )
    args = parser.parse_args()
    return args.inference_dir, args.config, args.max_llm_judge_samples


def main():
    """Main function for VLM scoring program."""
    try:
        inference_dir, config_cli_path, max_llm_judge_samples = parse_args()

        pred_path = os.path.join(inference_dir, 'predictions.jsonl')
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f'Pred path does not exist: {pred_path}')

        output_dir = os.path.join(inference_dir, 'llm_judge')
        os.makedirs(output_dir, exist_ok=True)
        output_json_file = os.path.join(output_dir, 'llm_judge_predictions.json')
        
        # Load config from CLI/env/default
        cfg = load_config_file(config_path=config_cli_path)

        # Required
        base_url = cfg.get("base_url")
        model = cfg.get("model")
        if not base_url:
            raise ValueError("Missing base_url in config")
        if not model:
            raise ValueError("Missing model in config")

        # make openai client
        CLIENT_API_KEY = os.getenv("CLIENT_API_KEY")
        if not CLIENT_API_KEY:
            raise ValueError("No client api key set")

        client = OpenAI(
            api_key=CLIENT_API_KEY,
            base_url=base_url
        )

        # Optional with defaults
        max_workers = int(cfg.get("max_workers", 10))
        system_prompt_text = cfg.get("system_prompt")
        user_prompt_text = cfg.get("user_prompt")
        api_rate_limit = float(cfg.get("api_rate_limit", 0.01))
        max_retries = int(cfg.get("max_retries", 10))
        retry_delay = float(cfg.get("retry_delay", 0.1))
        stats_only = bool(cfg.get("stats_only", False))
        request_timeout_seconds = float(cfg.get("request_timeout_seconds", 60))
        per_entry_timeout_seconds = float(cfg.get("per_entry_timeout_seconds", 180))
        
        # New: drop failed results from output
        drop_failed = bool(cfg.get("drop_failed", False))

        # New: selective result fields
        result_fields_cfg = cfg.get("result_fields")
        result_fields: Optional[List[str]] = None
        if isinstance(result_fields_cfg, str):
            parts = [p.strip() for p in result_fields_cfg.split(',')]
            result_fields = [p for p in parts if p]
        elif isinstance(result_fields_cfg, list):
            result_fields = [str(p) for p in result_fields_cfg]

        # Metadata handling
        metadata_mode = cfg.get("metadata_mode", "auto")
        metadata_max_chars = int(cfg.get("metadata_max_chars", 0))
        metadata_compact = bool(cfg.get("metadata_compact", True))
          
        # Load entries # TODO harry have to do
        entries = load_entries(pred_path)
        open_ended_entries = []

        for entry in entries:
            evaluation_type = entry.get("evaluation_type")
            if evaluation_type is None:
                warnings.warn(
                    f"Missing evaluation_type for prediction: {entry.get('id')}"
                )
            elif evaluation_type == "open_ended":
                open_ended_entries.append(entry)

        entries = open_ended_entries

        total_loaded = len(entries)
        # Subset: CLI --limit overrides config max_entries
        max_entries = max_llm_judge_samples if max_llm_judge_samples is not None else cfg.get("max_llm_judge_samples")
        if max_entries is not None:
            n = int(max_entries)
            if n < 1:
                n = 1
            entries = entries[:n]
            print(f"Limited to first {len(entries)} of {total_loaded} entries")
        else:
            print(f"Found {len(entries)} entries to process")

        # Compose system prompt
        final_system_prompt = compose_system_prompt(system_prompt_text)
     
        # Create shared API manager
        api_manager = APICallManager(client=client, 
                                     max_workers=max_workers, 
                                     api_rate_limit=api_rate_limit)
     
        # Process entries
        results = process_entries_parallel_scoring(
            entries,
            model,
            max_workers,
            final_system_prompt,
            metadata_mode,
            metadata_max_chars,
            metadata_compact,
            api_manager,
            max_retries,
            retry_delay,
            request_timeout_seconds,
            per_entry_timeout_seconds,
            result_fields,
            user_prompt_text,
        )

        # Optionally drop failed results from output only (not from statistics)
        def _is_failed_result(r: Dict[str, Any]) -> bool:
            if not isinstance(r, dict):
                return False
            if "error" in r:
                return True
            score_val = r.get("score")
            try:
                if score_val is not None and int(score_val) == 0:
                    return True
            except Exception:
                pass
            return False

        results_for_output = [r for r in results if not _is_failed_result(r)] if drop_failed else results
        dropped_failed_count = (len(results) - len(results_for_output)) if drop_failed else 0
     
        # Calculate statistics
        stats = calculate_statistics(results)

        # Collect failures (entries with "error" key) for JSON and console
        failures = [{"id": r.get("id", "unknown"), "error": r.get("error", "")} for r in results if r.get("error")]
        error_counts_by_category = count_errors_by_category(failures)
     
        # Prepare output
        output_data = {
            "statistics": stats,
            "results": results_for_output if not stats_only else [],
            "failures": failures,
            "error_counts_by_category": error_counts_by_category,
            "metadata": {
                "inference_dir": inference_dir,
                "model_used": model,
                "max_workers": max_workers,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata_mode": metadata_mode,
                "metadata_max_chars": metadata_max_chars,
                "metadata_compact": metadata_compact,
                "api_rate_limit": api_rate_limit,
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "stats_only": stats_only,
                "drop_failed": drop_failed,
                "dropped_failed_count": dropped_failed_count,
            }
        }

        # Save results
        os.makedirs(os.path.dirname(output_json_file) if os.path.dirname(output_json_file) else ".", exist_ok=True)

        write_json(output_json_file, output_data)

        # Print summary
        print(f"\n{'='*50}")
        print("SCORING SUMMARY (1-10)")
        print(f"{'='*50}")
        print(f"Total entries processed: {stats['total_entries']}")
        print(f"Successful evaluations: {stats['successful_evaluations']}")
        print(f"Failed evaluations: {stats['failed_evaluations']}")
        print(f"  ├─ Explicit errors: {stats['explicit_errors']}")
        if error_counts_by_category:
            items = sorted(error_counts_by_category.items(), key=lambda x: -x[1])
            for i, (cat, count) in enumerate(items):
                branch = "└─" if i == len(items) - 1 else "├─"
                print(f"  │   {branch} {cat}: {count}")
        print(f"  └─ Invalid/unparseable responses: {stats['invalid_responses']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")

        if stats['successful_evaluations'] > 0:
            print(f"\nSCORE STATISTICS (1-10):")
            print(f"Mean score: {stats['mean_score']:.2f}")
            print(f"Median score: {stats['median_score']:.2f}")
            print(f"Standard deviation: {stats['std_dev']:.2f}")
            print(f"Score range: {stats['min_score']} - {stats['max_score']}")

            print(f"\nSCORE DISTRIBUTION:")
            dist = stats['score_distribution']
            total_entries = stats['total_entries']
            successful = stats['successful_evaluations']
            print(f"  (% of successful evaluations = count/{successful}, % of total entries = count/{total_entries})")
            print(f"Excellent (9-10): {dist['excellent_9_10']}  |  {dist['excellent_9_10']/successful*100:.1f}% of successful  |  {dist['excellent_9_10']/total_entries*100:.1f}% of total")
            print(f"Good (8):         {dist['good_8']}  |  {dist['good_8']/successful*100:.1f}% of successful  |  {dist['good_8']/total_entries*100:.1f}% of total")
            print(f"Fair (7):         {dist['fair_7']}  |  {dist['fair_7']/successful*100:.1f}% of successful  |  {dist['fair_7']/total_entries*100:.1f}% of total")
            print(f"Poor (6):         {dist['poor_6']}  |  {dist['poor_6']/successful*100:.1f}% of successful  |  {dist['poor_6']/total_entries*100:.1f}% of total")
            print(f"Very Poor (1-5):  {dist['very_poor_1_5']}  |  {dist['very_poor_1_5']/successful*100:.1f}% of successful  |  {dist['very_poor_1_5']/total_entries*100:.1f}% of total")

        # Print API stats
        api_stats = api_manager.get_stats()
        print_api_statistics(api_stats, max_workers, api_rate_limit)

        print(f"\nResults saved to: {output_json_file}")
        
    except SystemExit:
        # Re-raise SystemExit to allow proper program termination
        raise
    except Exception as e:
        print(f"\n[FATAL ERROR] Program terminated with error:")
        print(f"[FATAL ERROR] {type(e).__name__}: {str(e)}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main() 
