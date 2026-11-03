#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-only barrier tests for vLLM admission and GPU running capacity.

Modes (``VLLM_QUEUE_TEST``):

* ``503`` (default) — burst ``max_num_queued_reqs + 2``; expect some HTTP 503.
* ``running_cap`` — burst ``data_parallel_size × max_num_seqs`` (e.g. 32); expect
  all HTTP 200 when ``max_num_queued_reqs`` is at least that large. vLLM load-balances
  across DP ranks; each rank runs up to ``max_num_seqs`` sequences at once (continuous
  batching). This script does not expose per-GPU batch counts from the API.
* ``both`` — ``running_cap`` then ``503`` (waits for the server to drain between).

  python3 avlm/deployment/vLLM/client/queue_admission_test.py
  VLLM_QUEUE_TEST=running_cap python3 avlm/deployment/vLLM/client/queue_admission_test.py

Override burst with ``VLLM_QUEUE_BURST``. For ``running_cap``, default ``max_tokens``
is 64 so work overlaps on GPU; override with ``VLLM_QUEUE_MAX_TOKENS``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _serve_config_path() -> Path:
    deploy = Path(__file__).resolve().parents[1]
    override = os.environ.get("SERVE_CONFIG", "").strip()
    if override:
        path = Path(override)
        return path if path.is_absolute() else _repo_root() / path
    return deploy / "configs" / "vllm_server_params.yaml"


def _load_serve_scalars(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if not body or ":" not in body:
            continue
        key, raw = body.split(":", 1)
        values[key.strip()] = raw.strip().strip("\"'")
    return values


def _base_url(cfg: dict[str, str]) -> str:
    raw = os.environ.get("VLLM_BASE_URL", "").strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    if raw:
        return raw
    port = os.environ.get("VLLM_PORT", "").strip() or cfg.get("vllm_port", "12500")
    return f"http://127.0.0.1:{port}"


def _check_health(base_url: str) -> None:
    url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            if resp.status != 200:
                raise SystemExit(f"error: health HTTP {resp.status} at {url}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"error: vLLM not healthy at {base_url} — run: bash avlm/deployment/vLLM/vllm_server.sh start\n{exc}"
        ) from exc


@dataclass
class ShotResult:
    index: int
    status: int
    latency_s: float
    error: str


def _text_payload(model: str, max_tokens: int, prompt: str) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }


def _post(base_url: str, payload: dict, timeout: float) -> tuple[int, float, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}"
        exc.read()
    except urllib.error.URLError as exc:
        error = str(exc.reason)
    return status, time.perf_counter() - started, error


def _run_burst(
    base_url: str,
    model: str,
    burst: int,
    max_tokens: int,
    timeout: float,
    prompt: str,
) -> list[ShotResult]:
    barrier = threading.Barrier(burst + 1)
    results: list[ShotResult | None] = [None] * burst
    lock = threading.Lock()

    def worker(i: int) -> None:
        barrier.wait()
        status, latency, error = _post(base_url, _text_payload(model, max_tokens, f"{prompt} #{i}"), timeout)
        with lock:
            results[i] = ShotResult(index=i, status=status, latency_s=latency, error=error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(burst)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


def _print_results(results: list[ShotResult]) -> tuple[int, int, int]:
    burst = len(results)
    ok = sum(1 for r in results if r.status == 200)
    rejected = sum(1 for r in results if r.status == 503)
    other = burst - ok - rejected
    print("Results")
    print(f"  HTTP 200: {ok}")
    print(f"  HTTP 503: {rejected}")
    if other:
        print(f"  other:    {other}")
        for r in results:
            if r.status not in (200, 503):
                print(f"    [{r.index}] {r.error or r.status}  {r.latency_s:.3f}s")
    return ok, rejected, other


def _run_mode(
    mode: str,
    base_url: str,
    cfg_path: Path,
    model: str,
    running_cap: int,
    max_queued: int,
    dp: int,
    max_num_seqs: int,
    timeout: float,
    prompt: str,
) -> int:
    burst_override = os.environ.get("VLLM_QUEUE_BURST", "").strip()
    if mode == "running_cap":
        burst = int(burst_override) if burst_override else running_cap
        default_tokens = "64"
    else:
        burst = int(burst_override) if burst_override else (max_queued + 2)
        default_tokens = "4"
    if burst < 1:
        raise SystemExit("error: VLLM_QUEUE_BURST must be at least 1")

    max_tokens = int(os.environ.get("VLLM_QUEUE_MAX_TOKENS", default_tokens))

    print(f"=== VLLM_QUEUE_TEST={mode} ===")
    print(f"Server:  {base_url}")
    print(f"Config:  {cfg_path}")
    print(f"Running: up to {running_cap} sequences on GPU ({dp} x max_num_seqs={max_num_seqs})")
    print(f"Limit:   max_num_queued_reqs={max_queued} (API admission, unfinished waiting+running)")
    print(f"Burst:   {burst} text requests released together (max_tokens={max_tokens})")
    if mode == "running_cap":
        print(
            "Expect:  all HTTP 200 — cluster can run up to "
            f"{running_cap} sequences ({max_num_seqs} per DP rank); LB assigns ranks internally."
        )
    print()

    results = _run_burst(base_url, model, burst, max_tokens, timeout, prompt)
    ok, rejected, other = _print_results(results)

    if mode == "running_cap":
        if ok == burst and rejected == 0 and other == 0:
            print()
            print(
                f"All {burst} requests completed — admission allows full GPU running cap "
                f"({running_cap}) with max_num_queued_reqs={max_queued}."
            )
            return 0
        if rejected:
            print()
            print(
                f"error: got HTTP 503 with burst={burst} but max_num_queued_reqs={max_queued}. "
                "Restart the server after YAML edits or raise max_num_queued_reqs."
            )
            return 1
        print()
        print("error: not all requests returned HTTP 200.")
        return 1

    # mode == "503"
    if rejected:
        print()
        print("503 rejections observed — max_num_queued_reqs admission control is active.")
        return 0
    print()
    print("No HTTP 503 in this burst. Try a larger VLLM_QUEUE_BURST, confirm the server")
    print("was started with max_num_queued_reqs, or grep server.log for 'Request queue full'.")
    return 1


def main() -> int:
    cfg_path = _serve_config_path()
    cfg = _load_serve_scalars(cfg_path)
    base_url = _base_url(cfg)
    _check_health(base_url)

    dp = int(cfg.get("data_parallel_size") or "1")
    max_num_seqs = int(cfg.get("max_num_seqs") or "4")
    running_cap = max(1, dp * max_num_seqs)
    max_queued = int(cfg.get("max_num_queued_reqs") or "8")

    test = os.environ.get("VLLM_QUEUE_TEST", "503").strip().lower() or "503"
    if test not in ("503", "running_cap", "both"):
        raise SystemExit("error: VLLM_QUEUE_TEST must be 503, running_cap, or both")

    model = os.environ.get("VLLM_CHAT_MODEL", "").strip() or cfg.get("served_model_name") or "nemotron_3_nano_omni"
    timeout = float(os.environ.get("VLLM_QUEUE_TIMEOUT", "120"))
    prompt = os.environ.get("VLLM_QUEUE_PROMPT", "Reply with one word: ok.")
    drain = float(os.environ.get("VLLM_QUEUE_DRAIN_SEC", "2"))

    if test == "both":
        code = _run_mode(
            "running_cap", base_url, cfg_path, model, running_cap, max_queued, dp, max_num_seqs, timeout, prompt
        )
        if code != 0:
            return code
        print()
        time.sleep(drain)
        print()
        return _run_mode("503", base_url, cfg_path, model, running_cap, max_queued, dp, max_num_seqs, timeout, prompt)

    return _run_mode(test, base_url, cfg_path, model, running_cap, max_queued, dp, max_num_seqs, timeout, prompt)


if __name__ == "__main__":
    sys.exit(main())
