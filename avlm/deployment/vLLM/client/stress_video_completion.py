#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent video chat against a running vLLM server.

Repeats the questions and video from chat_video_completion.sh. Each level keeps
that many requests in flight, then records latency and throughput. Writes latency.svg, summary.csv, and requests.csv under .cache/vllm_server/stress/<timestamp>/
(or VLLM_STRESS_OUT). A level stays inside
the latency budget when its p95 is at most VLLM_STRESS_LATENCY_FACTOR times the
concurrency-1 p95 (default 2) and every request succeeds.

  python3 avlm/deployment/vLLM/client/stress_video_completion.py
  python3 avlm/deployment/vLLM/client/stress_video_completion.py --regen-figure .cache/vllm_server/stress/<timestamp>

Run it from the repo root, on the same machine as the server. ``--regen-figure`` rebuilds
latency.svg from an existing summary.csv. Override the sweep
with VLLM_STRESS_CONCURRENCY=1,2,4,8,16,32 and the sample size with
VLLM_STRESS_REQUESTS (0 = two passes of the question list, and at least one full
wave at that concurrency).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Same questions and token limits as chat_video_completion.sh.
_MCQ_MAX_TOKENS = 512
_CAPTION_MAX_TOKENS = 256

QUESTIONS: list[tuple[str, str, int]] = [
    (
        "MCQ: serving attempts",
        "How many serving attempts were made in this point?\n(A) 3\n(B) 4\n(C) 1\n(D) 2",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: server location",
        "Was the server on the near or far side of the visible court?\n(A) far-court\n(B) near-court",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: receiver location",
        "Was the receiver on the near or far side of the visible court?\n(A) near-court\n(B) far-court",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: score after the point",
        "Major vs. Mustermann:{6-4, 5-3, 30-15}. What was the score after this point?\n"
        "(A) Major vs. Mustermann:{6-4, 5-3, 40-30}\n"
        "(B) Major vs. Mustermann:{6-4, 5-3, 40-15}\n"
        "(C) Major vs. Mustermann:{6-4, 5-3, 30-30}\n"
        "(D) Major vs. Mustermann:{6-4, 5-3, 30-15}",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: losing player's lateral movement",
        "What was the losing player's lateral movement?\n(A) right to left\n(B) left to right\n(C) no lateral movement",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: losing player's depth movement",
        "How did the losing player move forward or backward?\n(A) backward\n(B) forward\n(C) no depth movement",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: server and winner",
        "Who served and who won this tennis point?\n"
        "(A) Mustermann served and Mustermann won\n"
        "(B) Major served and Mustermann won\n"
        "(C) Mustermann served and Major won\n"
        "(D) Major served and Major won",
        _MCQ_MAX_TOKENS,
    ),
    (
        "MCQ: winner's role",
        "Did the server or receiver win the point?\n(A) receiver\n(B) server",
        _MCQ_MAX_TOKENS,
    ),
    (
        "Detailed caption",
        'What happened in this point? Provide a detailed caption. (Major - "Major" is the player in the white sleeveless tennis dress with a pleated skirt. She is wearing white shoes and has dark hair tied back in a ponytail.; Mustermann - "Mustermann" is the player in the black sleeveless top and black pleated skirt with white piping along the sides and hem. She is wearing white shoes and has blonde hair tied back in a ponytail. She is holding a green and black tennis racket with a red grip.)',
        _CAPTION_MAX_TOKENS,
    ),
]


@dataclass
class RequestResult:
    concurrency: int
    index: int
    label: str
    ok: bool
    http_status: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str


@dataclass
class LevelSummary:
    concurrency: int
    n: int
    ok: int
    errors: int
    wall_s: float
    mean_s: float
    p50_s: float
    p95_s: float
    max_s: float
    req_per_s: float
    completion_tokens_per_s: float


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


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _concurrency_levels(slot_cap: int) -> list[int]:
    raw = os.environ.get("VLLM_STRESS_CONCURRENCY", "").strip()
    if raw:
        levels = [int(part) for part in raw.split(",") if part.strip()]
    else:
        levels = []
        level = 1
        while level < slot_cap:
            levels.append(level)
            level *= 2
        levels.append(slot_cap)
    levels = sorted({level for level in levels if level > 0})
    if not levels:
        raise SystemExit("error: VLLM_STRESS_CONCURRENCY needs at least one positive integer")
    return levels


def _requests_for_level(concurrency: int, requested: int) -> int:
    if requested > 0:
        return requested
    return max(len(QUESTIONS) * 2, concurrency)


def _video_path() -> Path:
    raw = os.environ.get("VLLM_CHAT_VIDEO", "assets/tennis_demo_video/full_tennis_point.mp4")
    path = Path(raw)
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.is_file():
        raise SystemExit(f"error: video not found: {path}")
    return path.resolve()


def _base_url(cfg: dict[str, str]) -> str:
    raw = os.environ.get("VLLM_BASE_URL", "").strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    if raw:
        return raw
    port = os.environ.get("VLLM_PORT", "").strip() or cfg.get("vllm_port", "12500")
    return f"http://127.0.0.1:{port}"


def _payload(video: Path, message: str, max_tokens: int, model: str) -> dict:
    video_token = os.environ.get("VLLM_VIDEO_TOKEN", "<video>")
    user_text = message if message.startswith(video_token) else f"{video_token}{message}"
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": float(os.environ.get("VLLM_CHAT_TEMPERATURE", "0")),
        "top_p": float(os.environ.get("VLLM_CHAT_TOP_P", "1")),
        "top_k": int(os.environ.get("VLLM_CHAT_TOP_K", "1")),
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video.as_uri()}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    }


def _check_health(base_url: str, timeout: float) -> None:
    url = f"{base_url}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                raise SystemExit(f"error: vLLM health HTTP {resp.status} at {url}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"error: vLLM not healthy at {base_url} — run: bash avlm/deployment/vLLM/vllm_server.sh start\n{exc}"
        ) from exc


def _one_request(
    base_url: str,
    payload: dict,
    timeout: float,
    concurrency: int,
    index: int,
    label: str,
) -> RequestResult:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status = 0
    body = b""
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read()
        error = f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        error = str(exc.reason)
    latency = time.perf_counter() - started
    prompt_tokens = 0
    completion_tokens = 0
    ok = status == 200 and not error
    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            usage = parsed.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            if not ok and not error:
                err_obj = parsed.get("error")
                error = json.dumps(err_obj) if err_obj else f"HTTP {status}"
    return RequestResult(
        concurrency=concurrency,
        index=index,
        label=label,
        ok=ok,
        http_status=status,
        latency_s=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=error,
    )


def _run_level(
    base_url: str,
    video: Path,
    model: str,
    timeout: float,
    concurrency: int,
    n_requests: int,
    print_lock: threading.Lock,
) -> tuple[list[RequestResult], float]:
    jobs = [QUESTIONS[i % len(QUESTIONS)] for i in range(n_requests)]
    results: list[RequestResult] = []
    done = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _one_request,
                base_url,
                _payload(video, message, max_tokens, model),
                timeout,
                concurrency,
                index,
                label,
            )
            for index, (label, message, max_tokens) in enumerate(jobs)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            status = "ok" if result.ok else f"FAIL {result.error}"
            with print_lock:
                print(
                    f"  [{done:3d}/{n_requests}] c={concurrency:<3d} {result.latency_s:7.3f}s  {status}  {result.label}",
                    flush=True,
                )
    wall = time.perf_counter() - started
    results.sort(key=lambda item: item.index)
    return results, wall


def _summarize(concurrency: int, results: list[RequestResult], wall_s: float) -> LevelSummary:
    ok_rows = [row for row in results if row.ok]
    latencies = [row.latency_s for row in ok_rows]
    completion_tokens = sum(row.completion_tokens for row in ok_rows)
    n_ok = len(ok_rows)
    return LevelSummary(
        concurrency=concurrency,
        n=len(results),
        ok=n_ok,
        errors=len(results) - n_ok,
        wall_s=wall_s,
        mean_s=(sum(latencies) / n_ok) if n_ok else float("nan"),
        p50_s=_percentile(latencies, 50),
        p95_s=_percentile(latencies, 95),
        max_s=max(latencies) if latencies else float("nan"),
        req_per_s=(n_ok / wall_s) if wall_s > 0 else 0.0,
        completion_tokens_per_s=(completion_tokens / wall_s) if wall_s > 0 else 0.0,
    )


def _fmt(value: float, digits: int = 3) -> str:
    if value != value:
        return "-"
    return f"{value:.{digits}f}"


def _print_table(rows: list[LevelSummary]) -> None:
    header = ("c", "ok", "mean", "p50", "p95", "max", "req/s", "out tok/s", "wall")
    body = [
        (
            str(row.concurrency),
            f"{row.ok}/{row.n}",
            _fmt(row.mean_s),
            _fmt(row.p50_s),
            _fmt(row.p95_s),
            _fmt(row.max_s),
            _fmt(row.req_per_s),
            _fmt(row.completion_tokens_per_s, 1),
            _fmt(row.wall_s, 1),
        )
        for row in rows
    ]
    widths = [max(len(header[col]), *(len(line[col]) for line in body)) for col in range(len(header))]
    print()
    print("Load balance sweep")
    print("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(header)))
    for line in body:
        print("  ".join(cell.ljust(widths[col]) for col, cell in enumerate(line)))


def _budget_concurrency(rows: list[LevelSummary], factor: float) -> tuple[LevelSummary | None, float]:
    baseline = next((row for row in rows if row.concurrency == 1 and row.ok), None)
    if baseline is None:
        baseline = next((row for row in rows if row.ok), None)
    if baseline is None or baseline.p95_s != baseline.p95_s:
        return None, float("nan")
    limit = baseline.p95_s * factor
    within = [row for row in rows if row.errors == 0 and row.ok == row.n and row.p95_s <= limit]
    if not within:
        return None, limit
    return max(within, key=lambda row: row.concurrency), limit


def _print_conclusion(rows: list[LevelSummary], factor: float, slot_cap: int) -> None:
    chosen, limit = _budget_concurrency(rows, factor)
    peak = max(rows, key=lambda row: row.req_per_s)
    baseline = next((row for row in rows if row.concurrency == 1 and row.ok), None)
    if baseline is None:
        baseline = next((row for row in rows if row.ok), None)
    print()
    if chosen is None or baseline is None:
        print(
            f"No concurrency kept p95 within {factor:g}x the single-request p95 "
            f"with zero errors. The in-flight slot cap from the serve config is {slot_cap}."
        )
    else:
        ratio = chosen.p95_s / baseline.p95_s if baseline.p95_s else float("nan")
        print(
            f"Latency stays within {factor:g}x the single-request p95 "
            f"({_fmt(limit)}s) up to {chosen.concurrency} requests in flight "
            f"(p95 {_fmt(chosen.p95_s)}s, {ratio:.2f}x, {chosen.req_per_s:.2f} req/s)."
        )
        print(
            f"The serve config allows {slot_cap} sequences in flight "
            f"(data_parallel_size x max_num_seqs)."
        )
    print(
        f"Peak throughput in this sweep is {peak.req_per_s:.2f} req/s "
        f"at concurrency {peak.concurrency} (p95 {_fmt(peak.p95_s)}s)."
    )
    print(
        "Each request sends the same video and cycles the same questions. "
        "If prefix caching is on, later copies of a prompt can be cheaper than the first."
    )


def _write_svg(path: Path, rows: list[LevelSummary], limit_s: float) -> None:
    width, height = 980, 460
    panel_w, panel_h = 420, 300
    left_x, right_x, top = 70, 530, 70

    def ticks(max_value: float, count: int = 4) -> list[float]:
        if max_value <= 0:
            return [0.0]
        return [max_value * i / count for i in range(count + 1)]

    latency_max = max([row.p95_s for row in rows if row.p95_s == row.p95_s] + [limit_s or 0.0, 1.0])
    latency_max *= 1.15
    throughput_max = max((row.req_per_s for row in rows), default=1.0) * 1.15 or 1.0
    n = len(rows)
    gap = panel_w / max(n, 1)

    def x_at(panel: int, index: int) -> float:
        origin = left_x if panel == 0 else right_x
        return origin + gap * (index + 0.5)

    def y_latency(value: float) -> float:
        return top + panel_h - (value / latency_max) * panel_h

    def y_throughput(value: float) -> float:
        return top + panel_h - (value / throughput_max) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="36" font-family="sans-serif" font-size="20" fill="#1a1a1a">vLLM load-balance sweep</text>',
    ]
    for panel, (title, y_of, _series) in enumerate(
        (
            ("Latency (seconds)", y_latency, None),
            ("Throughput (requests/s)", y_throughput, None),
        )
    ):
        origin = left_x if panel == 0 else right_x
        parts.append(
            f'<text x="{origin}" y="{top - 16}" font-family="sans-serif" font-size="14" fill="#1a1a1a">{title}</text>'
        )
        parts.append(
            f'<line x1="{origin}" y1="{top}" x2="{origin}" y2="{top + panel_h}" stroke="#cccccc"/>'
        )
        parts.append(
            f'<line x1="{origin}" y1="{top + panel_h}" x2="{origin + panel_w}" y2="{top + panel_h}" stroke="#cccccc"/>'
        )
        scale_max = latency_max if panel == 0 else throughput_max
        for tick in ticks(scale_max):
            y = (y_latency if panel == 0 else y_throughput)(tick)
            parts.append(
                f'<line x1="{origin - 4}" y1="{y:.1f}" x2="{origin + panel_w}" y2="{y:.1f}" stroke="#eeeeee"/>'
            )
            parts.append(
                f'<text x="{origin - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#666666">{tick:.2f}</text>'
            )
        for index, row in enumerate(rows):
            x = x_at(panel, index)
            parts.append(
                f'<text x="{x:.1f}" y="{top + panel_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#333333">{row.concurrency}</text>'
            )
        parts.append(
            f'<text x="{origin + panel_w / 2}" y="{top + panel_h + 40}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#666666">requests in flight</text>'
        )

    if limit_s == limit_s:
        y = y_latency(min(limit_s, latency_max))
        parts.append(
            f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x + panel_w}" y2="{y:.1f}" stroke="#c0392b" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{left_x + panel_w - 4}" y="{y - 6:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#c0392b">p95 budget</text>'
        )

    def polyline(panel: int, color: str, values: list[float], y_of) -> None:
        points: list[str] = []
        for index, value in enumerate(values):
            if value != value:
                continue
            point = f"{x_at(panel, index):.1f},{y_of(value):.1f}"
            points.append(point)
            parts.append(f'<circle cx="{x_at(panel, index):.1f}" cy="{y_of(value):.1f}" r="3.5" fill="{color}"/>')
        if len(points) >= 2:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>'
            )

    polyline(0, "#1f6feb", [row.p50_s for row in rows], y_latency)
    polyline(0, "#d97706", [row.p95_s for row in rows], y_latency)
    bar_w = min(28, gap * 0.45)
    for index, row in enumerate(rows):
        x = x_at(1, index) - bar_w / 2
        y = y_throughput(row.req_per_s)
        h = top + panel_h - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#1a7f4b"/>')

    legend_y = height - 28
    parts.append('<circle cx="70" cy="' + str(legend_y) + '" r="5" fill="#1f6feb"/>')
    parts.append(f'<text x="82" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#333333">p50</text>')
    parts.append('<circle cx="140" cy="' + str(legend_y) + '" r="5" fill="#d97706"/>')
    parts.append(f'<text x="152" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#333333">p95</text>')
    parts.append(f'<rect x="210" y="{legend_y - 6}" width="12" height="12" fill="#1a7f4b"/>')
    parts.append(f'<text x="228" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#333333">throughput</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_csv(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "concurrency",
                "index",
                "label",
                "ok",
                "http_status",
                "latency_s",
                "prompt_tokens",
                "completion_tokens",
                "error",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.concurrency,
                    row.index,
                    row.label,
                    int(row.ok),
                    row.http_status,
                    f"{row.latency_s:.6f}",
                    row.prompt_tokens,
                    row.completion_tokens,
                    row.error,
                ]
            )


def _write_summary_csv(path: Path, rows: list[LevelSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "concurrency",
                "n",
                "ok",
                "errors",
                "wall_s",
                "mean_s",
                "p50_s",
                "p95_s",
                "max_s",
                "req_per_s",
                "completion_tokens_per_s",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.concurrency,
                    row.n,
                    row.ok,
                    row.errors,
                    f"{row.wall_s:.3f}",
                    _fmt(row.mean_s),
                    _fmt(row.p50_s),
                    _fmt(row.p95_s),
                    _fmt(row.max_s),
                    _fmt(row.req_per_s),
                    _fmt(row.completion_tokens_per_s, 3),
                ]
            )


def _load_summary_csv(path: Path) -> list[LevelSummary]:
    rows: list[LevelSummary] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for line in csv.DictReader(fh):
            rows.append(
                LevelSummary(
                    concurrency=int(line["concurrency"]),
                    n=int(line["n"]),
                    ok=int(line["ok"]),
                    errors=int(line["errors"]),
                    wall_s=float(line["wall_s"]),
                    mean_s=float(line["mean_s"]),
                    p50_s=float(line["p50_s"]),
                    p95_s=float(line["p95_s"]),
                    max_s=float(line["max_s"]),
                    req_per_s=float(line["req_per_s"]),
                    completion_tokens_per_s=float(line["completion_tokens_per_s"]),
                )
            )
    return rows


def _regen_figure(stress_dir: Path, factor: float) -> int:
    stress_dir = stress_dir.expanduser().resolve()
    summary_path = stress_dir / "summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"error: missing {summary_path}")
    summaries = _load_summary_csv(summary_path)
    if not summaries:
        raise SystemExit(f"error: no rows in {summary_path}")
    _, limit = _budget_concurrency(summaries, factor)
    svg_path = stress_dir / "latency.svg"
    _write_svg(svg_path, summaries, limit)
    print(f"Figure:  {svg_path}")
    return 0


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--regen-figure":
        factor = _float_env("VLLM_STRESS_LATENCY_FACTOR", 2.0)
        return _regen_figure(Path(sys.argv[2]), factor)

    cfg_path = _serve_config_path()
    cfg = _load_serve_scalars(cfg_path)
    dp = int(cfg.get("data_parallel_size") or "1")
    max_num_seqs = int(cfg.get("max_num_seqs") or "4")
    max_queued = int(cfg.get("max_num_queued_reqs") or "0")
    slot_cap = max(1, dp * max_num_seqs)
    levels = _concurrency_levels(slot_cap)
    requested = _int_env("VLLM_STRESS_REQUESTS", 0)
    factor = _float_env("VLLM_STRESS_LATENCY_FACTOR", 2.0)
    timeout = _float_env("VLLM_STRESS_TIMEOUT", 600.0)
    video = _video_path()
    base_url = _base_url(cfg)
    model = os.environ.get("VLLM_CHAT_MODEL", "").strip() or cfg.get("served_model_name") or "nemotron_3_nano_omni"
    _check_health(base_url, timeout=30)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(os.environ.get("VLLM_STRESS_OUT", "")).expanduser() if os.environ.get("VLLM_STRESS_OUT") else (
        _repo_root() / ".cache" / "vllm_server" / "stress" / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Server: {base_url}  model={model}")
    print(f"Config: {cfg_path}")
    print(f"Video:  {video}")
    queued_info = f"  admission cap {max_queued}" if max_queued > 0 else ""
    print(
        f"Sweep:  concurrency {','.join(str(level) for level in levels)}  "
        f"latency budget {factor:g}x single-request p95  "
        f"running cap {slot_cap} ({dp} copies x {max_num_seqs} seqs){queued_info}"
    )

    print_lock = threading.Lock()
    print("Warmup: 1 request (not included in the table)")
    warmup_label, warmup_message, warmup_tokens = QUESTIONS[0]
    warmup = _one_request(
        base_url,
        _payload(video, warmup_message, warmup_tokens, model),
        timeout,
        1,
        0,
        warmup_label,
    )
    warmup_state = "ok" if warmup.ok else f"FAIL {warmup.error}"
    print(f"  warmup {warmup.latency_s:.3f}s  {warmup_state}")
    if not warmup.ok:
        return 1

    all_results: list[RequestResult] = []
    summaries: list[LevelSummary] = []
    for concurrency in levels:
        n_requests = _requests_for_level(concurrency, requested)
        print()
        print(f"Concurrency {concurrency}: {n_requests} requests, {concurrency} in flight")
        results, wall = _run_level(base_url, video, model, timeout, concurrency, n_requests, print_lock)
        summary = _summarize(concurrency, results, wall)
        all_results.extend(results)
        summaries.append(summary)
        print(
            f"  level done  ok={summary.ok}/{summary.n}  p50={_fmt(summary.p50_s)}s  "
            f"p95={_fmt(summary.p95_s)}s  {summary.req_per_s:.2f} req/s  wall={summary.wall_s:.1f}s"
        )

    _print_table(summaries)
    _print_conclusion(summaries, factor, slot_cap)

    _write_csv(out_dir / "requests.csv", all_results)
    _write_summary_csv(out_dir / "summary.csv", summaries)
    _, limit = _budget_concurrency(summaries, factor)
    svg_path = out_dir / "latency.svg"
    _write_svg(svg_path, summaries, limit)
    print()
    print(f"Figure:  {svg_path}")
    print(f"CSV:     {out_dir / 'summary.csv'}")
    print(f"Rows:    {out_dir / 'requests.csv'}")
    return 0 if any(row.ok for row in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
