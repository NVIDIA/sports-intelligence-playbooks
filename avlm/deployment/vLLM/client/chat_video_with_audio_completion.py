#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One video (+ optional extracted audio) chat completion against a running vLLM server.

Same request shape as ``chat_video_completion.sh`` (greedy sampling, ``<video>`` in text,
``file://`` video URL). When the MP4 has an audio track, ffmpeg extracts 16 kHz mono WAV
and the payload adds ``audio_url`` with ``use_audio_in_video: false`` (AAC-in-MP4 safe).

  python3 avlm/deployment/vLLM/client/chat_video_with_audio_completion.py \\
    --video /path/to/point_with_audio.mp4 \\
    --question 'Describe what you hear and see in this tennis point.'

Uses the same env vars as the bash client where applicable: ``VLLM_CHAT_VIDEO``,
``VLLM_CHAT_MESSAGE``, ``VLLM_BASE_URL``, ``VLLM_PORT``, ``VLLM_CHAT_MODEL``,
``VLLM_CHAT_MAX_TOKENS``, ``VLLM_CHAT_TEMPERATURE``, ``VLLM_CHAT_TOP_P``,
``VLLM_CHAT_TOP_K``, ``VLLM_CHAT_REPETITION_PENALTY``, ``VLLM_CHAT_ENABLE_THINKING``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stress_video_completion import (  # noqa: E402
    _base_url,
    _check_health,
    _load_serve_scalars,
    _repo_root,
    _serve_config_path,
)


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ffmpeg_candidates() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in ("VLLM_CHAT_FFMPEG", "VLLM_BATCH_FFMPEG"):
        override = os.environ.get(key, "").strip()
        if override:
            ordered.append(override)
            break
    try:
        import imageio_ffmpeg

        ordered.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    path_ff = shutil.which("ffmpeg")
    if path_ff:
        ordered.append(path_ff)
    result: list[str] = []
    for exe in ordered:
        if exe and exe not in seen:
            seen.add(exe)
            result.append(exe)
    return result


def _video_has_audio_stream(video_path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return bool(proc.stdout.strip())
    candidates = _ffmpeg_candidates()
    ffmpeg = candidates[0] if candidates else shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    stderr = probe.stderr or ""
    return any(line.strip().startswith("Stream") and " Audio:" in line for line in stderr.splitlines())


def _audio_cache_root() -> Path:
    raw = os.environ.get("VLLM_CHAT_AUDIO_CACHE", "").strip()
    if raw:
        root = Path(raw).expanduser()
        return root if root.is_absolute() else _repo_root() / root
    return _repo_root() / ".cache" / "vllm_chat_audio"


def _wav_cache_path(video: Path, cache_root: Path, sampling_rate: int) -> Path:
    stat = video.stat()
    digest = hashlib.sha256(
        f"{video.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{sampling_rate}".encode()
    ).hexdigest()
    return cache_root / digest[:2] / f"{digest}.wav"


def _extract_wav(video: Path, sampling_rate: int = 16000) -> Path:
    cache_root = _audio_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    out = _wav_cache_path(video, cache_root, sampling_rate)
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.part{out.suffix}")
    candidates = _ffmpeg_candidates()
    if not candidates:
        raise SystemExit(
            "error: no ffmpeg found; install ffmpeg, pip install imageio-ffmpeg, "
            "or set VLLM_CHAT_FFMPEG"
        )
    errors: list[str] = []
    for ffmpeg in candidates:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-y",
                    "-i",
                    str(video),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sampling_rate),
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    str(tmp),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            tmp.unlink(missing_ok=True)
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            tail = stderr.strip()[-800:] if stderr else ""
            errors.append(f"{ffmpeg}: {tail}")
            continue
        tmp.replace(out)
        return out
    raise SystemExit(
        "error: ffmpeg audio extract failed:\n" + "\n---\n".join(errors)
    )


def _user_text(message: str, *, with_audio: bool) -> str:
    """Match chat_video_completion.sh for video-only; add audio tokens when WAV is sent."""
    video_token = os.environ.get("VLLM_VIDEO_TOKEN", "<video>")
    if not with_audio:
        return message if message.startswith(video_token) else f"{video_token}{message}"
    audio_token = os.environ.get("VLLM_AUDIO_TOKEN", "<so_embedding>")
    prefix = f"{video_token}\n{audio_token}\n"
    if message.startswith(video_token):
        rest = message[len(video_token) :].lstrip("\n")
        return prefix + rest if rest else prefix + message
    if message.startswith(prefix):
        return message
    return prefix + message


def _build_payload(
    video: Path,
    audio_wav: Path | None,
    message: str,
    *,
    model: str,
    max_tokens: int,
    enable_thinking: bool,
) -> dict:
    """Same fields as chat_video_completion.sh _vllm_build_payload, plus audio_url when set."""
    with_audio = audio_wav is not None
    user_text = _user_text(message, with_audio=with_audio)
    content: list[dict] = [
        {"type": "video_url", "video_url": {"url": video.as_uri()}},
    ]
    if audio_wav is not None:
        content.append({"type": "audio_url", "audio_url": {"url": audio_wav.as_uri()}})
    content.append({"type": "text", "text": user_text})
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "mm_processor_kwargs": {"use_audio_in_video": False},
        "temperature": float(os.environ.get("VLLM_CHAT_TEMPERATURE", "0")),
        "top_p": float(os.environ.get("VLLM_CHAT_TOP_P", "1")),
        "top_k": int(os.environ.get("VLLM_CHAT_TOP_K", "1")),
    }
    rep = os.environ.get("VLLM_CHAT_REPETITION_PENALTY", "").strip()
    if rep:
        payload["repetition_penalty"] = float(rep)
    return payload


def _completion_text(body: bytes) -> str:
    parsed = json.loads(body)
    if parsed.get("error"):
        raise SystemExit(f"error: {parsed['error']}")
    choices = parsed.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content or "")
    text = text.strip()
    if not text:
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            text = rc.strip()
    return text


def _enable_thinking(cfg: dict[str, str]) -> bool:
    raw = os.environ.get("VLLM_CHAT_ENABLE_THINKING", "").strip()
    if raw:
        return _truthy(raw)
    env_reason = os.environ.get("VLLM_ENABLE_REASONING", "").strip()
    if env_reason:
        return _truthy(env_reason)
    return _truthy(cfg.get("enable_reasoning", "false"))


def _default_max_tokens(cfg: dict[str, str]) -> int:
    for key in ("VLLM_CHAT_MAX_TOKENS", "VLLM_DEFAULT_MAX_TOKENS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return int(raw)
    raw = cfg.get("max_tokens", "").strip()
    if raw:
        return int(raw)
    return 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default=os.environ.get(
            "VLLM_CHAT_VIDEO", "assets/tennis_demo_video/full_tennis_point.mp4"
        ),
    )
    parser.add_argument(
        "--question",
        default=os.environ.get("VLLM_CHAT_MESSAGE", ""),
        help="Question text (same as VLLM_CHAT_MESSAGE for the bash client)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not extract audio (payload matches bash with use_audio_in_video=false)",
    )
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="Fail if the MP4 has no audio stream",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("VLLM_CHAT_TIMEOUT", "600")),
    )
    args = parser.parse_args()

    message = args.question.strip()
    if not message:
        raise SystemExit("error: pass --question or set VLLM_CHAT_MESSAGE")

    video = Path(args.video.strip())
    if not video.is_absolute():
        video = (_repo_root() / video).resolve()
    if not video.is_file():
        raise SystemExit(f"error: video not found: {video}")

    cfg = _load_serve_scalars(_serve_config_path())
    base = _base_url(cfg)
    _check_health(base, timeout=10.0)
    model = os.environ.get("VLLM_CHAT_MODEL", "").strip() or cfg.get(
        "served_model_name", "nemotron_3_nano_omni"
    )
    max_tokens = args.max_tokens if args.max_tokens is not None else _default_max_tokens(cfg)
    enable_thinking = _enable_thinking(cfg)

    audio_wav: Path | None = None
    if not args.no_audio:
        if not _video_has_audio_stream(video):
            if args.require_audio:
                raise SystemExit(f"error: no audio stream in {video}")
            print(
                f"[WARN] {video.name} has no audio track; video-only payload.",
                file=sys.stderr,
            )
        else:
            audio_wav = _extract_wav(video)
            print(f"[INFO] audio WAV: {audio_wav}", file=sys.stderr)

    payload = _build_payload(
        video,
        audio_wav,
        message,
        model=model,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = resp.read()
        if resp.status != 200:
            raise SystemExit(f"error: HTTP {resp.status}: {body[:500]!r}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"error: HTTP {exc.code}: {detail[:2000]}") from exc
    print(_completion_text(body))


if __name__ == "__main__":
    main()
