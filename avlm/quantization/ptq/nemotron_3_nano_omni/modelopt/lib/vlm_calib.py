# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sports video+text JSONL calibration for HF PTQ (match-diverse subset + forwards).

Goal: diverse match videos and user text (MCQ prompts, captions, etc.). JSONL PTQ forwards
use video frames only. Clip groups
use the first path segment under video_root (for example a match folder). Within each
match, rows dedupe on (video path, normalized user text); selection round-robins one
row per match per pass until num_samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PROCESSOR_METADATA_KEYS = ("num_patches", "num_tokens", "imgs_sizes")


def video_rel_from_row(row: dict) -> str | None:
    for turn in row.get("conversation", row.get("conversations", [])):
        for part in turn.get("content", []):
            if isinstance(part, dict) and part.get("type") in {"video", "video-sound", "video_sound"}:
                return part.get("path") or part.get("video") or part.get("video-sound")
    return None


def user_text_from_row(row: dict) -> str:
    for turn in row.get("conversation", row.get("conversations", [])):
        if turn.get("role") != "user":
            continue
        parts = []
        for part in turn.get("content", []):
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        text = "\n".join(p for p in parts if p).strip()
        if text:
            return text.replace("<video-sound>", "").replace("<video>", "").strip()
    raise ValueError(f"no user text in row id={row.get('id')}")


def clip_group_id_from_video_rel(rel: str) -> str:
    """Group id for diversity (e.g. tennis match folder under video_root)."""
    return rel.split("/")[0] if rel else "unknown"


def user_text_key_from_row(row: dict) -> str | None:
    try:
        text = user_text_from_row(row)
    except ValueError:
        return None
    text = " ".join(text.split()).casefold()
    return text or None


def row_clip_text_key(row: dict, rel: str) -> tuple[str, str] | None:
    text_key = user_text_key_from_row(row)
    if text_key is None:
        return None
    return (rel, text_key)


def iter_jsonl_rows(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_calib_rows(jsonl_path: Path, num_samples: int) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= num_samples:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_match_diverse_calib_jsonl(
    *,
    source_jsonls: list[Path],
    output_jsonl: Path,
    num_samples: int,
    video_root: Path,
    seed: int = 42,
    max_rows_per_match: int = 32,
    require_video_exists: bool = True,
) -> dict:
    """Build per-match pools, then round-robin across matches until num_samples."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    seen_in_match: dict[str, set[tuple[str, str]]] = {}
    skipped_duplicate = 0

    def try_add(group_id: str, rel: str, row: dict) -> None:
        nonlocal skipped_duplicate
        key = row_clip_text_key(row, rel)
        if key is None:
            skipped_duplicate += 1
            return
        seen = seen_in_match.setdefault(group_id, set())
        if key in seen:
            skipped_duplicate += 1
            return
        bucket = buckets.setdefault(group_id, [])
        if len(bucket) < max_rows_per_match:
            seen.add(key)
            bucket.append(row)
            return
        j = rng.randrange(len(bucket) + 1)
        if j >= max_rows_per_match:
            return
        old = bucket[j]
        old_rel = video_rel_from_row(old)
        if old_rel:
            old_key = row_clip_text_key(old, old_rel)
            if old_key is not None:
                seen.discard(old_key)
        seen.add(key)
        bucket[j] = row

    for row in iter_jsonl_rows(source_jsonls):
        rel = video_rel_from_row(row)
        if not rel:
            continue
        group_id = clip_group_id_from_video_rel(rel)
        try_add(group_id, rel, row)

    group_ids = list(buckets.keys())
    rng.shuffle(group_ids)
    selected: list[dict] = []
    round_idx = 0
    while len(selected) < num_samples:
        progressed = False
        for group_id in group_ids:
            bucket = buckets[group_id]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                progressed = True
                if len(selected) >= num_samples:
                    break
        if not progressed:
            break
        round_idx += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_missing = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row in selected:
            rel = video_rel_from_row(row)
            if require_video_exists and rel:
                if not (video_root / rel).is_file():
                    skipped_missing += 1
                    continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written >= num_samples:
                break

    stats = {
        "num_clip_groups_in_pool": len(group_ids),
        "num_samples_requested": num_samples,
        "num_samples_written": written,
        "skipped_duplicate_clip_or_text": skipped_duplicate,
        "skipped_missing_video": skipped_missing,
        "source_jsonls": [str(p) for p in source_jsonls],
        "output_jsonl": str(output_jsonl),
        "video_root": str(video_root),
        "seed": seed,
    }
    stats_path = output_jsonl.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote calib JSONL %s (%s)", output_jsonl, stats)
    return stats


def get_jsonl_vlm_forward_loop(
    model,
    processor,
    tokenizer,
    *,
    calib_jsonl: Path,
    video_root: Path,
    num_samples: int,
    video_fps: int = 2,
    max_video_frames: int = 128,
    include_video_metadata: bool = True,
) -> Callable[[], None]:
    from avlm.inference.common.utils.video_io import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    rows = load_calib_rows(calib_jsonl, num_samples)
    if len(rows) < num_samples:
        logger.warning("Calib JSONL has %d rows (requested %d)", len(rows), num_samples)

    device = model.device
    media_prefix = processor.video_token
    logger.info("JSONL calib: video only")

    calib_stats = {"completed": 0}

    def forward_loop() -> None:
        for row in rows:
            if calib_stats["completed"] >= num_samples:
                break
            rel = video_rel_from_row(row)
            if not rel:
                logger.warning("Skipping row without video: id=%s", row.get("id"))
                continue
            video_path = video_root / rel
            if not video_path.is_file():
                logger.warning("Missing video %s (id=%s)", video_path, row.get("id"))
                continue
            prompt_text = user_text_from_row(row)
            image_urls, video_metadata = maybe_path_or_url_to_data_urls(
                str(video_path),
                fps=max(0, int(video_fps)),
                nframe_max=int(max_video_frames),
            )
            frames = [pil_image_from_base64(u) for u in image_urls]
            messages = [{"role": "user", "content": f"{media_prefix}{prompt_text}"}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            processor_kwargs: dict = {
                "text": [prompt],
                "videos": frames,
                "return_tensors": "pt",
                "do_sample_frames": False,
            }
            if include_video_metadata:
                processor_kwargs["videos_kwargs"] = {"video_metadata": video_metadata}
            batch = processor(**processor_kwargs)
            for key in _PROCESSOR_METADATA_KEYS:
                batch.pop(key, None)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            calib_stats["completed"] += 1
            logger.info(
                "Running calibration forward %d (target %d JSONL VLM samples)",
                calib_stats["completed"],
                num_samples,
            )
            with torch.no_grad():
                model.generate(
                    **batch,
                    max_new_tokens=4,
                    do_sample=False,
                    output_hidden_states=False,
                    eos_token_id=tokenizer.eos_token_id,
                )

    forward_loop.calib_stats = calib_stats
    return forward_loop


_MODEOPT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_QUANT_PARAMS = _MODEOPT_DIR / "configs" / "quantization_hf_tennis.yaml"
_CFG_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _find_repo_root() -> Path:
    d = _MODEOPT_DIR
    for _ in range(16):
        if (d / "avlm" / "utils" / "_source_params.sh").is_file():
            return d
        if d.parent == d:
            break
        d = d.parent
    raise RuntimeError("Could not find repo root (avlm/utils/_source_params.sh)")


def _substitute_cfg_refs(value: Any, cfg: Mapping[str, Any]) -> Any:
    if isinstance(value, str):

        def _repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in cfg:
                return match.group(0)
            inner = cfg[key]
            return str(inner) if inner is not None else ""

        out = value
        for _ in range(16):
            new = _CFG_REF.sub(_repl, out)
            if new == out:
                return new
            out = new
        return out
    if isinstance(value, list):
        return [_substitute_cfg_refs(item, cfg) for item in value]
    if isinstance(value, dict):
        return {k: _substitute_cfg_refs(v, cfg) for k, v in value.items()}
    return value


def _load_quant_params_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required for --quant-params") from exc
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"quant params root must be a mapping: {path}")
    cfg: dict[str, Any] = dict(raw)
    for _ in range(16):
        expanded = _substitute_cfg_refs(cfg, cfg)
        if expanded == cfg:
            break
        cfg = expanded if isinstance(expanded, dict) else cfg
    return cfg


def _resolve_quant_path(path_str: str, *, repo_root: Path, cache_dir: Path) -> Path:
    if path_str.startswith("@repo/"):
        return repo_root / path_str.removeprefix("@repo/")
    p = Path(path_str)
    if p.is_absolute():
        return p
    return cache_dir / path_str


def _cli_defaults_from_quant_params(
    quant_params: Path,
) -> tuple[list[Path], Path, Path, int]:
    cfg = _load_quant_params_yaml(quant_params)
    repo_root = _find_repo_root()
    cache_dir = Path(str(cfg.get("CACHE_DIR") or repo_root / ".cache"))

    sources_raw = cfg.get("CALIB_JSONL_SOURCES")
    if not sources_raw:
        raise ValueError(f"{quant_params}: set CALIB_JSONL_SOURCES")
    if isinstance(sources_raw, str):
        sources_raw = sources_raw.split()
    sources = [Path(str(p)) for p in sources_raw]

    video_root = cfg.get("VIDEO_ROOT")
    calib_jsonl = cfg.get("CALIB_JSONL")
    if not video_root or not calib_jsonl:
        raise ValueError(f"{quant_params}: set VIDEO_ROOT and CALIB_JSONL")

    num_samples = int(cfg.get("NUM_SAMPLES") or 512)
    output_jsonl = _resolve_quant_path(str(calib_jsonl), repo_root=repo_root, cache_dir=cache_dir)
    return sources, Path(str(video_root)), output_jsonl, num_samples


def _cli_main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build a clip-group-diverse sports JSONL subset for ModelOpt PTQ calibration.",
    )
    parser.add_argument(
        "--quant-params",
        type=Path,
        nargs="?",
        const=_DEFAULT_QUANT_PARAMS,
        default=_DEFAULT_QUANT_PARAMS if _DEFAULT_QUANT_PARAMS.is_file() else None,
        help="PTQ recipe YAML (default: configs/quantization_hf_tennis.yaml). "
        "Uncomment the JSONL block there first. "
        "Uses CALIB_JSONL_SOURCES, VIDEO_ROOT, CALIB_JSONL, NUM_SAMPLES unless overridden.",
    )
    parser.add_argument("--sources", nargs="+", help="Input JSONL files (overrides YAML)")
    parser.add_argument("--output-jsonl", type=Path, help="Output path (overrides YAML)")
    parser.add_argument("--video-root", type=Path, help="Video root (overrides YAML)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-match", type=int, default=32)
    args = parser.parse_args(argv)

    sources: list[Path] | None = [Path(p) for p in args.sources] if args.sources else None
    output_jsonl = args.output_jsonl
    video_root = args.video_root
    num_samples = args.num_samples

    if args.quant_params is not None:
        if not args.quant_params.is_file():
            logger.error("Missing --quant-params file: %s", args.quant_params)
            return 1
        y_sources, y_video_root, y_output, y_num = _cli_defaults_from_quant_params(args.quant_params)
        sources = sources or y_sources
        video_root = video_root or y_video_root
        output_jsonl = output_jsonl or y_output
        if num_samples is None:
            num_samples = y_num
        logger.info("Using quant params from %s", args.quant_params)

    if not sources or video_root is None or output_jsonl is None:
        parser.error("Provide --quant-params or all of --sources, --video-root, --output-jsonl")
    num_samples = num_samples if num_samples is not None else 512

    for path in sources:
        if not path.is_file():
            logger.error("Missing source JSONL: %s", path)
            return 1

    build_match_diverse_calib_jsonl(
        source_jsonls=sources,
        output_jsonl=output_jsonl,
        num_samples=num_samples,
        video_root=video_root,
        seed=args.seed,
        max_rows_per_match=args.max_rows_per_match,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
