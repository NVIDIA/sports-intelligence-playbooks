#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _score_int(raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _llm_judge_metrics(inference_dir: Path) -> dict[str, float]:
    pred_path = inference_dir / "llm_judge" / "llm_judge_predictions.json"
    data = _load_json(pred_path)
    results = data.get("results", [])

    scores: list[int] = []
    for row in results:
        if row.get("error"):
            continue
        score = _score_int(row.get("score"))
        if score > 0:
            scores.append(score)

    total = len(scores)
    if total == 0:
        return {
            "llm_judge/pct_ge_6": 0.0,
            "llm_judge/pct_ge_9": 0.0,
        }

    return {
        "llm_judge/pct_ge_6": sum(score >= 6 for score in scores) / total,
        "llm_judge/pct_ge_9": sum(score >= 9 for score in scores) / total,
    }


def _mcq_metrics(inference_dir: Path) -> dict[str, float]:
    results_path = inference_dir / "mcq" / "mcq_results.json"
    data = _load_json(results_path)
    return {"mcq/pct_correct": float(data.get("pct_correct", 0.0))}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _wandb_kwargs() -> dict[str, Any] | None:
    mode = os.environ.get("WANDB_MODE", "").strip()
    if mode == "disabled":
        return None

    project = os.environ.get("WANDB_PROJECT", "").strip()
    if not project:
        return None

    kwargs: dict[str, Any] = {"project": project}
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    name = os.environ.get("WANDB_NAME")
    run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if entity:
        kwargs["entity"] = entity
    if name:
        kwargs["name"] = name
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
    if mode:
        kwargs["mode"] = mode
    return kwargs


def _maybe_log_wandb(metrics: dict[str, float], step: int) -> None:
    kwargs = _wandb_kwargs()
    if kwargs is None:
        if os.environ.get("WANDB_MODE", "").strip() != "disabled":
            print("warning: skipped inference validation wandb logging: WANDB_PROJECT is unset", flush=True)
        return
    try:
        import wandb

        step_metric = "inference_validation/checkpoint_step"
        wandb_metrics = {
            "inference_validation/mcq_pct_correct": metrics["mcq/pct_correct"],
            "inference_validation/llm_judge_pct_ge_6": metrics["llm_judge/pct_ge_6"],
            "inference_validation/llm_judge_pct_ge_9": metrics["llm_judge/pct_ge_9"],
        }

        wandb.init(**kwargs)
        wandb.define_metric(step_metric)
        for key in wandb_metrics:
            wandb.define_metric(key, step_metric=step_metric)
        wandb.log({step_metric: step, **wandb_metrics})
        wandb.finish()
    except Exception as exc:  # noqa: BLE001 - metrics must not fail the eval pipeline.
        print(f"warning: failed to log inference validation metrics to wandb: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inference_dir = Path(args.inference_dir)
    metrics = {}
    metrics.update(_mcq_metrics(inference_dir))
    metrics.update(_llm_judge_metrics(inference_dir))

    row = {
        "step": args.step,
        "checkpoint_path": args.checkpoint_path,
        "inference_dir": str(inference_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    _append_jsonl(Path(args.output_jsonl), row)
    _maybe_log_wandb(metrics, args.step)
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
