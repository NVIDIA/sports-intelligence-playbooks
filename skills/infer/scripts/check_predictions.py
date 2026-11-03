#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that an inference output directory holds a complete, well-formed predictions.jsonl.

Prints a JSON summary to stdout. Exits 0 when the file is well formed, 1 when a
check fails, and 2 on usage errors.
"""

import argparse
import json
import sys
from pathlib import Path

# Fields written by avlm/inference/common/run_inference.py:to_record() that scoring relies on.
REQUIRED_KEYS = ("index", "id", "prediction", "ground_truth", "inference_status")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inference_dir", help="Inference output directory, e.g. avlm/inference/outputs/<INFERENCE_NAME>.")
    parser.add_argument(
        "--expected",
        type=int,
        default=None,
        help="Expected row count (MAX_INFERENCE_SAMPLES, or the dataset row count for a full run).",
    )
    args = parser.parse_args()

    out_dir = Path(args.inference_dir)
    path = out_dir / "predictions.jsonl"
    if not path.is_file():
        hint = ""
        if (out_dir / "rank_results").is_dir():
            hint = " (rank_results/ exists: the run did not finish; read the Slurm log, then rerun with RESUME=1)"
        print(f"error: {path} not found{hint}", file=sys.stderr)
        return 1

    errors, rows = [], []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        missing = [key for key in REQUIRED_KEYS if key not in rec]
        if missing:
            errors.append(f"line {line_no}: missing keys {missing}")
        rows.append(rec)

    if [rec.get("index") for rec in rows] != list(range(len(rows))):
        errors.append("'index' values are not 0..N-1 in order")
    if args.expected is not None and len(rows) != args.expected:
        errors.append(f"row count {len(rows)} != expected {args.expected}")

    json_path = out_dir / "predictions.json"
    if json_path.is_file():
        try:
            with open(json_path, encoding="utf-8") as f:
                parsed = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"predictions.json: invalid JSON ({exc.msg})")
        else:
            if not isinstance(parsed, list):
                errors.append("predictions.json is not a JSON array")
            elif len(parsed) != len(rows):
                errors.append("predictions.json and predictions.jsonl row counts differ")

    summary = {
        "predictions": str(path),
        "rows": len(rows),
        "empty_predictions": sum(1 for rec in rows if not str(rec.get("prediction") or "").strip()),
        "non_success_status": sum(1 for rec in rows if rec.get("inference_status") != "success"),
        "first": {"id": rows[0].get("id"), "prediction": str(rows[0].get("prediction"))[:120]} if rows else None,
        "errors": errors[:20],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
