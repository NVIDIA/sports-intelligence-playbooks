#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize one AutoModel or Megatron-Bridge training log as a JSON row.

Reads the per-job ``*.log`` written by the Slurm launchers, checks that every
logged loss is finite and that the run reached its last step, and reports step
time and token throughput under the field names used by the published
performance dashboard:
https://nvidia.github.io/sports-intelligence-playbooks/latest/multimodal/perf_dash/avlm_dashboard.html

Exit status: 0 = run passed, 1 = no steps found, a non-finite loss, or the run
stopped before its last step, 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
# AutoModel: "<ts> | INFO | root | step 3 | epoch 0 | loss 0.93 | ... | tps 31970.36(499.54/gpu) | ..."
AM_STEP = re.compile(TS + r" \| \w+ \| [\w.]+ \| step (\d+) \| epoch \d+ \| loss (\S+)")
AM_TPS = re.compile(r"\| tps ([\d.]+)\(([\d.]+)/gpu\)")
AM_GBS = re.compile(r"global_batch_size=(\d+) \(DP=\d+, (\d+) GPU ranks\)")
AM_TOTAL = re.compile(r"Training:\s+\d+%\|[^|]*\|\s*\d+/(\d+)")
# Megatron-Bridge: "[<ts>] iteration 5/ 10 | ... | elapsed time per iteration (ms): 1234.5 | ..."
MB_ITER = re.compile(r"\[" + TS + r"\] iteration\s+(\d+)/\s*(\d+) \|([^\n]*)")
MB_MS = re.compile(r"elapsed time per iteration \(ms\): ([\d.]+)")
MB_GBS = re.compile(r"global batch size:\s+(\d+)")
MB_LOSS = re.compile(r"lm loss: (\S+)")
MB_NAN = re.compile(r"number of nan iterations:\s+(\d+)")


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def parse_automodel(text: str) -> dict:
    steps: dict[int, dict] = {}
    for line in text.splitlines():
        starts = list(AM_STEP.finditer(line))
        for i, m in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(line)
            tps = AM_TPS.search(line, m.end(), end)
            steps[int(m.group(2))] = {
                "ts": datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                "loss": _float(m.group(3)),
                "tps": float(tps.group(1)) if tps else None,
                "tps_gpu": float(tps.group(2)) if tps else None,
            }
    gbs = AM_GBS.search(text)
    totals = AM_TOTAL.findall(text)
    return {
        "backend": "AutoModel",
        "steps": [dict(step=k + 1, **v) for k, v in sorted(steps.items())],
        "gbs": int(gbs.group(1)) if gbs else None,
        "expected": int(totals[-1]) if totals else None,
    }


def parse_megatron(text: str) -> dict:
    steps: dict[int, dict] = {}
    expected = None
    for m in MB_ITER.finditer(text):
        rest = m.group(4)
        ms, gbs, loss, nan = (r.search(rest) for r in (MB_MS, MB_GBS, MB_LOSS, MB_NAN))
        loss_value = _float(loss.group(1)) if loss else math.nan
        if nan and int(nan.group(1)) > 0:
            loss_value = math.nan
        steps[int(m.group(2))] = {
            "loss": loss_value,
            "step_time": float(ms.group(1)) / 1000.0 if ms else None,
            "gbs": int(gbs.group(1)) if gbs else None,
        }
        expected = int(m.group(3))
    ordered = [dict(step=k, **v) for k, v in sorted(steps.items())]
    return {
        "backend": "Megatron",
        "steps": ordered,
        "gbs": ordered[-1]["gbs"] if ordered else None,
        "expected": expected,
    }


def summarize(log: Path, warmup: int, expect: int | None, method: str | None, hardware: str | None) -> dict:
    text = log.read_text(errors="replace")
    parsed = parse_automodel(text)
    if not parsed["steps"]:
        parsed = parse_megatron(text)
    steps = parsed["steps"]
    expected = expect or parsed["expected"]
    notes: list[str] = []

    measured = steps[warmup:]
    step_times: list[float] = []
    tps: list[float] = []
    tps_gpu: list[float] = []
    if parsed["backend"] == "AutoModel":
        by_step = {s["step"]: s for s in steps}
        for s in measured:
            prev = by_step.get(s["step"] - 1)
            if prev:
                step_times.append((s["ts"] - prev["ts"]).total_seconds())
            if s["tps"] is not None:
                tps.append(s["tps"])
                tps_gpu.append(s["tps_gpu"])
        notes.append("Step time from 1-second log timestamps between consecutive steps.")
    else:
        step_times = [s["step_time"] for s in measured if s["step_time"] is not None]
        notes.append("Megatron-Bridge iteration lines carry no token counts; Tokens/s left null.")
    if not measured:
        notes.append(f"No steps after {warmup} warm-up step(s); raise MAX_STEPS or lower --warmup-steps.")

    bad = [s["step"] for s in steps if not math.isfinite(s["loss"])]
    last = steps[-1]["step"] if steps else 0
    failures = []
    if not steps:
        failures.append("no training steps found in log")
    if bad:
        failures.append(f"non-finite loss at step(s) {bad[:10]}")
    if expected and last < expected:
        failures.append(f"stopped at step {last} of {expected}")

    return {
        "Backend": parsed["backend"] if steps else None,
        "Training Method": method.upper() if method else None,
        "Hardware": hardware,
        "Run status": "Failed" if failures else "Passed",
        "Total training steps": last,
        "Warm-up steps": warmup,
        "Measured steps": len(measured),
        "Global batch size": parsed["gbs"],
        "Step time (s)": _mean(step_times),
        "Tokens/s": _mean(tps),
        "Tokens/s/GPU": _mean(tps_gpu),
        "Run ID": log.stem,
        "expected_steps": expected,
        "final_loss": steps[-1]["loss"] if steps and math.isfinite(steps[-1]["loss"]) else None,
        "failures": failures,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="Training log: <LOGS_DIR>/<job-name>_<jobid>_<timestamp>.log")
    parser.add_argument("--warmup-steps", type=int, default=3,
                        help="Logged steps excluded from timing (dashboard methodology: 3). Default: 3")
    parser.add_argument("--expect-steps", type=int,
                        help="Steps the run must reach (default: total parsed from the log progress line)")
    parser.add_argument("--method", choices=("sft", "lora"), help="Recorded as 'Training Method'")
    parser.add_argument("--hardware", help="Recorded as 'Hardware', for example A100 or H100")
    parser.add_argument("--output", type=Path, help="Also append the JSON row to this JSONL file")
    args = parser.parse_args(argv)
    if not args.log.is_file():
        parser.error(f"log file not found: {args.log}")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be >= 0")

    row = summarize(args.log, args.warmup_steps, args.expect_steps, args.method, args.hardware)
    line = json.dumps(row)
    print(line)
    if args.output:
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0 if row["Run status"] == "Passed" else 1


if __name__ == "__main__":
    sys.exit(main())
