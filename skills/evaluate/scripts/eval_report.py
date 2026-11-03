#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check, break down and compare AVLM eval results (read-only).

For each inference directory, reads predictions.jsonl, mcq/mcq_results.json
and, when present, llm_judge/llm_judge_results.json. MCQ counts are recomputed
with the parser and counting rules of avlm/evals/mcq/eval_mcq.py, cross-checked
against mcq_results.json, compared with a chance baseline, and broken down by
super_category, fine_category and class. With several directories, the first
is the baseline for deltas. Prints Markdown; writes JSON only with --json-out.

Exit codes: 0 = no FAIL checks, 1 = at least one FAIL, 2 = unusable input.
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
try:
    from avlm.evals.mcq.eval_mcq import parse_mcq_answer
except ImportError as exc:
    print(f"error: cannot import avlm.evals.mcq.eval_mcq from {REPO_ROOT} ({exc}); "
          "run this script from its location inside a sports_intelligence checkout", file=sys.stderr)
    sys.exit(2)

LEVELS = ("super_category", "fine_category", "class")
OUTCOMES = ("correct", "incorrect", "parse_failed", "infer_failed")
RESULT_FIELDS = {
    "total_mcq_questions": "n",
    "num_correct": "correct",
    "num_incorrect": "incorrect",
    "num_parse_failed": "parse_failed",
    "num_infer_failed": "infer_failed",
}
SCORED_TYPES = ("mcq", "open_ended")
OPTION_RE = re.compile(r"^\s*\(([A-Za-z])\)\s", re.MULTILINE)


def die(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inference_dirs", nargs="+", metavar="INFERENCE_DIR",
                        help="Directory containing predictions.jsonl (first one is the baseline)")
    parser.add_argument("--expected-rows", type=int, default=None,
                        help="Expected predictions.jsonl row count (dataset rows or MAX_INFERENCE_SAMPLES)")
    parser.add_argument("--json-out", default=None,
                        help="Also write the full report as JSON to this path")
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    die(f"{path}:{line_no} is not valid JSON ({exc})")
    return rows


def prompt_strings(node):
    """Yield strings from the non-assistant turns of a conversations field."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from prompt_strings(item)
    elif isinstance(node, dict):
        if node.get("role") == "assistant" or node.get("from") == "gpt":
            return
        for value in node.values():
            yield from prompt_strings(value)


def num_options(row):
    text = "\n".join(prompt_strings(row.get("conversations") or row.get("conversation") or []))
    return len({m.lower() for m in OPTION_RE.findall(text)})


def new_bucket():
    return {"n": 0, **{k: 0 for k in OUTCOMES}, "chance_sum": 0.0, "chance_n": 0}


def score_mcq(rows):
    """Mirror eval_mcq.py: inference and parse failures stay in the denominator;
    rows whose ground truth does not parse are skipped (counted separately)."""
    overall, skipped_gt, ids = new_bucket(), 0, set()
    letters = {"prediction": Counter(), "ground_truth": Counter()}
    by_level = {level: defaultdict(new_bucket) for level in LEVELS}
    for row in rows:
        if row.get("evaluation_type") != "mcq":
            continue
        if row.get("inference_status") != "success":
            outcome = "infer_failed"
        else:
            gt = parse_mcq_answer(row.get("ground_truth") or "")
            if gt is None:
                skipped_gt += 1
                continue
            pred = parse_mcq_answer(row.get("prediction") or "")
            outcome = "parse_failed" if pred is None else ("correct" if pred == gt else "incorrect")
            letters["ground_truth"][gt] += 1
            letters["prediction"][pred or "unparsed"] += 1
        ids.add(row.get("id"))
        k = num_options(row)
        buckets = [overall] + [by_level[level][str(row.get(level) or "Unknown")] for level in LEVELS]
        for bucket in buckets:
            bucket["n"] += 1
            bucket[outcome] += 1
            if k >= 2:
                bucket["chance_sum"] += 1.0 / k
                bucket["chance_n"] += 1
    return overall, {level: dict(b) for level, b in by_level.items()}, skipped_gt, ids, letters


def acc(bucket):
    return bucket["correct"] / bucket["n"] if bucket["n"] else 0.0


def chance(bucket):
    return bucket["chance_sum"] / bucket["chance_n"] if bucket["chance_n"] else None


def pct(value):
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def check_mcq_results(run_dir, overall, per_class):
    path = run_dir / "mcq" / "mcq_results.json"
    if not path.exists():
        return [("mcq_results", "FAIL",
                 "mcq/mcq_results.json missing; run INFERENCE_DIR=<dir> bash ./avlm/evals/scripts/run_mcq_eval.sh")]
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    diffs = [f"{key}={results.get(key)} vs recomputed {overall[field]}"
             for key, field in RESULT_FIELDS.items() if results.get(key) != overall[field]]
    for cls in sorted(set(results.get("per_class", {})) | set(per_class)):
        stored, mine = results.get("per_class", {}).get(cls), per_class.get(cls)
        if stored is None or mine is None or any(stored.get(k) != mine[f] for k, f in RESULT_FIELDS.items()):
            diffs.append(f"per_class[{cls}] differs")
    checks = [("mcq_results", "FAIL" if diffs else "PASS",
               "stale or foreign mcq_results.json, re-run run_mcq_eval.sh: " + "; ".join(diffs[:4]) if diffs
               else f"mcq_results.json matches predictions.jsonl ({overall['n']} scored MCQ)")]
    if (run_dir / "predictions.jsonl").stat().st_mtime > path.stat().st_mtime:
        checks.append(("mcq_results_age", "WARN", "predictions.jsonl is newer than mcq_results.json"))
    return checks


def judge_summary(run_dir, n_open):
    judge_dir = run_dir / "llm_judge"
    path = judge_dir / "llm_judge_results.json"
    if not path.exists():
        if (judge_dir / "llm_judge_predictions.json").exists():
            return None, [("llm_judge", "WARN", "llm_judge_predictions.json not aggregated; "
                           "run INFERENCE_DIR=<dir> bash ./avlm/evals/scripts/run_qa_llm_judge_eval.sh")]
        status = "WARN" if n_open else "PASS"
        return None, [("llm_judge", status, f"{n_open} open_ended rows not judged (judge needs CLIENT_API_KEY)")]
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    checks = []
    total = results.get("total_judged", 0)
    if total < n_open:
        checks.append(("llm_judge_coverage", "WARN",
                       f"judged {total} of {n_open} open_ended rows (MAX_LLM_JUDGE_SAMPLES, drop_failed or a crash)"))
    else:
        checks.append(("llm_judge_coverage", "PASS", f"judged {total} of {n_open} open_ended rows"))
    if results.get("num_failed"):
        checks.append(("llm_judge_failures", "WARN", f"{results['num_failed']} judge calls failed or scored 0"))
    excellent = results.get("score_distribution", {}).get("excellent_9_10", 0)
    summary = {
        "mean_score": results.get("mean_score"),
        "num_success": results.get("num_success"),
        "total_judged": total,
        "pass9_rate": excellent / total if total else None,
        "per_class_mean": {k: v.get("mean_score") for k, v in results.get("per_class", {}).items()},
    }
    return summary, checks


def analyze(run_dir, expected_rows):
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        die(f"{pred_path} not found; INFERENCE_DIR must contain predictions.jsonl from inference")
    rows = load_jsonl(pred_path)
    types = defaultdict(int)
    for row in rows:
        types[row.get("evaluation_type")] += 1
    ids = [row.get("id") for row in rows]
    overall, by_level, skipped_gt, mcq_ids, letters = score_mcq(rows)

    checks = [("rows", "PASS" if rows else "FAIL",
               f"{len(rows)} rows: " + ", ".join(f"{k}={v}" for k, v in sorted(types.items(), key=str)))]
    if expected_rows is not None:
        checks.append(("expected_rows", "PASS" if len(rows) == expected_rows else "FAIL",
                       f"{len(rows)} rows, expected {expected_rows}"))
    dupes = len(ids) - len(set(ids))
    checks.append(("unique_ids", "FAIL" if dupes else "PASS", f"{dupes} duplicate ids"))
    unscored = {str(k): v for k, v in types.items() if k not in SCORED_TYPES}
    checks.append(("evaluation_type", "FAIL" if unscored else "PASS",
                   f"rows with missing or unknown evaluation_type, silently skipped by both scorers: {unscored}"
                   if unscored else "every row is mcq or open_ended"))
    uncategorized = by_level["super_category"].get("Unknown", {}).get("n", 0)
    if uncategorized:
        checks.append(("categories", "WARN",
                       f"{uncategorized} MCQ rows lack super_category (pre-categorical dataset); category tables show 'Unknown'"))
    if overall["infer_failed"]:
        checks.append(("inference_status", "WARN", f"{overall['infer_failed']} MCQ rows with inference_status != success"))
    if skipped_gt:
        checks.append(("ground_truth", "WARN", f"{skipped_gt} MCQ rows skipped: ground truth does not start with '(X)'"))
    if overall["parse_failed"]:
        checks.append(("answer_format", "WARN",
                       f"{overall['parse_failed']} predictions do not start with '(X)'; counted wrong"))
    base = chance(overall)
    if overall["n"] and base is not None:
        margin = 1.96 * math.sqrt(base * (1 - base) / overall["n"])
        above = acc(overall) - base > margin
        checks.append(("above_chance", "PASS" if above else "WARN",
                       f"accuracy {pct(acc(overall))} vs chance {pct(base)} (95% noise band +/-{100 * margin:.1f}pp)"))
    checks += check_mcq_results(run_dir, overall, by_level["class"])
    judge, judge_checks = judge_summary(run_dir, types.get("open_ended", 0))
    checks += judge_checks
    return {"path": str(run_dir), "checks": checks, "mcq_ids": mcq_ids,
            "mcq": {"overall": overall, "by_level": by_level, "skipped_invalid_gt": skipped_gt,
                    "letters": {k: dict(sorted(v.items())) for k, v in letters.items()}},
            "judge": judge}


def render_run(label, run):
    out = [f"## Run `{label}`", f"Path: `{run['path']}`", "", "| Check | Status | Detail |", "|---|---|---|"]
    out += [f"| {name} | {status} | {detail} |" for name, status, detail in run["checks"]]
    o = run["mcq"]["overall"]
    out += ["", f"MCQ accuracy {pct(acc(o))} ({o['correct']}/{o['n']}), chance {pct(chance(o))}, "
            f"parse_failed {o['parse_failed']}, infer_failed {o['infer_failed']}"]
    for kind, counts in run["mcq"]["letters"].items():
        out.append(f"{kind} letters: " + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none"))
    if run["judge"]:
        j = run["judge"]
        out.append(f"LLM judge mean {j['mean_score']:.2f}/10 over {j['num_success']}/{j['total_judged']} judged, "
                   f"score>=9 rate {pct(j['pass9_rate'])}")
    for level in LEVELS:
        out += ["", f"### MCQ by {level}", "", "| Category | n | Accuracy | Chance | Parse failed |", "|---|---:|---:|---:|---:|"]
        for name, b in sorted(run["mcq"]["by_level"][level].items()):
            out.append(f"| {name} | {b['n']} | {pct(acc(b))} | {pct(chance(b))} | {b['parse_failed']} |")
    return out


def render_comparison(labels, runs):
    base = runs[0]
    out = ["## Comparison (baseline = first run)", "", "| Check | Status | Detail |", "|---|---|---|"]
    checks = []
    for label, run in zip(labels[1:], runs[1:]):
        same = run["mcq_ids"] == base["mcq_ids"]
        detail = (f"{label}: same {len(run['mcq_ids'])} MCQ ids as baseline" if same else
                  f"{label}: {len(run['mcq_ids'])} MCQ ids vs {len(base['mcq_ids'])} in baseline, "
                  f"{len(run['mcq_ids'] & base['mcq_ids'])} shared; deltas are not comparable")
        checks.append(("same_mcq_samples", "PASS" if same else "FAIL", detail))
    out += [f"| {n} | {s} | {d} |" for n, s, d in checks]
    header = "| Level | Category | n | " + " | ".join(labels) + " |"
    out += ["", header, "|" + "---|" * (len(labels) + 3)]

    def cells(values):
        first = values[0]
        return [pct(first)] + [pct(v) if v is None or first is None else f"{pct(v)} ({100 * (v - first):+.1f}pp)"
                               for v in values[1:]]

    out.append("| overall | MCQ accuracy | " + str(base["mcq"]["overall"]["n"]) + " | "
               + " | ".join(cells([acc(r["mcq"]["overall"]) for r in runs])) + " |")
    if all(r["judge"] for r in runs):
        means = [r["judge"]["mean_score"] for r in runs]
        out.append("| overall | LLM judge mean (1-10) | " + str(base["judge"]["total_judged"]) + " | "
                   + " | ".join([f"{means[0]:.2f}"] + [f"{m:.2f} ({m - means[0]:+.2f})" for m in means[1:]]) + " |")
    for level in LEVELS:
        for name in sorted(set().union(*(r["mcq"]["by_level"][level] for r in runs))):
            buckets = [r["mcq"]["by_level"][level].get(name) for r in runs]
            n = buckets[0]["n"] if buckets[0] else 0
            out.append(f"| {level} | {name} | {n} | "
                       + " | ".join(cells([acc(b) if b else None for b in buckets])) + " |")
    return out, checks


def main():
    args = parse_args()
    dirs = [Path(d) for d in args.inference_dirs]
    labels = [d.resolve().name if [x.resolve().name for x in dirs].count(d.resolve().name) == 1
              else f"{i}:{d.resolve().name}" for i, d in enumerate(dirs)]
    runs = [analyze(d, args.expected_rows) for d in dirs]
    lines = ["# AVLM eval report", ""]
    for label, run in zip(labels, runs):
        lines += render_run(label, run) + [""]
    comparison_checks = []
    if len(runs) > 1:
        comparison, comparison_checks = render_comparison(labels, runs)
        lines += comparison
    print("\n".join(lines))

    failed = [f"{label}:{name}" for label, run in zip(labels, runs)
              for name, status, _ in run["checks"] if status == "FAIL"]
    failed += [name for name, status, _ in comparison_checks if status == "FAIL"]
    if args.json_out:
        payload = {"runs": {label: {k: v for k, v in run.items() if k != "mcq_ids"}
                            for label, run in zip(labels, runs)},
                   "comparison_checks": comparison_checks, "failed_checks": failed}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print(f"\nRESULT: {'FAIL ' + ', '.join(failed) if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
