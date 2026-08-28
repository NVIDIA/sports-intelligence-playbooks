#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply split-aware paraphrases to closed-vocab MCQ options.

Rewrites option/answer surface forms using a paraphrase bank JSON, e.g.:
  - ``configs/paraphrases/serve_rule_paraphrase_bank.json``
  - ``configs/paraphrases/point_outcome_paraphrase_bank.json``

Pools are disjoint:
  train -> 8 paraphrases
  validation -> 3 paraphrases
  test (test_seen + test_unseen) -> 3 paraphrases

One paraphrase style index is sampled per record (seeded by split + id) and
applied to all options in that question so wording stays stylistically matched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

OPTION_RE = re.compile(r"^(\(([A-Z])\)\s+)(.*)$")
DEFAULT_TEMPLATES = (
    "serve_rule_consistency",
    "grounded_serve_rule_consistency",
    "serve_sequence_outcome",
)

SPLIT_TO_POOL = {
    "train": "train",
    "validation": "validation",
    "test_seen": "test",
    "test_unseen": "test",
    # filename stems used by split_dataset.py
    "tennis_mcq_qa_train": "train",
    "tennis_mcq_qa_validation": "validation",
    "tennis_mcq_qa_test_seen_videos": "test",
    "tennis_mcq_qa_test_unseen_videos": "test",
}


def load_bank(path: str | Path) -> dict[str, Any]:
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"pool_sizes", "paraphrases"}
    missing = required - set(bank)
    if missing:
        raise ValueError(f"Paraphrase bank missing keys: {sorted(missing)}")
    sizes = bank["pool_sizes"]
    for pool, n in sizes.items():
        for canonical, pools in bank["paraphrases"].items():
            vals = pools.get(pool) or []
            if len(vals) != int(n):
                raise ValueError(
                    f"Bank entry {canonical!r} pool {pool!r} has "
                    f"{len(vals)} paraphrases; expected {n}"
                )
    validate_disjoint_pools(bank)
    return bank


def validate_disjoint_pools(bank: dict[str, Any]) -> None:
    """Ensure paraphrases are substantive and disjoint across data splits."""
    seen: dict[str, str] = {}
    for canonical, pools in bank["paraphrases"].items():
        canonical_seen: set[str] = set()
        for pool, texts in pools.items():
            for text in texts:
                normalized = re.sub(r"[\W_]+", " ", text.casefold()).strip()
                if normalized in canonical_seen:
                    raise ValueError(
                        f"Duplicate paraphrase for {canonical!r}: {text!r}"
                    )
                canonical_seen.add(normalized)
                if re.search(
                    r"\brule note\b|^\s*rules say:|^\s*standard interpretation:",
                    text,
                    flags=re.IGNORECASE,
                ) or text.rstrip().endswith("?."):
                    raise ValueError(
                        f"Low-quality paraphrase for {canonical!r}: {text!r}"
                    )

                prev = seen.get(normalized)
                key = f"{pool}:{canonical}"
                if prev is None:
                    seen[normalized] = key
                    continue
                prev_pool = prev.split(":", 1)[0]
                if prev_pool != pool:
                    raise ValueError(
                        "Paraphrase leaks across pools: "
                        f"{text!r} in {prev} and {key}"
                    )


def pool_for_split_name(split_name: str) -> str:
    stem = Path(split_name).stem
    if stem in SPLIT_TO_POOL:
        return SPLIT_TO_POOL[stem]
    for key, pool in SPLIT_TO_POOL.items():
        if key in stem:
            return pool
    raise ValueError(f"Cannot map split name to paraphrase pool: {split_name}")


def style_index(record_id: str, split_name: str, pool_size: int) -> int:
    digest = hashlib.sha256(f"{split_name}::{record_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % pool_size


def paraphrase_text(
    canonical: str,
    *,
    bank: dict[str, Any],
    pool: str,
    index: int,
    passthrough_missing: bool = False,
) -> str | None:
    entry = bank["paraphrases"].get(canonical)
    if entry is None:
        if passthrough_missing:
            return None
        raise KeyError(f"Canonical option missing from paraphrase bank: {canonical!r}")
    options = entry[pool]
    return options[index % len(options)]


def rewrite_question_and_answer(
    question: str,
    answer: str,
    *,
    bank: dict[str, Any],
    pool: str,
    index: int,
) -> tuple[str, str, dict[str, str]]:
    """Rewrite MCQ option lines and matching answer text.

    Options absent from the bank are left unchanged when
    ``bank["passthrough_missing"]`` is true (used for none-phrase banks).

    When ``bank["rewrite_stems"]`` is true, non-option lines that exactly match
    a bank key (e.g. rules-knowledge question stems) are also paraphrased.

    Returns (new_question, new_answer, canonical_to_paraphrase).
    """
    passthrough = bool(bank.get("passthrough_missing"))
    rewrite_stems = bool(bank.get("rewrite_stems"))
    mapping: dict[str, str] = {}
    new_lines: list[str] = []
    for line in question.split("\n"):
        match = OPTION_RE.match(line.strip())
        if not match:
            stem = line.strip()
            if rewrite_stems and stem:
                paraphrased = paraphrase_text(
                    stem,
                    bank=bank,
                    pool=pool,
                    index=index,
                    passthrough_missing=True,
                )
                if paraphrased is not None:
                    mapping[stem] = paraphrased
                    indent = line[: len(line) - len(line.lstrip(" "))]
                    new_lines.append(f"{indent}{paraphrased}")
                    continue
            new_lines.append(line)
            continue
        prefix, _label, canonical = match.group(1), match.group(2), match.group(3).strip()
        paraphrased = paraphrase_text(
            canonical,
            bank=bank,
            pool=pool,
            index=index,
            passthrough_missing=passthrough,
        )
        if paraphrased is None:
            new_lines.append(line)
            continue
        mapping[canonical] = paraphrased
        indent = line[: len(line) - len(line.lstrip(" "))]
        new_lines.append(f"{indent}{prefix}{paraphrased}")

    new_question = "\n".join(new_lines)
    ans = answer.strip()
    ans_match = OPTION_RE.match(ans)
    if ans_match:
        prefix, _label, canonical = (
            ans_match.group(1),
            ans_match.group(2),
            ans_match.group(3).strip(),
        )
        paraphrased = mapping.get(canonical)
        if paraphrased is None:
            paraphrased = paraphrase_text(
                canonical,
                bank=bank,
                pool=pool,
                index=index,
                passthrough_missing=passthrough,
            )
            if paraphrased is not None:
                mapping[canonical] = paraphrased
        new_answer = f"{prefix}{paraphrased}" if paraphrased is not None else answer
    else:
        new_answer = mapping.get(ans, answer)
    return new_question, new_answer, mapping


def rewrite_record(
    record: dict[str, Any],
    *,
    bank: dict[str, Any],
    pool: str,
    split_name: str,
    templates: set[str],
) -> bool:
    """Rewrite one HF record in place. Returns True if modified."""
    if record.get("class") not in templates:
        return False
    conversation = record.get("conversation") or []
    if len(conversation) < 2:
        return False
    user = conversation[0]
    assistant = conversation[1]
    q_block = next(
        (c for c in user.get("content") or [] if c.get("type") == "text"),
        None,
    )
    a_block = next(
        (c for c in assistant.get("content") or [] if c.get("type") == "text"),
        None,
    )
    if not q_block or not a_block:
        return False

    index = style_index(str(record.get("id", "")), split_name, bank["pool_sizes"][pool])
    new_q, new_a, mapping = rewrite_question_and_answer(
        q_block["text"],
        a_block["text"],
        bank=bank,
        pool=pool,
        index=index,
    )
    if not mapping:
        return False
    q_block["text"] = new_q
    a_block["text"] = new_a

    # Do not create metadata on MCQ records. Point-level metadata is reserved
    # for open-ended evaluation; paraphrase provenance is optional and is only
    # appended when a record already carries a metadata mapping.
    meta = record.get("metadata")
    if isinstance(meta, dict):
        meta_key = bank.get("metadata_key") or "option_paraphrase"
        meta[meta_key] = {
            "pool": pool,
            "style_index": index,
            "canonical_to_paraphrase": mapping,
            "templates": sorted(templates),
        }
    return True


def rewrite_jsonl_file(
    path: Path,
    *,
    bank: dict[str, Any],
    templates: set[str],
) -> dict[str, int]:
    pool = pool_for_split_name(path.name)
    split_name = path.stem
    tmp = path.with_suffix(path.suffix + ".tmp")
    stats = {"records": 0, "rewritten": 0, "skipped": 0}
    with path.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            stats["records"] += 1
            changed = rewrite_record(
                record,
                bank=bank,
                pool=pool,
                split_name=split_name,
                templates=templates,
            )
            if changed:
                stats["rewritten"] += 1
            else:
                stats["skipped"] += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)
    stats["pool"] = pool  # type: ignore[assignment]
    return stats


def assert_no_train_eval_option_overlap(
    splits_dir: Path,
    templates: set[str],
    bank: dict[str, Any] | None = None,
) -> None:
    """Hard check: train option strings must not appear in val/test for these templates.

    When ``bank`` is provided, only option strings that belong to the bank's
    paraphrase pools are checked. That lets none-phrase banks leave distractors
    unchanged without failing on shared shot/trajectory labels.
    """
    opt_re = re.compile(r"^\(([A-Z])\)\s+(.*)$")
    bank_texts: set[str] | None = None
    if bank is not None:
        bank_texts = set()
        for pools in bank["paraphrases"].values():
            for texts in pools.values():
                bank_texts.update(texts)

    def collect(path: Path) -> set[str]:
        texts: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("class") not in templates:
                    continue
                q = next(
                    c["text"]
                    for c in rec["conversation"][0]["content"]
                    if c.get("type") == "text"
                )
                for raw in q.split("\n"):
                    m = opt_re.match(raw.strip())
                    if m:
                        texts.add(m.group(2).strip())
        if bank_texts is not None:
            texts &= bank_texts
        return texts

    files = {p.stem: p for p in splits_dir.glob("tennis_mcq_qa_*.jsonl")}
    train = next(p for stem, p in files.items() if stem.endswith("_train") or stem == "tennis_mcq_qa_train")
    val = next(p for stem, p in files.items() if "validation" in stem)
    tests = [p for stem, p in files.items() if "test_" in stem]
    train_opts = collect(train)
    eval_opts: set[str] = set()
    eval_opts |= collect(val)
    for p in tests:
        eval_opts |= collect(p)
    overlap = train_opts & eval_opts
    if overlap:
        examples = sorted(overlap)[:10]
        raise AssertionError(
            f"Train/eval paraphrase overlap detected ({len(overlap)} strings). "
            f"Examples: {examples}"
        )


def run_rewrite(
    splits_dir: str | Path,
    bank_path: str | Path,
    templates: list[str] | None = None,
    *,
    check_overlap: bool = True,
) -> list[dict[str, int]]:
    splits_dir = Path(splits_dir)
    bank = load_bank(bank_path)
    template_set = set(templates or bank.get("templates") or DEFAULT_TEMPLATES)
    results = []
    for path in sorted(splits_dir.glob("tennis_mcq_qa_*.jsonl")):
        stats = rewrite_jsonl_file(path, bank=bank, templates=template_set)
        stats_out = {"file": path.name, **stats}
        results.append(stats_out)
        print(
            f"  {path.name}: pool={stats['pool']} "
            f"rewritten={stats['rewritten']}/{stats['records']}"
        )
    if check_overlap:
        assert_no_train_eval_option_overlap(splits_dir, template_set, bank=bank)
        print("  Overlap check passed (train option texts ∩ eval option texts = ∅)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="Directory containing tennis_mcq_qa_*.jsonl splits",
    )
    parser.add_argument(
        "--bank",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            "paraphrases",
            "serve_rule_paraphrase_bank.json",
        ),
        help="Paraphrase bank JSON path",
    )
    parser.add_argument(
        "--skip-overlap-check",
        action="store_true",
        help="Skip post-rewrite train/eval overlap assertion",
    )
    args = parser.parse_args()
    print("MCQ option paraphrase rewrite")
    print(f"  splits: {args.splits_dir}")
    print(f"  bank:   {args.bank}")
    run_rewrite(
        args.splits_dir,
        args.bank,
        check_overlap=not args.skip_overlap_check,
    )


if __name__ == "__main__":
    main()
