#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Categorical-evaluation MCQ/QA generator.

Reads ``configs/generation/tennis_categorical_eval_config.json`` (legacy-compatible
``question_templates`` + ``qa_templates`` shape) and writes HuggingFace
conversation JSON per video. Does not modify ``generate_mcq_qa.py``.

Usage:
    python generate_categorical_mcq_qa.py
    python generate_categorical_mcq_qa.py \
        --config configs/generation/tennis_categorical_eval_config.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from generate_mcq_qa import (
    ConfigDrivenMCQGenerator,
    MatchContext,
    Metadata,
    PointData,
    TennisDataset,
    VideoInfo,
    clean_freetext_answer,
    is_empty_answer_value,
    load_question_types,
    normalize_option_text,
    normalize_question_text,
    option_dedupe_key,
    unique_mcq_options,
)
from tennis_rules_knowledge import build_default_selector

from llm_helper import generate_tennis_mcq_with_llm

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_UNAVAILABLE_MARKERS = ["n/a", "unknown", "unknow"]

EXTRA_POINT_BASE_FIELDS = [
    "winner_shot_type",
    "loser_shot_type",
    "winner_hand_used",
    "loser_hand_used",
    "winner_shot_trajectory",
    "loser_shot_trajectory",
    "server_location",
    "receiver_location",
    "loser_lateral_movement",
    "loser_depth_movement",
    "loser_point_ended",
]
EXTRA_POINT_FIELDS = [
    *EXTRA_POINT_BASE_FIELDS,
    *(f"{name}_other" for name in EXTRA_POINT_BASE_FIELDS),
]


def canonicalize_categorical_text(value: Any) -> str:
    """Lowercase and collapse inconsequential whitespace/punctuation variants."""
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    return text.rstrip(".").strip()


def coerce_positive_int(value: Any) -> Optional[int]:
    """Return a positive integer from noisy annotations, else ``None``.

    Serve-attempt counts must be >= 1; values such as ``0``, blanks, or dirty
    strings (``"`1"``, ``"  "``) are treated as unusable so they never become an
    MCQ answer or option.
    """
    text = normalize_option_text(value)
    if not text:
        return None
    try:
        parsed = int(float(text))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


_LIVE_POINT_SCORE_RE = re.compile(
    r"(?P<prefix>\{\s*\d+\s*[-–]\s*\d+\s*,\s*)"
    r"\((?P<points>\d+\s*[-–]\s*\d+)\)"
    r"(?P<suffix>\s*\})"
)


def canonicalize_live_score_display(value: Any) -> str:
    """Render live point scores without tiebreak-only parentheses.

    Only the point-score component inside ``{games, points}`` is changed.
    Historical set tiebreak results such as ``7-6(7-5)`` remain untouched.
    """
    text = normalize_option_text(value)
    return _LIVE_POINT_SCORE_RE.sub(
        r"\g<prefix>\g<points>\g<suffix>",
        text,
    )


def score_display_key(value: Any) -> str:
    """Return a comparison key that ignores live-score display variants."""
    return option_dedupe_key(canonicalize_live_score_display(value))


def alternate_next_tennis_score(score_before: Any, score_after: Any) -> str:
    """Return the valid next score if the other player had won the point."""
    before = str(score_before)
    match = re.match(
        r"^(?P<prefix>.*\{)\s*(?P<games1>\d+)-(?P<games2>\d+)\s*,\s*"
        r"(?P<points1>[A-Za-z0-9]+)-(?P<points2>[A-Za-z0-9]+)\s*"
        r"(?P<suffix>\}.*)$",
        before,
    )
    if not match:
        tie_break = re.match(
            r"^(?P<prefix>.*\{.*?\d+-\d+\s*,?\s*\()"
            r"(?P<points1>\d+)-(?P<points2>\d+)(?P<suffix>\).*)$",
            before,
        )
        if not tie_break:
            return before
        tie_points = [
            int(tie_break.group("points1")),
            int(tie_break.group("points2")),
        ]
        candidates = []
        for side in (0, 1):
            next_points = tie_points.copy()
            next_points[side] += 1
            candidates.append(
                f"{tie_break.group('prefix')}{next_points[0]}-{next_points[1]}"
                f"{tie_break.group('suffix')}"
            )
        normalized_after = score_display_key(score_after)
        for candidate in candidates:
            if score_display_key(candidate) != normalized_after:
                return candidate
        return before

    games = [int(match.group("games1")), int(match.group("games2"))]
    points = [match.group("points1").upper(), match.group("points2").upper()]

    def award_point(side: int) -> str:
        next_games = games.copy()
        next_points = points.copy()
        other = 1 - side
        game_won = False

        if next_points[side] == "AD":
            game_won = True
        elif next_points[other] == "AD":
            next_points[other] = "40"
        elif next_points[side] == "40":
            if next_points[other] == "40":
                next_points[side] = "AD"
            else:
                game_won = True
        else:
            progression = {"0": "15", "15": "30", "30": "40"}
            if next_points[side] in progression:
                next_points[side] = progression[next_points[side]]
            elif next_points[side].isdigit() and next_points[other].isdigit():
                next_points[side] = str(int(next_points[side]) + 1)
            else:
                return before

        if game_won:
            next_games[side] += 1
            next_points = ["0", "0"]
        return (
            f"{match.group('prefix')}{next_games[0]}-{next_games[1]}, "
            f"{next_points[0]}-{next_points[1]}{match.group('suffix')}"
        )

    candidates = [award_point(0), award_point(1)]
    normalized_after = score_display_key(score_after)
    for candidate in candidates:
        if score_display_key(candidate) != normalized_after:
            return candidate
    return before


SUPPORTED_RALLY_OUTCOMES = ("Winner", "Forced Error", "Unforced Error")


def serve_sequence_summary(attempts: Any, outcome: str) -> Optional[str]:
    """Describe the serve sequence without equating two attempts with a double fault."""
    attempt_text = str(attempts).strip()
    outcome_text = str(outcome).strip()

    if attempt_text not in {"1", "2"}:
        return None
    if outcome_text == "Double Fault":
        if attempt_text != "2":
            return None
        return "First and second serves were faults; point outcome: Double Fault"
    if outcome_text == "Ace":
        if attempt_text == "1":
            return "First serve was an ace; point outcome: Ace"
        return "First serve was a fault; second serve was an ace; point outcome: Ace"
    if outcome_text not in SUPPORTED_RALLY_OUTCOMES:
        return None
    if attempt_text == "1":
        return f"First serve was successful; point outcome: {outcome_text}"
    return (
        "First serve was a fault; second serve was successful; "
        f"point outcome: {outcome_text}"
    )


def build_serve_sequence_options(
    attempts: Any,
    outcome: str,
    maximum_options: int = 4,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return one annotated answer and coherent alternative serve sequences."""
    answer = serve_sequence_summary(attempts, outcome)
    if answer is None:
        return None, None

    scenarios = [
        serve_sequence_summary("1", "Ace"),
        serve_sequence_summary("2", "Ace"),
        serve_sequence_summary("2", "Double Fault"),
        serve_sequence_summary("1", "Winner"),
        serve_sequence_summary("2", "Winner"),
        serve_sequence_summary("1", "Forced Error"),
        serve_sequence_summary("2", "Forced Error"),
        serve_sequence_summary("1", "Unforced Error"),
        serve_sequence_summary("2", "Unforced Error"),
    ]
    options = [answer]
    options.extend(
        scenario
        for scenario in scenarios
        if scenario is not None and scenario != answer
    )
    return answer, options[:maximum_options]


def build_serve_rule_consistency(
    attempts: Any,
    outcome: str,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Build one rule-consistent interpretation and three contradictions."""
    correct = serve_sequence_summary(attempts, outcome)
    if correct is None:
        return None, None

    attempt_text = str(attempts).strip()
    outcome_text = str(outcome).strip()
    if outcome_text == "Double Fault":
        contradictions = [
            "Second serve was successful, but point outcome was Double Fault",
            "Second serve was an ace, but the rally continued afterward",
            "Both serves were faults, but the rally continued to a Winner",
        ]
    elif outcome_text == "Ace" and attempt_text == "2":
        contradictions = [
            "Second serve was successful and the rally continued, despite the Ace outcome",
            "Both serves were faults, but point outcome was Ace",
            "First serve was an ace, but a second serve was attempted",
        ]
    elif outcome_text == "Ace":
        contradictions = [
            "First serve was an ace, but the rally continued to a Winner",
            "First serve was a fault, but no second serve was taken and outcome was Ace",
            "First serve was successful, but point outcome was Double Fault",
        ]
    elif attempt_text == "2":
        contradictions = [
            f"Both serves were faults, but the rally continued to {outcome_text}",
            "Second serve was successful, but point outcome was Double Fault",
            f"Second serve was an ace, but the rally continued to {outcome_text}",
        ]
    else:
        contradictions = [
            f"First serve was an ace, but the rally continued to {outcome_text}",
            "First serve was a fault, but the rally continued without a second serve",
            f"Both serves were faults, but the rally continued to {outcome_text}",
        ]
    return correct, [correct, *contradictions]


def build_grounded_serve_rule_consistency(
    attempts: Any,
    outcome: str,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Mix a coherent video distractor with two tennis-rule contradictions."""
    correct, coherent_options = build_serve_sequence_options(
        attempts,
        outcome,
        maximum_options=4,
    )
    _, rule_options = build_serve_rule_consistency(attempts, outcome)
    if correct is None or coherent_options is None or rule_options is None:
        return None, None

    coherent_wrong = next(
        (option for option in coherent_options if option != correct),
        None,
    )
    contradictions = [
        option for option in rule_options if option != correct
    ][:2]
    if coherent_wrong is None or len(contradictions) < 2:
        return None, None
    return correct, [correct, coherent_wrong, *contradictions]


def build_point_outcome_reasoning(
    winner: Any,
    outcome: str,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Explain how the point-ending mechanism determines the winning role."""
    winner_role = str(winner).strip().casefold()
    if winner_role not in {"server", "receiver"}:
        return None, None
    loser_role = "receiver" if winner_role == "server" else "server"
    outcome_text = str(outcome).strip()

    if outcome_text == "Ace":
        if winner_role != "server":
            return None, None
        correct = "server hit an ace and won the point immediately"
    elif outcome_text == "Double Fault":
        if winner_role != "receiver":
            return None, None
        correct = "server double-faulted, so receiver won the point"
    elif outcome_text == "Winner":
        correct = f"{winner_role} hit a point-ending winner and won the point"
    elif outcome_text == "Forced Error":
        correct = (
            f"{winner_role} forced an error from {loser_role} and won the point"
        )
    elif outcome_text == "Unforced Error":
        correct = (
            f"{loser_role} committed an unforced error, so "
            f"{winner_role} won the point"
        )
    else:
        return None, None

    coherent_wrong = (
        f"{loser_role} hit a point-ending winner and won the point"
    )
    contradictions = [
        (
            f"{winner_role} committed the point-ending error "
            f"but still won the point"
        ),
        (
            f"{loser_role} hit the point-ending winner, "
            f"but {winner_role} won the point"
        ),
    ]
    return correct, [correct, coherent_wrong, *contradictions]


# ---------------------------------------------------------------------------
# Caption action-mismatch MCQ construction
#
# Leave the original caption intact and append a controlled sentence that
# states exactly one verified structured action. Distractors alter only the
# value in that appended sentence (never mid-caption phrase swaps), so grammar
# stays intact even when values have different articles/prepositions.
# ---------------------------------------------------------------------------

ACTION_FIELD_KIND = {
    "winner_shot_type": "shot_type",
    "loser_shot_type": "shot_type",
    "winner_hand_used": "hand_used",
    "loser_hand_used": "hand_used",
    "winner_shot_trajectory": "shot_trajectory",
    "loser_shot_trajectory": "shot_trajectory",
    "loser_lateral_movement": "lateral_movement",
    "loser_depth_movement": "depth_movement",
}

ACTION_ROLE = {
    "winner_shot_type": "winning",
    "loser_shot_type": "losing",
    "winner_hand_used": "winning",
    "loser_hand_used": "losing",
    "winner_shot_trajectory": "winning",
    "loser_shot_trajectory": "losing",
    "loser_lateral_movement": "losing",
    "loser_depth_movement": "losing",
}

DEFAULT_ACTION_ALTERNATIVES = {
    "shot_type": [
        "flat groundstroke",
        "slice",
        "volley",
        "drop shot",
        "lob",
        "half-volley",
        "overhead smash",
        "passing shot",
        "block shot",
    ],
    "hand_used": ["forehand", "backhand"],
    "shot_trajectory": [
        "cross-court",
        "down the line",
        "down the middle",
        "deuce court",
        "ad court",
        "out of court",
    ],
    "lateral_movement": [
        "left to right",
        "right to left",
        "no lateral movement",
    ],
    "depth_movement": ["forward", "backward", "no depth movement"],
}


@dataclass(frozen=True)
class ActionAttribute:
    field_name: str
    kind: str
    role: str
    display: str
    aliases: tuple[str, ...]
    matched_alias: Optional[str]
    in_caption: bool


def normalize_action_with_mapping(value: Any, mapping: dict[str, Any] | None) -> str:
    text = str(value).strip()
    if not mapping or not isinstance(mapping, dict):
        return canonicalize_categorical_text(text)
    casefold_mapping = {
        canonicalize_categorical_text(key): str(mapped).strip()
        for key, mapped in mapping.items()
        if not str(key).startswith("ball_outcome_")
    }
    canonical = canonicalize_categorical_text(text)
    if canonical in casefold_mapping:
        return casefold_mapping[canonical]
    return canonicalize_categorical_text(text)


def _unique_action_phrases(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in phrases:
        cleaned = re.sub(r"\s+", " ", str(phrase)).strip()
        if not cleaned:
            continue
        key = canonicalize_categorical_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def action_display_aliases(
    field_name: str,
    raw_value: Any,
    display: str,
    normalizations: dict[str, Any],
) -> list[str]:
    kind = ACTION_FIELD_KIND[field_name]
    mapping = normalizations.get(kind) or normalizations.get(
        "shot_type_aliases" if kind == "shot_type" else kind,
        {},
    )
    aliases = [display, str(raw_value).strip(), canonicalize_categorical_text(display)]
    if isinstance(mapping, dict):
        target = canonicalize_categorical_text(display)
        for key, mapped in mapping.items():
            if (
                canonicalize_categorical_text(mapped) == target
                or canonicalize_categorical_text(key) == target
            ):
                aliases.extend([str(key), str(mapped)])
    if kind == "hand_used":
        if "forehand" in canonicalize_categorical_text(display):
            aliases.extend(["forehand", "FH", "fh"])
        if "backhand" in canonicalize_categorical_text(display):
            aliases.extend(["backhand", "BH", "bh"])
    if kind == "shot_trajectory":
        aliases.extend(
            [
                display.replace("-", " "),
                display.replace("cross-court", "cross court"),
            ]
        )
    if kind == "lateral_movement":
        if "left to right" in canonicalize_categorical_text(display):
            aliases.extend(["left to right", "L2R", "l2r"])
        if "right to left" in canonicalize_categorical_text(display):
            aliases.extend(["right to left", "R2L", "r2l"])
    if kind == "depth_movement":
        if canonicalize_categorical_text(display) == "forward":
            aliases.extend(["forward", "back to front", "B2F", "b2f"])
        if canonicalize_categorical_text(display) == "backward":
            aliases.extend(["backward", "front to back", "F2B", "f2b"])
    return _unique_action_phrases(aliases)


def find_action_alias_in_caption(caption: str, aliases: Iterable[str]) -> Optional[str]:
    lower = caption.casefold()
    for alias in sorted(aliases, key=lambda item: len(item), reverse=True):
        if alias and alias.casefold() in lower:
            return alias
    return None


def action_alternatives_for(
    kind: str,
    display: str,
    normalizations: dict[str, Any],
) -> list[str]:
    pool = list(DEFAULT_ACTION_ALTERNATIVES.get(kind, []))
    mapping_key = "shot_type_aliases" if kind == "shot_type" else kind
    mapping = normalizations.get(mapping_key, {})
    if isinstance(mapping, dict):
        pool.extend(str(value).strip() for value in mapping.values())
    current = canonicalize_categorical_text(display)
    return [
        phrase
        for phrase in _unique_action_phrases(pool)
        if canonicalize_categorical_text(phrase) != current
    ]


def indefinite_article(phrase: str) -> str:
    """Return ``a`` / ``an`` for the leading word of ``phrase``."""
    word = re.sub(r"[^a-z]", "", phrase.casefold().split(None, 1)[0] if phrase else "")
    return "an" if word[:1] in "aeiou" else "a"


def format_action_clause(attribute: "ActionAttribute", value: str) -> str:
    """Build a complete, grammatical action sentence for one structured field."""
    role = attribute.role
    kind = attribute.kind
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if kind == "shot_type":
        return (
            f"The {role} player's shot type was "
            f"{indefinite_article(cleaned)} {cleaned}."
        )
    if kind == "hand_used":
        return f"The {role} player's stroke side was {cleaned}."
    if kind == "shot_trajectory":
        return f"The {role} player's shot direction was {cleaned}."
    if kind == "lateral_movement":
        return f"The {role} player's lateral movement was {cleaned}."
    if kind == "depth_movement":
        return f"The {role} player's depth movement was {cleaned}."
    raise ValueError(f"Unsupported action kind: {kind}")


def append_action_clause(caption: str, clause: str) -> str:
    """Append ``clause`` to ``caption`` without mutating the original wording."""
    base = re.sub(r"\s+", " ", str(caption or "")).strip()
    clause = re.sub(r"\s+", " ", str(clause or "")).strip()
    if not clause:
        return base
    if base and base[-1] not in ".!?":
        base = f"{base}."
    if canonicalize_categorical_text(clause) in canonicalize_categorical_text(base):
        return base
    return f"{base} {clause}".strip() if base else clause


def inject_action_into_caption(caption: str, attribute: ActionAttribute) -> str:
    """Compatibility wrapper: append the controlled action sentence."""
    return append_action_clause(caption, format_action_clause(attribute, attribute.display))


def replace_action_phrase_in_caption(
    caption: str,
    attribute: ActionAttribute,
    replacement: str,
) -> str:
    """Compatibility wrapper: rebuild caption with a substituted action value.

    Always rebuilds from the original caption + a controlled sentence so mid-
    caption swaps (and article mismatches) cannot occur.
    """
    # Prefer the caption text that precedes any previously appended action clause.
    # Fall back to the full caption if no known clause marker is present.
    original = caption
    markers = (
        " The winning player's ",
        " The losing player's ",
    )
    for marker in markers:
        idx = original.find(marker)
        if idx > 0:
            original = original[:idx].rstrip()
            break
    return append_action_clause(
        original, format_action_clause(attribute, replacement)
    )


def collect_action_attributes(
    caption: str,
    field_values: dict[str, Any],
    normalizations: dict[str, Any],
    is_skip: Callable[[Any], bool],
    action_fields: Iterable[str],
) -> list[ActionAttribute]:
    attributes: list[ActionAttribute] = []
    for field_name in action_fields:
        kind = ACTION_FIELD_KIND.get(field_name)
        if not kind:
            continue
        raw = field_values.get(field_name)
        if is_skip(raw):
            continue
        mapping_key = "shot_type_aliases" if kind == "shot_type" else kind
        display = normalize_action_with_mapping(raw, normalizations.get(mapping_key))
        if is_skip(display) or not display:
            continue
        aliases = action_display_aliases(field_name, raw, display, normalizations)
        matched = find_action_alias_in_caption(caption, aliases)
        attributes.append(
            ActionAttribute(
                field_name=field_name,
                kind=kind,
                role=ACTION_ROLE[field_name],
                display=display,
                aliases=tuple(aliases),
                matched_alias=matched,
                in_caption=matched is not None,
            )
        )
    return attributes


def build_caption_action_mismatch(
    caption: str,
    field_values: dict[str, Any],
    normalizations: dict[str, Any],
    is_skip: Callable[[Any], bool],
    action_fields: Iterable[str],
    minimum_options: int = 2,
    maximum_options: int = 4,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return (correct_caption, options) or (None, None) when unusable.

    The original caption is preserved verbatim. A controlled sentence stating
    one verified structured action is appended for the correct option;
    distractors change only that sentence's value.
    """
    caption_text = clean_freetext_answer(caption)
    if not caption_text or is_skip(caption_text):
        return None, None

    attributes = collect_action_attributes(
        caption_text,
        field_values,
        normalizations,
        is_skip,
        action_fields,
    )
    if not attributes:
        return None, None

    # Prefer attributes already mentioned in the caption; otherwise use any
    # verified structured field. Pick the first attribute that has alternatives.
    preferred = [item for item in attributes if item.in_caption]
    fallback = [item for item in attributes if not item.in_caption]
    attribute: Optional[ActionAttribute] = None
    alternatives: list[str] = []
    for candidate in preferred + fallback:
        alts = action_alternatives_for(
            candidate.kind, candidate.display, normalizations
        )
        if alts:
            attribute = candidate
            alternatives = alts
            break
    if attribute is None:
        return None, None

    correct = append_action_clause(
        caption_text, format_action_clause(attribute, attribute.display)
    )
    options = [correct]
    for alt in alternatives:
        wrong = append_action_clause(
            caption_text, format_action_clause(attribute, alt)
        )
        if wrong == correct or wrong in options:
            continue
        options.append(wrong)
        if len(options) >= maximum_options:
            break

    if len(options) < minimum_options:
        return None, None
    return correct, options


@dataclass
class CategoricalPointData(PointData):
    """PointData plus categorical visual / movement fields."""

    winner_shot_type: str = ""
    loser_shot_type: str = ""
    winner_hand_used: str = ""
    loser_hand_used: str = ""
    winner_shot_trajectory: str = ""
    loser_shot_trajectory: str = ""
    server_location: str = ""
    receiver_location: str = ""
    loser_lateral_movement: str = ""
    loser_depth_movement: str = ""
    loser_point_ended: str = ""
    winner_shot_type_other: str = ""
    loser_shot_type_other: str = ""
    winner_hand_used_other: str = ""
    loser_hand_used_other: str = ""
    winner_shot_trajectory_other: str = ""
    loser_shot_trajectory_other: str = ""
    server_location_other: str = ""
    receiver_location_other: str = ""
    loser_lateral_movement_other: str = ""
    loser_depth_movement_other: str = ""
    loser_point_ended_other: str = ""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CategoricalPointData":
        base = PointData.from_json(data)
        extras = {name: str(data.get(name, "") or "") for name in EXTRA_POINT_FIELDS}
        return cls(**{**base.__dict__, **extras})


@dataclass
class CategoricalPointAnnotations:
    points: List[CategoricalPointData] = field(default_factory=list)

    @classmethod
    def from_json(cls, instances: List[Dict[str, Any]]) -> "CategoricalPointAnnotations":
        annotations = cls()
        for instance in instances:
            if instance.get("className") != "table":
                continue
            attributes = instance.get("attributes", [])
            if not attributes:
                continue
            table_data = attributes[0].get("name", [])
            if isinstance(table_data, list):
                for point_data in table_data:
                    annotations.points.append(CategoricalPointData.from_json(point_data))
        return annotations


@dataclass
class CategoricalTennisDataset(TennisDataset):
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CategoricalTennisDataset":
        metadata = Metadata.from_json(data.get("metadata", {}))
        instances = data.get("instances", [])
        dataset = cls(
            metadata=metadata,
            video_info=VideoInfo.from_json(instances),
            match_context=MatchContext.from_json(instances),
            point_annotations=CategoricalPointAnnotations.from_json(instances),
        )
        dataset._raw_instances = instances
        return dataset


class CategoricalMCQGenerator(ConfigDrivenMCQGenerator):
    """Generator for the standalone categorical-evaluation config."""

    def __init__(
        self,
        categorical_config_file: str,
        max_workers: int = 4,
        api_rate_limit: float = 1.0,
        enable_progress_bar: bool = True,  # Kept for caller compatibility.
        question_types: Optional[List[str]] = None,
        offline: bool = False,
        video_root: Optional[str] = None,
    ):
        _ = enable_progress_bar
        self.categorical_config = self.load_config(categorical_config_file)
        self.normalizations = self.categorical_config.get("normalizations", {})
        self.generation_policy = self.categorical_config.get("generation_policy", {})
        self.unavailable_markers = {
            str(marker).strip().casefold()
            for marker in self.generation_policy.get(
                "unavailable_value_markers", DEFAULT_UNAVAILABLE_MARKERS
            )
            if marker is not None and str(marker).strip() != ""
        }
        self.offline = offline
        self.video_root = os.path.abspath(video_root) if video_root else None
        self._skipped_missing_video_ids: set[str] = set()

        all_mcq = self.categorical_config.get("question_templates", {})
        all_qa = self.categorical_config.get("qa_templates", {})
        known = set(all_mcq) | set(all_qa)

        if question_types is None:
            allowed: Optional[Set[str]] = None
        else:
            allowed = set(question_types)
            unknown = allowed - known
            if unknown:
                logging.warning(f"Unknown question types (ignored): {sorted(unknown)}")
            allowed &= known

        self.question_templates = self._filter_templates(all_mcq, allowed)
        self.qa_templates = self._filter_templates(all_qa, allowed)
        self.observed_option_vocabularies: Dict[str, List[str]] = {}
        self.observed_raw_values: Dict[str, List[str]] = {}
        self.observed_unavailable_values: Dict[str, List[str]] = {}

        self.max_workers = max_workers
        self.api_rate_limit = api_rate_limit
        self._api_call_times: List[float] = []
        from threading import Lock

        self._api_call_lock = Lock()
        self._api_call_count = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._retry_attempts = 0
        self._total_api_time = 0.0
        # Annotation-triggered tennis-rules knowledge MCQs (optional configs).
        try:
            self.rules_knowledge_selector = build_default_selector(
                Path(__file__).resolve().parent
            )
        except FileNotFoundError:
            self.rules_knowledge_selector = None
            logging.warning(
                "Rules-knowledge configs not found; "
                "rules_knowledge_contextual templates will be skipped."
            )

    def build_context_data(
        self,
        point: CategoricalPointData,
        dataset: CategoricalTennisDataset,
        player_names: Dict[str, str],
    ) -> Dict[str, Any]:
        """Keep raw scores for derivation while canonicalizing displayed scores."""
        context_data = super().build_context_data(point, dataset, player_names)
        context_data["<raw_score_before>"] = point.score_before or ""
        context_data["<raw_score_after>"] = point.score_after or ""
        context_data["<score_before>"] = canonicalize_live_score_display(
            point.score_before
        )
        context_data["<score_after>"] = canonicalize_live_score_display(
            point.score_after
        )
        return context_data

    def normalize_value(self, value: Any, normalization: Optional[str]) -> str:
        text = str(value).strip()
        if not normalization:
            return text
        mapping = self.normalizations.get(normalization, {})
        if isinstance(mapping, dict):
            casefold_mapping = {
                canonicalize_categorical_text(key): canonicalize_categorical_text(mapped)
                for key, mapped in mapping.items()
            }
            canonical = canonicalize_categorical_text(text)
            if canonical in casefold_mapping:
                return casefold_mapping[canonical]
            if normalization == "ball_outcome":
                own_side = self._normalize_ball_outcome_own_side(canonical)
                if own_side is not None:
                    return own_side
            return canonical
        return canonicalize_categorical_text(text)

    def _normalize_ball_outcome_own_side(self, canonical: str) -> Optional[str]:
        """Map free-text own-side endings onto the dedicated ball-outcome option."""
        include = [
            canonicalize_categorical_text(marker)
            for marker in self.normalizations.get("ball_outcome_own_side_markers", [])
        ]
        exclude = [
            canonicalize_categorical_text(marker)
            for marker in self.normalizations.get(
                "ball_outcome_own_side_exclude_markers", []
            )
        ]
        if not include:
            return None
        if any(marker and marker in canonical for marker in exclude):
            return None
        if any(marker and marker in canonical for marker in include):
            return canonicalize_categorical_text(
                "landed on the losing player's side of the court"
            )
        return None

    def is_skip_value(self, value: Any) -> bool:
        """True for empty values or values matching configured unavailable markers."""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, str):
            return value.strip().casefold() in self.unavailable_markers
        return False

    def annotation_value(self, source: Any, field_name: str) -> Any:
        """Resolve an annotation value, including Data Factory's `other` field."""
        if isinstance(source, dict):
            value = source.get(field_name)
            other_value = source.get(f"{field_name}_other")
        else:
            value = getattr(source, field_name, None)
            other_value = getattr(source, f"{field_name}_other", None)
        if isinstance(value, str) and value.strip().casefold() == "other":
            return other_value.strip() if isinstance(other_value, str) else other_value
        return value

    def prescreen_observed_option_vocabularies(
        self, json_files: List[str]
    ) -> None:
        """Build option vocabularies from observed annotation values only."""
        dynamic_templates = {
            name: template
            for name, template in self.question_templates.items()
            if template.get("options_from_observed_fields")
        }
        raw_by_field: Dict[str, List[Any]] = {}
        unavailable_by_field: Dict[str, List[Any]] = {}
        counts = {name: Counter() for name in dynamic_templates}

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for instance in data.get("instances", []):
                if instance.get("className") != "table":
                    continue
                attributes = instance.get("attributes", [])
                if not attributes:
                    continue
                rows = attributes[0].get("name", [])
                if not isinstance(rows, list):
                    continue
                for point in rows:
                    for name, template in dynamic_templates.items():
                        for field_name in template["options_from_observed_fields"]:
                            if field_name not in point:
                                continue
                            value = self.annotation_value(point, field_name)
                            field_raw = raw_by_field.setdefault(field_name, [])
                            if value not in field_raw:
                                field_raw.append(value)
                            if self.is_skip_value(value):
                                field_unavailable = unavailable_by_field.setdefault(
                                    field_name, []
                                )
                                if value not in field_unavailable:
                                    field_unavailable.append(value)
                                continue
                            normalized = self.normalize_value(
                                value, template.get("answer_normalization")
                            )
                            if not self.is_skip_value(normalized):
                                counts[name][normalized] += 1

        default_minimum = int(
            self.generation_policy.get("minimum_observed_option_count", 1)
        )
        vocabularies = {}
        for name, template in dynamic_templates.items():
            minimum = int(
                template.get("minimum_observed_option_count", default_minimum)
            )
            vocabularies[name] = [
                value
                for value, count in counts[name].most_common()
                if count >= minimum
            ]

        self.observed_raw_values = {
            field_name: [str(value) for value in values]
            for field_name, values in raw_by_field.items()
        }
        self.observed_unavailable_values = {
            field_name: [str(value) for value in values]
            for field_name, values in unavailable_by_field.items()
        }
        self.observed_option_vocabularies = vocabularies

        for name, values in vocabularies.items():
            minimum = int(
                dynamic_templates[name].get(
                    "minimum_observed_option_count", default_minimum
                )
            )
            print(
                f"Prescreened {len(values)} lowercase option values observed "
                f"at least {minimum} times for {name}: {values}"
            )

    def outcome_label(self, point: CategoricalPointData) -> str:
        return str(point.how_point_ended or "").split(":", 1)[0].strip()

    def audio_groups_for_point(self, point: CategoricalPointData) -> List[str]:
        cue_text = str(point.audio_cues or "").lower()
        groups = []
        for group, terms in self.normalizations.get("audio_groups", {}).items():
            if any(term.lower() in cue_text for term in terms):
                groups.append(group)
        return groups

    def player_name_for_role(
        self, point: CategoricalPointData, role: str, context_data: Dict[str, Any]
    ) -> str:
        value = getattr(point, role, "")
        if value == "Player 1":
            return context_data.get("<player1>", "Player 1")
        if value == "Player 2":
            return context_data.get("<player2>", "Player 2")
        return str(value)

    def winning_player_name(
        self, point: CategoricalPointData, context_data: Dict[str, Any]
    ) -> str:
        if point.winner == "Server":
            return self.player_name_for_role(point, "server", context_data)
        if point.winner == "Receiver":
            return self.player_name_for_role(point, "receiver", context_data)
        return str(point.winner)

    def resolve_field_value(
        self,
        point: CategoricalPointData,
        template: Dict[str, Any],
        context_data: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[List[str]]]:
        """Return (field_value, optional_prebuilt_options)."""
        template_type = template.get("type", "categorical")
        field_name = template["field"]
        depends_on = template.get("depends_on") or [field_name]

        action_fields = template.get("action_fields")
        if action_fields:
            minimum = int(template.get("minimum_non_null_action_fields", 1))
            present = sum(
                1
                for name in action_fields
                if not self.is_skip_value(self.annotation_value(point, name))
            )
            if present < minimum:
                return None, None

        if template_type in {"derived_categorical", "derived_open_ended"}:
            for name in depends_on:
                if not hasattr(point, name):
                    continue
                if self.is_skip_value(self.annotation_value(point, name)):
                    # Audio presence / event-set templates still run on empty cues.
                    if template.get("derivation") in {
                        "audio_group_presence",
                    }:
                        continue
                    return None, None
            return self.derive_value(template, point, context_data)

        # Composite roles used by legacy-compatible templates.
        if field_name == "server_winner":
            server = point.server
            receiver = point.receiver
            winner = point.winner
            if (
                not server
                or not receiver
                or server == receiver
                or winner not in ("Server", "Receiver")
            ):
                return None, None
            server_player = self.player_name_for_role(point, "server", context_data)
            receiver_player = self.player_name_for_role(point, "receiver", context_data)
            winner_player = server_player if winner == "Server" else receiver_player
            return f"{server_player} served and {winner_player} won", None

        if field_name == "receiver_winner":
            server = point.server
            receiver = point.receiver
            winner = point.winner
            if (
                not server
                or not receiver
                or server == receiver
                or winner not in ("Server", "Receiver")
            ):
                return None, None
            server_player = self.player_name_for_role(point, "server", context_data)
            receiver_player = self.player_name_for_role(point, "receiver", context_data)
            winner_player = server_player if winner == "Server" else receiver_player
            return f"{receiver_player} received and {winner_player} won", None

        raw = self.annotation_value(point, field_name)
        if self.is_skip_value(raw):
            return None, None

        if template.get("answer_transform") == "leading_outcome_label":
            return self.outcome_label(point), None

        if template.get("answer_normalization"):
            return self.normalize_value(raw, template["answer_normalization"]), None

        if field_name == "num_serving_attempts_until_successful":
            attempts = coerce_positive_int(raw)
            if attempts is None:
                return None, None
            return str(attempts), None

        return raw, None

    def derive_value(
        self,
        template: Dict[str, Any],
        point: CategoricalPointData,
        context_data: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[List[str]]]:
        derivation = template.get("derivation")
        if derivation == "server_receiver_location":
            server = str(point.server_location).lower()
            receiver = str(point.receiver_location).lower()
            if "near" in server and "far" in receiver:
                answer = "server near-court; receiver far-court"
            elif "far" in server and "near" in receiver:
                answer = "server far-court; receiver near-court"
            elif "near" in server and "near" in receiver:
                answer = "both near-court"
            elif "far" in server and "far" in receiver:
                answer = "both far-court"
            else:
                answer = f"server {server}; receiver {receiver}"
            return answer, None

        if derivation == "combined_movement":
            raw_lateral = self.annotation_value(point, "loser_lateral_movement")
            raw_depth = self.annotation_value(point, "loser_depth_movement")
            lateral = self.normalize_value(raw_lateral, "lateral_movement")
            depth = self.normalize_value(raw_depth, "depth_movement")
            if self.is_skip_value(raw_lateral) or self.is_skip_value(raw_depth):
                return None, None
            return f"{lateral}; {depth}", None

        if derivation == "score_transition":
            if self.is_skip_value(point.score_before) or self.is_skip_value(point.score_after):
                return None, None
            winner_role = str(point.winner).strip().casefold()
            if winner_role not in {"server", "receiver"}:
                return None, None
            loser_role = "receiver" if winner_role == "server" else "server"
            winner_name = self.winning_player_name(point, context_data)
            loser_name = self.player_name_for_role(point, loser_role, context_data)
            # Distinct player names are required: otherwise the winner/loser and
            # role-swap options collapse to the same text once deduplicated.
            if (
                not str(winner_name).strip()
                or not str(loser_name).strip()
                or option_dedupe_key(winner_name) == option_dedupe_key(loser_name)
            ):
                return None, None
            wrong_next_score = alternate_next_tennis_score(
                point.score_before, point.score_after
            )
            if score_display_key(wrong_next_score) == score_display_key(
                point.score_before
            ):
                return None, None
            score_after_display = canonicalize_live_score_display(point.score_after)
            wrong_next_score_display = canonicalize_live_score_display(
                wrong_next_score
            )
            answer = (
                f"{winner_name} ({winner_role}) won the point; "
                f"new score: {score_after_display}"
            )
            options = [
                answer,
                (
                    f"{loser_name} ({loser_role}) won the point; "
                    f"new score: {score_after_display}"
                ),
                (
                    f"{winner_name} ({loser_role}) won the point; "
                    f"new score: {score_after_display}"
                ),
                (
                    f"{winner_name} ({winner_role}) won the point; "
                    f"new score: {wrong_next_score_display}"
                ),
            ]
            return answer, options

        if derivation == "serve_sequence_outcome":
            outcome = self.outcome_label(point)
            attempts = point.num_serving_attempts_until_successful
            return build_serve_sequence_options(
                attempts,
                outcome,
                maximum_options=int(
                    self.generation_policy.get("maximum_mcq_options", 4)
                ),
            )

        if derivation == "serve_rule_consistency":
            outcome = self.outcome_label(point)
            attempts = point.num_serving_attempts_until_successful
            return build_serve_rule_consistency(attempts, outcome)

        if derivation == "grounded_serve_rule_consistency":
            outcome = self.outcome_label(point)
            attempts = point.num_serving_attempts_until_successful
            return build_grounded_serve_rule_consistency(attempts, outcome)

        if derivation == "point_outcome_reasoning":
            return build_point_outcome_reasoning(
                point.winner,
                self.outcome_label(point),
            )

        if derivation == "rules_knowledge_contextual":
            selector = getattr(self, "rules_knowledge_selector", None)
            if selector is None:
                return None, None
            selected = selector.select_for_point(
                point,
                selection_rank=int(template.get("selection_rank", 0)),
            )
            if selected is None:
                return None, None
            context_data["<rules_question_stem>"] = selected.question
            context_data["<rules_question_id>"] = str(selected.question_id)
            context_data["<rules_question_name>"] = selected.name
            return selected.correct_answer, list(selected.options)

        if derivation == "audio_group_presence":
            group = template["audio_group"]
            answer = "Yes" if group in self.audio_groups_for_point(point) else "No"
            return answer, None

        if derivation == "caption_action_mismatch":
            action_fields = template.get("action_fields") or []
            field_values = {
                name: self.annotation_value(point, name) for name in action_fields
            }
            return build_caption_action_mismatch(
                caption=point.point_caption,
                field_values=field_values,
                normalizations=self.normalizations,
                is_skip=self.is_skip_value,
                action_fields=action_fields,
                minimum_options=int(
                    self.generation_policy.get("minimum_mcq_options", 2)
                ),
                maximum_options=int(
                    self.generation_policy.get("maximum_mcq_options", 4)
                ),
            )

        return None, None

    def filter_unavailable_options(self, options: List[str]) -> List[str]:
        """Drop empty/unavailable markers so they never appear as MCQ choices."""
        filtered: List[str] = []
        seen: Set[str] = set()
        for option in options:
            if option is None:
                continue
            text = normalize_option_text(option)
            if not text or self.is_skip_value(text):
                continue
            key = option_dedupe_key(text)
            if key in seen:
                continue
            filtered.append(text)
            seen.add(key)
        return filtered

    def format_mcq(
        self, question: str, correct_answer: str, options: List[str]
    ) -> Tuple[str, str, List[str]]:
        correct_answer = normalize_option_text(correct_answer)
        if self.is_skip_value(correct_answer):
            return None, None, []
        options = self.filter_unavailable_options(options)
        # Prefer vocabulary spelling when the answer is only a whitespace/quote
        # variant of a configured option (e.g. annotation "  1" vs option "1").
        correct_key = option_dedupe_key(correct_answer)
        for option in options:
            if option_dedupe_key(option) == correct_key:
                correct_answer = option
                break
        distractors = [
            option
            for option in options
            if option_dedupe_key(option) != correct_key
        ]
        max_options = int(self.generation_policy.get("maximum_mcq_options", 4))
        if len(distractors) > max_options - 1:
            distractors = random.sample(distractors, max_options - 1)
        unique = unique_mcq_options(
            correct_answer, distractors, max_options=max_options
        )
        minimum_options = int(self.generation_policy.get("minimum_mcq_options", 2))
        if len(unique) < minimum_options:
            return None, None, []
        random.shuffle(unique)
        labels = ["(A)", "(B)", "(C)", "(D)"]
        parts = [question]
        labeled_answer = ""
        for i, option in enumerate(unique):
            if i >= len(labels):
                break
            parts.append(f"{labels[i]} {option}")
            if option_dedupe_key(option) == correct_key:
                labeled_answer = f"{labels[i]} {option}"
        return normalize_question_text("\n".join(parts)), labeled_answer, unique

    def generate_question_answer(
        self,
        field_value: Any,
        template: Dict[str, Any],
        question_type: str,
        context_data: Dict[str, Any],
        prebuilt_options: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        question = self.replace_placeholders(template["question"], context_data)
        template_type = template.get("type", "categorical")

        if template_type == "llm_gen_distractor":
            if self.offline:
                return None, None, []
            try:
                metadata = context_data.copy()
                metadata["field_value"] = str(field_value) if field_value else ""
                generated_question, correct_answer, distractors = self._rate_limited_api_call(
                    generate_tennis_mcq_with_llm,
                    metadata=metadata,
                    field_value=str(field_value) if field_value else "",
                    question_template=question,
                    answer_prompt=template.get("llm_answer_prompt"),
                    distractor_prompt=template.get("llm_question_prompt"),
                )
                return self.format_mcq(
                    generated_question, correct_answer, [correct_answer] + distractors
                )
            except Exception as exc:
                logging.error(f"LLM MCQ failed for {question_type}: {exc}")
                return None, None, []

        if template_type == "freeform_with_distractors":
            raw_correct_answer = (
                normalize_option_text(field_value) if field_value else ""
            )
            if question_type != "score_after" or not raw_correct_answer:
                return None, None, []
            distractors = self.generate_score_distractors(
                context_data.get(
                    "<raw_score_before>",
                    context_data.get("<score_before>", ""),
                ),
                raw_correct_answer,
                context_data,
            )
            correct_answer = canonicalize_live_score_display(raw_correct_answer)
            display_options = [
                canonicalize_live_score_display(option)
                for option in [raw_correct_answer, *distractors]
            ]
            return self.format_mcq(
                question,
                correct_answer,
                display_options,
            )

        if template_type == "numeric_range":
            if not isinstance(field_value, (int, float)):
                try:
                    field_value = int(field_value)
                except (TypeError, ValueError):
                    return None, None, []
            correct_answer = None
            for range_config in template.get("ranges", []):
                if range_config["min"] <= field_value <= range_config["max"]:
                    correct_answer = range_config["answer"]
                    break
            if not correct_answer:
                return None, None, []
            options = template.get("options", [])
            return self.format_mcq(question, correct_answer, options)

        if template_type in {"categorical", "derived_categorical"}:
            correct_answer = (
                normalize_option_text(field_value) if field_value is not None else ""
            )
            correct_answer = normalize_option_text(
                self.replace_placeholders(correct_answer, context_data)
            )
            answer_normalization = template.get("answer_normalization")
            if answer_normalization:
                # Apply the option canonicalization at the final presentation
                # boundary as well. This keeps rare/raw answers that were not
                # part of the prescreened distractor vocabulary normalized.
                correct_answer = self.normalize_value(
                    correct_answer, answer_normalization
                )
            if not correct_answer.strip():
                return None, None, []
            if prebuilt_options is not None:
                options = [
                    self.replace_placeholders(opt, context_data) for opt in prebuilt_options
                ]
            else:
                configured_options = self.observed_option_vocabularies.get(
                    question_type
                )
                options = [
                    self.replace_placeholders(opt, context_data)
                    for opt in (
                        configured_options
                        if configured_options is not None
                        else template.get("options", [])
                    )
                ]
            if answer_normalization:
                options = [
                    self.normalize_value(option, answer_normalization)
                    for option in options
                ]
            options = self.filter_unavailable_options(options)
            if template.get("require_answer_in_options"):
                answer_keys = {option_dedupe_key(option) for option in options}
                if option_dedupe_key(correct_answer) not in answer_keys:
                    return None, None, []
            if self.is_skip_value(correct_answer):
                return None, None, []
            none_option = template.get("optional_none_option")
            if none_option:
                none_option = normalize_option_text(
                    self.replace_placeholders(none_option, context_data)
                )
                # The frequency threshold filters distractors only. A rare raw
                # answer remains the ground truth and format_mcq inserts it.
                if template.get("force_none_correct"):
                    max_options = int(
                        self.generation_policy.get("maximum_mcq_options", 4)
                    )
                    regular_distractors = [
                        option
                        for option in options
                        if option not in {correct_answer, none_option}
                    ]
                    random.shuffle(regular_distractors)
                    if not regular_distractors:
                        return None, None, []
                    correct_answer = none_option
                    options = [
                        *regular_distractors[: max(0, max_options - 1)],
                        none_option,
                    ]
                elif random.random() < float(
                    template.get("optional_none_probability", 0.0)
                ):
                    max_options = int(
                        self.generation_policy.get("maximum_mcq_options", 4)
                    )
                    regular_distractors = [
                        option
                        for option in options
                        if option not in {correct_answer, none_option}
                    ]
                    random.shuffle(regular_distractors)
                    if regular_distractors and random.random() < float(
                        template.get("none_option_correct_probability", 0.0)
                    ):
                        correct_answer = none_option
                        options = [
                            *regular_distractors[: max(0, max_options - 1)],
                            none_option,
                        ]
                    else:
                        options = [
                            correct_answer,
                            *regular_distractors[: max(0, max_options - 2)],
                            none_option,
                        ]
            return self.format_mcq(question, correct_answer, options)

        return None, None, []

    def generate_qa_question_answer(
        self,
        field_value: Any,
        template: Dict[str, Any],
        question_type: str,
        context_data: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        question = normalize_question_text(
            self.replace_placeholders(template["question"], context_data)
        )
        template_type = template.get("type", "open_ended")
        if template_type not in {"open_ended", "derived_open_ended"}:
            return None, None
        answer = str(field_value) if field_value is not None else ""
        answer = self.replace_placeholders(answer, context_data)
        # Tidy annotation noise (leading segment tag, wrapping quotes/backticks,
        # collapsed whitespace) without disturbing wording or punctuation.
        answer = clean_freetext_answer(answer)
        # Drop non-answer placeholder values (e.g. "n/a") so they never ship.
        if is_empty_answer_value(answer):
            return None, None
        min_length = template.get("min_length", 0)
        max_length = template.get("max_length", 2000)
        if len(answer) < min_length:
            return None, None
        if len(answer) > max_length:
            answer = answer[:max_length]
        return question, answer

    def generate_qa_from_dataset(
        self, dataset: CategoricalTennisDataset, entry_id: str
    ) -> List[Dict[str, Any]]:
        qa_pairs: List[Dict[str, Any]] = []
        if not dataset.point_annotations.points:
            return qa_pairs
        player_names = self.extract_player_names(dataset)
        for point in dataset.point_annotations.points:
            if not self.has_valid_video(point):
                self._skipped_missing_video_ids.add(str(point.id))
                continue
            context_data = self.build_context_data(point, dataset, player_names)
            for question_type, template in self.qa_templates.items():
                field_value, _ = self.resolve_field_value(
                    point, template, context_data
                )
                if field_value is None:
                    continue
                question, answer = self.generate_qa_question_answer(
                    field_value, template, question_type, context_data
                )
                if question and answer:
                    qa_pairs.append(
                        {
                            "question_type": question_type,
                            "question": question,
                            "answer": answer,
                            "file_path": entry_id,
                            "point_data": {"id": point.id, "video": point.video},
                        }
                    )
        return qa_pairs

    def generate_mcq_from_dataset_threaded(
        self, dataset: CategoricalTennisDataset, entry_id: str
    ) -> List[Dict[str, Any]]:
        if not dataset.point_annotations.points:
            return []
        player_names = self.extract_player_names(dataset)
        tasks = []
        total_points = len(dataset.point_annotations.points)
        valid_points = sum(1 for p in dataset.point_annotations.points if self.has_valid_video(p))
        if self.video_root:
            print(
                f"  Points with existing clip: {valid_points}/"
                f"{total_points} (skipping {total_points - valid_points})"
            )
        for point in dataset.point_annotations.points:
            if not self.has_valid_video(point):
                self._skipped_missing_video_ids.add(str(point.id))
                continue
            context_data = self.build_context_data(point, dataset, player_names)
            for question_type, template in self.question_templates.items():
                generation_probability = float(
                    template.get("generation_probability", 1.0)
                )
                if generation_probability < 1.0 and random.random() >= generation_probability:
                    continue
                # Per-template copy: some derivations (e.g. rules_knowledge) write
                # placeholders into context_data; sharing one dict across templates
                # would let a later resolve overwrite an earlier stem.
                template_context = dict(context_data)
                field_value, prebuilt_options = self.resolve_field_value(
                    point, template, template_context
                )
                if field_value is None or field_value == "":
                    continue
                # For numeric_range, keep numeric field value for range matching.
                if template.get("type") == "numeric_range" and template["field"] == "num_shots_exchanged":
                    field_value = point.num_shots_exchanged
                tasks.append(
                    (
                        point,
                        template,
                        question_type,
                        template_context,
                        field_value,
                        prebuilt_options,
                        entry_id,
                    )
                )

        mcq_pairs: List[Dict[str, Any]] = []
        if not tasks:
            return mcq_pairs

        def _run(task):
            point, template, question_type, context_data, field_value, prebuilt, entry = task
            try:
                question, correct_answer, _ = self.generate_question_answer(
                    field_value,
                    template,
                    question_type,
                    context_data,
                    prebuilt_options=prebuilt,
                )
                if question and correct_answer:
                    emit_type = template.get("hf_class") or question_type
                    return {
                        "question_type": emit_type,
                        "id_question_type": question_type,
                        "question": question,
                        "correct_answer": correct_answer,
                        "file_path": entry,
                        "point_data": {"id": point.id, "video": point.video},
                        "super_category": template.get("super_category"),
                        "fine_category": template.get("fine_category"),
                    }
            except Exception as exc:
                logging.error(f"MCQ failed for {question_type}: {exc}")
            return None

        workers = 1 if self.offline else self.max_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    mcq_pairs.append(result)
        return mcq_pairs

    def process_data(
        self,
        input_dir: str,
        per_video_output_dir: str = None,
        video_ids: Optional[List[str]] = None,
    ) -> None:
        print("Tennis Categorical MCQ/QA Generator")
        print("=" * 60)
        input_dir_abs = os.path.abspath(input_dir)
        per_video_dir = (
            os.path.abspath(per_video_output_dir)
            if per_video_output_dir
            else os.path.join(os.path.dirname(input_dir_abs), "mcq_qa_per_video")
        )
        os.makedirs(per_video_dir, exist_ok=True)
        all_json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
        if not all_json_files:
            raise ValueError(f"No JSON files found in {input_dir}")
        # Prescreen the complete corpus so a small test run uses the same option
        # vocabularies that full generation will use.
        self.prescreen_observed_option_vocabularies(all_json_files)
        json_files = all_json_files
        if video_ids:
            requested = set(video_ids)
            json_files = [
                path
                for path in all_json_files
                if os.path.splitext(os.path.basename(path))[0] in requested
            ]
            found = {
                os.path.splitext(os.path.basename(path))[0] for path in json_files
            }
            missing = sorted(requested - found)
            if missing:
                raise ValueError(f"Requested video IDs not found: {', '.join(missing)}")

        print(f"Found {len(json_files)} files")
        print(f"MCQ templates: {len(self.question_templates)}")
        print(f"QA templates: {len(self.qa_templates)}")
        print(f"Offline (skip LLM): {self.offline}")
        print(f"Video root: {self.video_root or '(unset; path-string only)'}")
        print(f"Output: {per_video_dir}")

        processed = failed = skipped = 0
        all_mcq: List[Dict[str, Any]] = []
        all_qa: List[Dict[str, Any]] = []

        for idx, json_file in enumerate(json_files, 1):
            video_name = os.path.basename(json_file).replace(".json", "")
            video_output_file = os.path.join(per_video_dir, f"{video_name}_derived_mcq_qa.json")
            if os.path.exists(video_output_file):
                skipped += 1
                continue
            try:
                print(f"\n[{idx}/{len(json_files)}] {os.path.basename(json_file)}")
                with open(json_file, "r", encoding="utf-8") as handle:
                    entry_data = json.load(handle)
                if "instances" not in entry_data:
                    failed += 1
                    continue
                dataset = CategoricalTennisDataset.from_json(entry_data)
                if not dataset.point_annotations.points:
                    failed += 1
                    continue
                mcq_pairs = self.generate_mcq_from_dataset_threaded(dataset, json_file)
                qa_pairs = self.generate_qa_from_dataset(dataset, json_file)
                if not mcq_pairs and not qa_pairs:
                    failed += 1
                    continue
                conversations = self.convert_to_conversation_format(mcq_pairs, qa_pairs)
                with open(video_output_file, "w", encoding="utf-8") as handle:
                    json.dump(conversations, handle, indent=2, ensure_ascii=False)
                print(f"  MCQ={len(mcq_pairs)} QA={len(qa_pairs)} -> {os.path.basename(video_output_file)}")
                all_mcq.extend(mcq_pairs)
                all_qa.extend(qa_pairs)
                processed += 1
            except Exception as exc:
                logging.error(f"Failed on {json_file}: {exc}")
                failed += 1

        print("\n" + "=" * 60)
        print(f"Processed={processed} skipped={skipped} failed={failed}")
        print(f"Points skipped for missing clips: {len(self._skipped_missing_video_ids)}")
        print(f"New MCQ={len(all_mcq)} New QA={len(all_qa)}")
        if all_mcq or all_qa:
            self.print_statistics(all_mcq, all_qa)
        self.print_api_statistics()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generation/tennis_categorical_eval_config.json",
        help="Path to categorical evaluation config",
    )
    parser.add_argument("--input-dir", default="raw_tennis_data_input")
    parser.add_argument(
        "--output-dir",
        default="training_tennis_data_output/categorical_mcq_qa_per_video",
    )
    parser.add_argument("--question-types", nargs="+")
    parser.add_argument("--question-types-config")
    parser.add_argument(
        "--video-ids",
        nargs="+",
        help="Generate only these video IDs (useful for focused test runs)",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help=(
            "Directory used to resolve relative point clip paths. Points whose "
            "clips are missing/empty are skipped. Pass empty string to disable."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--api-rate-limit", type=float, default=0.01)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip llm_gen_distractor templates (no API calls)",
    )
    args = parser.parse_args()

    question_types = None
    if args.question_types_config:
        question_types = load_question_types(args.question_types_config)
    elif args.question_types:
        question_types = args.question_types

    video_root = args.video_root or None
    generator = CategoricalMCQGenerator(
        categorical_config_file=args.config,
        max_workers=args.max_workers,
        api_rate_limit=args.api_rate_limit,
        enable_progress_bar=True,
        question_types=question_types,
        offline=args.offline,
        video_root=video_root,
    )
    generator.process_data(args.input_dir, args.output_dir, video_ids=args.video_ids)


if __name__ == "__main__":
    main()
