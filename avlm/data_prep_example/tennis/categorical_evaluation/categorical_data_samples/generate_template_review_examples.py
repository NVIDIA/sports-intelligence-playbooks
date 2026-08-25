#!/usr/bin/env python3
"""Generate one review example per categorical-evaluation template.

Writes review artifacts into this directory
(``categorical_evaluation/categorical_data_samples/``).

The latest generated sample is the preferred example source. Templates omitted
from that sample are documented without generated examples.
"""

from __future__ import annotations

import json
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

TENNIS_DIR_BOOTSTRAP = Path(__file__).resolve().parent.parent.parent
if str(TENNIS_DIR_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(TENNIS_DIR_BOOTSTRAP))

from generate_categorical_mcq_qa import (
    build_caption_action_mismatch,
    build_grounded_serve_rule_consistency,
    build_point_outcome_reasoning,
    build_serve_rule_consistency,
    build_serve_sequence_options,
)


def md_code_span(text: str) -> str:
    """Wrap text in a Markdown code span so preview keeps ``<placeholder>`` literal.

    HTML-entity escaping is not enough: Cursor/VS Code preview decodes
    ``&lt;player1&gt;`` back to ``<player1>`` and then strips it as an HTML tag.
    """
    value = str(text).replace("\n", " ")
    fence_len = max((len(match) for match in re.findall(r"`+", value)), default=0) + 1
    fence = "`" * fence_len
    return f"{fence}{value}{fence}"


TYPE_OPTION_SPACE = {
    "categorical": "closed",
    "derived_categorical": "closed",
    "numeric_range": "closed",
    "llm_gen_distractor": "open",
    "freeform_with_distractors": "open",
    "open_ended": "open",
    "derived_open_ended": "open",
}

TYPE_DISTRACTOR_GENERATION = {
    "categorical": "fixed",
    "derived_categorical": "derived",
    "numeric_range": "fixed",
    "llm_gen_distractor": "llm",
    "freeform_with_distractors": "heuristic",
    "open_ended": "none",
    "derived_open_ended": "none",
}


def infer_option_space(implementation_type: str) -> str:
    return TYPE_OPTION_SPACE.get(implementation_type, "unknown")


def infer_distractor_generation(implementation_type: str) -> str:
    return TYPE_DISTRACTOR_GENERATION.get(implementation_type, "unknown")


SAMPLES_DIR = Path(__file__).resolve().parent
TENNIS_DIR = SAMPLES_DIR.parent.parent
CONFIG_PATH = (
    TENNIS_DIR
    / "configs"
    / "generation"
    / "tennis_categorical_eval_config.json"
)
RAW_DIR = TENNIS_DIR / "raw_tennis_data_input"
AGGREGATED_DIR = TENNIS_DIR / "aggregated_data_complete"
GENERATED_DIR = (
    TENNIS_DIR
    / "training_tennis_data_output"
    / "categorical_mcq_qa_sample"
)
JSON_OUTPUT = SAMPLES_DIR / "template_review_examples.json"
MD_OUTPUT = SAMPLES_DIR / "TEMPLATE_REVIEW.MD"
DEFAULT_UNAVAILABLE_MARKERS = ["n/a", "unknown", "unknow"]


def unavailable_markers(config: dict[str, Any]) -> set[str]:
    markers = config.get("generation_policy", {}).get(
        "unavailable_value_markers", DEFAULT_UNAVAILABLE_MARKERS
    )
    return {
        str(marker).strip().casefold()
        for marker in markers
        if marker is not None and str(marker).strip() != ""
    }


def is_skip_value(value: Any, markers: set[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, str):
        return value.strip().casefold() in markers
    return False


def annotation_value(point: dict[str, Any], field_name: str) -> Any:
    """Resolve Data Factory's `other` sentinel to its free-text companion."""
    value = point.get(field_name)
    if isinstance(value, str) and value.strip().casefold() == "other":
        other_value = point.get(f"{field_name}_other")
        return other_value.strip() if isinstance(other_value, str) else other_value
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def instance_value(data: dict[str, Any], class_name: str) -> str:
    """Return the first non-empty attribute value for ``class_name``."""
    for instance in data.get("instances", []):
        if instance.get("className") != class_name:
            continue
        attributes = instance.get("attributes", [])
        if not attributes:
            continue
        value = str(attributes[0].get("name", "") or "").strip()
        if value:
            return value
    return ""


_SCORE_PLAYER_RE = re.compile(
    r'^\s*[\'"]?(?P<p1>.+?)[\'"]?\s+vs\.?\s+[\'"]?(?P<p2>.+?)[\'"]?\s*[:，,]',
    re.IGNORECASE,
)


def player_names_from_scores(data: dict[str, Any]) -> tuple[str, str]:
    """Fallback: parse ``PlayerA vs. PlayerB`` from score strings."""
    for instance in data.get("instances", []):
        if instance.get("className") != "table":
            continue
        attributes = instance.get("attributes", [])
        if not attributes:
            continue
        rows = attributes[0].get("name", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            for key in ("score_before", "score_after"):
                match = _SCORE_PLAYER_RE.match(str(row.get(key) or ""))
                if match:
                    return match.group("p1").strip(), match.group("p2").strip()
    return "", ""


def collect_points(data: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    player1 = instance_value(data, "player_1_name")
    player2 = instance_value(data, "player_2_name")
    if not player1 or not player2:
        score_p1, score_p2 = player_names_from_scores(data)
        player1 = player1 or score_p1 or "Player 1"
        player2 = player2 or score_p2 or "Player 2"
    else:
        player1 = player1 or "Player 1"
        player2 = player2 or "Player 2"
    player1_description = instance_value(data, "player_1_description")
    player2_description = instance_value(data, "player_2_description")
    points: list[dict[str, Any]] = []
    for instance in data.get("instances", []):
        if instance.get("className") != "table":
            continue
        attributes = instance.get("attributes", [])
        if not attributes:
            continue
        rows = attributes[0].get("name", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            point = dict(row)
            point["_video_id"] = video_id
            point["_player1"] = player1
            point["_player2"] = player2
            point["_player1_description"] = player1_description
            point["_player2_description"] = player2_description
            points.append(point)
    return points


def load_points(directory: Path = RAW_DIR) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "build_sample_from_raw.py":
            continue
        points.extend(collect_points(load_json(path), path.stem))
    return points


def load_generated_examples() -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(GENERATED_DIR.glob("*_derived_mcq_qa.json")):
        for record in load_json(path):
            question = next(
                (
                    item.get("text", "")
                    for item in record["conversation"][0]["content"]
                    if item.get("type") == "text"
                ),
                "",
            )
            video_path = next(
                (
                    item.get("path", "")
                    for item in record["conversation"][0]["content"]
                    if item.get("type") == "video"
                ),
                "",
            )
            answer = record["conversation"][1]["content"][0].get("text", "")
            question_lines = question.splitlines()
            option_start = next(
                (
                    index
                    for index, line in enumerate(question_lines)
                    if re.match(r"^\s*\([A-Z]\)\s+", line)
                ),
                None,
            )
            options = []
            if option_start is not None:
                options = [
                    re.sub(r"^\s*\([A-Z]\)\s+", "", line).strip()
                    for line in question_lines[option_start:]
                    if re.match(r"^\s*\([A-Z]\)\s+", line)
                ]
                question = "\n".join(question_lines[:option_start]).strip()
            examples[record["class"]].append(
                {
                    "source_video": video_path.split("/", 1)[0] if video_path else path.stem,
                    "point_id": record["id"],
                    "question": question,
                    "answer": answer,
                    **({"options": options} if options else {}),
                }
            )
    return examples


def normalize(value: Any, normalization: str | None, config: dict[str, Any]) -> str:
    # Strip surrounding markdown backticks from noisy annotations (e.g. "`1").
    text = str(value).strip().strip("`").strip()
    if not normalization:
        return text

    def canonicalize(item: Any) -> str:
        normalized = re.sub(r"\s+", " ", str(item)).strip().casefold()
        return normalized.rstrip(".").strip()

    mapping = config["normalizations"].get(normalization, {})
    casefold_mapping = {
        canonicalize(key): canonicalize(mapped)
        for key, mapped in mapping.items()
    }
    canonical = canonicalize(text)
    if canonical in casefold_mapping:
        return casefold_mapping[canonical]
    if normalization == "ball_outcome":
        include = [
            canonicalize(marker)
            for marker in config["normalizations"].get(
                "ball_outcome_own_side_markers", []
            )
        ]
        exclude = [
            canonicalize(marker)
            for marker in config["normalizations"].get(
                "ball_outcome_own_side_exclude_markers", []
            )
        ]
        if include and not any(marker and marker in canonical for marker in exclude):
            if any(marker and marker in canonical for marker in include):
                return canonicalize(
                    "landed on the losing player's side of the court"
                )
    return canonical


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
        normalized_after = re.sub(r"\s+", "", str(score_after)).casefold()
        for candidate in candidates:
            if re.sub(r"\s+", "", candidate).casefold() != normalized_after:
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
    normalized_after = re.sub(r"\s+", "", str(score_after)).casefold()
    for candidate in candidates:
        if re.sub(r"\s+", "", candidate).casefold() != normalized_after:
            return candidate
    return before


def outcome_label(point: dict[str, Any]) -> str:
    return str(point.get("how_point_ended", "")).split(":", 1)[0].strip()


def audio_groups(point: dict[str, Any], config: dict[str, Any]) -> list[str]:
    cue_text = str(point.get("audio_cues", "")).lower()
    groups = []
    for group, terms in config["normalizations"]["audio_groups"].items():
        if any(term.lower() in cue_text for term in terms):
            groups.append(group)
    return groups


def player_name(point: dict[str, Any], role: str) -> str:
    player_ref = point.get(role, "")
    if player_ref == "Player 1":
        return point["_player1"]
    if player_ref == "Player 2":
        return point["_player2"]
    return str(player_ref)


def winning_player_name(point: dict[str, Any]) -> str:
    winner_role = point.get("winner")
    if winner_role == "Server":
        return player_name(point, "server")
    if winner_role == "Receiver":
        return player_name(point, "receiver")
    return str(winner_role)


def derived_answer(
    derivation: str,
    point: dict[str, Any],
    config: dict[str, Any],
    template: dict[str, Any] | None = None,
) -> tuple[str | None, list[str] | None]:
    if derivation == "server_receiver_location":
        answer = (
            f"server {str(point.get('server_location', '')).lower()}; "
            f"receiver {str(point.get('receiver_location', '')).lower()}"
        )
        return answer, None
    if derivation == "combined_movement":
        lateral = normalize(
            annotation_value(point, "loser_lateral_movement"),
            "lateral_movement",
            config,
        )
        depth = normalize(
            annotation_value(point, "loser_depth_movement"),
            "depth_movement",
            config,
        )
        return f"{lateral}; {depth}", None
    if derivation == "score_transition":
        winner_role = str(point.get("winner", "")).strip().casefold()
        if winner_role not in {"server", "receiver"}:
            return None, None
        loser_role = "receiver" if winner_role == "server" else "server"
        winner_name = winning_player_name(point)
        loser_name = player_name(point, loser_role)
        score_before = point.get("score_before")
        score_after = point.get("score_after")
        wrong_next_score = alternate_next_tennis_score(score_before, score_after)
        if wrong_next_score == str(score_before):
            return None, None
        answer = (
            f"{winner_name} ({winner_role}) won the point; "
            f"new score: {score_after}"
        )
        return answer, [
            answer,
            (
                f"{loser_name} ({loser_role}) won the point; "
                f"new score: {score_after}"
            ),
            (
                f"{winner_name} ({loser_role}) won the point; "
                f"new score: {score_after}"
            ),
            (
                f"{winner_name} ({winner_role}) won the point; "
                f"new score: {wrong_next_score}"
            ),
        ]
    if derivation == "serve_sequence_outcome":
        outcome = outcome_label(point)
        attempts = point.get("num_serving_attempts_until_successful")
        return build_serve_sequence_options(
            attempts,
            outcome,
            maximum_options=int(
                config.get("generation_policy", {}).get("maximum_mcq_options", 4)
            ),
        )
    if derivation == "serve_rule_consistency":
        return build_serve_rule_consistency(
            point.get("num_serving_attempts_until_successful"),
            outcome_label(point),
        )
    if derivation == "grounded_serve_rule_consistency":
        return build_grounded_serve_rule_consistency(
            point.get("num_serving_attempts_until_successful"),
            outcome_label(point),
        )
    if derivation == "point_outcome_reasoning":
        return build_point_outcome_reasoning(
            point.get("winner"),
            outcome_label(point),
        )
    if derivation == "caption_action_mismatch":
        markers = unavailable_markers(config)
        action_fields = list(
            (template or {}).get("action_fields")
            or [
                "winner_shot_type",
                "winner_hand_used",
                "winner_shot_trajectory",
                "loser_shot_type",
                "loser_hand_used",
                "loser_shot_trajectory",
                "loser_lateral_movement",
                "loser_depth_movement",
            ]
        )
        field_values = {
            name: annotation_value(point, name) for name in action_fields
        }
        return build_caption_action_mismatch(
            caption=point.get("point_caption", ""),
            field_values=field_values,
            normalizations=config.get("normalizations", {}),
            is_skip=lambda value: is_skip_value(value, markers),
            action_fields=action_fields,
            minimum_options=int(
                config.get("generation_policy", {}).get("minimum_mcq_options", 2)
            ),
            maximum_options=int(
                config.get("generation_policy", {}).get("maximum_mcq_options", 4)
            ),
        )
    return None, None


def template_format(_template: dict[str, Any], section: str) -> str:
    return "qa" if section == "qa_templates" else "mcq"


def source_fields(template: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    if template.get("depends_on"):
        fields.extend(template["depends_on"])
    elif template.get("field"):
        fields.append(template["field"])
    for name in template.get("action_fields") or []:
        if name not in fields:
            fields.append(name)
    return fields or [template["field"]]


def source_is_valid(
    template: dict[str, Any],
    point: dict[str, Any],
    markers: set[str],
) -> bool:
    action_fields = template.get("action_fields")
    if action_fields:
        minimum = int(template.get("minimum_non_null_action_fields", 1))
        if (
            sum(
                not is_skip_value(annotation_value(point, field), markers)
                for field in action_fields
            )
            < minimum
        ):
            return False
    fields = source_fields(template)
    derivation = template.get("derivation")
    if derivation in {"audio_group_presence"}:
        return point.get("audio_cues") is not None
    for field_name in fields:
        # Virtual/derived field names are not present on the raw point.
        skip_names = {
            "server_receiver_location",
            "loser_combined_movement",
            "score_transition",
            "serve_sequence_outcome",
            "serve_rule_consistency",
            "grounded_serve_rule_consistency",
            "point_outcome_reasoning",
            "server_winner",
            "receiver_winner",
        }
        if action_fields:
            skip_names.update(action_fields)
        if field_name in skip_names:
            continue
        if is_skip_value(annotation_value(point, field_name), markers):
            return False
    return True


def trim_options(answer: str, options: list[str], maximum: int = 4) -> list[str]:
    unique = []
    for option in [answer, *options]:
        if option and option not in unique:
            unique.append(option)
    return unique[:maximum]


def observed_field_options(
    fields: list[str],
    points: list[dict[str, Any]],
    normalization: str | None,
    config: dict[str, Any],
    markers: set[str],
) -> list[str]:
    """Prescreen all review input points for a normalized option vocabulary."""
    counts: Counter[str] = Counter()
    for point in points:
        for field_name in fields:
            value = annotation_value(point, field_name)
            if is_skip_value(value, markers):
                continue
            option = normalize(value, normalization, config)
            if is_skip_value(option, markers):
                continue
            counts[option] += 1
    minimum = int(
        config.get("generation_policy", {}).get("minimum_observed_option_count", 1)
    )
    return [
        option
        for option, count in counts.most_common()
        if count >= minimum
    ]


def preview_mcq_options(
    name: str,
    point_id: str,
    answer: str,
    options: list[str],
    template: dict[str, Any],
    markers: set[str],
    maximum: int = 4,
    minimum: int = 2,
) -> tuple[str, list[str]]:
    """Build a deterministic preview of runtime option sampling/shuffling."""
    seed_text = f"{name}:{point_id}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    none_option = template.get("optional_none_option")
    options = [
        option
        for option in options
        if option and not is_skip_value(option, markers)
    ]

    if is_skip_value(answer, markers):
        return answer, []

    # Thresholding applies only to distractors; a rare raw answer stays valid.
    if none_option and template.get("force_none_correct"):
        distractors = [
            option for option in options if option not in {answer, none_option}
        ]
        rng.shuffle(distractors)
        if not distractors:
            return answer, []
        answer = str(none_option)
        selected = [
            *distractors[: max(0, maximum - 1)],
            answer,
        ]
    elif none_option and rng.random() < float(
        template.get("optional_none_probability", 0.0)
    ):
        distractors = [
            option for option in options if option not in {answer, none_option}
        ]
        rng.shuffle(distractors)
        if distractors and rng.random() < float(
            template.get("none_option_correct_probability", 0.0)
        ):
            answer = str(none_option)
            selected = [
                *distractors[: max(0, maximum - 1)],
                answer,
            ]
        else:
            selected = [
                answer,
                *distractors[: max(0, maximum - 2)],
                str(none_option),
            ]
    else:
        selected = trim_options(answer, options, maximum)

    selected = [
        option
        for option in selected
        if option == none_option or not is_skip_value(option, markers)
    ]
    # Sparse observed vocabularies can leave only the annotated label. When an
    # optional None choice is configured, include it so the preview still meets
    # the minimum-option policy (same salvage path as successful runtime samples).
    if (
        none_option
        and len(selected) < minimum
        and answer != none_option
        and str(none_option) not in selected
    ):
        selected = [answer, str(none_option)]
    if len(selected) < minimum:
        return answer, []
    rng.shuffle(selected)
    return answer, selected


def render_new_example(
    name: str,
    template: dict[str, Any],
    point: dict[str, Any],
    points: list[dict[str, Any]],
    config: dict[str, Any],
    fmt: str,
    markers: set[str],
) -> dict[str, Any] | None:
    if not source_is_valid(template, point, markers):
        return None
    answer: str | None = None
    derived_options: list[str] | None = None
    derivation = template.get("derivation")
    template_type = template.get("type", "categorical")

    if derivation == "audio_group_presence":
        answer = "Yes" if template["audio_group"] in audio_groups(point, config) else "No"
    elif derivation:
        answer, derived_options = derived_answer(derivation, point, config, template)
    elif template.get("answer_transform") == "leading_outcome_label":
        answer = outcome_label(point)
    elif template_type == "numeric_range":
        try:
            value = int(point.get(template["field"]))
        except (TypeError, ValueError):
            return None
        for range_config in template.get("ranges", []):
            if range_config["min"] <= value <= range_config["max"]:
                answer = range_config["answer"]
                break
    else:
        value = annotation_value(point, template["field"])
        answer = normalize(value, template.get("answer_normalization"), config)

    if not answer:
        return None

    question = template["question"]
    replacements = {
        "<score_before>": point.get("score_before", ""),
        "<player1>": point.get("_player1", "Player 1"),
        "<player2>": point.get("_player2", "Player 2"),
        "<player1_description>": point.get("_player1_description", ""),
        "<player2_description>": point.get("_player2_description", ""),
    }
    for placeholder, value in replacements.items():
        question = question.replace(placeholder, str(value))
    example = {
        "source_video": point["_video_id"],
        "point_id": point.get("id", ""),
        "question": question,
        "answer": answer,
    }

    if fmt == "mcq":
        annotated_answer = answer
        options = list(template.get("options", []))
        if template.get("options_from_observed_fields"):
            options = observed_field_options(
                template["options_from_observed_fields"],
                points,
                template.get("answer_normalization"),
                config,
                markers,
            )
        if derived_options:
            options = derived_options
        answer_normalization = template.get("answer_normalization")
        if answer_normalization:
            answer = normalize(answer, answer_normalization, config)
            options = [
                normalize(option, answer_normalization, config)
                for option in options
            ]
        if template.get("require_answer_in_options") and answer not in options:
            return None
        minimum = int(
            config.get("generation_policy", {}).get("minimum_mcq_options", 2)
        )
        maximum = int(
            config.get("generation_policy", {}).get("maximum_mcq_options", 4)
        )
        answer, options = preview_mcq_options(
            name,
            str(point.get("id", "")),
            answer,
            [str(option) for option in options],
            template,
            markers,
            maximum=maximum,
            minimum=minimum,
        )
        if not options:
            return None
        example["answer"] = answer
        if answer != annotated_answer:
            example["source_answer"] = annotated_answer
        example["options"] = options
    return example


def choose_two(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(examples) <= 2:
        return examples
    chosen = [examples[0]]
    different = next(
        (example for example in examples[1:] if example["answer"] != chosen[0]["answer"]),
        None,
    )
    chosen.append(different or examples[1])
    return chosen


def choose_new_examples(
    name: str,
    examples: list[dict[str, Any]],
    template: dict[str, Any],
    minimum_options: int = 2,
) -> list[dict[str, Any]]:
    if name in {
        "serve_sequence_outcome",
        "serve_rule_consistency",
        "grounded_serve_rule_consistency",
    }:
        double_fault = next(
            (
                example
                for example in examples
                if "point outcome: Double Fault" in example["answer"]
            ),
            None,
        )
        successful_second_serve = next(
            (
                example
                for example in examples
                if "second serve was successful" in example["answer"]
            ),
            None,
        )
        selected = [
            example
            for example in (double_fault, successful_second_serve)
            if example is not None
        ]
        if len(selected) == 2:
            return selected
    selected = choose_two(examples)
    none_option = template.get("optional_none_option")
    if template.get("force_none_correct") and none_option and selected and all(
        example["answer"] != none_option for example in selected
    ):
        example = max(
            selected,
            key=lambda item: len(
                [
                    option
                    for option in item.get("options", [])
                    if option not in {item["answer"], none_option}
                ]
            ),
        )
        annotated_answer = example["answer"]
        distractors = [
            option
            for option in example.get("options", [])
            if option not in {annotated_answer, none_option}
        ]
        # None-as-correct needs at least one real distractor so the MCQ
        # still satisfies the minimum-option policy.
        if distractors:
            options = [*distractors[:3], none_option]
            if len(options) >= minimum_options:
                seed = int(
                    hashlib.sha256(
                        f"{name}:{example['point_id']}:review-none".encode("utf-8")
                    ).hexdigest()[:16],
                    16,
                )
                random.Random(seed).shuffle(options)
                example["source_answer"] = annotated_answer
                example["answer"] = none_option
                example["options"] = options
    return selected


def label_mcq_answers(
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefix every MCQ answer with its displayed option letter."""
    for example in examples:
        options = example.get("options") or []
        answer = re.sub(
            r"^\([A-Z]\)\s*",
            "",
            str(example.get("answer", "")),
        )
        try:
            answer_index = options.index(answer)
        except ValueError:
            continue
        example["answer"] = f"({chr(ord('A') + answer_index)}) {answer}"
    return examples


def answer_for_template(
    template: dict[str, Any],
    point: dict[str, Any],
    config: dict[str, Any],
    markers: set[str],
) -> tuple[str | None, list[str] | None]:
    """Resolve a template's annotated answer without rendering its question."""
    if not source_is_valid(template, point, markers):
        return None, None

    field_name = template["field"]
    derivation = template.get("derivation")
    if derivation == "audio_group_presence":
        answer = "Yes" if template["audio_group"] in audio_groups(point, config) else "No"
        return answer, None
    if derivation:
        return derived_answer(derivation, point, config, template)

    if field_name in {"server_winner", "receiver_winner"}:
        server = point.get("server")
        receiver = point.get("receiver")
        winner = point.get("winner")
        if (
            not server
            or not receiver
            or server == receiver
            or winner not in {"Server", "Receiver"}
        ):
            return None, None
        server_player = player_name(point, "server")
        receiver_player = player_name(point, "receiver")
        winner_player = server_player if winner == "Server" else receiver_player
        if field_name == "server_winner":
            return f"{server_player} served and {winner_player} won", None
        return f"{receiver_player} received and {winner_player} won", None

    if template.get("answer_transform") == "leading_outcome_label":
        answer = outcome_label(point)
        return (answer or None), None

    if template.get("type") == "numeric_range":
        try:
            value = int(point.get(field_name))
        except (TypeError, ValueError):
            return None, None
        for range_config in template.get("ranges", []):
            if range_config["min"] <= value <= range_config["max"]:
                return str(range_config["answer"]), None
        return None, None

    value = annotation_value(point, field_name)
    if is_skip_value(value, markers):
        return None, None
    answer = normalize(value, template.get("answer_normalization"), config)
    return (answer or None), None


def distribution_summary(
    counts: Counter[str],
    total: int,
    limit: int = 10,
    *,
    value_mode: str = "single",
    min_percent: float = 0.1,
) -> dict[str, Any]:
    """Return compact, JSON-serializable counts and percentages.

    Values below ``min_percent`` are omitted from the shown rows and folded
    into the ``other_*`` aggregates.
    """
    shown: list[dict[str, Any]] = []
    for value, count in counts.most_common():
        percent = round(100.0 * count / total, 2) if total else 0.0
        if percent < min_percent:
            break
        shown.append(
            {
                "value": value,
                "count": count,
                "percent": percent,
            }
        )
        if len(shown) >= limit:
            break
    return {
        "total": total,
        "unique_values": len(counts),
        "value_mode": value_mode,
        "min_percent": min_percent,
        "values": shown,
        "other_value_count": max(
            0,
            sum(counts.values()) - sum(item["count"] for item in shown),
        ),
        "other_unique_values": max(0, len(counts) - len(shown)),
    }


def parse_audio_cue_set(value: Any, config: dict[str, Any]) -> set[str]:
    """Split free-text ``audio_cues`` into a per-point set of individual cues."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return set()

    known_terms = {
        str(term).strip().casefold()
        for terms in config.get("normalizations", {}).get("audio_groups", {}).values()
        for term in terms
        if str(term).strip()
    }
    # Spelling variants that should count as one cue, without collapsing
    # distinct sounds from the same audio group (e.g. thwack vs pop).
    alias_map = {
        "umpire call": "umpire's call",
        "umpire's call": "umpire's call",
        "umpire shout": "umpire's shout",
        "umpire's shout": "umpire's shout",
        "clapping": "clap",
        "claps": "clap",
        "cheers": "cheer",
        "cheering": "cheer",
        "murmurs": "murmur",
        "gasps": "gasp",
        "fault call": "fault call",
    }

    def canonicalize_cue(token: str) -> str:
        key = token.casefold()
        if key in alias_map:
            return alias_map[key]
        if key in known_terms:
            if " " not in key and key.endswith("s") and key[:-1] in known_terms:
                return key[:-1]
            return key
        if " " not in key and key.endswith("s"):
            singular = key[:-1]
            if singular in known_terms or singular in alias_map:
                return alias_map.get(singular, singular)
        return key

    cues: set[str] = set()
    for raw_token in text.split(","):
        token = raw_token.strip().strip(".").strip()
        if not token:
            continue
        cues.add(canonicalize_cue(token))
    return cues


def field_values_for_stats(
    point: dict[str, Any],
    field_name: str,
    template: dict[str, Any],
    config: dict[str, Any],
    markers: set[str],
) -> tuple[str, set[str]]:
    """Return ``(value_mode, values)`` for source-field distribution counting."""
    value = annotation_value(point, field_name)
    if is_skip_value(value, markers):
        return "single", set()
    if field_name == "audio_cues":
        return "set", parse_audio_cue_set(value, config)

    normalization = None
    if field_name == template.get("field"):
        normalization = template.get("answer_normalization")
    elif field_name in {"winner_shot_type", "loser_shot_type"}:
        normalization = "shot_type_aliases"
    elif field_name in {
        "winner_hand_used",
        "loser_hand_used",
        "winner_shot_trajectory",
        "loser_shot_trajectory",
        "loser_lateral_movement",
        "loser_depth_movement",
    }:
        normalization = {
            "winner_hand_used": "hand_used",
            "loser_hand_used": "hand_used",
            "winner_shot_trajectory": "shot_trajectory",
            "loser_shot_trajectory": "shot_trajectory",
            "loser_lateral_movement": "lateral_movement",
            "loser_depth_movement": "depth_movement",
        }[field_name]
    normalized_value = normalize(value, normalization, config)
    if is_skip_value(normalized_value, markers):
        return "single", set()
    return "single", {normalized_value}


def statistics_source_fields(template: dict[str, Any]) -> list[str]:
    """Return the concrete annotations that determine a template answer."""
    field_name = template.get("field")
    if field_name in {"server_winner", "receiver_winner"}:
        return ["winner"]
    # Skip identity/score fields that are nearly unique or not useful as
    # class-balance diagnostics (Player 1/2 assignment, exact score strings).
    skip = {"server", "receiver", "score_before", "score_after"}
    return [field for field in source_fields(template) if field not in skip]


def answer_mirrors_single_source_field(
    answer_counts: Counter[str],
    source_counts: dict[str, Counter[str]],
    source_modes: dict[str, str],
) -> bool:
    """True when the answer table would duplicate the sole source-field table.

    Keep a separate answer distribution for Yes/No, multi-field compounds,
    bucketed numeric ranges, force-none, and other derived answers.
    """
    if len(source_counts) != 1:
        return False
    field_name = next(iter(source_counts))
    if source_modes.get(field_name) == "set":
        return False
    return answer_counts == source_counts[field_name]


def template_statistics(
    template: dict[str, Any],
    fmt: str,
    points: list[dict[str, Any]],
    config: dict[str, Any],
    markers: set[str],
) -> dict[str, Any]:
    """Measure answer and source-field imbalance over the corpus."""
    source_fields_for_stats = statistics_source_fields(template)
    source_counts = {field: Counter() for field in source_fields_for_stats}
    source_modes = {field: "single" for field in source_fields_for_stats}
    source_missing = Counter()
    answer_counts: Counter[str] = Counter()
    eligible_points = 0

    for point in points:
        answer, derived_options = answer_for_template(template, point, config, markers)
        if not answer:
            continue

        normalized_answer = normalize(
            answer,
            template.get("answer_normalization"),
            config,
        )

        options = [str(option) for option in template.get("options", [])]
        if derived_options:
            options = [str(option) for option in derived_options]
        if template.get("require_answer_in_options") and normalized_answer not in options:
            continue
        if fmt == "mcq" and derived_options:
            minimum = int(
                config.get("generation_policy", {}).get("minimum_mcq_options", 2)
            )
            if len(set(options)) < minimum:
                continue

        eligible_points += 1
        displayed_answer = (
            str(template["optional_none_option"])
            if template.get("force_none_correct")
            and template.get("optional_none_option")
            else normalized_answer
        )
        answer_counts[displayed_answer] += 1

        for field_name, counts in source_counts.items():
            value_mode, values = field_values_for_stats(
                point,
                field_name,
                template,
                config,
                markers,
            )
            source_modes[field_name] = value_mode
            if not values:
                source_missing[field_name] += 1
                continue
            counts.update(values)

    generation_probability = float(template.get("generation_probability", 1.0))
    expected_data_points = round(eligible_points * generation_probability)
    mirrors_source = answer_mirrors_single_source_field(
        answer_counts,
        source_counts,
        source_modes,
    )
    return {
        "corpus_points": len(points),
        "eligible_points": eligible_points,
        "coverage_percent": round(
            100.0 * eligible_points / len(points), 2
        ) if points else 0.0,
        "generation_probability": generation_probability,
        "expected_data_points": expected_data_points,
        "answer_mirrors_single_source_field": mirrors_source,
        # Omit duplicate answer table when it is identical to the sole
        # source-field distribution (e.g. num_serving_attempts_until_successful).
        "answer_distribution": (
            None
            if mirrors_source
            else distribution_summary(answer_counts, eligible_points)
        ),
        "source_field_distributions": {
            field_name: {
                **distribution_summary(
                    counts,
                    eligible_points,
                    value_mode=source_modes[field_name],
                ),
                "missing_among_eligible": source_missing[field_name],
            }
            for field_name, counts in source_counts.items()
        },
    }


def build_review() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    points = load_points()
    vocabulary_points = (
        load_points(AGGREGATED_DIR)
        if AGGREGATED_DIR.is_dir()
        else points
    )
    generated = load_generated_examples()
    markers = unavailable_markers(config)
    reviewed: dict[str, Any] = {}
    sections = {
        "question_templates": config.get("question_templates", {}),
        "qa_templates": config.get("qa_templates", {}),
    }

    for section, templates in sections.items():
        for name, template in templates.items():
            fmt = template_format(template, section)
            generated_examples = generated.get(name, [])
            if generated_examples:
                examples = generated_examples[:1]
                example_source = "latest generated categorical sample"
            else:
                candidates = [
                    rendered
                    for point in points
                    if (
                        rendered := render_new_example(
                            name,
                            template,
                            point,
                            vocabulary_points,
                            config,
                            fmt,
                            markers,
                        )
                    )
                ]
                examples = choose_new_examples(
                    name,
                    candidates,
                    template,
                    minimum_options=int(
                        config.get("generation_policy", {}).get(
                            "minimum_mcq_options", 2
                        )
                    ),
                )
                examples = examples[:1]
                example_source = "deterministic preview from raw annotations"
            if fmt == "mcq":
                examples = label_mcq_answers(examples)
            reviewed[name] = {
                "super_category": template["super_category"],
                "fine_category": template["fine_category"],
                "format": fmt,
                "option_space": template.get(
                    "option_space",
                    infer_option_space(template.get("type", "")),
                ),
                "distractor_generation": template.get(
                    "distractor_generation",
                    infer_distractor_generation(template.get("type", "")),
                ),
                "implementation_type": template["type"],
                "availability": template["availability"],
                "example_source": example_source,
                "source_fields": source_fields(template),
                "option_vocabulary_source": template.get(
                    "option_vocabulary_source", []
                ),
                "question_template": template["question"],
                "reasoning_note": template.get("reasoning_note"),
                "llm_question_prompt": template.get("llm_question_prompt"),
                "review_warning": template.get("review_warning"),
                "statistics": template_statistics(
                    template,
                    fmt,
                    vocabulary_points,
                    config,
                    markers,
                ),
                "examples": examples,
            }

    # Keep MCQ and QA variants together under one super-category heading.
    # The config stores them in separate sections for runtime compatibility,
    # but the review artifact is organized by evaluation taxonomy.
    category_order = list(
        dict.fromkeys(item["super_category"] for item in reviewed.values())
    )
    reviewed = {
        name: item
        for category in category_order
        for name, item in reviewed.items()
        if item["super_category"] == category
    }

    total_eligible = sum(
        item["statistics"]["eligible_points"] for item in reviewed.values()
    )
    total_expected = sum(
        item["statistics"]["expected_data_points"] for item in reviewed.values()
    )
    aggregate_video_count = len(
        {point["_video_id"] for point in vocabulary_points}
    )
    example_video_count = len({point["_video_id"] for point in points})
    generated_files = sorted(GENERATED_DIR.glob("*_derived_mcq_qa.json"))
    generated_conversation_count = sum(
        len(load_json(path)) for path in generated_files
    )
    return {
        "config": str(CONFIG_PATH.relative_to(TENNIS_DIR)),
        "purpose": "Template review using the latest generated categorical sample.",
        "example_corpus": {
            "source": str(GENERATED_DIR.relative_to(TENNIS_DIR)),
            "video_files": len(generated_files),
            "conversations": generated_conversation_count,
            "fallback_source": "raw_tennis_data_input",
            "fallback_video_files": example_video_count,
            "fallback_points": len(points),
            "note": "Generated examples are used for every available class.",
        },
        "statistics_corpus": {
            "source": "aggregated_data_complete",
            "video_files": aggregate_video_count,
            "points": len(vocabulary_points),
        },
        "examples_per_template": 1,
        "class_family_count": 33,
        "template_count": len(reviewed),
        "generated_template_count": sum(
            item["availability"] == "generated" for item in reviewed.values()
        ),
        "new_template_count": sum(
            item["availability"] == "new_template" for item in reviewed.values()
        ),
        "template_count_note": "Winner/loser shot type and trajectory each have a sampled none-correct variant, producing 37 concrete templates from 33 class families.",
        "aggregated_data_complete": {
            "video_files": aggregate_video_count,
            "points": len(vocabulary_points),
            "eligible_template_point_pairs": total_eligible,
            "expected_data_points": total_expected,
        },
        "templates": reviewed,
    }


def append_template_statistics(
    lines: list[str],
    stats: dict[str, Any],
) -> None:
    """Append complete-corpus statistics after a template's examples."""
    lines.extend(
        [
            "### Complete-corpus statistics",
            "",
            f"- Total corpus points: **{stats['corpus_points']:,}**",
            f"- Eligible points: **{stats['eligible_points']:,}** ({stats['coverage_percent']:.2f}%)",
            (
                f"- Expected generated data points: **{stats['expected_data_points']:,}**"
                f" (generation probability: {stats['generation_probability']:.2f})"
            ),
        ]
    )

    answer_distribution = stats.get("answer_distribution")
    if answer_distribution is not None:
        lines.extend(
            [
                "",
                "**Correct-answer / composed-answer distribution among eligible points** "
                f"({answer_distribution['unique_values']:,} unique answers)",
                "",
                "| Answer | Count | Percent |",
                "|---|---:|---:|",
            ]
        )
        for value in answer_distribution["values"]:
            lines.append(
                f"| {md_code_span(value['value'])} | {value['count']:,} | "
                f"{value['percent']:.2f}% |"
            )
        if not answer_distribution["values"]:
            lines.append("| _No eligible answers_ | 0 | 0.00% |")
        other_unique = answer_distribution.get("other_unique_values", 0)
        if other_unique:
            other_percent = (
                100.0
                * answer_distribution["other_value_count"]
                / answer_distribution["total"]
                if answer_distribution["total"]
                else 0.0
            )
            lines.append(
                f"| _Other answers ({other_unique:,} unique)_ "
                f"| {answer_distribution['other_value_count']:,} | "
                f"{other_percent:.2f}% |"
            )

    for field_name, distribution in stats["source_field_distributions"].items():
        value_mode = distribution.get("value_mode", "single")
        if value_mode == "set":
            heading = (
                f"**Source field `{field_name}` cue presence among eligible points** "
                f"({distribution['unique_values']:,} unique cues; percentages are "
                "per-point presence rates and may sum to more than 100%)"
            )
        else:
            heading = (
                f"**Source field `{field_name}` distribution among eligible points** "
                f"({distribution['unique_values']:,} unique values)"
            )
        lines.extend(
            [
                "",
                heading,
                "",
                "| Value | Count | Percent |",
                "|---|---:|---:|",
            ]
        )
        for value in distribution["values"]:
            lines.append(
                f"| {md_code_span(value['value'])} | {value['count']:,} | "
                f"{value['percent']:.2f}% |"
            )
        if not distribution["values"]:
            lines.append("| _No non-missing values_ | 0 | 0.00% |")
        other_unique = distribution.get("other_unique_values", 0)
        if other_unique:
            if value_mode == "set":
                lines.append(
                    f"| _Other cues ({other_unique:,} unique)_ "
                    f"| {distribution['other_value_count']:,} point-cue occurrences | — |"
                )
            else:
                other_percent = (
                    100.0
                    * distribution["other_value_count"]
                    / distribution["total"]
                    if distribution["total"]
                    else 0.0
                )
                lines.append(
                    f"| _Other values ({other_unique:,} unique)_ "
                    f"| {distribution['other_value_count']:,} | {other_percent:.2f}% |"
                )
        if distribution["missing_among_eligible"]:
            missing_percent = (
                100.0
                * distribution["missing_among_eligible"]
                / stats["eligible_points"]
                if stats["eligible_points"]
                else 0.0
            )
            lines.append(
                f"| _Missing/unavailable_ | {distribution['missing_among_eligible']:,} "
                f"| {missing_percent:.2f}% |"
            )
    lines.append("")


def write_markdown(review: dict[str, Any]) -> None:
    aggregate = review["aggregated_data_complete"]
    lines = [
        "# Categorical Evaluation Template Review",
        "",
        "This document contains one example per generated template from the latest categorical sample.",
        "",
        f"- Example corpus: `{review['example_corpus']['source']}` "
        f"(**{review['example_corpus']['video_files']}** video files; "
        f"**{review['example_corpus']['conversations']:,}** generated records)",
        f"- Fallback preview corpus: `{review['example_corpus']['fallback_source']}` "
        f"(**{review['example_corpus']['fallback_video_files']}** video files; "
        f"**{review['example_corpus']['fallback_points']:,}** points)",
        f"- Statistics corpus: `{review['statistics_corpus']['source']}` "
        f"(**{review['statistics_corpus']['video_files']}** video files; "
        f"**{review['statistics_corpus']['points']:,}** points)",
        f"- Standalone config: `{review['config']}`",
        f"- Templates: **{review['template_count']}**",
        f"- Existing generated templates: **{review['generated_template_count']}**",
        f"- New templates requiring review: **{review['new_template_count']}**",
        f"- Evaluation class families: **{review['class_family_count']}**",
        f"- Count note: {review['template_count_note']}",
        f"- Requested examples per template: **{review['examples_per_template']}**",
        "- Review tip: search for `Availability: **new_template**` to jump between the new templates.",
        "",
        "## Statistics conventions",
        "",
        f"- Corpus: `aggregated_data_complete` (**{aggregate['video_files']:,} video files; {aggregate['points']:,} annotated points**).",
        "- Per-template statistics below use the full statistics corpus (`aggregated_data_complete`).",
        "- Example blocks use the latest generated categorical sample.",
        "- Eligible points are points with the required source annotations and a resolvable correct answer.",
        "- Expected data points equal eligible points after applying a template's configured generation probability. They are estimates for probabilistically sampled templates; an actual run can differ slightly.",
        "- Counts are annotation-derived capacity estimates.",
        "- Source-field distributions are computed among eligible points only and show at most the 10 most frequent values; remaining observations are reported as `other values`.",
        "- Every template also reports the correct/composed answer distribution (e.g. Yes/No for audio presence, or the joint label for multi-field templates).",
        "- For `audio_cues`, free-text lists are treated as a set: statistics are per individual cue (point-presence rates), so percentages can sum to more than 100%.",
        "",
    ]
    current_super = None
    for name, item in review["templates"].items():
        if item["super_category"] != current_super:
            current_super = item["super_category"]
            lines.extend([f"# {current_super}", ""])
        lines.extend(
            [
                f"## `{name}`",
                "",
                f"- Fine category: **{item['fine_category']}**",
                f"- Format: **{item['format'].upper()}**",
                f"- Option space: **`{item['option_space']}`**",
                f"- Distractor generation: **`{item['distractor_generation']}`**",
                f"- Implementation type: **`{item['implementation_type']}`**",
                f"- Availability: **{item['availability']}**",
                f"- Example source: {item['example_source']}",
                f"- Source fields: `{', '.join(item['source_fields'])}`",
                f"- Question template: {md_code_span(item['question_template'])}",
            ]
        )
        if item["option_vocabulary_source"]:
            lines.append(f"- Option vocabulary source: {item['option_vocabulary_source']}")
        if item["reasoning_note"]:
            lines.append(f"- Why reasoning: {item['reasoning_note']}")
        if item["llm_question_prompt"]:
            lines.append(f"- LLM distractor prompt: {md_code_span(item['llm_question_prompt'])}")
        if item["review_warning"]:
            lines.append(f"- Review warning: **{item['review_warning']}**")
        lines.append("")
        for index, example in enumerate(item["examples"], 1):
            lines.extend(
                [
                    f"### Example {index}",
                    "",
                    f"- Source: `{example['source_video']}` / `{example['point_id']}`",
                    f"- Question: {example['question']}",
                ]
            )
            if example.get("options"):
                for option_index, option in enumerate(example["options"]):
                    label = chr(ord("A") + option_index)
                    lines.append(f"  - ({label}) {option}")
            lines.append(f"- Answer: {example['answer']}")
            if example.get("source_answer"):
                lines.append(
                    f"- Underlying annotated value: {example['source_answer']}"
                )
            lines.append("")
        if len(item["examples"]) < review["examples_per_template"]:
            lines.extend(
                [
                    f"> Only {len(item['examples'])} valid example(s) could be produced from the current sample.",
                    "",
                ]
            )
        append_template_statistics(lines, item["statistics"])
    lines.extend(
        [
            "# Complete aggregated dataset totals",
            "",
            "Source: `aggregated_data_complete`",
            "",
            f"- Video files: **{aggregate['video_files']:,}**",
            f"- Annotated points: **{aggregate['points']:,}**",
            f"- Eligible template–point pairs across all templates: **{aggregate['eligible_template_point_pairs']:,}**",
            f"- Expected generated data points across all templates: **{aggregate['expected_data_points']:,}**",
            "",
            "> The last two totals count a point once for every template it can produce. They are therefore larger than the number of unique annotated points.",
            "",
        ]
    )
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    review = build_review()
    JSON_OUTPUT.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(review)
    missing = [
        name
        for name, item in review["templates"].items()
        if len(item["examples"]) < review["examples_per_template"]
    ]
    print(f"Wrote {JSON_OUTPUT}")
    print(f"Wrote {MD_OUTPUT}")
    print(
        f"Templates: {review['template_count']}; "
        f"missing requested examples: {len(missing)}"
    )
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
