#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Categorical-evaluation MCQ/QA generator for the BARD basketball dataset.

Reads ``dataset_paths.csv`` and ``captions/caption.json`` from the BARD tree plus the media
table written by ``probe_clip_media.py``, applies the 16 templates and data-quality gates of
``categorical_evaluation/CATEGORICAL_EVAL.MD``, splits by game and writes HuggingFace
conversation JSONL per split. ``categorical_evaluation/TEMPLATE_EXAMPLES_CATEGORICAL.MD`` lists
counts, baselines and one example per class. The clip is the sample: nothing is cut and
``video_root`` is the BARD tree.

Usage:
    python generate_categorical_mcq_qa.py \
        --dataset-root raw_basketball_data_input/BARD --out-dir training_basketball_data_output
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SHOT_ACTIONS = ("2PT Shot", "3PT Shot", "Free Throw")
FIELD_GOALS = ("2PT Shot", "3PT Shot")
ALL_ACTIONS = (
    "2PT Shot", "3PT Shot", "Free Throw", "Rebound",
    "Foul", "Turnover", "Steal", "Block", "Violation",
)
ACTION_PHRASE = {
    "2PT Shot": "a two point shot",
    "3PT Shot": "a three point shot",
    "Free Throw": "a free throw",
    "Rebound": "a rebound",
    "Foul": "a foul",
    "Turnover": "a turnover",
    "Steal": "a steal",
    "Block": "a block",
    "Violation": "a violation",
}
ACTION_VERB = {
    "2PT Shot": "attempts a two point shot",
    "3PT Shot": "attempts a three point shot",
    "Free Throw": "attempts a free throw",
    "Rebound": "grabs a rebound",
    "Foul": "commits a foul",
    "Turnover": "commits a turnover",
    "Steal": "makes a steal",
    "Block": "blocks a shot",
    "Violation": "commits a violation",
}

# Verbatim from categorical_evaluation/TEMPLATE_EXAMPLES_CATEGORICAL.MD's clip_caption example.
CAPTION_PROMPT = (
    "Analyze the basketball video and extract all player actions. For each action, identify "
    "the player's jersey number, jersey colour, and action type, using the format: \"A player "
    "with jersey number <n> and jersey_color <colour> made a <action> which result was "
    "made/miss and was assisted by other player with jersey number <n>\"."
)

SUPER_VISUAL = "Fine-Grained Visual Recognition"
SUPER_STRUCTURE = "Match Structure & Reasoning"
SUPER_IDENTITY = "Player Identification"
SUPER_REPORTING = "Commentary & Reporting"

DEFAULT_NUM_UNSEEN_GAMES = 9
DEFAULT_TEST_SEEN_CLIP_FRACTION = 0.10
DEFAULT_VAL_SAMPLE_FRACTION = 0.05
DEFAULT_SEED = 42
DEFAULT_PREFIX = "bard_cat"

CLASS_META = {
    "player_action": ("mcq", SUPER_VISUAL, "Action Attribution"),
    "shot_type": ("mcq", SUPER_VISUAL, "Shot Mechanics"),
    "shot_type_targeted": ("mcq", SUPER_VISUAL, "Shot Mechanics"),
    "shot_result": ("mcq", SUPER_VISUAL, "Shot Outcome"),
    "shot_result_targeted": ("mcq", SUPER_VISUAL, "Shot Outcome"),
    "shot_assisted": ("mcq", SUPER_VISUAL, "Play Construction"),
    "rebound_side": ("mcq", SUPER_VISUAL, "Rebounding"),
    "block_present": ("mcq", SUPER_VISUAL, "Shot Outcome"),
    "possession_loss": ("mcq", SUPER_STRUCTURE, "Possession Outcome"),
    "action_order": ("mcq", SUPER_STRUCTURE, "Play Structure"),
    "action_order_last": ("mcq", SUPER_STRUCTURE, "Play Structure"),
    "actor_jersey_number": ("mcq", SUPER_IDENTITY, "Player Identification"),
    "actor_team_colour": ("mcq", SUPER_IDENTITY, "Team Attribution"),
    "assister_jersey_number": ("mcq", SUPER_IDENTITY, "Play Construction"),
    "clip_caption": ("open_ended", SUPER_REPORTING, "Structured Reporting"),
    "player_action_description": ("open_ended", SUPER_REPORTING, "Action Attribution"),
}

# Both teams of this game are intermittently labelled `white` and two players share white
# #23, so every class whose question or answer depends on jersey colour/number drops it.
# Clip-level classes and the verbatim clip_caption keep the game.
COLOUR_GATE_GAME = "nyk-vs-mia-0022400869"
COLOUR_GATED_CLASSES = {
    "player_action", "player_action_description",
    "shot_type_targeted", "shot_result_targeted",
    "shot_assisted", "rebound_side",
    "actor_jersey_number", "actor_team_colour", "assister_jersey_number",
}

# Post-gate corpus totals from categorical_evaluation/TEMPLATE_EXAMPLES_CATEGORICAL.MD (re-verified
# against the raw annotations when the catalog was curated). A mismatch means the raw
# annotations drifted or this script's filters diverged from the catalog -- either way, stop.
EXPECTED_ROWS = {
    "player_action": 20401,
    "shot_type": 11664,
    "shot_type_targeted": 12464,
    "shot_result": 11087,
    "shot_result_targeted": 12464,
    "shot_assisted": 4428,
    "rebound_side": 3538,
    "block_present": 4249,
    "possession_loss": 2267,
    "action_order": 6127,
    "action_order_last": 6127,
    "actor_jersey_number": 21079,
    "actor_team_colour": 21079,
    "assister_jersey_number": 3142,
    "clip_caption": 14676,
    "player_action_description": 20401,
}

POSSESSION_LOSS_OPTIONS = (
    "a foul is committed",
    "a turnover that is not forced by a steal",
    "a defender steals the ball",
)
POSSESSION_LOSS_LABEL = {
    frozenset({"Foul"}): "a foul is committed",
    frozenset({"Turnover"}): "a turnover that is not forced by a steal",
    frozenset({"Steal", "Turnover"}): "a defender steals the ball",
}

_LETTERS = "ABCDE"


@dataclass(frozen=True)
class Clip:
    """One annotated BARD clip."""

    relative_path: str  # "bkn-vs-det-0022400861/2.mp4" -- relative to video_root
    game: str
    clip_index: str
    actions: tuple[dict[str, Any], ...]
    duration_s: float
    caption: str

    @property
    def clip_id(self) -> str:
        return f"{self.game}_{self.clip_index}"


@dataclass(frozen=True)
class Question:
    """One row: a clip (optionally a specific actor/slot in it) asked about one class."""

    clip: Clip
    question_class: str
    question: str
    answer: str  # the correct option text, or the free-text target for open_ended
    options: tuple[str, ...] | None  # None for open_ended
    suffix: str | None  # disambiguates several rows of one class per clip, e.g. "black8"
    label: str  # short label for stats/confusion

    @property
    def question_id(self) -> str:
        suffix = f"_{self.suffix}" if self.suffix else ""
        return f"{self.clip.clip_id}_{self.question_class}{suffix}"


def load_clips(dataset_root: Path, media: dict[str, list]) -> list[Clip]:
    """Read dataset_paths.csv + caption.json + the probed media table into Clip records.

    Hard-fails on any row that does not match the documented schema, on a missing video and
    on a missing caption or media probe: a silently skipped row shows up only as a
    mysteriously smaller dataset.
    """
    captions = {
        entry["video"]: next(t["value"] for t in entry["conversations"] if t["from"] == "gpt")
        for entry in json.loads(
            (dataset_root / "captions" / "caption.json").read_text(encoding="utf-8")
        )
    }
    paths_csv = dataset_root / "dataset_paths.csv"

    clips: list[Clip] = []
    with paths_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        if header[:3] != ["urls", "actions", "numerosity"]:
            raise ValueError(f"unexpected header in {paths_csv}: {header!r}")
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"malformed row in {paths_csv}: {row!r}")
            relative_path, raw_actions, numerosity = row
            actions = ast.literal_eval(raw_actions)
            if len(actions) != int(numerosity):
                raise ValueError(
                    f"{relative_path}: numerosity {numerosity} != {len(actions)} actions"
                )
            if not (dataset_root / relative_path).is_file():
                raise FileNotFoundError(f"missing clip: {dataset_root / relative_path}")
            if relative_path not in media:
                raise KeyError(f"{relative_path} absent from the media table")
            if relative_path not in captions:
                raise KeyError(f"{relative_path} absent from caption.json")
            game, _, clip_file = relative_path.partition("/")
            clips.append(
                Clip(
                    relative_path=relative_path,
                    game=game,
                    clip_index=clip_file.rsplit(".", 1)[0],
                    actions=tuple(actions),
                    duration_s=float(media[relative_path][0]),
                    caption=captions[relative_path].strip(),
                )
            )

    if not clips:
        raise ValueError(f"no clips parsed from {paths_csv}")
    return clips


def build_game_colours(clips: list[Clip]) -> dict[str, dict[str, set]]:
    """game -> colour -> jersey numbers seen on that colour anywhere in the game.

    These are the identification classes' distractor pools: same team, same game, so the
    colour in the question cannot be used to eliminate options.
    """
    game_colours: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for clip in clips:
        for action in clip.actions:
            game_colours[clip.game][action["color"]].add(action["player"])
    for game, colours in game_colours.items():
        assert len(colours) == 2, (
            f"{game}: expected exactly 2 jersey colours, got {sorted(colours)}"
        )
    return game_colours


def unique_actors(clip: Clip) -> set:
    """(colour, number) pairs that appear exactly once, so a question can name them."""
    counts = Counter((a["color"], a["player"]) for a in clip.actions)
    return {key for key, n in counts.items() if n == 1}


def describe(action: dict[str, Any]) -> str:
    return f"the player wearing the {action['color']} number {action['player']} jersey"


def actor_key(action: dict[str, Any]) -> str:
    return f"{action['color']}{action['player']}"


def build_questions(
    clip: Clip, game_colours: dict[str, dict[str, set]], seed: int,
    exclusions: Counter,
) -> list[Question]:
    """Every row this clip contributes, across all 16 templates (colour gate NOT applied
    here -- main() filters gated classes out of the gate game and counts what it removed)."""
    actions = list(clip.actions)
    addressable = unique_actors(clip)
    colours = game_colours[clip.game]
    questions: list[Question] = []

    def rng_for(question_class: str, suffix: str | None) -> random.Random:
        return random.Random(f"{seed}:{clip.clip_id}:{question_class}:{suffix or ''}")

    # ---- per-slot classes -------------------------------------------------------------
    shots_by_actor = Counter(
        (a["color"], a["player"]) for a in actions if a["action"] in SHOT_ACTIONS
    )
    action_type_counts = Counter(a["action"] for a in actions)

    for action in actions:
        who = describe(action)
        key = actor_key(action)

        # player_action + player_action_description: the addressed actor must be unique by
        # (colour, number), which leaves them exactly one logged action.
        if (action["color"], action["player"]) in addressable:
            distractors = rng_for("player_action", key).sample(
                [a for a in ALL_ACTIONS if a != action["action"]], 3
            )
            questions.append(
                Question(
                    clip, "player_action",
                    f"What does {who} do in this video?",
                    ACTION_PHRASE[action["action"]],
                    tuple(ACTION_PHRASE[a] for a in [action["action"], *distractors]),
                    key, action["action"],
                )
            )
            answer = f"{who.capitalize()} {ACTION_VERB[action['action']]}."
            if action["result"] is not None:
                answer = answer[:-1] + (" and scores." if action["result"] else " and misses.")
            questions.append(
                Question(
                    clip, "player_action_description",
                    f"Describe what {who} does in this video.",
                    answer, None, key, "description",
                )
            )

            # shot_assisted: made field goals only (BARD sets `assisted` only there).
            if action["action"] in FIELD_GOALS and action["result"]:
                questions.append(
                    Question(
                        clip, "shot_assisted",
                        f"The field goal by {who} goes in. Was it assisted by a teammate?",
                        "it was assisted by a teammate" if action["assisted"]
                        else "it was unassisted",
                        ("it was assisted by a teammate", "it was unassisted"),
                        key, "assisted" if action["assisted"] else "unassisted",
                    )
                )

        # Targeted shot classes: looser filter than player_action -- unambiguous iff the
        # actor attempts exactly ONE shot; their non-shot slots don't disqualify.
        if action["action"] in SHOT_ACTIONS:
            if shots_by_actor[(action["color"], action["player"])] == 1:
                questions.append(
                    Question(
                        clip, "shot_type_targeted",
                        f"What type of shot does {who} attempt in this video?",
                        ACTION_PHRASE[action["action"]],
                        tuple(ACTION_PHRASE[s] for s in SHOT_ACTIONS),
                        key, action["action"],
                    )
                )
                made = bool(action["result"])
                questions.append(
                    Question(
                        clip, "shot_result_targeted",
                        f"Does the shot taken by {who} go in?",
                        "the shot is made" if made else "the shot is missed",
                        ("the shot is made", "the shot is missed"),
                        key, "made" if made else "missed",
                    )
                )
            else:
                exclusions["targeted_shot_slot_actor_has_multiple_shots"] += 1

        # Identification classes: the asked action type must be unique in the clip, so
        # exactly one player performs it.
        if action_type_counts[action["action"]] == 1:
            suffix = f"{key}_{action['action'].replace(' ', '')}"
            pool = sorted(colours[action["color"]] - {action["player"]})
            assert len(pool) >= 3, f"{clip.relative_path}: <3 same-colour distractor numbers"
            sampled = rng_for("actor_jersey_number", suffix).sample(pool, 3)
            questions.append(
                Question(
                    clip, "actor_jersey_number",
                    f"Which {action['color']}-jersey player "
                    f"{ACTION_VERB[action['action']]} in this video?",
                    f"number {action['player']}",
                    tuple([f"number {action['player']}"] + [f"number {n}" for n in sampled]),
                    suffix, f"number {action['player']}",
                )
            )
            other = next(c for c in colours if c != action["color"])
            questions.append(
                Question(
                    clip, "actor_team_colour",
                    "What colour jersey is worn by the player who "
                    f"{ACTION_VERB[action['action']]} in this video?",
                    action["color"], (action["color"], other),
                    suffix, action["color"],
                )
            )

        # assister_jersey_number: every assisted basket; no clip has two assisted baskets
        # by the same (colour, number), verified zero cases.
        if action["assisted"]:
            if not str(action["other_player"]).isdigit():
                exclusions["assister_corrupt_jersey_rows"] += 1
            else:
                pool = sorted(colours[action["color"]] - {action["player"], action["other_player"]})
                assert len(pool) >= 3, f"{clip.relative_path}: <3 assister distractor numbers"
                sampled = rng_for("assister_jersey_number", key).sample(pool, 3)
                questions.append(
                    Question(
                        clip, "assister_jersey_number",
                        f"The basket scored by {who} was assisted. Which teammate made the assist?",
                        f"number {action['other_player']}",
                        tuple(
                            [f"number {action['other_player']}"] + [f"number {n}" for n in sampled]
                        ),
                        key, f"number {action['other_player']}",
                    )
                )

    # ---- clip-level classes -----------------------------------------------------------
    shot_slots = [a for a in actions if a["action"] in SHOT_ACTIONS]

    shot_types = {a["action"] for a in shot_slots}
    if len(shot_types) == 1:
        shot_type = next(iter(shot_types))
        questions.append(
            Question(
                clip, "shot_type",
                "What type of shot attempt is shown in this video?",
                ACTION_PHRASE[shot_type],
                tuple(ACTION_PHRASE[s] for s in SHOT_ACTIONS),
                None, shot_type,
            )
        )
    elif len(shot_types) > 1:
        exclusions["shot_type_mixed_type_clips"] += 1

    if len(shot_slots) == 1:
        made = bool(shot_slots[0]["result"])
        questions.append(
            Question(
                clip, "shot_result",
                "The video shows a single shot attempt. Does the shot go in?",
                "the shot is made" if made else "the shot is missed",
                ("the shot is made", "the shot is missed"),
                None, "made" if made else "missed",
            )
        )
        # block_present: missed field goals only (a free throw cannot be blocked).
        if not made and shot_slots[0]["action"] != "Free Throw":
            blocked = any(a["action"] == "Block" for a in actions)
            questions.append(
                Question(
                    clip, "block_present",
                    "The field goal attempt in this video is missed. "
                    "Is the shot blocked by a defender?",
                    "the shot is blocked" if blocked else "the shot is not blocked",
                    ("the shot is blocked", "the shot is not blocked"),
                    None, "blocked" if blocked else "not blocked",
                )
            )

    # rebound_side: derived -- offensive iff the rebounder's colour equals the shooter's.
    rebounds = [a for a in actions if a["action"] == "Rebound"]
    if rebounds:
        if len(rebounds) != 1:
            exclusions["rebound_side_multi_rebound_clips"] += 1
        elif len(shot_slots) != 1:
            exclusions["rebound_side_not_one_shot_clips"] += 1
        elif actions.index(rebounds[0]) < actions.index(shot_slots[0]):
            exclusions["rebound_side_rebound_precedes_shot"] += 1
        elif shot_slots[0]["result"]:
            exclusions["rebound_side_made_shot_clips"] += 1
        else:
            offensive = rebounds[0]["color"] == shot_slots[0]["color"]
            questions.append(
                Question(
                    clip, "rebound_side",
                    "A shot is missed and rebounded in this video. Which team gets the rebound?",
                    "the shooting team gets the offensive rebound" if offensive
                    else "the defending team gets the defensive rebound",
                    ("the shooting team gets the offensive rebound",
                     "the defending team gets the defensive rebound"),
                    None, "offensive" if offensive else "defensive",
                )
            )

    if not shot_slots:
        label = POSSESSION_LOSS_LABEL.get(frozenset(a["action"] for a in actions))
        if label is not None:
            questions.append(
                Question(
                    clip, "possession_loss",
                    "No shot is attempted in this video. Which of the following happens?",
                    label, POSSESSION_LOSS_OPTIONS, None, label,
                )
            )
        else:
            exclusions["possession_loss_mixed_no_shot_clips"] += 1

    # action_order: options are the action types actually present (play-by-play order is
    # temporal order), so the model must order them rather than detect them. Options are
    # shuffled per row later, which removes the listed-in-true-order giveaway.
    distinct_types = list(dict.fromkeys(a["action"] for a in actions))
    if len(distinct_types) >= 2:
        options = tuple(ACTION_PHRASE[t] for t in distinct_types)
        questions.append(
            Question(
                clip, "action_order",
                "Which of these actions happens FIRST in this video?",
                ACTION_PHRASE[actions[0]["action"]], options, None, actions[0]["action"],
            )
        )
        questions.append(
            Question(
                clip, "action_order_last",
                "Which of these actions happens LAST in this video?",
                ACTION_PHRASE[actions[-1]["action"]], options, None, actions[-1]["action"],
            )
        )

    questions.append(
        Question(clip, "clip_caption", CAPTION_PROMPT, clip.caption, None, None, "caption")
    )
    return questions


def assign_clip_splits(
    clips: list[Clip], num_unseen_games: int, test_seen_clip_fraction: float, seed: int
) -> dict[str, list[Clip]]:
    """Split CLIPS three ways: unseen games / held-out clips of seen games / the train+val
    pool. Seed 42 reproduces the catalog's duration-baseline holdout games, so test_unseen
    stays comparable with the catalog's baselines."""
    games = sorted({clip.game for clip in clips})
    if num_unseen_games >= len(games):
        raise ValueError(f"{num_unseen_games} unseen games requested of only {len(games)}")

    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)
    unseen_games = set(shuffled[:num_unseen_games])

    by_seen_game: dict[str, list[Clip]] = defaultdict(list)
    splits: dict[str, list[Clip]] = {"pool": [], "test_seen": [], "test_unseen": []}
    for clip in clips:
        if clip.game in unseen_games:
            splits["test_unseen"].append(clip)
        else:
            by_seen_game[clip.game].append(clip)

    for game in sorted(by_seen_game):
        members = sorted(by_seen_game[game], key=lambda c: c.relative_path)
        held_out = max(1, round(len(members) * test_seen_clip_fraction))
        chosen = set(
            random.Random(f"{seed}:test_seen:{game}").sample(range(len(members)), held_out)
        )
        for index, clip in enumerate(members):
            splits["test_seen" if index in chosen else "pool"].append(clip)
    return splits


def split_pool_rows(
    pool_questions: list[Question], val_sample_fraction: float, seed: int
) -> tuple[list[Question], list[Question]]:
    """Partition the pool's ROWS into train/val, stratified per class: train and val share
    clips (by design -- val tracks fitting), but no sample appears in both."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for question in pool_questions:
        by_class[question.question_class].append(question.question_id)

    val_ids = set()
    for question_class, ids in sorted(by_class.items()):
        ids = sorted(ids)
        num_val = round(len(ids) * val_sample_fraction)
        val_ids.update(random.Random(f"{seed}:val:{question_class}").sample(ids, num_val))

    train = [q for q in pool_questions if q.question_id not in val_ids]
    val = [q for q in pool_questions if q.question_id in val_ids]
    return train, val


def build_row(question: Question, seed: int) -> dict[str, Any]:
    """One JSONL row. MCQ options are shuffled per row, seeded off the row identity, so the
    answer letter carries no signal but the dataset still regenerates byte-identically."""
    evaluation_type, super_category, fine_category = CLASS_META[question.question_class]

    if question.options is None:
        prompt, answer = question.question, question.answer
    else:
        options = list(question.options)
        random.Random(f"{seed}:{question.question_id}").shuffle(options)
        lines = [question.question]
        lines.extend(f"({letter}) {text}" for letter, text in zip(_LETTERS, options))
        prompt = "\n".join(lines)
        letter = _LETTERS[options.index(question.answer)]
        answer = f"({letter}) {question.answer}"

    return {
        "id": question.question_id,
        "class": question.question_class,
        "evaluation_type": evaluation_type,
        "super_category": super_category,
        "fine_category": fine_category,
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": question.clip.relative_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
    }


def duration_only_accuracy(
    train: list[Question], held_out: list[Question], question_class: str, num_bins: int = 10
) -> float | None:
    """Accuracy of predicting the label from clip duration alone, fit on train rows.

    BARD cuts clips on play-by-play boundaries, so length tracks event count -- reporting
    this next to chance and majority is what keeps a model result interpretable. On val it
    is flattered (val shares clips with train); the test splits carry the honest numbers.
    """
    fit = [q for q in train if q.question_class == question_class]
    score = [q for q in held_out if q.question_class == question_class]
    if not fit or not score:
        return None

    durations = sorted(q.clip.duration_s for q in fit)
    edges = [durations[int(i * len(durations) / num_bins)] for i in range(num_bins)]
    per_bin: dict[int, Counter] = defaultdict(Counter)
    for q in fit:
        per_bin[bisect.bisect_right(edges, q.clip.duration_s) - 1][q.label] += 1
    fallback = Counter(q.label for q in fit).most_common(1)[0][0]
    rule = {index: counts.most_common(1)[0][0] for index, counts in per_bin.items()}
    correct = sum(
        rule.get(bisect.bisect_right(edges, q.clip.duration_s) - 1, fallback) == q.label
        for q in score
    )
    return round(correct / len(score), 4)


def summarize_split(
    questions: list[Question], train_questions: list[Question], rows_by_class: dict[str, list]
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for question_class in CLASS_META:
        members = [q for q in questions if q.question_class == question_class]
        if not members:
            continue
        counts = Counter(q.label for q in members)
        entry: dict[str, Any] = {
            "num_rows": len(members),
            "evaluation_type": CLASS_META[question_class][0],
        }
        if CLASS_META[question_class][0] == "mcq":
            option_counts = Counter(len(q.options) for q in members)
            # "Always answer X" is only a strategy when the answer vocabulary is CLOSED.
            # Jersey numbers are per-instance (options shuffled), so majority and the
            # duration rule are meaningless there and chance is the real floor.
            closed = len(counts) <= 12
            entry.update(
                num_options=dict(sorted(option_counts.items())),
                chance_accuracy=round(sum(1 / len(q.options) for q in members) / len(members), 4),
                answer_letter_counts=dict(
                    sorted(Counter(r["conversation"][1]["content"][0]["text"][1]
                                   for r in rows_by_class[question_class]).items())
                ),
            )
            if closed:
                entry.update(
                    class_counts=dict(sorted(counts.items())),
                    majority_class_accuracy=round(max(counts.values()) / len(members), 4),
                    duration_only_accuracy=duration_only_accuracy(
                        train_questions, questions, question_class
                    ),
                )
            else:
                entry["distinct_answers"] = len(counts)
        else:
            lengths = sorted(len(q.answer) for q in members)
            entry["target_chars"] = {
                "min": lengths[0],
                "median": lengths[len(lengths) // 2],
                "p90": lengths[int(0.9 * len(lengths))],
                "max": lengths[-1],
            }
        per_class[question_class] = entry

    clips = {q.clip.relative_path: q.clip for q in questions}
    durations = sorted(clip.duration_s for clip in clips.values())
    return {
        "num_rows": len(questions),
        "num_clips": len(clips),
        "num_games": len({clip.game for clip in clips.values()}),
        "games": sorted({clip.game for clip in clips.values()}),
        "clip_duration_s": {
            "min": round(durations[0], 2),
            "median": round(durations[len(durations) // 2], 2),
            "p90": round(durations[int(0.9 * len(durations))], 2),
            "max": round(durations[-1], 2),
        },
        "per_class": per_class,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(
    dataset_root: Path,
    out_dir: Path,
    media_json: Path | None = None,
    num_unseen_games: int = DEFAULT_NUM_UNSEEN_GAMES,
    test_seen_clip_fraction: float = DEFAULT_TEST_SEEN_CLIP_FRACTION,
    val_sample_fraction: float = DEFAULT_VAL_SAMPLE_FRACTION,
    seed: int = DEFAULT_SEED,
    prefix: str = DEFAULT_PREFIX,
) -> None:
    """Write the four splits, their metadata and ``split_stats.json`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    media_path = media_json or out_dir / "clip_media.json"
    media = json.loads(media_path.read_text(encoding="utf-8"))

    clips = load_clips(dataset_root, media)
    logger.info("parsed %d clips across %d games", len(clips), len({c.game for c in clips}))

    game_colours = build_game_colours(clips)
    exclusions: Counter = Counter()
    by_clip = {
        clip.relative_path: build_questions(clip, game_colours, seed, exclusions)
        for clip in clips
    }

    # Colour gate: drop identity-dependent rows of the corrupt-colour game, counting what
    # was removed so the exclusion stays auditable.
    for questions in by_clip.values():
        for question in list(questions):
            if (question.clip.game == COLOUR_GATE_GAME
                    and question.question_class in COLOUR_GATED_CLASSES):
                questions.remove(question)
                exclusions[f"colour_gate_{question.question_class}"] += 1

    all_questions = [q for clip in clips for q in by_clip[clip.relative_path]]
    logger.info("built %d rows from %d clips", len(all_questions), len(clips))
    class_totals = Counter(q.question_class for q in all_questions)
    for question_class in CLASS_META:
        members = [q for q in all_questions if q.question_class == question_class]
        labels = Counter(q.label for q in members)
        logger.info(
            "  %-26s %6d rows  %s",
            question_class, len(members),
            dict(labels.most_common(4)) if len(labels) <= 12 else f"{len(labels)} distinct answers",
        )
    diverged = {
        k: (class_totals.get(k), v) for k, v in EXPECTED_ROWS.items() if class_totals.get(k) != v
    }
    assert dict(class_totals) == EXPECTED_ROWS, (
        "per-class totals diverged from "
        f"categorical_evaluation/TEMPLATE_EXAMPLES_CATEGORICAL.MD: {diverged}"
    )
    logger.info("exclusion counts: %s", dict(sorted(exclusions.items())))

    clip_splits = assign_clip_splits(
        clips, num_unseen_games, test_seen_clip_fraction, seed
    )
    pool_questions = [q for clip in clip_splits["pool"] for q in by_clip[clip.relative_path]]
    train_questions, val_questions = split_pool_rows(
        pool_questions, val_sample_fraction, seed
    )
    split_questions = {
        "train": train_questions,
        "val": val_questions,
        "test_seen": [q for clip in clip_splits["test_seen"] for q in by_clip[clip.relative_path]],
        "test_unseen": [
            q for clip in clip_splits["test_unseen"] for q in by_clip[clip.relative_path]
        ],
    }

    stats: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "video_root": str(dataset_root),
        "seed": seed,
        "split_design": {
            "num_unseen_games": num_unseen_games,
            "test_seen_clip_fraction": test_seen_clip_fraction,
            "val_sample_fraction": val_sample_fraction,
            "semantics": (
                "test_unseen holds out whole games (leak-free); test_seen holds out clips "
                "inside the train games (same broadcasts, unseen plays); train and val share "
                "the remaining clips with rows partitioned per class, so val tracks fitting, "
                "not generalization."
            ),
        },
        "num_clips": len(clips),
        "num_rows": len(all_questions),
        "classes": {
            name: {"evaluation_type": e, "super_category": s, "fine_category": f}
            for name, (e, s, f) in CLASS_META.items()
        },
        "caption_prompt": CAPTION_PROMPT,
        "quality_gates": {
            "colour_gate_game": COLOUR_GATE_GAME,
            "colour_gated_classes": sorted(COLOUR_GATED_CLASSES),
            "exclusion_counts": dict(sorted(exclusions.items())),
        },
        "note": (
            "Full curated categorical set per avlm/data_prep_example/basketball/"
            "categorical_evaluation/CATEGORICAL_EVAL.MD and its TEMPLATE_EXAMPLES_CATEGORICAL.MD "
            "(post-gate totals asserted against the catalog). "
            "Clip classification: the file is the sample, nothing is cut, video_root is the "
            "BARD tree. Clip length tracks event count, hence duration_only_accuracy per "
            "class (fit on train rows; honest on the test splits only). rebound_side and "
            "block_present are heavily imbalanced -- report macro accuracy and confusion "
            "matrices, never pooled accuracy. action_order_last is the weak LAST variant "
            "and must be reported separately from action_order. At video_sample_fps 60 with "
            "max_video_frames 128 the cap binds on 100% of clips (shortest clip yields 163 "
            "pre-cap frames), so every sample is exactly 128 frames."
        ),
        "splits": {},
    }

    for split_name in ("train", "val", "test_seen", "test_unseen"):
        questions = split_questions[split_name]
        rows = [build_row(q, seed) for q in questions]
        split_clips = sorted(
            {q.clip.relative_path: q.clip for q in questions}.values(),
            key=lambda c: c.relative_path,
        )
        meta_rows = [{"id": clip.clip_id, "video": clip.relative_path} for clip in split_clips]

        jsonl_path = out_dir / f"{prefix}_{split_name}.jsonl"
        meta_path = out_dir / f"{prefix}_{split_name}_meta_input.jsonl"
        write_jsonl(jsonl_path, rows)
        write_jsonl(meta_path, meta_rows)
        # What build_video_metadata.py would produce for this meta_input, taken from the
        # already-probed media table instead of re-decoding 14,676 videos.
        metadata_path = out_dir / f"{prefix}_{split_name}_meta_input_video_metadata.json"
        metadata_path.write_text(
            json.dumps({clip.relative_path: media[clip.relative_path] for clip in split_clips})
            + "\n",
            encoding="utf-8",
        )

        rows_by_class: dict[str, list] = defaultdict(list)
        for question, row in zip(questions, rows):
            rows_by_class[question.question_class].append(row)

        split_stats = summarize_split(questions, train_questions, rows_by_class)
        split_stats["path"] = str(jsonl_path)
        split_stats["meta_input_path"] = str(meta_path)
        split_stats["video_metadata_path"] = str(metadata_path)
        split_stats["num_meta_rows"] = len(meta_rows)
        stats["splits"][split_name] = split_stats
        logger.info(
            "%s: %d rows / %d clips -> %s", split_name, len(rows), len(meta_rows), jsonl_path
        )

    # Invariants the rest of the pipeline assumes. Assertions rather than defensive guards:
    # each is guaranteed by the code above, so a failure is a bug, not bad input.
    assert len({q.question_id for q in all_questions}) == len(all_questions), "duplicate row ids"
    assert sum(len(v) for v in split_questions.values()) == len(all_questions), "split lost rows"
    for question_class in CLASS_META:
        for split_name, questions in split_questions.items():
            assert any(q.question_class == question_class for q in questions), (
                f"{question_class} has no rows in {split_name}"
            )
    games_of = {name: {q.clip.game for q in qs} for name, qs in split_questions.items()}
    seen_games = games_of["train"] | games_of["val"] | games_of["test_seen"]
    assert not (games_of["test_unseen"] & seen_games), "test_unseen games leaked into another split"
    clips_of = {name: {q.clip.relative_path for q in qs} for name, qs in split_questions.items()}
    assert not (clips_of["test_seen"] & (clips_of["train"] | clips_of["val"])), \
        "test_seen clips leaked into train/val"
    assert not ({q.question_id for q in split_questions["train"]}
                & {q.question_id for q in split_questions["val"]}), "train/val share a sample"
    assert clips_of["val"] & clips_of["train"], "expected train and val to share clips by design"
    for q in all_questions:
        assert q.options is None or q.answer in q.options, (
            f"{q.question_id}: answer not among options"
        )
        assert q.options is None or len(q.options) == len(set(q.options)), (
            f"{q.question_id}: duplicate options"
        )
        assert q.options is None or 2 <= len(q.options) <= 5, f"{q.question_id}: bad option count"

    stats_path = out_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote stats -> %s", stats_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="the BARD tree (dataset_paths.csv, captions/, <game>/)",
    )
    parser.add_argument(
        "--out-dir", required=True, help="output dir for the JSONL splits and stats"
    )
    parser.add_argument(
        "--media-json",
        default=None,
        help="probed media table mapping relative path -> [duration, frames, fps] "
        "(written by probe_clip_media.py); defaults to <out-dir>/clip_media.json",
    )
    parser.add_argument(
        "--num-unseen-games", type=int, default=DEFAULT_NUM_UNSEEN_GAMES,
        help="games held out whole for test_unseen (seed 42 reproduces the catalog's holdout)",
    )
    parser.add_argument(
        "--test-seen-clip-fraction", type=float, default=DEFAULT_TEST_SEEN_CLIP_FRACTION,
        help="fraction of each seen game's clips held out for test_seen",
    )
    parser.add_argument(
        "--val-sample-fraction", type=float, default=DEFAULT_VAL_SAMPLE_FRACTION,
        help="per-class fraction of the train+val pool's rows assigned to val",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    generate_dataset(
        dataset_root=Path(args.dataset_root),
        out_dir=Path(args.out_dir),
        media_json=Path(args.media_json) if args.media_json else None,
        num_unseen_games=args.num_unseen_games,
        test_seen_clip_fraction=args.test_seen_clip_fraction,
        val_sample_fraction=args.val_sample_fraction,
        seed=args.seed,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
