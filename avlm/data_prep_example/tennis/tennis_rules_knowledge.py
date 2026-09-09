#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Annotation-triggered tennis-rules knowledge MCQs.

Loads ``configs/rules/tennis_rules_question_bank.json`` and
``configs/rules/tennis_rules_trigger_map.json``, evaluates predicates against a
point annotation, and returns up to ``max_per_point`` matching MCQs.

Selection modes (``trigger_map.selection_mode``):

- ``priority``: top ``max_per_point`` by priority (legacy).
- ``diversity`` (default): at most one question per ``family`` first, with
  family order rotated by point id so secondary slots spread across topics.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SCORE_BRACE_RE = re.compile(r"\{([^}]+)\}")
_POINT_SCORE_RE = re.compile(
    r"(?P<a>0|15|30|40|AD|ad|love)\s*[-–]\s*(?P<b>0|15|30|40|AD|ad|love)",
    re.IGNORECASE,
)
_GAMES_RE = re.compile(r"(?P<g1>\d+)\s*[-–]\s*(?P<g2>\d+)")

# Fallback when an entry omits ``family``. Keep in sync with trigger map.
_DEFAULT_FAMILY_BY_ID: dict[int, str] = {
    1: "serve",
    2: "serve",
    3: "outcome",
    6: "outcome",
    7: "geometry",
    12: "serve",
    13: "outcome",
    15: "geometry",
    16: "geometry",
    18: "score",
    19: "score",
    20: "score",
    21: "score",
    22: "score",
    23: "score",
    24: "score",
    25: "score",
    26: "outcome",
    27: "rally",
    29: "rally",
    30: "serve",
    31: "serve",
    35: "rally",
    36: "serve",
    37: "score",
    38: "score",
    39: "geometry",
}

_DEFAULT_FAMILY_ORDER = ("outcome", "serve", "score", "geometry", "rally")


@dataclass(frozen=True)
class RulesQuestion:
    id: int
    name: str
    title: str
    question_stems: list[str]
    options: list[str]
    correct_answer: str


@dataclass(frozen=True)
class TriggeredRulesMCQ:
    question_id: int
    name: str
    title: str
    question: str
    options: list[str]
    correct_answer: str
    priority: int


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _casefold(text: Any) -> str:
    return _norm(text).casefold()


def load_question_bank(path: Path | str) -> dict[int, RulesQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, RulesQuestion] = {}
    for item in data.get("questions", []):
        qid = int(item["id"])
        stems = item.get("question_stems") or [item["question"]]
        out[qid] = RulesQuestion(
            id=qid,
            name=str(item.get("name") or f"rules_{qid}"),
            title=str(item.get("title") or ""),
            question_stems=[_norm(s) for s in stems if _norm(s)],
            options=[_norm(o) for o in item["options"]],
            correct_answer=_norm(item["correct_answer"]),
        )
    return out


def load_trigger_map(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_score_inside_braces(score_text: Any) -> str:
    text = _norm(score_text).strip('"').strip("'")
    match = _SCORE_BRACE_RE.search(text)
    return match.group(1).strip() if match else ""


def parse_point_score(score_text: Any) -> Optional[tuple[str, str]]:
    """Return (server_points, receiver_points) labels, or None."""
    inside = extract_score_inside_braces(score_text)
    if not inside:
        return None
    # Prefer the segment after the comma (games, point-score).
    point_part = inside.split(",")[-1].strip() if "," in inside else inside.strip()
    match = _POINT_SCORE_RE.search(point_part)
    if not match:
        # Bare 0-0 / love-love style without words already handled by regex;
        # also accept numeric-only already covered.
        return None
    a = match.group("a").upper().replace("LOVE", "0")
    b = match.group("b").upper().replace("LOVE", "0")
    if a == "AD":
        a = "AD"
    if b == "AD":
        b = "AD"
    # Canonicalize love aliases already mapped; keep 0/15/30/40/AD.
    def canon(x: str) -> str:
        x = x.upper()
        if x in {"0", "15", "30", "40", "AD"}:
            return x
        return x

    return canon(a), canon(b)


def parse_games(score_text: Any) -> Optional[tuple[int, int]]:
    inside = extract_score_inside_braces(score_text)
    if not inside or "," not in inside:
        return None
    games_part = inside.split(",", 1)[0].strip()
    match = _GAMES_RE.search(games_part)
    if not match:
        return None
    return int(match.group("g1")), int(match.group("g2"))


def is_deuce(points: Optional[tuple[str, str]]) -> bool:
    return points == ("40", "40")


def is_advantage(points: Optional[tuple[str, str]]) -> bool:
    if not points:
        return False
    return points in {("AD", "40"), ("40", "AD")}


def is_game_start(points: Optional[tuple[str, str]]) -> bool:
    return points == ("0", "0")


def mentions_love(score_text: Any) -> bool:
    text = _casefold(score_text)
    if "love" in text:
        return True
    points = parse_point_score(score_text)
    return bool(points and ("0" in points))


def is_server_game_point(points: Optional[tuple[str, str]]) -> bool:
    """Server can win the game on the next point (server score first)."""
    if not points:
        return False
    s, r = points
    if s == "AD" and r == "40":
        return True
    if s == "40" and r in {"0", "15", "30"}:
        return True
    return False


def is_break_point(points: Optional[tuple[str, str]]) -> bool:
    """Receiver can win the server's game on the next point."""
    if not points:
        return False
    s, r = points
    if r == "AD" and s == "40":
        return True
    if r == "40" and s in {"0", "15", "30"}:
        return True
    return False


def game_won_from_scores(score_before: Any, score_after: Any) -> bool:
    """True when games component advanced (service game finished)."""
    before = parse_games(score_before)
    after = parse_games(score_after)
    if before is None or after is None:
        return False
    return after != before


def server_won_game(score_before: Any, score_after: Any, winner_role: Any) -> bool:
    if not game_won_from_scores(score_before, score_after):
        return False
    return _casefold(winner_role) == "server"


def receiver_won_game(score_before: Any, score_after: Any, winner_role: Any) -> bool:
    if not game_won_from_scores(score_before, score_after):
        return False
    return _casefold(winner_role) == "receiver"


def outcome_label(how_point_ended: Any) -> str:
    return _norm(how_point_ended).split(":", 1)[0].strip()


def _style_index(*parts: Any, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def entry_family(entry: dict[str, Any]) -> str:
    raw = entry.get("family")
    if raw:
        return str(raw).strip().casefold()
    qid = int(entry["id"])
    return _DEFAULT_FAMILY_BY_ID.get(qid, "other")


class RulesKnowledgeSelector:
    """Select contextual rules MCQs for one annotated point."""

    def __init__(
        self,
        bank: dict[int, RulesQuestion],
        trigger_map: dict[str, Any],
    ):
        self.bank = bank
        self.trigger_map = trigger_map
        self.max_per_point = int(trigger_map.get("max_per_point", 2))
        mode = str(trigger_map.get("selection_mode") or "diversity").strip().casefold()
        self.selection_mode = mode if mode in {"diversity", "priority"} else "diversity"
        order = trigger_map.get("family_order") or list(_DEFAULT_FAMILY_ORDER)
        self.family_order = [str(f).strip().casefold() for f in order if str(f).strip()]
        if not self.family_order:
            self.family_order = list(_DEFAULT_FAMILY_ORDER)
        # Rotate among the best N priority matches inside a family (not only #1).
        self.family_top_k = max(1, int(trigger_map.get("family_top_k", 4)))
        self.entries = list(trigger_map.get("questions", []))

    def _predicate(self, name: str, point: Any) -> bool:
        outcome = outcome_label(getattr(point, "how_point_ended", ""))
        attempts_raw = getattr(point, "num_serving_attempts_until_successful", 0)
        try:
            attempts = int(str(attempts_raw).strip())
        except (TypeError, ValueError):
            attempts = 0
        loser = _casefold(getattr(point, "loser_point_ended", ""))
        caption = _casefold(getattr(point, "point_caption", ""))
        description = _casefold(getattr(point, "how_point_ended_description", ""))
        text_blob = f"{caption} {description} {_casefold(getattr(point, 'how_point_ended', ''))}"
        score_before = getattr(point, "score_before", "")
        score_after = getattr(point, "score_after", "")
        before_pts = parse_point_score(score_before)
        after_pts = parse_point_score(score_after)
        winner = getattr(point, "winner", "")
        winner_shot = _casefold(getattr(point, "winner_shot_type", ""))
        loser_shot = _casefold(getattr(point, "loser_shot_type", ""))

        if name == "has_serve_attempt":
            return attempts >= 1 or outcome in {"Ace", "Double Fault"}
        if name.startswith("serve_attempts_eq:"):
            target = int(name.split(":", 1)[1])
            return attempts == target
        if name.startswith("outcome_in:"):
            values = {v.strip() for v in name.split(":", 1)[1].split(",") if v.strip()}
            return outcome in values
        if name.startswith("loser_point_ended_in:"):
            values = {
                v.strip().casefold()
                for v in name.split(":", 1)[1].split(",")
                if v.strip()
            }
            return any(v and v in loser for v in values)
        if name.startswith("text_mentions:"):
            needles = [
                v.strip().casefold()
                for v in name.split(":", 1)[1].split(",")
                if v.strip()
            ]
            return any(n and n in text_blob for n in needles)
        if name.startswith("shot_type_mentions:"):
            needle = name.split(":", 1)[1].strip().casefold()
            return bool(needle) and (needle in winner_shot or needle in loser_shot)
        if name == "has_parseable_scores":
            return before_pts is not None and after_pts is not None
        if name == "score_before_is_deuce":
            return is_deuce(before_pts)
        if name == "score_after_is_deuce":
            return is_deuce(after_pts)
        if name == "score_before_is_advantage":
            return is_advantage(before_pts)
        if name == "score_after_is_advantage":
            return is_advantage(after_pts)
        if name == "score_mentions_love":
            return mentions_love(score_before) or mentions_love(score_after)
        if name == "score_before_is_break_point":
            return is_break_point(before_pts)
        if name == "score_before_is_server_game_point":
            return is_server_game_point(before_pts)
        if name == "score_before_is_game_start":
            return is_game_start(before_pts)
        if name == "not_score_before_is_game_start":
            return before_pts is not None and not is_game_start(before_pts)
        if name == "game_won_from_score":
            return game_won_from_scores(score_before, score_after)
        if name == "server_won_game":
            return server_won_game(score_before, score_after, winner)
        if name == "receiver_won_game":
            return receiver_won_game(score_before, score_after, winner)
        return False

    def _triggers_match(self, triggers: dict[str, Any], point: Any) -> bool:
        any_preds = triggers.get("any") or []
        all_preds = triggers.get("all") or []
        if any_preds and not any(self._predicate(p, point) for p in any_preds):
            return False
        if all_preds and not all(self._predicate(p, point) for p in all_preds):
            return False
        # Require at least one predicate group to be present and non-empty.
        return bool(any_preds or all_preds)

    def matching_entries(self, point: Any) -> list[dict[str, Any]]:
        matched = []
        for entry in self.entries:
            if entry.get("mode") != "contextual":
                continue
            qid = int(entry["id"])
            if qid not in self.bank:
                continue
            if not self._triggers_match(entry.get("triggers") or {}, point):
                continue
            matched.append(entry)
        matched.sort(key=lambda e: (int(e.get("priority", 100)), int(e["id"])))
        return matched

    def _rank_priority(self, matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return matched[: self.max_per_point]

    def _family_representative(
        self,
        candidates: list[dict[str, Any]],
        *,
        point: Any,
        family: str,
    ) -> dict[str, Any]:
        """Pick among top-priority matches in a family (rotated by point id)."""
        top = candidates[: self.family_top_k]
        idx = _style_index(
            getattr(point, "id", ""),
            "rules_within_family",
            family,
            modulo=len(top),
        )
        return top[idx]

    def _rank_diversity(
        self,
        matched: list[dict[str, Any]],
        point: Any,
    ) -> list[dict[str, Any]]:
        """One pick per family (rotated), families rotated by point id, then fill."""
        if not matched:
            return []
        by_family: dict[str, list[dict[str, Any]]] = {}
        for entry in matched:  # priority-sorted
            fam = entry_family(entry)
            by_family.setdefault(fam, []).append(entry)

        best_by_family = {
            fam: self._family_representative(cands, point=point, family=fam)
            for fam, cands in by_family.items()
        }

        # Include any unexpected families after the configured order.
        extra = sorted(f for f in best_by_family if f not in self.family_order)
        base_order = list(self.family_order) + extra
        offset = _style_index(
            getattr(point, "id", ""),
            "rules_family_rotate",
            modulo=len(base_order),
        )
        rotated = base_order[offset:] + base_order[:offset]

        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        for fam in rotated:
            if len(selected) >= self.max_per_point:
                break
            entry = best_by_family.get(fam)
            if entry is None:
                continue
            qid = int(entry["id"])
            if qid in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(qid)

        if len(selected) < self.max_per_point:
            for entry in matched:
                if len(selected) >= self.max_per_point:
                    break
                qid = int(entry["id"])
                if qid in selected_ids:
                    continue
                selected.append(entry)
                selected_ids.add(qid)
        return selected

    def ranked_entries(self, point: Any) -> list[dict[str, Any]]:
        matched = self.matching_entries(point)
        if self.selection_mode == "priority":
            return self._rank_priority(matched)
        return self._rank_diversity(matched, point)

    def select_for_point(
        self,
        point: Any,
        *,
        selection_rank: int = 0,
    ) -> Optional[TriggeredRulesMCQ]:
        ranked = self.ranked_entries(point)
        if selection_rank < 0 or selection_rank >= len(ranked):
            return None
        if selection_rank >= self.max_per_point:
            return None
        entry = ranked[selection_rank]
        qid = int(entry["id"])
        question = self.bank[qid]
        stem_idx = _style_index(
            getattr(point, "id", ""),
            qid,
            selection_rank,
            modulo=len(question.question_stems),
        )
        stem = question.question_stems[stem_idx]
        return TriggeredRulesMCQ(
            question_id=qid,
            name=question.name,
            title=question.title,
            question=stem,
            options=list(question.options),
            correct_answer=question.correct_answer,
            priority=int(entry.get("priority", 100)),
        )


def default_config_paths(base_dir: Path | str | None = None) -> tuple[Path, Path]:
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    cfg = root / "configs" / "rules"
    return cfg / "tennis_rules_question_bank.json", cfg / "tennis_rules_trigger_map.json"


def build_default_selector(base_dir: Path | str | None = None) -> RulesKnowledgeSelector:
    bank_path, map_path = default_config_paths(base_dir)
    return RulesKnowledgeSelector(load_question_bank(bank_path), load_trigger_map(map_path))
