#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tennis MCQ + QA Generator - Configuration Driven

Generates multiple-choice and open-ended question/answer pairs from preprocessed
tennis annotations and writes them in **HuggingFace conversation format** (one
per-video JSON file). Question templates and options are defined in the JSON
config files under ``configs/``.

Video Processing:
----------------
Reads the relative video path for each point from the per-video annotation JSON
(each point should carry a ``video`` field with the relative clip path) and emits
it as a ``{"type": "video", "path": ...}`` content block in each conversation
record.

Usage:
------
1. Place per-video annotation JSONs under ``raw_tennis_data_input/`` (one file per game).
2. Run this script to generate conversations:
   python generate_mcq_qa.py

3. Use in code:
   generator = ConfigDrivenMCQGenerator()
   generator.process_data("raw_tennis_data_input", "training_tennis_data_output/mcq_qa_per_video")
"""

import argparse
import json
import os
import random
import re
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from tqdm import tqdm

# Import LLM helper functions
from llm_helper import generate_tennis_mcq_with_llm

# Configure logging - set to WARNING to reduce API call logs
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_option_text(value: Any) -> str:
    """Strip annotation noise so MCQ options compare/display cleanly.

    Ground-truth scores are often stored with wrapping quotes
    (``"Player vs. Player: {0-0, 15-0}"``), and numeric fields sometimes carry
    leading/trailing whitespace (``"  1"``). Those variants must collapse to the
    same option text or distractor dedupe fails.
    """
    text = str(value if value is not None else "").strip().strip("`").strip()
    text = _SEGMENT_TAG_RE.sub("", text)
    while True:
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1].strip()
            continue
        if text.startswith('"'):
            text = text[1:].strip()
            continue
        if text.endswith('"'):
            text = text[:-1].strip()
            continue
        break
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize common score punctuation spacing for stable display.
    if "{" in text and "vs" in text.casefold():
        text = re.sub(r"\s*:\s*", ": ", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\s*\{\s*", "{", text)
        text = re.sub(r"\s*\}\s*", "}", text)
        text = re.sub(r"\s*\(\s*", "(", text)
        text = re.sub(r"\s*\)\s*", ")", text)
        text = re.sub(r"\s+", " ", text).strip()
    return fix_common_indefinite_articles(text)


# Leading annotation segment tag such as ``[dEBO7DcYhtQ_seg_04]`` that some
# ground-truth captions/descriptions carry and that must not leak into answers.
_SEGMENT_TAG_RE = re.compile(r"^\s*\[[\w-]+_seg_\d+\]\s*")

# Runs of spaces/tabs (but not newlines) used to tidy assembled prompt text.
_HSPACE_RUN_RE = re.compile(r"[ \t]{2,}")

# Open-ended answers whose entire content is a non-answer placeholder value.
_EMPTY_ANSWER_VALUES = {"", "n/a", "na", "none", "null", "unknown", "unknow"}


_MCQ_OPTION_LINE_RE = re.compile(r"^\([A-D]\)\s+")
# Common tennis annotation article mistakes (``a overhead``, ``a ace``, ``an flat``).
_A_BEFORE_VOWEL_RE = re.compile(
    r"\ba\s+(?=ace\b|ad\b|approach\b|inside\b|open\b|overhead\b|umpire\b)",
    re.IGNORECASE,
)
_AN_BEFORE_CONSONANT_RE = re.compile(
    r"\ban\s+(?=flat\b|slice\b|volley\b|drop\b|lob\b|half\b|passing\b|block\b|forehand\b|backhand\b)",
    re.IGNORECASE,
)


def fix_common_indefinite_articles(text: str) -> str:
    """Correct frequent a/an mistakes in tennis annotation free text."""
    text = _A_BEFORE_VOWEL_RE.sub("an ", text)
    text = _AN_BEFORE_CONSONANT_RE.sub("a ", text)
    return text


def normalize_question_text(text: Any) -> str:
    """Tidy assembled question/prompt text.

    Player descriptions sometimes contain embedded newlines and trailing spaces
    before separators (``...earrings.\\n, Rafael...``, ``...shoes. )``). Collapse
    the question stem onto one line, fix punctuation artifacts there, and keep
    MCQ option rows on their own lines so option text stays aligned with the
    labeled answer.
    """
    if not text:
        return text
    lines = [_HSPACE_RUN_RE.sub(" ", line).strip() for line in str(text).split("\n")]
    stem_parts: list[str] = []
    option_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        if _MCQ_OPTION_LINE_RE.match(line):
            option_lines.append(line)
        else:
            stem_parts.append(line)
    stem = " ".join(stem_parts)
    stem = re.sub(r"\.\s*,\s*", "; ", stem)
    stem = re.sub(r"\s+,", ",", stem)
    stem = re.sub(r"\s+\)", ")", stem)
    stem = _HSPACE_RUN_RE.sub(" ", stem).strip()
    stem = fix_common_indefinite_articles(stem)
    parts = ([stem] if stem else []) + option_lines
    return "\n".join(parts).strip()


def is_empty_answer_value(value: Any) -> bool:
    """True when a free-text answer is effectively a non-answer placeholder."""
    return str(value if value is not None else "").strip().casefold() in _EMPTY_ANSWER_VALUES


def clean_freetext_answer(value: Any) -> str:
    """Tidy a free-text (open-ended) answer without altering its meaning.

    Ground-truth captions and descriptions occasionally carry annotation noise
    that should never surface to a model: a leading segment tag
    (``[<video>_seg_NN] ``), wrapping quotes, stray backticks, doubled
    single-quote wrappers, runs of internal whitespace, and common a/an
    mistakes.
    """
    text = str(value if value is not None else "")
    # Remove stray backticks and collapse whitespace runs.
    text = text.replace("`", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Some player descriptions quote names with doubled apostrophes
    # (``''Rafael Nadal''``). Remove only balanced doubled wrappers, preserving
    # ordinary apostrophes in names and contractions.
    text = re.sub(r"''([^']+?)''", r"\1", text)
    # Alternately peel wrapping quotes and a leading segment identifier tag
    # (either may nest the other) until neither remains.
    while True:
        stripped = _SEGMENT_TAG_RE.sub("", text)
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            stripped = stripped[1:-1].strip()
        if stripped == text:
            break
        text = stripped
    return fix_common_indefinite_articles(text.strip())


def option_dedupe_key(value: Any) -> str:
    """Casefold identity key used to reject duplicate MCQ choices.

    For score-like strings, insignificant whitespace around punctuation
    (``Kenin :`` vs ``Kenin:``, ``7-6 (7-5)`` vs ``7-6(7-5)``) must also
    collapse, or annotated answers and generated distractors collide.
    """
    text = normalize_option_text(value).casefold()
    # Drop spaces next to score punctuation, then remove remaining spaces so
    # comparison is whitespace-invariant while display text stays readable.
    text = re.sub(r"\s*([:{},()])\s*", r"\1", text)
    return re.sub(r"\s+", "", text)


def unique_mcq_options(
    correct_answer: Any,
    candidates: List[Any],
    *,
    max_options: int = 4,
) -> List[str]:
    """Build a unique option list with the correct answer first.

    Candidates that normalize to the same key as the correct answer (or an
    earlier choice) are dropped. Returns at most ``max_options`` entries.
    """
    correct = normalize_option_text(correct_answer)
    if not correct:
        return []
    unique = [correct]
    seen = {option_dedupe_key(correct)}
    for candidate in candidates:
        text = normalize_option_text(candidate)
        if not text:
            continue
        key = option_dedupe_key(text)
        if key in seen:
            continue
        unique.append(text)
        seen.add(key)
        if len(unique) >= max_options:
            break
    return unique


# Data Classes
@dataclass
class Metadata:
    """Metadata information for tennis dataset."""
    name: str = ""
    status: str = ""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Metadata':
        """Create Metadata from JSON data."""
        return cls(
            name=data.get('name', ''),
            status=data.get('status', ''),
        )


@dataclass
class VideoInfo:
    """Video information data."""
    video_id: str = ""
    video_duration: float = 0.0

    @classmethod
    def from_json(cls, instances: List[Dict[str, Any]]) -> 'VideoInfo':
        """Create VideoInfo from instances data."""
        video_info = cls()
        
        for instance in instances:
            attributes = instance.get('attributes', [])
            if not attributes:
                continue
                
            if instance.get('className') == 'video_id':
                video_info.video_id = attributes[0].get('name', '')
            elif instance.get('className') == 'video_duration':
                video_info.video_duration = attributes[0].get('name', 0.0)
        
        return video_info


@dataclass
class Player:
    """Player information."""
    name: str = ""
    description: str = ""

    @classmethod
    def from_json(cls, name_instance: Dict[str, Any], desc_instance: Dict[str, Any] = None) -> 'Player':
        """Create Player from name and description instances."""
        name_attrs = name_instance.get('attributes', [])
        desc_attrs = desc_instance.get('attributes', []) if desc_instance else []
        
        return cls(
            name=name_attrs[0].get('name', '') if name_attrs else '',
            description=desc_attrs[0].get('name', '') if desc_attrs else ''
        )


@dataclass
class MatchContext:
    """Match context information."""
    player_1: Optional[Player] = None
    player_2: Optional[Player] = None
    general_context: str = ""
    
    @classmethod
    def from_json(cls, instances: List[Dict[str, Any]]) -> 'MatchContext':
        """Create MatchContext from instances data."""
        context = cls()
        
        # Extract player information - find first non-empty instances
        player_1_name = None
        player_1_desc = None
        player_2_name = None
        player_2_desc = None
        
        for instance in instances:
            attributes = instance.get('attributes', [])
            if not attributes:
                continue
                
            class_name = instance.get('className')
            name_value = attributes[0].get('name', '')
            
            # Only use the first non-empty player information found
            if class_name == 'player_1_name' and name_value and name_value.strip() and not player_1_name:
                player_1_name = instance
            elif class_name == 'player_1_description' and name_value and name_value.strip() and not player_1_desc:
                player_1_desc = instance
            elif class_name == 'player_2_name' and name_value and name_value.strip() and not player_2_name:
                player_2_name = instance
            elif class_name == 'player_2_description' and name_value and name_value.strip() and not player_2_desc:
                player_2_desc = instance
        
        # Create player objects if we have at least name (description is optional)
        if player_1_name:
            context.player_1 = Player.from_json(player_1_name, player_1_desc)
        if player_2_name:
            context.player_2 = Player.from_json(player_2_name, player_2_desc)
        
        return context


@dataclass
class PointData:
    """Individual point data."""
    id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    server: str = ""
    receiver: str = ""
    winner: str = ""
    num_serving_attempts_until_successful: int = 0
    num_shots_exchanged: int = 0
    score_before: str = ""
    score_after: str = ""
    how_point_ended: str = ""
    how_point_ended_description: str = ""
    point_caption: str = ""
    
    # Score parsing for placeholders
    current_score: str = ""
    serving_score: str = ""
    receiving_score: str = ""
    
    # Audio cues field
    audio_cues: str = ""
    
    # Video field (extracted video filename)
    video: str = ""
    
    def extract_score_info(self):
        """Extract score information for placeholders."""
        # Extract current score from score_before or score_after
        if self.score_before:
            # Extract score from format like "Federer vs. Benneteau: {1-0, 30-30}"
            import re
            score_match = re.search(r'\{([^}]+)\}', self.score_before)
            if score_match:
                self.current_score = score_match.group(1)
        
        if self.score_after:
            # Extract score from format like "Federer vs. Benneteau: {1-0, 30-40}"
            import re
            score_match = re.search(r'\{([^}]+)\}', self.score_after)
            if score_match:
                self.current_score = score_match.group(1)
        
        # Set serving and receiving scores based on server
        if self.server == "Player 1":
            self.serving_score = self.current_score
            self.receiving_score = self.current_score  # Could be enhanced with actual receiving score
        elif self.server == "Player 2":
            self.serving_score = self.current_score
            self.receiving_score = self.current_score  # Could be enhanced with actual receiving score

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Coerce annotation values (often strings like "2") to int."""
        if value is None or value == "":
            return default
        try:
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return default

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'PointData':
        """Create PointData from JSON point data."""
        point = cls(
            id=data.get('id', ''),
            start_time=data.get('start_time', 0.0),
            end_time=data.get('end_time', 0.0),
            server=data.get('server', ''),
            receiver=data.get('receiver', ''),
            winner=data.get('winner', ''),
            # Annotations store these as strings ("1", "2"); coerce so
            # numeric_range / categorical matching works reliably.
            num_serving_attempts_until_successful=cls._safe_int(data.get('num_serving_attempts_until_successful'), 0),
            num_shots_exchanged=cls._safe_int(data.get('num_shots_exchanged'), 0),
            score_before=normalize_option_text(data.get('score_before', '')),
            score_after=normalize_option_text(data.get('score_after', '')),
            how_point_ended=data.get('how_point_ended', ''),
            how_point_ended_description=data.get('how_point_ended_description', ''),
            point_caption=data.get('point_caption', ''),
            audio_cues=data.get('audio_cues', ''),
            video=data.get('video', '')  # Read video field from preprocessed data
        )
        
        # Extract score information for placeholders
        point.extract_score_info()
        
        return point


@dataclass
class PointAnnotations:
    """Point-by-point annotations data."""
    points: List[PointData] = field(default_factory=list)

    @classmethod
    def from_json(cls, instances: List[Dict[str, Any]]) -> 'PointAnnotations':
        """Create PointAnnotations from instances data."""
        annotations = cls()
        
        for instance in instances:
            if instance.get('className') == 'table':
                attributes = instance.get('attributes', [])
                if not attributes:
                    continue
                    
                table_data = attributes[0].get('name', [])
                if isinstance(table_data, list):
                    for point_data in table_data:
                        point = PointData.from_json(point_data)
                        annotations.points.append(point)
                # Remove the break statement to process all table instances
        
        return annotations


@dataclass
class TennisDataset:
    """Complete tennis dataset wrapper."""
    metadata: Metadata
    video_info: VideoInfo
    match_context: MatchContext
    point_annotations: PointAnnotations

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'TennisDataset':
        """Create TennisDataset from complete JSON data."""
        metadata = Metadata.from_json(data.get('metadata', {}))
        instances = data.get('instances', [])
        
        video_info = VideoInfo.from_json(instances)
        match_context = MatchContext.from_json(instances)
        point_annotations = PointAnnotations.from_json(instances)
        
        dataset = cls(
            metadata=metadata,
            video_info=video_info,
            match_context=match_context,
            point_annotations=point_annotations,
        )
        
        dataset._raw_instances = instances
        
        return dataset

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset."""
        return {
            "video_id": self.video_info.video_id,
            "video_name": self.metadata.name,
            "duration": self.video_info.video_duration,
            "status": self.metadata.status,
            "player_1": self.match_context.player_1.name if self.match_context.player_1 else "",
            "player_2": self.match_context.player_2.name if self.match_context.player_2 else "",
            "total_points": len(self.point_annotations.points)
        }

class ConfigDrivenMCQGenerator:
    """Configuration-driven MCQ and QA generator."""

    @staticmethod
    def _filter_templates(templates: Dict[str, Any],
                          allowed: Optional[Set[str]]) -> Dict[str, Any]:
        """Return templates whose names are in *allowed*, or all if *allowed* is None."""
        if allowed is None:
            return dict(templates)
        return {name: tmpl for name, tmpl in templates.items() if name in allowed}

    def __init__(self, mcq_config_file: str = None, qa_config_file: str = None,
                 max_workers: int = 4, api_rate_limit: float = 1.0,
                 enable_progress_bar: bool = True,
                 question_types: Optional[List[str]] = None,
                 video_root: Optional[str] = None):
        """
        Initialize MCQ and QA generator with configuration files.

        Args:
            mcq_config_file: Path to MCQ configuration JSON file
            qa_config_file: Path to QA configuration JSON file
            max_workers: Maximum number of threads for parallel API calls
            api_rate_limit: Minimum seconds between API calls (rate limiting)
            enable_progress_bar: Whether to show progress bars
            question_types: Names of question templates to generate (e.g.
                ``"server_winner"``, ``"point_caption"``). If ``None``, every
                template defined in the MCQ and QA config files is generated.
            video_root: Directory used to resolve relative point clip paths.
                When set, points whose resolved clip is missing/empty are skipped.
        """
        mcq_config = self.load_config(mcq_config_file) if mcq_config_file else {}
        qa_config = self.load_config(qa_config_file) if qa_config_file else {}

        all_mcq = mcq_config.get('question_templates', {})
        all_qa = qa_config
        known = set(all_mcq) | set(all_qa)

        if question_types is None:
            allowed: Optional[Set[str]] = None
        else:
            allowed = set(question_types)
            unknown = allowed - known
            if unknown:
                logging.warning(f"Unknown question types (ignored): {sorted(unknown)}")
            allowed &= known
            skipped = known - allowed
            if skipped:
                logging.info(f"Not generating {len(skipped)} question types: {sorted(skipped)}")

        self.question_templates = self._filter_templates(all_mcq, allowed)
        self.qa_templates = self._filter_templates(all_qa, allowed)
        self.video_root = os.path.abspath(video_root) if video_root else None
        
        # Multithreading configuration
        self.max_workers = max_workers
        self.api_rate_limit = api_rate_limit
        self.enable_progress_bar = enable_progress_bar
        
        # Rate limiting for parallel API calls
        self._api_call_times = []  # Queue of recent API call timestamps
        self._api_call_lock = Lock()  # Only for managing the timestamp queue
        
        # Statistics tracking
        self._api_call_count = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._retry_attempts = 0
        self._total_api_time = 0.0
        self._skipped_missing_video_ids: set = set()

    def resolve_video_path(self, relative_or_absolute: Optional[str]) -> Optional[str]:
        """Resolve a point clip path against ``video_root`` when needed."""
        if not relative_or_absolute:
            return None
        path = str(relative_or_absolute).strip()
        if not path:
            return None
        if os.path.isabs(path):
            return path
        if not self.video_root:
            return path
        return os.path.join(self.video_root, path)

    def has_valid_video(self, point: Any) -> bool:
        """True when the point has a resolvable, existing, non-empty clip.

        If ``video_root`` is unset, only require a non-empty path string (legacy
        behavior). When ``video_root`` is set, missing/empty files are rejected.
        """
        video = getattr(point, "video", None)
        if not video or not str(video).strip():
            return False
        if not self.video_root:
            return True
        resolved = self.resolve_video_path(video)
        return bool(
            resolved
            and os.path.isfile(resolved)
            and os.path.getsize(resolved) > 0
        )

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not config_file:
            return {}
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logging.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration from {config_file}: {e}")
            raise
    
    def _rate_limited_api_call(self, api_func: Callable, *args, **kwargs) -> Any:
        """
        Execute API call with parallel-friendly rate limiting and error handling.
        
        Args:
            api_func: The API function to call
            *args: Arguments for the API function
            **kwargs: Keyword arguments for the API function
            
        Returns:
            Result of the API call
        """
        max_retries = kwargs.pop('max_retries', 2)
        retry_delay = kwargs.pop('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            # Parallel-friendly rate limiting
            self._wait_for_rate_limit()
            
            # Execute API call with timing
            start_time = time.time()
            try:
                with self._api_call_lock:
                    self._api_call_count += 1
                
                result = api_func(*args, **kwargs)
                
                with self._api_call_lock:
                    self._successful_calls += 1
                return result
            except Exception as e:
                with self._api_call_lock:
                    self._failed_calls += 1
                
                is_last_attempt = attempt == max_retries - 1
                
                if is_last_attempt:
                    logging.error(f"API call failed after {max_retries} attempts, discarding: {e}")
                    raise
                else:
                    with self._api_call_lock:
                        self._retry_attempts += 1
                    logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(retry_delay)
            finally:
                with self._api_call_lock:
                    self._total_api_time += time.time() - start_time
    
    def _wait_for_rate_limit(self) -> None:
        """
        Wait if necessary to respect rate limits in a parallel-friendly way.
        Uses a simplified approach for maximum throughput.
        """
        current_time = time.time()
        
        with self._api_call_lock:
            # Clean up old timestamps (only keep recent ones)
            cutoff_time = current_time - 1.0  # Keep last 1 second of calls
            self._api_call_times = [t for t in self._api_call_times if t > cutoff_time]
            
            # Simple rate limiting: if we have too many recent calls, wait briefly
            if len(self._api_call_times) >= self.max_workers:
                # If we have max_workers calls in the last second, wait for rate limit
                wait_time = self.api_rate_limit
            else:
                wait_time = 0
            
            # Record this call's timestamp
            self._api_call_times.append(current_time)
        
        # Brief wait outside the lock
        if wait_time > 0:
            time.sleep(wait_time)
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API call statistics."""
        return {
            "total_calls": self._api_call_count,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "retry_attempts": self._retry_attempts,
            "success_rate": self._successful_calls / max(self._api_call_count, 1) * 100,
            "total_api_time": self._total_api_time,
            "average_call_time": self._total_api_time / max(self._successful_calls, 1)
        }
    
    def extract_player_names(self, dataset: TennisDataset) -> Dict[str, str]:
        """
        Extract player names from dataset using object access.
        
        Args:
            dataset: TennisDataset object
            
        Returns:
            Dictionary with player1_name and player2_name
        """
        # Debug: Check if player objects exist
        logging.info(f"Player 1 object: {dataset.match_context.player_1}")
        logging.info(f"Player 2 object: {dataset.match_context.player_2}")
        
        # Direct access to player names from match context
        player1_name = dataset.match_context.player_1.name if dataset.match_context.player_1 else ""
        player2_name = dataset.match_context.player_2.name if dataset.match_context.player_2 else ""
        
        logging.info(f"Initial player names: '{player1_name}' vs '{player2_name}'")
        
        # If player names are empty, try to extract from general context
        if not player1_name or player1_name == "":
            general_context = dataset.match_context.general_context
            logging.info(f"General context: {general_context}")
            
            if general_context:
                # Try to extract player names from general context
                # Look for patterns like "Player1 vs Player2" or "Player1 against Player2"
                import re
                vs_pattern = r'(\w+(?:\s+\w+)*)\s+(?:vs\.?|against|v\.?)\s+(\w+(?:\s+\w+)*)'
                match = re.search(vs_pattern, general_context, re.IGNORECASE)
                
                if match:
                    player1_name = match.group(1).strip()
                    player2_name = match.group(2).strip()
                    logging.info(f"Extracted from general context: '{player1_name}' vs '{player2_name}'")
                else:
                    # Try to extract from colon format like "Federer vs. Benneteau: {1-0, 30-30}"
                    colon_pattern = r'(\w+(?:\s+\w+)*)\s+vs\.?\s+(\w+(?:\s+\w+)*)\s*:'
                    match = re.search(colon_pattern, general_context, re.IGNORECASE)
                    
                    if match:
                        player1_name = match.group(1).strip()
                        player2_name = match.group(2).strip()
                        logging.info(f"Extracted from colon format: '{player1_name}' vs '{player2_name}'")
        
        # If still empty, try to extract from score_before/score_after in points
        if (not player1_name or player1_name == "") and dataset.point_annotations.points:
            for point in dataset.point_annotations.points:
                if point.score_before:
                    # Try to extract from score format like "Federer vs. Benneteau: {1-0, 30-30}"
                    import re
                    score_pattern = r'(\w+(?:\s+\w+)*)\s+vs\.?\s+(\w+(?:\s+\w+)*)\s*:'
                    match = re.search(score_pattern, point.score_before, re.IGNORECASE)
                    
                    if match:
                        player1_name = match.group(1).strip()
                        player2_name = match.group(2).strip()
                        logging.info(f"Extracted from score_before: '{player1_name}' vs '{player2_name}'")
                        break
        
        # Final fallback to default values
        if not player1_name or player1_name == "":
            player1_name = "Player 1"
        if not player2_name or player2_name == "":
            player2_name = "Player 2"
        
        logging.info(f"Final player names: {player1_name} vs {player2_name}")
        
        return {
            "player1_name": player1_name,
            "player2_name": player2_name
        }
    
    def replace_placeholders(self, text: str, context_data: Dict[str, Any]) -> str:
        """
        Replace all placeholders with actual values from context data.
        
        Args:
            text: Text containing placeholders
            context_data: Dictionary with placeholder values
            
        Returns:
            Text with replaced placeholders
        """
        if not text:
            return text
        
        # Replace all placeholders found in the text
        for placeholder, value in context_data.items():
            if placeholder in text and value is not None:
                text = text.replace(placeholder, str(value))
        
        return text
    
    def build_context_data(self, point: PointData, dataset: TennisDataset, player_names: Dict[str, str]) -> Dict[str, Any]:
        """
        Build context data dictionary for placeholder replacement using object access.
        
        Args:
            point: PointData object for the current point
            dataset: TennisDataset object
            player_names: Dictionary with player1_name and player2_name
            
        Returns:
            Dictionary with placeholder values
        """
        context_data = {}
        
        # Player information
        context_data["<player1>"] = player_names.get("player1_name", "Player 1")
        context_data["<player2>"] = player_names.get("player2_name", "Player 2")
        
        # Player descriptions - use object access
        player1_desc = ""
        player2_desc = ""
        general_context = ""
        
        if dataset.match_context.player_1:
            player1_desc = dataset.match_context.player_1.description
        if dataset.match_context.player_2:
            player2_desc = dataset.match_context.player_2.description
        if dataset.match_context.general_context:
            general_context = dataset.match_context.general_context
        
        # Try to extract better descriptions from dataset instances
        if not player1_desc or player1_desc == "":
            player1_desc = self.extract_player_description_from_instances(dataset, "player_1", player_names.get('player1_name', 'Player 1'))
        if not player2_desc or player2_desc == "":
            player2_desc = self.extract_player_description_from_instances(dataset, "player_2", player_names.get('player2_name', 'Player 2'))
        if not general_context or general_context == "":
            general_context = f"tennis match between {player_names.get('player1_name', 'Player 1')} and {player_names.get('player2_name', 'Player 2')}"
        
        context_data["<player1_description>"] = clean_freetext_answer(player1_desc)
        context_data["<player2_description>"] = clean_freetext_answer(player2_desc)
        context_data["<general_context>"] = clean_freetext_answer(general_context)
        
        # Score context - direct attribute access
        context_data["<current_score>"] = point.current_score or "0-0"
        context_data["<serving_score>"] = point.serving_score or point.current_score or "0-0"
        context_data["<receiving_score>"] = point.receiving_score or point.current_score or "0-0"
        context_data["<score_before>"] = point.score_before or ""
        context_data["<score_after>"] = point.score_after or ""
        
        # Point context - direct attribute access
        context_data["<rally_length>"] = str(point.num_shots_exchanged or 3)
        
        # Video context - now using point's video field directly
        context_data["<video_filename>"] = point.video or "unknown.mp4"
        
        return context_data
    
    def extract_player_description_from_instances(self, dataset: TennisDataset, player_key: str, player_name: str) -> str:
        """
        Extract player description from dataset instances.
        
        Args:
            dataset: TennisDataset object
            player_key: Either "player_1" or "player_2"
            player_name: Player name
            
        Returns:
            Player description string
        """
        # Look for player description in raw instances - find the first non-empty description
        if hasattr(dataset, '_raw_instances'):
            for instance in dataset._raw_instances:
                class_name = instance.get("className", "")
                if class_name == f"{player_key}_description":
                    attributes = instance.get("attributes", [])
                    if attributes:
                        desc = attributes[0].get("name", "")
                        if desc and desc.strip() and desc.strip() != "":
                            logging.info(f"Found {player_key} description: {desc[:100]}...")
                            return desc.strip()
        
        # If no description found, try to create a more detailed fallback
        # Check for general context to provide richer fallback
        context_info = ""
        if hasattr(dataset, 'match_context') and dataset.match_context.general_context:
            context_info = f" in {dataset.match_context.general_context}"
        
        # Create a more informative fallback description
        fallback_desc = f"tennis player {player_name}{context_info}"
        logging.info(f"Using fallback description for {player_key}: {fallback_desc}")
        return fallback_desc
    
    def generate_score_distractors(self, score_before: str, correct_answer: str, context_data: Dict[str, Any]) -> List[str]:
        """
        Generate plausible but incorrect score options for score_after questions.
        Uses a state machine approach to generate realistic distractors.
        
        Args:
            score_before: The score before the point
            correct_answer: The correct score after the point
            context_data: Context data with player names
            
        Returns:
            List of distractor score options
        """
        from collections import deque
        import re
        
        # Extract player names
        player1 = context_data.get("<player1>", "Player 1")
        player2 = context_data.get("<player2>", "Player 2")

        # Annotations often wrap scores in quotes; normalize before parsing and
        # before comparing against generated distractors.
        score_before = normalize_option_text(score_before)
        correct_answer = normalize_option_text(correct_answer)
        
        # Parse score format. Allow hyphenated surnames and optional quotes.
        score_pattern = (
            r'(.+?)\s+vs\.?\s+(.+?)\s*:\s*(.*?)\s*\{([^}]+)\}'
        )
        
        before_match = re.search(score_pattern, score_before) if score_before else None
        after_match = re.search(score_pattern, correct_answer) if correct_answer else None
        
        if not before_match or not after_match:
            return self._generate_fallback_distractors(player1, player2, correct_answer)
        
        # Extract score components
        before_sets = before_match.group(3).strip()
        after_sets = after_match.group(3).strip()
        before_games_points = before_match.group(4).strip()
        after_games_points = after_match.group(4).strip()
        
        # Parse games and points
        before_games, before_points = self._parse_games_points(before_games_points)
        after_games, after_points = self._parse_games_points(after_games_points)
        
        # Generate distractors using state machine
        distractors = self._generate_distractors_with_state_machine(
            player1, player2, before_sets, after_sets,
            before_games, before_points, after_games, after_points, correct_answer
        )
        
        return distractors
    
    def _generate_fallback_distractors(self, player1: str, player2: str, correct_answer: str = "") -> List[str]:
        """Generate fallback distractors when parsing fails."""
        correct_answer = normalize_option_text(correct_answer)
        correct_key = option_dedupe_key(correct_answer)
        # Try to extract the format from correct_answer if available
        if ':' in correct_answer:
            name_part = correct_answer.split(':', 1)[0].strip()
            candidates = [
                f"{name_part}: {{0-0, 15-0}}",
                f"{name_part}: {{0-0, 0-15}}",
                f"{name_part}: {{1-0, 0-0}}",
                f"{name_part}: {{0-1, 0-0}}",
                f"{name_part}: {{0-0, 30-0}}",
            ]
        else:
            candidates = [
                f"{player1} vs. {player2}: {{0-0, 15-0}}",
                f"{player1} vs. {player2}: {{0-0, 0-15}}",
                f"{player1} vs. {player2}: {{1-0, 0-0}}",
                f"{player1} vs. {player2}: {{0-1, 0-0}}",
                f"{player1} vs. {player2}: {{0-0, 30-0}}",
            ]
        distractors = []
        for option in candidates:
            option = normalize_option_text(option)
            if option_dedupe_key(option) == correct_key:
                continue
            distractors.append(option)
            if len(distractors) >= 3:
                break
        return distractors
    
    def _parse_games_points(self, games_points: str) -> tuple:
        """Parse games and points from string like '3-2, 40-30'."""
        if ',' in games_points:
            games, points = [x.strip() for x in games_points.split(',', 1)]
        else:
            games, points = games_points, "0-0"
        return games, points
    
    def _get_tennis_state_transitions(self) -> Dict[str, List[str]]:
        """
        Define possible state transitions for tennis scoring.
        Returns a dictionary mapping current score to possible next scores.
        """
        return {
            # Regular scoring
            "0-0": ["15-0", "0-15"],
            "15-0": ["30-0", "15-15"],
            "0-15": ["15-15", "0-30"],
            "30-0": ["40-0", "30-15"],
            "15-15": ["30-15", "15-30"],
            "0-30": ["15-30", "0-40"],
            "40-0": ["0-0", "40-15"],  # Game can end or continue
            "30-15": ["40-15", "30-30"],
            "15-30": ["30-30", "15-40"],
            "0-40": ["15-40", "0-0"],  # Game can end or continue
            "40-15": ["0-0", "40-30"],  # Game can end or continue
            "30-30": ["40-30", "30-40"],
            "15-40": ["30-40", "0-0"],  # Game can end or continue
            "40-30": ["0-0", "40-40"],  # Game can end or go to deuce
            "30-40": ["40-40", "0-0"],  # Game can end or go to deuce
            
            # Deuce and advantage
            "40-40": ["AD-40", "40-AD"],
            "AD-40": ["0-0", "40-40"],  # Game can end or back to deuce
            "40-AD": ["40-40", "0-0"],  # Game can end or back to deuce
        }
    
    def _generate_distractors_with_state_machine(self, player1: str, player2: str, 
                                                before_sets: str, after_sets: str,
                                                before_games: str, before_points: str,
                                                after_games: str, after_points: str,
                                                correct_answer: str) -> List[str]:
        """Generate distractors using tennis state machine."""
        from collections import deque
        
        distractors = set()
        state_transitions = self._get_tennis_state_transitions()
        
        # Helper function to format score - extract the exact format from correct_answer
        def format_score(sets_info: str, games: str, points: str) -> str:
            sets_part = f"{sets_info} " if sets_info.strip() else ""
            # Extract the exact player name format from the correct answer
            if ':' in correct_answer:
                # Use the exact format from the correct answer (including player name format and spacing)
                name_part = correct_answer.split(':', 1)[0].strip()
                return normalize_option_text(
                    f"{name_part}: {sets_part}{{{games}, {points}}}"
                )
            else:
                # Fallback to original format
                return normalize_option_text(
                    f"{player1} vs. {player2}: {sets_part}{{{games}, {points}}}"
                )
        
        # Strategy 1: Wrong player gets the point
        if before_points in state_transitions:
            possible_next = state_transitions[before_points]
            for next_score in possible_next:
                if next_score != after_points:
                    distractor = format_score(after_sets, after_games, next_score)
                    distractors.add(distractor)
        
        # Strategy 2: Stay at previous state (no progression)
        if before_points != after_points:
            distractor = format_score(after_sets, before_games, before_points)
            distractors.add(distractor)
        
        # Strategy 3: Wrong game count progression
        if before_games != after_games:
            distractor = format_score(after_sets, before_games, "0-0")
            distractors.add(distractor)
        
        # Strategy 4: Skip intermediate states
        if before_points in state_transitions:
            possible_next = state_transitions[before_points]
            for next_score in possible_next:
                if next_score in state_transitions:
                    skip_to = state_transitions[next_score]
                    for skip_score in skip_to:
                        if skip_score != after_points:
                            distractor = format_score(after_sets, after_games, skip_score)
                            distractors.add(distractor)
        
        # Strategy 5: Set completion errors
        if before_sets != after_sets:
            distractor = format_score(before_sets, before_games, "0-0")
            distractors.add(distractor)
        
        # Strategy 6: Advantage direction errors
        if "AD-40" in after_points:
            distractor = format_score(after_sets, after_games, "40-AD")
            distractors.add(distractor)
        elif "40-AD" in after_points:
            distractor = format_score(after_sets, after_games, "AD-40")
            distractors.add(distractor)
        
        # Strategy 7: Game completion errors
        if after_points == "0-0" and before_points in ["AD-40", "40-AD", "40-30", "30-40", "40-0", "0-40", "40-15", "15-40"]:
            # Should complete game, but distractor shows it doesn't
            if before_points in ["AD-40", "40-AD", "30-40", "40-30"]:
                # From advantage or game point, distractor: back to deuce
                distractor = format_score(after_sets, before_games, "40-40")
                distractors.add(distractor)
            else:
                # From clear game point, distractor: game doesn't end
                distractor = format_score(after_sets, before_games, before_points)
                distractors.add(distractor)
        
        # Strategy 8: Tiebreak transition errors
        if "6-6(" in after_games:
            # Should go to tiebreak, but distractor shows regular game
            distractor = format_score(after_sets, "6-6", "0-0")
            distractors.add(distractor)
            # Or wrong tiebreak score
            if "(0-0)" in after_games:
                distractor = format_score(after_sets, "6-6(1-0)", "")
                distractors.add(distractor)
                distractor = format_score(after_sets, "6-6(0-1)", "")
                distractors.add(distractor)
        
        # Strategy 9: Wrong game progression after tiebreak
        if "7-6(" in after_games or "6-7(" in after_games:
            # After tiebreak win, should start new set
            if after_points == "0-0":
                # Distractor: continue in same set
                distractor = format_score(after_sets, before_games, "0-0")
                distractors.add(distractor)
        
        # Convert to list and limit
        correct_key = option_dedupe_key(correct_answer)
        distractors_list = [
            d for d in distractors if option_dedupe_key(d) != correct_key
        ]
        if len(distractors_list) > 3:
            distractors_list = distractors_list[:3]
        
        # Fill with generic options if needed
        while len(distractors_list) < 3:
            generic_options = [
                format_score(after_sets, "0-0", "15-15"),
                format_score(after_sets, "0-0", "30-30"),
                format_score(after_sets, "1-1", "0-0"),
                format_score(after_sets, "0-1", "0-0"),
                format_score(after_sets, "2-0", "0-0"),
                format_score(after_sets, after_games, "40-40")
            ]
            added = False
            existing_keys = {option_dedupe_key(d) for d in distractors_list}
            existing_keys.add(correct_key)
            for option in generic_options:
                if option_dedupe_key(option) not in existing_keys:
                    distractors_list.append(option)
                    added = True
                    break
            if not added:
                break
        
        return distractors_list
    
    def generate_question_answer(self, field_value: Any, template: Dict[str, Any], question_type: str, context_data: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """
        Generate question, correct answer, and options based on template configuration.
        
        Args:
            field_value: Actual value from the data
            template: Question template from config
            question_type: Type of question
            context_data: Dictionary with placeholder values
            
        Returns:
            Tuple of (question, correct_answer, options)
        """
        question = template["question"]
        template_type = template.get("type", "categorical")
        
        # Handle LLM-based distractor generation type
        if template_type == "llm_gen_distractor":
            try:
                # Get LLM prompts from template
                llm_answer_prompt = template.get("llm_answer_prompt")
                llm_question_prompt = template.get("llm_question_prompt") 
                
                # Prepare metadata for LLM
                metadata = context_data.copy()
                metadata["field_value"] = str(field_value) if field_value else ""
                
                # Generate question, answer, and distractors using LLM with rate limiting
                generated_question, correct_answer, distractors = self._rate_limited_api_call(
                    generate_tennis_mcq_with_llm,
                    metadata=metadata,
                    field_value=str(field_value) if field_value else "",
                    question_template=question,
                    answer_prompt=llm_answer_prompt,
                    distractor_prompt=llm_question_prompt
                    # model parameter will use DEFAULT_MODEL from llm_helper
                )
                
                # Create final options and shuffle
                options = [correct_answer] + distractors
                random.shuffle(options)
                
                # Create question with options
                option_labels = ["(A)", "(B)", "(C)", "(D)"]
                question_parts = [generated_question]
                
                correct_answer_label = ""
                for i, option in enumerate(options):
                    if i < len(option_labels):
                        question_parts.append(f"{option_labels[i]} {option}")
                        if option == correct_answer:
                            correct_answer_label = f"{option_labels[i]} {option}"
                
                final_question = normalize_question_text("\n".join(question_parts))
                return final_question, correct_answer_label, options
                
            except Exception as e:
                logging.error(f"Error generating LLM-based MCQ for question type {question_type}: {e}")
                # Discard this question after retries failed
                return None, None, []
        
        # Handle freeform_with_distractors type early
        if template_type == "freeform_with_distractors":
            # For score_after type questions, use the actual field value as correct answer
            correct_answer = normalize_option_text(field_value) if field_value else ""
            
            # Generate score distractors if this is a score-related question
            if question_type == "score_after":
                # Generate distractors based on score_before from context
                score_before = context_data.get("<score_before>", "")
                distractors = self.generate_score_distractors(score_before, correct_answer, context_data)
                
                options = unique_mcq_options(
                    correct_answer, distractors, max_options=4
                )
                if len(options) < 2:
                    return None, None, []
                random.shuffle(options)
                
                # Replace placeholders and create formatted question
                question = self.replace_placeholders(question, context_data)
                correct_answer = normalize_option_text(
                    self.replace_placeholders(correct_answer, context_data)
                )
                
                # Create question with options
                option_labels = ["(A)", "(B)", "(C)", "(D)"]
                question_parts = [question]
                
                correct_answer_label = ""
                for i, option in enumerate(options):
                    if i < len(option_labels):
                        question_parts.append(f"{option_labels[i]} {option}")
                        if option_dedupe_key(option) == option_dedupe_key(correct_answer):
                            correct_answer_label = f"{option_labels[i]} {option}"
                
                final_question = normalize_question_text("\n".join(question_parts))
                return final_question, correct_answer_label, options
            else:
                # For other freeform types, return empty for now
                return None, None, []
        
        # For other types, get options from template
        base_options = template.get("options", [])
        
        if not base_options:
            logging.warning(f"No options found in template for question type: {question_type}")
            return None, None, []
        
        # Handle different template types
        if template_type == "categorical":
            # Map Player 1/Player 2 to actual player names for server/receiver questions
            if question_type in ["server", "receiver"] and field_value in ["Player 1", "Player 2"]:
                if field_value == "Player 1":
                    correct_answer = context_data.get("<player1>", "Player 1")
                else:  # Player 2
                    correct_answer = context_data.get("<player2>", "Player 2")
            else:
                correct_answer = str(field_value) if field_value else ""
        
        elif template_type == "numeric_range":
            if not isinstance(field_value, (int, float)):
                return None, None, []
            
            ranges = template.get("ranges", [])
            correct_answer = None
            for range_config in ranges:
                if range_config["min"] <= field_value <= range_config["max"]:
                    correct_answer = range_config["answer"]
                    break
            
            if not correct_answer:
                return None, None, []
        
        elif template_type == "text_pattern":
            if not isinstance(field_value, str):
                return None, None, []
            
            patterns = template.get("patterns", [])
            field_lower = field_value.lower()
            correct_answer = None
            
            for pattern_config in patterns:
                pattern = pattern_config["pattern"]
                if re.search(pattern, field_lower):
                    correct_answer = pattern_config["answer"]
                    break
            
            if not correct_answer:
                return None, None, []
        
        else:
            # Default to categorical
            correct_answer = str(field_value) if field_value else ""
        
        # Replace placeholders in question, answer, and options
        question = self.replace_placeholders(question, context_data)
        correct_answer = normalize_option_text(
            self.replace_placeholders(correct_answer, context_data)
        )
        
        # Skip if correct_answer is empty
        if not correct_answer or correct_answer.strip() == "":
            return None, None, []
        
        # Replace placeholders in base options
        processed_options = []
        for option in base_options:
            processed_option = normalize_option_text(
                self.replace_placeholders(option, context_data)
            )
            processed_options.append(processed_option)

        # Prefer a configured option spelling when the answer is only a
        # whitespace/quote variant (e.g. annotation "  1" vs option "1").
        correct_key = option_dedupe_key(correct_answer)
        for option in processed_options:
            if option_dedupe_key(option) == correct_key:
                correct_answer = option
                break
        
        # Generate distractors (wrong options)
        # Filter out "Unknown" and other unwanted options
        unwanted_options = {"unknown", "not specified", "n/a", ""}
        distractors = [
            opt
            for opt in processed_options
            if option_dedupe_key(opt) != correct_key
            and option_dedupe_key(opt) not in unwanted_options
        ]
        if len(distractors) >= 3:
            distractors = random.sample(distractors, 3)
        else:
            # Only use available distractors from configuration, no generic fallbacks
            # If not enough distractors, use what we have
            pass
        
        # Create final options and shuffle
        # Limit to maximum 4 options (ABCD), but allow fewer if not enough distractors
        max_options = 4
        options = unique_mcq_options(
            correct_answer, distractors, max_options=max_options
        )
        if len(options) < 2:
            return None, None, []
        random.shuffle(options)
        
        # Create question with options
        option_labels = ["(A)", "(B)", "(C)", "(D)"]
        question_parts = [question]
        
        correct_answer_label = ""
        for i, option in enumerate(options):
            if i < len(option_labels):  # Make sure we don't exceed ABCD
                question_parts.append(f"{option_labels[i]} {option}")
                if option_dedupe_key(option) == option_dedupe_key(correct_answer):
                    correct_answer_label = f"{option_labels[i]} {option}"
        
        final_question = normalize_question_text("\n".join(question_parts))
        
        return final_question, correct_answer_label, options
    
    def generate_qa_from_dataset(self, dataset: TennisDataset, entry_id: str) -> List[Dict[str, Any]]:
        """
        Generate QA pairs from a dataset using QA configuration.
        
        Args:
            dataset: TennisDataset object
            entry_id: Entry identifier (file path)
            
        Returns:
            List of QA question-answer pairs
        """
        qa_pairs = []
        
        # Skip if no points
        if not dataset.point_annotations.points:
            logging.warning(f"No points found in dataset for {entry_id}")
            return qa_pairs
        
        logging.info(f"Generating QA pairs for {len(dataset.point_annotations.points)} points")
        
        # Extract player names for this dataset
        player_names = self.extract_player_names(dataset)
        
        # Track video statistics in source data for QA
        total_points = len(dataset.point_annotations.points)
        points_with_video = sum(1 for p in dataset.point_annotations.points if p.video)
        points_without_video = total_points - points_with_video
        points_with_valid_video = sum(
            1 for p in dataset.point_annotations.points if self.has_valid_video(p)
        )
        
        print(f"  QA Source data video statistics:")
        print(f"    Total points: {total_points}")
        print(f"    Points with video path: {points_with_video}")
        print(f"    Points without video path: {points_without_video}")
        if self.video_root:
            print(f"    Points with existing clip: {points_with_valid_video}")
            print(f"    Points skipped (missing clip): {total_points - points_with_valid_video}")
        
        for i, point in enumerate(dataset.point_annotations.points):
            if not self.has_valid_video(point):
                self._skipped_missing_video_ids.add(str(point.id))
                continue
            # Build context data for placeholder replacement
            context_data = self.build_context_data(point, dataset, player_names)
            
            for question_type, template in self.qa_templates.items():
                field_name = template["field"]
                
                # Use getattr for dynamic attribute access
                field_value = getattr(point, field_name, None)
                
                if field_value is None or field_value == "":
                    continue
                
                # Generate question and answer for open-ended QA
                question, answer = self.generate_qa_question_answer(
                    field_value, template, question_type, context_data
                )
                
                if question and answer:
                    qa_pair = {
                        "question_type": question_type,
                        "question": question,
                        "answer": answer,
                        "file_path": entry_id,
                        "point_data": {
                            "id": point.id,
                            "video": point.video,
                        }
                    }
                    qa_pairs.append(qa_pair)
        
        logging.info(f"Generated {len(qa_pairs)} QA pairs for {entry_id}")
        return qa_pairs
    
    def generate_qa_question_answer(self, field_value: Any, template: Dict[str, Any], question_type: str, context_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate open-ended question and answer based on QA template configuration.
        
        Args:
            field_value: Actual value from the data
            template: QA template from config
            question_type: Type of question
            context_data: Dictionary with placeholder values
            
        Returns:
            Tuple of (question, answer)
        """
        question = template["question"]
        template_type = template.get("type", "open_ended")
        
        # Only handle open_ended type for QA
        if template_type != "open_ended":
            return None, None
        
        # For open-ended questions, the field value is the answer
        answer = str(field_value) if field_value else ""
        # Tidy annotation noise (leading segment tag, wrapping quotes/backticks,
        # collapsed whitespace) without disturbing wording or punctuation.
        answer = clean_freetext_answer(answer)
        # Drop non-answer placeholder values (e.g. "n/a") so they never ship.
        if is_empty_answer_value(answer):
            return None, None
        
        # Check minimum length requirement
        min_length = template.get("min_length", 0)
        if len(answer) < min_length:
            return None, None
        
        # Check maximum length requirement
        max_length = template.get("max_length", 1000)
        if len(answer) > max_length:
            # Truncate the answer to max_length
            answer = answer[:max_length]
        
        # Replace placeholders in question and answer
        question = normalize_question_text(self.replace_placeholders(question, context_data))
        answer = self.replace_placeholders(answer, context_data)
        
        # Skip if answer is empty after processing
        if is_empty_answer_value(answer):
            return None, None
        
        return question, answer
    
    # Maps question_type -> underscore-free task name (so point IDs stay
    # "videoId_Task_uuid" and remain parseable by the splitter).
    TASK_NAME_MAPPING = {
        'server': 'Serve',
        'receiver': 'Receive',
        'winner': 'Winner',
        'server_winner': 'ServerWinner',
        'receiver_winner': 'ReceiverWinner',
        'how_point_ended': 'PointEnd',
        'num_shots_exchanged': 'RallyLength',
        'num_serving_attempts_until_successful': 'ServingAttempts',
        'point_caption': 'Caption',
        'score_before': 'ScoreBefore',
        'score_after': 'ScoreAfter',
        'how_point_ended_description': 'PointEndDescription',
        'audio_cues': 'AudioCues',
    }

    def _template_for_question_type(self, question_type: str) -> Dict[str, Any]:
        """Return the MCQ or QA template dict for ``question_type``, if known."""
        template = self.question_templates.get(question_type)
        if template is None:
            template = self.qa_templates.get(question_type)
        return template or {}

    def _evaluation_type_for(self, question_type: str) -> str:
        """Return ``mcq`` or ``open_ended`` for a template/question type."""
        if question_type in self.qa_templates:
            return "open_ended"
        template = self._template_for_question_type(question_type)
        if template.get("type") in {"open_ended", "derived_open_ended"}:
            return "open_ended"
        return "mcq"

    def _build_hf_entry(self, pair: Dict[str, Any], question: str, answer: str) -> Dict[str, Any]:
        """Build a single HuggingFace conversation record from a QA/MCQ pair.

        Output schema (the only format this pipeline produces)::

            {"id", "class", "evaluation_type", "super_category", "fine_category",
             "conversation": [
               {"role": "user", "content": [{"type": "video", "path": <rel>},
                                            {"type": "text", "text": <question>}]},
               {"role": "assistant", "content": [{"type": "text", "text": <answer>}]}]}

        ``class`` is the question-type / template name (e.g. ``server_winner``,
        ``score_after``). ``evaluation_type`` is ``mcq`` or ``open_ended``.
        ``super_category`` / ``fine_category`` come from the template metadata
        (e.g. ``Match Facts`` / ``Point Roles and Outcome``) when present. The
        video ``path`` stays relative; a training/eval config resolves it
        against a ``video_root``.
        """
        point_data = pair['point_data']
        question_type = pair['question_type']
        id_question_type = pair.get("id_question_type", question_type)
        task_name = self.TASK_NAME_MAPPING.get(
            id_question_type,
            id_question_type.title().replace('_', ''),
        )

        base_name = os.path.basename(pair['file_path']).replace('.json', '')
        task_id = f"{base_name}_{task_name}_{point_data['id']}"

        user_content: List[Dict[str, Any]] = []
        video_path = point_data.get('video')
        if video_path:
            user_content.append({"type": "video", "path": video_path})
        user_content.append({"type": "text", "text": question})

        entry: Dict[str, Any] = {
            "id": task_id,
            "class": question_type,
            "evaluation_type": self._evaluation_type_for(question_type),
        }
        template = self._template_for_question_type(question_type)
        super_category = template.get("super_category") or pair.get("super_category")
        fine_category = template.get("fine_category") or pair.get("fine_category")
        if super_category:
            entry["super_category"] = super_category
        if fine_category:
            entry["fine_category"] = fine_category
        entry["conversation"] = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return entry

    def convert_to_conversation_format(self, mcq_pairs: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Convert MCQ and QA pairs into HuggingFace conversation records.

        Args:
            mcq_pairs: List of MCQ question-answer pairs
            qa_pairs: List of open-ended QA question-answer pairs (optional)

        Returns:
            List of HF conversation records (see ``_build_hf_entry``).
        """
        conversation_entries = []

        for mcq_pair in mcq_pairs:
            entry = self._build_hf_entry(
                mcq_pair, mcq_pair['question'], mcq_pair['correct_answer']
            )
            entry["evaluation_type"] = "mcq"
            conversation_entries.append(entry)

        for qa_pair in (qa_pairs or []):
            entry = self._build_hf_entry(
                qa_pair, qa_pair['question'], qa_pair['answer']
            )
            entry["evaluation_type"] = "open_ended"
            conversation_entries.append(entry)

        total = len(conversation_entries)
        with_video = sum(
            1 for e in conversation_entries
            if any(c.get("type") == "video" for c in e["conversation"][0]["content"])
        )
        print(f"    HF records: {total} ({with_video} with video, {total - with_video} without)")

        return conversation_entries


    def process_data(self, input_dir: str, per_video_output_dir: str = None) -> None:
        """
        Process data and generate MCQ and QA pairs using configuration.
        Saves individual JSON files per video in the specified output directory.
        
        Args:
            input_dir: Path to directory containing preprocessed JSON files
            per_video_output_dir: Directory to save per-video JSON files (default: mcq_qa_per_video)
        """
        print("Tennis MCQ and QA Generator - Configuration Driven")
        print("=" * 60)
        
        # Get absolute path for input directory
        input_dir_abs = os.path.abspath(input_dir)
        
        # Create per-video output directory
        if per_video_output_dir is None:
            # Default: create as sibling to input_dir
            per_video_dir = os.path.join(os.path.dirname(input_dir_abs), "mcq_qa_per_video")
        else:
            per_video_dir = os.path.abspath(per_video_output_dir)
        
        os.makedirs(per_video_dir, exist_ok=True)
        
        # Find all preprocessed JSON files and sort them
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {input_dir}")
        
        json_files.sort()  # Sort for consistent ordering
        
        print(f"Found {len(json_files)} preprocessed files (sorted)")
        print(f"Per-video output directory: {per_video_dir}")
        
        all_mcq_pairs = []
        all_qa_pairs = []
        processed_entries = 0
        failed_entries = 0
        skipped_entries = 0
        
        # Check for existing per-video files and load them
        print("\nChecking for already processed videos...")
        for json_file in json_files:
            video_name = os.path.basename(json_file).replace('.json', '')
            video_output_file = os.path.join(per_video_dir, f"{video_name}_derived_mcq_qa.json")
            
            if os.path.exists(video_output_file):
                try:
                    with open(video_output_file, 'r', encoding='utf-8') as f:
                        video_conversations = json.load(f)
                        # Note: We can't easily extract MCQ/QA counts from conversation format,
                        # but we know the video is complete
                        skipped_entries += 1
                except Exception as e:
                    logging.warning(f"Could not load existing video file {video_output_file}: {e}")
        
        if skipped_entries > 0:
            print(f"Found {skipped_entries} already processed videos (will skip)")
        
        for idx, json_file in enumerate(json_files, 1):
            video_name = os.path.basename(json_file).replace('.json', '')
            video_output_file = os.path.join(per_video_dir, f"{video_name}_derived_mcq_qa.json")
            
            # Skip if already processed
            if os.path.exists(video_output_file):
                continue
            
            try:
                print(f"\n[{idx}/{len(json_files)}] Processing: {os.path.basename(json_file)}")
                
                # Load preprocessed data
                with open(json_file, 'r', encoding='utf-8') as f:
                    entry_data = json.load(f)
                
                # Validate JSON structure
                if 'instances' not in entry_data:
                    logging.error(f"Invalid JSON structure in {json_file}: missing 'instances'")
                    failed_entries += 1
                    continue
                
                # Convert to TennisDataset object
                dataset = TennisDataset.from_json(entry_data)
                
                # Validate dataset
                if not dataset.point_annotations.points:
                    logging.warning(f"No points found in {json_file}")
                    failed_entries += 1
                    continue
                
                # Generate MCQ pairs using multithreading
                mcq_pairs = self.generate_mcq_from_dataset_threaded(dataset, json_file)
                mcq_generated = len(mcq_pairs) > 0
                
                # Generate QA pairs
                qa_pairs = self.generate_qa_from_dataset(dataset, json_file)
                qa_generated = len(qa_pairs) > 0
                
                # Track successful generation
                if mcq_generated or qa_generated:
                    if mcq_generated:
                        all_mcq_pairs.extend(mcq_pairs)
                        print(f"  ✓ Generated {len(mcq_pairs)} MCQ pairs")
                    if qa_generated:
                        all_qa_pairs.extend(qa_pairs)
                        print(f"  ✓ Generated {len(qa_pairs)} QA pairs")
                    processed_entries += 1
                    
                    # Convert this video's data to conversation format
                    video_conversation_entries = self.convert_to_conversation_format(mcq_pairs, qa_pairs)
                    
                    # Save per-video conversation file
                    try:
                        with open(video_output_file, 'w', encoding='utf-8') as f:
                            json.dump(video_conversation_entries, f, indent=2, ensure_ascii=False)
                        print(f"  ✓ Saved {len(video_conversation_entries)} conversations to {video_name}_derived_mcq_qa.json")
                    except Exception as e:
                        logging.error(f"Could not save per-video file: {e}")
                        failed_entries += 1
                        continue
                else:
                    logging.warning(f"No MCQ or QA pairs generated for {json_file}")
                    failed_entries += 1
                    
            except Exception as e:
                logging.error(f"Error processing file {json_file}: {e}")
                failed_entries += 1
                continue
        
        # Print processing summary
        print(f"\n" + "=" * 60)
        print(f"PROCESSING SUMMARY:")
        print(f"  - Total files found: {len(json_files)}")
        print(f"  - Skipped (already processed): {skipped_entries}")
        print(f"  - Successfully processed: {processed_entries}")
        print(f"  - Failed to process: {failed_entries}")
        print(f"  - New MCQ questions generated: {len(all_mcq_pairs)}")
        print(f"  - New QA questions generated: {len(all_qa_pairs)}")
        print(f"  - Total new questions: {len(all_mcq_pairs) + len(all_qa_pairs)}")
        print(f"\nPer-video output directory: {per_video_dir}")
        print(f"All per-video JSON files saved successfully!")
        print(f"\nTo create train/val/test splits, run:")
        print(f"  python split_dataset.py")
        
        # Print statistics for newly processed videos
        if all_mcq_pairs or all_qa_pairs:
            self.print_statistics(all_mcq_pairs, all_qa_pairs)
        
        # Print API call statistics
        self.print_api_statistics()
    
    def print_statistics(self, mcq_pairs: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]]) -> None:
        """Print statistics about generated questions."""
        mcq_question_type_counts = {}
        qa_question_type_counts = {}
        
        # Count MCQ question types
        for pair in mcq_pairs:
            q_type = pair['question_type']
            mcq_question_type_counts[q_type] = mcq_question_type_counts.get(q_type, 0) + 1
        
        # Count QA question types
        for pair in qa_pairs:
            q_type = pair['question_type']
            qa_question_type_counts[q_type] = qa_question_type_counts.get(q_type, 0) + 1
        
        print(f"\nMCQ Question type distribution:")
        for q_type, count in mcq_question_type_counts.items():
            print(f"  {q_type}: {count} questions")
        
        print(f"\nQA Question type distribution:")
        for q_type, count in qa_question_type_counts.items():
            print(f"  {q_type}: {count} questions")

    @dataclass
    class QuestionTask:
        """Helper class for batching question generation tasks."""
        point: PointData
        template: Dict[str, Any]
        question_type: str
        context_data: Dict[str, Any]
        field_value: Any
        entry_id: str
    
    def _generate_single_mcq_threaded(self, task: 'ConfigDrivenMCQGenerator.QuestionTask') -> Optional[Dict[str, Any]]:
        """
        Generate a single MCQ question in a thread-safe manner.
        
        Args:
            task: QuestionTask containing all necessary data
            
        Returns:
            MCQ pair dictionary or None if generation failed
        """
        try:
            # Generate question and answer based on template type
            question, correct_answer, _options = self.generate_question_answer(
                task.field_value, task.template, task.question_type, task.context_data
            )
            
            if question and correct_answer:
                mcq_pair = {
                    "question_type": task.question_type,
                    "question": question,
                    "correct_answer": correct_answer,
                    "file_path": task.entry_id,
                    "point_data": {
                        "id": task.point.id,
                        "video": task.point.video,
                    }
                }
                return mcq_pair
            return None
        except Exception as e:
            logging.error(f"Error generating MCQ for {task.question_type} on point {task.point.id}: {e}")
            return None
    
    def generate_mcq_from_dataset_threaded(self, dataset: TennisDataset, entry_id: str) -> List[Dict[str, Any]]:
        """
        Generate MCQ questions from a dataset using multithreading for API calls.
        
        Args:
            dataset: TennisDataset object
            entry_id: Entry identifier (file path)
            
        Returns:
            List of MCQ question-answer pairs
        """
        # Debug: Print dataset summary
        logging.info(f"Processing dataset (threaded): {dataset.get_summary()}")
        
        # Skip if no points
        if not dataset.point_annotations.points:
            logging.warning(f"No points found in dataset for {entry_id}")
            return []
        
        logging.info(f"Found {len(dataset.point_annotations.points)} points in dataset")
        
        # Extract player names for this dataset
        player_names = self.extract_player_names(dataset)
        logging.info(f"Player names: {player_names}")
        
        # Track video statistics in source data
        total_points = len(dataset.point_annotations.points)
        points_with_video = sum(1 for p in dataset.point_annotations.points if p.video)
        points_without_video = total_points - points_with_video
        points_with_valid_video = sum(
            1 for p in dataset.point_annotations.points if self.has_valid_video(p)
        )
        
        print(f"  Source data video statistics:")
        print(f"    Total points: {total_points}")
        print(f"    Points with video path: {points_with_video}")
        print(f"    Points without video path: {points_without_video}")
        if self.video_root:
            print(f"    Points with existing clip: {points_with_valid_video}")
            print(f"    Points skipped (missing clip): {total_points - points_with_valid_video}")
        if points_without_video > 0:
            print(f"    Video missing rate: {(points_without_video/total_points)*100:.1f}%")
        
        # Collect all tasks for parallel processing
        tasks = []
        
        for i, point in enumerate(dataset.point_annotations.points):
            if not self.has_valid_video(point):
                self._skipped_missing_video_ids.add(str(point.id))
                continue
            # Debug: Print point info for first few points and any missing video
            if i < 3 or not point.video:
                logging.info(f"Point {i+1}: server={point.server}, receiver={point.receiver}, winner={point.winner}, video='{point.video}'")
                if not point.video:
                    logging.warning(f"Point {i+1} (id: {point.id}) missing video field in source data")
            
            # Build context data for placeholder replacement
            context_data = self.build_context_data(point, dataset, player_names)
            
            for question_type, template in self.question_templates.items():
                field_name = template["field"]
                
                # Handle special composite fields
                if field_name == "server_winner":
                    server = getattr(point, "server", "")
                    receiver = getattr(point, "receiver", "")
                    winner = getattr(point, "winner", "")
                    
                    if not server or not receiver or server == receiver or winner not in ("Server", "Receiver"):
                        continue
                    
                    server_player = context_data.get("<player1>", "Player 1") if server == "Player 1" else context_data.get("<player2>", "Player 2")
                    receiver_player = context_data.get("<player1>", "Player 1") if receiver == "Player 1" else context_data.get("<player2>", "Player 2")
                    winner_player = server_player if winner == "Server" else receiver_player
                    
                    field_value = f"{server_player} served and {winner_player} won"
                
                elif field_name == "receiver_winner":
                    server = getattr(point, "server", "")
                    receiver = getattr(point, "receiver", "")
                    winner = getattr(point, "winner", "")
                    
                    if not server or not receiver or server == receiver or winner not in ("Server", "Receiver"):
                        continue
                    
                    server_player = context_data.get("<player1>", "Player 1") if server == "Player 1" else context_data.get("<player2>", "Player 2")
                    receiver_player = context_data.get("<player1>", "Player 1") if receiver == "Player 1" else context_data.get("<player2>", "Player 2")
                    winner_player = server_player if winner == "Server" else receiver_player
                    
                    field_value = f"{receiver_player} received and {winner_player} won"
                
                else:
                    field_value = getattr(point, field_name, None)
                
                if field_value is None or field_value == "":
                    continue
                
                # Create task for parallel processing
                task = self.QuestionTask(
                    point=point,
                    template=template,
                    question_type=question_type,
                    context_data=context_data,
                    field_value=field_value,
                    entry_id=entry_id,
                )
                tasks.append(task)
        
        if not tasks:
            logging.warning(f"No valid tasks created for {entry_id}")
            return []
        
        print(f"  Created {len(tasks)} question generation tasks")
        
        # Process tasks in parallel with progress bar
        mcq_pairs = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit ALL tasks at once for maximum parallel processing
            print(f"  Submitting {len(tasks)} tasks to {self.max_workers} worker threads...")
            futures = [executor.submit(self._generate_single_mcq_threaded, task) for task in tasks]
            
            if self.enable_progress_bar:
                # Process all futures with progress bar (update every 5 seconds to reduce log spam)
                with tqdm(total=len(tasks), desc=f"Generating MCQs for {os.path.basename(entry_id)}", 
                         unit="question", mininterval=5.0) as pbar:
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=60)  # Increased timeout
                            if result:
                                mcq_pairs.append(result)
                            pbar.update(1)
                        except Exception as e:
                            logging.error(f"Task failed: {e}")
                            pbar.update(1)
            else:
                # Process without progress bar
                completed = 0
                print(f"  Processing {len(tasks)} tasks in parallel...")
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        if result:
                            mcq_pairs.append(result)
                        completed += 1
                        
                        # Print progress every 10% completion
                        if completed % max(1, len(tasks) // 10) == 0:
                            progress = (completed / len(tasks)) * 100
                            print(f"    Progress: {completed}/{len(tasks)} ({progress:.1f}%)")
                    except Exception as e:
                        logging.error(f"Task failed: {e}")
                        completed += 1
        
        logging.info(f"Generated {len(mcq_pairs)} MCQ pairs for {entry_id}")
        return mcq_pairs

    def print_api_statistics(self) -> None:
        """Print API call statistics."""
        stats = self.get_api_statistics()
        
        print(f"\n" + "=" * 50)
        print("API CALL STATISTICS")
        print("=" * 50)
        print(f"Total API calls: {stats['total_calls']}")
        print(f"Successful calls: {stats['successful_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Retry attempts: {stats['retry_attempts']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Total API time: {stats['total_api_time']:.2f} seconds")
        print(f"Average call time: {stats['average_call_time']:.2f} seconds")
        
        if stats['total_calls'] > 0:
            total_processing_time = stats['total_api_time']
            estimated_sequential_time = stats['total_calls'] * stats['average_call_time']
            speedup = estimated_sequential_time / total_processing_time if total_processing_time > 0 else 1
            print(f"Estimated speedup from threading: {speedup:.1f}x")
        
        print("=" * 50)



def load_question_types(path: str) -> List[str]:
    """Load a question-type list from JSON (``{"question_types": [...]}`` or a bare array)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data['question_types']


def main():
    """Main function to run configuration-driven MCQ and QA generation."""
    parser = argparse.ArgumentParser(description='Generate tennis MCQ/QA in HF conversation format.')
    parser.add_argument('--mcq-config', default='configs/generation/tennis_mcq_config.json')
    parser.add_argument('--qa-config', default='configs/generation/tennis_qa_config.json')
    parser.add_argument('--input-dir', default='raw_tennis_data_input')
    parser.add_argument('--output-dir', default='training_tennis_data_output/mcq_qa_per_video')
    parser.add_argument('--question-types', nargs='+',
                        help='Template names to generate (default: all types in configs)')
    parser.add_argument('--question-types-config',
                        help='JSON file with question_types list (overrides --question-types)')
    parser.add_argument('--max-workers', type=int, default=64)
    parser.add_argument('--api-rate-limit', type=float, default=0.01)
    parser.add_argument(
        '--video-root',
        default=None,
        help=(
            'Directory used to resolve relative point clip paths. When set, '
            'points whose clips are missing/empty are skipped.'
        ),
    )
    args = parser.parse_args()

    question_types: Optional[List[str]] = None
    if args.question_types_config:
        question_types = load_question_types(args.question_types_config)
    elif args.question_types:
        question_types = args.question_types

    mcq_config_file = args.mcq_config
    qa_config_file = args.qa_config
    input_dir = args.input_dir
    per_video_output_dir = args.output_dir
    max_workers = args.max_workers
    api_rate_limit = args.api_rate_limit
    enable_progress_bar = True
    print("Tennis MCQ and QA Generator - Parallel API Processing")
    print("=" * 60)
    print(f"Parallel API configuration:")
    print(f"  - Max concurrent API calls: {max_workers}")
    print(f"  - API rate limit: {api_rate_limit}s (sliding window)")
    print(f"  - Progress tracking: {enable_progress_bar}")
    print(f"  - Video root: {args.video_root or '(unset; path-string only)'}")
    if api_rate_limit > 0:
        print(f"  - Expected throughput: ~{max_workers/api_rate_limit:.0f} calls/second")
    else:
        print("  - Expected throughput: unlimited (no rate limit)")
    print(f"  - Aggressive mode: Minimized rate limiting for maximum speed")
    print("=" * 60)
    
    # Initialize generator with config and multithreading settings
    generator = ConfigDrivenMCQGenerator(
        mcq_config_file=mcq_config_file,
        qa_config_file=qa_config_file,
        max_workers=max_workers,
        api_rate_limit=api_rate_limit,
        enable_progress_bar=enable_progress_bar,
        question_types=question_types,
        video_root=args.video_root,
    )
    
    # Process data and generate per-video conversations
    generator.process_data(input_dir, per_video_output_dir)
    
    print(f"\nConfiguration-driven MCQ and QA generation completed successfully!")

if __name__ == "__main__":
    main()
