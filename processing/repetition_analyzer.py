# processing/repetition_analyzer.py
"""Analyze drafts for repeated text patterns."""

from __future__ import annotations

from collections import Counter

import structlog
from config import settings

import utils
from models import ProblemDetail
from processing.repetition_tracker import RepetitionTracker

logger = structlog.get_logger(__name__)


class RepetitionAnalyzer:
    """Detect repeated n-gram phrases within text."""

    def __init__(
        self,
        n: int = 4,
        threshold: int = 3,
        tracker: RepetitionTracker | None = None,
        cross_threshold: int | None = None,
    ) -> None:
        self.n = n
        self.threshold = threshold
        self.tracker = tracker
        self.cross_threshold = (
            cross_threshold
            if cross_threshold is not None
            else settings.REPETITION_TRACKER_THRESHOLD
        )

    async def analyze(self, text: str) -> list[ProblemDetail]:
        """Return repetition problems found in ``text``."""
        if not text.strip():
            return []

        # 1. Find all overused phrases first (both in-chapter and cross-chapter)
        tokens = text.split()
        counts: Counter[tuple[str, ...]] = Counter()
        overused_in_novel: set[str] = set()

        for i in range(len(tokens) - self.n + 1):
            ngram_tokens = tuple(tokens[i : i + self.n])
            counts[ngram_tokens] += 1
            if (
                self.tracker
                and self.tracker.phrase_counts.get(" ".join(ngram_tokens), 0)
                >= self.cross_threshold
            ):
                overused_in_novel.add(" ".join(ngram_tokens))

        overused_in_chapter = {
            " ".join(ngram)
            for ngram, count in counts.items()
            if count >= self.threshold
        }
        all_overused_phrases = overused_in_chapter.union(overused_in_novel)

        if not all_overused_phrases:
            return []

        # 2. Find the sentences that contain these overused phrases
        problems: list[ProblemDetail] = []
        processed_sentence_starts: set[int] = set()
        # Use our utility to get sentences with character offsets
        sentence_segments = utils.get_text_segments(text, "sentence")

        for sentence_text, start_char, end_char in sentence_segments:
            if start_char in processed_sentence_starts:
                continue

            # Find which overused phrases this sentence contains
            found_phrases = [
                f'"{phrase}"'
                for phrase in all_overused_phrases
                if phrase in sentence_text
            ]

            if found_phrases:
                description = (
                    f"Sentence contains overused phrases: {', '.join(found_phrases)}."
                )
                problems.append(
                    ProblemDetail(
                        issue_category="repetition_and_redundancy",
                        problem_description=description,
                        # The quote is now the full sentence, which is actionable
                        quote_from_original_text=sentence_text,
                        # We now have the correct location data
                        sentence_char_start=start_char,
                        sentence_char_end=end_char,
                        suggested_fix_focus="Rephrase this sentence to avoid using the repeated phrases, improving lexical diversity.",
                        severity="medium",
                    )
                )
                # Mark this sentence as processed to avoid creating duplicate problems for it
                processed_sentence_starts.add(start_char)

        if problems:
            logger.info(
                "RepetitionAnalyzer found %s problematic sentences.", len(problems)
            )
        return problems
