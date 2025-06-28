# processing/repetition_analyzer.py
"""Analyze drafts for repeated text patterns."""

from __future__ import annotations

from collections import Counter

import structlog
import utils
from config import settings

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
        deduped_text, _ = await utils.deduplicate_text_segments(text, "sentence")
        tokens = deduped_text.split()
        ngrams = [
            " ".join(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)
        ]
        counts = Counter(ngrams)
        problems: list[ProblemDetail] = []
        for ngram, count in counts.items():
            if count >= self.threshold:
                description = f'Phrase repeated {count} times: "{ngram}"'
                problems.append(
                    ProblemDetail(
                        issue_category="repetition_and_redundancy",
                        problem_description=description,
                        quote_from_original_text=ngram,
                        suggested_fix_focus="Rephrase or remove the repeated phrase.",
                        severity="medium",
                        related_spans=None,
                        rewrite_instruction=None,
                    )
                )
            if (
                self.tracker
                and self.tracker.phrase_counts.get(ngram, 0) >= self.cross_threshold
            ):
                description = (
                    f'Phrase "{ngram}" overused across novel '
                    f"({self.tracker.phrase_counts.get(ngram, 0)} times)"
                )
                problems.append(
                    ProblemDetail(
                        issue_category="repetition_and_redundancy",
                        problem_description=description,
                        quote_from_original_text=ngram,
                        suggested_fix_focus=(
                            "Replace or rephrase to avoid novel-wide repetition."
                        ),
                        severity="low",
                        related_spans=None,
                        rewrite_instruction=None,
                    )
                )
        if problems:
            logger.info("RepetitionAnalyzer found %s problems", len(problems))
        return problems
