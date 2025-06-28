# processing/repetition_analyzer.py
"""Analyze drafts for repeated text patterns."""

from __future__ import annotations

from collections import Counter

import structlog
import utils

from models import ProblemDetail

logger = structlog.get_logger(__name__)


class RepetitionAnalyzer:
    """Detect repeated n-gram phrases within text."""

    def __init__(self, n: int = 4, threshold: int = 3) -> None:
        self.n = n
        self.threshold = threshold

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
        if problems:
            logger.info("RepetitionAnalyzer found %s problems", len(problems))
        return problems
