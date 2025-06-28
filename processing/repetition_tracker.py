# processing/repetition_tracker.py
"""Track phrase usage across the novel for repetition detection."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections import Counter as CounterType

import structlog
from config import REPETITION_STATS_FILE_PATH, settings

logger = structlog.get_logger(__name__)


class RepetitionTracker:
    """Maintain counts of n-gram phrases across chapters."""

    def __init__(
        self,
        file_path: str = REPETITION_STATS_FILE_PATH,
        n: int = settings.REPETITION_TRACKER_NGRAM_SIZE,
    ) -> None:
        self.file_path = file_path
        self.n = n
        self.phrase_counts: CounterType[str] = Counter()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, encoding="utf-8") as f:
                    data = json.load(f)
                for phrase, count in data.items():
                    self.phrase_counts[phrase] = int(count)
            except Exception as exc:  # pragma: no cover - log and continue
                logger.error("Failed loading repetition stats", exc_info=exc)

    def _save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.phrase_counts, f)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error("Failed saving repetition stats", exc_info=exc)

    def update_from_text(self, text: str) -> None:
        tokens = text.split()
        for i in range(len(tokens) - self.n + 1):
            phrase = " ".join(tokens[i : i + self.n])
            self.phrase_counts[phrase] += 1
        self._save()

    def find_overused(self, text: str, threshold: int | None = None) -> set[str]:
        limit = (
            threshold
            if threshold is not None
            else settings.REPETITION_TRACKER_THRESHOLD
        )
        tokens = text.split()
        phrases = {
            " ".join(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)
        }
        return {p for p in phrases if self.phrase_counts.get(p, 0) >= limit}
