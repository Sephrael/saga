"""Service for running evaluation cycles."""

from __future__ import annotations

from dataclasses import dataclass

from kg_maintainer.models import EvaluationResult, ProblemDetail


@dataclass
class EvaluationCycleResult:
    """Result from a single evaluation cycle."""

    evaluation: EvaluationResult
    continuity_problems: list[ProblemDetail]
    eval_usage: dict[str, int] | None
    continuity_usage: dict[str, int] | None
    repetition_problems: list[ProblemDetail]
