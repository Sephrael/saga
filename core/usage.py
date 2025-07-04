from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """LLM token usage metrics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: TokenUsage | dict[str, int] | None) -> None:
        """Accumulate usage values from another instance or dictionary."""
        if not usage:
            return
        if isinstance(usage, TokenUsage):
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
        else:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
