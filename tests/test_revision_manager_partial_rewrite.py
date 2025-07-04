from unittest.mock import AsyncMock

import pytest
from core.llm_interface import llm_service
from processing.revision_manager import RevisionManager

from models import ProblemDetail


@pytest.mark.asyncio
async def test_rewrite_problem_paragraphs(monkeypatch):
    manager = RevisionManager()
    text = "Para1.\n\nBad para2.\n\nPara3."
    problems = [
        ProblemDetail(
            issue_category="style",
            problem_description="bad",
            quote_from_original_text="Bad para2",
            sentence_char_start=8,
            sentence_char_end=16,
            suggested_fix_focus="fix",
        )
    ]

    async def fake_call_llm(*_a, **_k):
        return "Fixed para2.", None

    monkeypatch.setattr(
        llm_service, "async_call_llm", AsyncMock(side_effect=fake_call_llm)
    )
    monkeypatch.setattr(llm_service, "clean_model_response", lambda t: t)

    new_text, raw, _ = await manager._rewrite_problem_scenes(
        {"plot_points": ["a"]}, text, 1, problems, "ctx", None
    )
    parts = new_text.split("\n\n")
    assert parts[0] == "Para1."
    assert parts[1] == "Fixed para2."
    assert parts[2] == "Para3."
