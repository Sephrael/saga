import numpy as np
import pytest

from chapter_revision_logic import _apply_patches_to_text
from llm_interface import llm_service


@pytest.mark.asyncio
async def test_patch_skipped_when_high_similarity(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hello",
            "reason_for_change": "same",
        }
    ]

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result = await _apply_patches_to_text(original, patches)
    assert result == original


@pytest.mark.asyncio
async def test_patch_applied_when_low_similarity(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        }
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result = await _apply_patches_to_text(original, patches)
    assert result == "Hi world!"
