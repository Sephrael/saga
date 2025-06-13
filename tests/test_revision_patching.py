import numpy as np
import pytest

from chapter_revision_logic import _apply_patches_to_text
from llm_interface import llm_service
import utils


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

    result, _ = await _apply_patches_to_text(original, patches)
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

    result, _ = await _apply_patches_to_text(original, patches)
    assert result == "Hi world!"


@pytest.mark.asyncio
async def test_dedup_prefer_newer(monkeypatch):
    text = "First\n\nSecond\n\nFirst"

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([0.5, 0.5])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    dedup, _ = await utils.deduplicate_text_segments(
        text,
        segment_level="sentence",
        use_semantic_comparison=False,
        min_segment_length_chars=0,
        prefer_newer=True,
    )


@pytest.mark.asyncio
async def test_skip_repatch_same_segment(monkeypatch):
    text = "Hello world!"
    first_patch = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greet",
        }
    ]

    second_patch = [
        {
            "original_problem_quote_text": "Hi",
            "target_char_start": 0,
            "target_char_end": 2,
            "replace_with": "Hey",
            "reason_for_change": "again",
        }
    ]

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([0.5, 0.5])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    patched1, spans1 = await _apply_patches_to_text(text, first_patch)
    patched2, _ = await _apply_patches_to_text(patched1, second_patch, spans1)

    assert patched2 == patched1
    assert dedup == "First\nSecond"


@pytest.mark.asyncio
async def test_multiple_patches_applied(monkeypatch):
    original = "Hello world! Bye world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        },
        {
            "original_problem_quote_text": "Bye",
            "target_char_start": 13,
            "target_char_end": 16,
            "replace_with": "See ya",
            "reason_for_change": "farewell",
        },
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
        "Bye": np.array([1.0, 0.0]),
        "See ya": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result = await _apply_patches_to_text(original, patches)
    assert result == "Hi world! See ya world!"


@pytest.mark.asyncio
async def test_duplicate_patch_skipped(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        },
        {
            "original_problem_quote_text": "Hello again",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hey",
            "reason_for_change": "greeting2",
        },
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
        "Hello again": np.array([1.0, 0.0]),
        "Hey": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result = await _apply_patches_to_text(original, patches)
    # Only the first patch should be applied because the second overlaps exactly
    assert result == "Hi world!"
