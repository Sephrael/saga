from unittest.mock import AsyncMock

import processing.patch as patch_generator
import pytest
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.patch_validation_agent import NoOpPatchValidator, PatchValidationAgent
from config import settings
from core.llm_interface import llm_service, truncate_text_by_tokens
from processing.revision_manager import RevisionManager

from models import EvaluationResult, ProblemDetail


@pytest.mark.asyncio
async def test_patch_generator_successful_application(monkeypatch):
    async def fake_generate(*_args, **_kwargs):
        return [
            {
                "original_problem_quote_text": "Hello",
                "target_char_start": 0,
                "target_char_end": 5,
                "replace_with": "Hi",
                "reason_for_change": "greet",
            }
        ], None

    async def fake_apply(orig, instr, *_args, **_kwargs):
        assert instr[0]["replace_with"] == "Hi"
        return "Hi world", [(0, 2)]

    monkeypatch.setattr(
        patch_generator, "_generate_patch_instructions_logic", fake_generate
    )
    monkeypatch.setattr(patch_generator, "_apply_patches_to_text", fake_apply)

    patcher = patch_generator.PatchGenerator()
    result, spans, usage = await patcher.generate_and_apply(
        {"plot_points": ["a"]},
        "Hello world",
        [
            ProblemDetail(
                issue_category="c",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="f",
            )
        ],
        1,
        "",
        None,
        None,
        PatchValidationAgent(),
    )

    assert result == "Hi world"
    assert spans == [(0, 2)]
    assert usage is None


@pytest.mark.asyncio
async def test_patch_generator_failed_validation(monkeypatch):
    async def fake_generate(plot, original, problems, chap, ctx, plan, validator):
        patch = {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greet",
        }
        ok, _reason, _ = await validator.validate_patch("", patch, problems)
        return ([patch] if ok else []), None

    async def fake_apply(orig, instr, *_args, **_kwargs):
        assert instr == []
        return orig, []

    class FakeValidator:
        async def validate_patch(self, *_a, **_k):
            return False, None, None

    monkeypatch.setattr(
        patch_generator, "_generate_patch_instructions_logic", fake_generate
    )
    monkeypatch.setattr(patch_generator, "_apply_patches_to_text", fake_apply)
    monkeypatch.setattr(
        patch_generator, "_get_sentence_embeddings", AsyncMock(return_value=[])
    )

    patcher = patch_generator.PatchGenerator()
    result, spans, usage = await patcher.generate_and_apply(
        {"plot_points": ["a"]},
        "Hello world",
        [
            ProblemDetail(
                issue_category="c",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="f",
            )
        ],
        1,
        "",
        None,
        None,
        FakeValidator(),
    )

    assert result == "Hello world"
    assert spans == []
    assert usage is None


@pytest.mark.asyncio
async def test_revision_manager_full_rewrite(monkeypatch):
    async def fake_generate_and_apply(*_args, **_kwargs):
        return "Hello world", [], None

    async def fake_call_llm(*_args, **_kwargs):
        return "Rewrite done", None

    monkeypatch.setattr(
        patch_generator.PatchGenerator, "generate_and_apply", fake_generate_and_apply
    )
    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_llm)
    monkeypatch.setattr(llm_service, "clean_model_response", lambda t: t)
    monkeypatch.setattr(
        truncate_text_by_tokens,
        "__call__",
        lambda text, *_args, **_k: text,
        raising=False,
    )

    manager = RevisionManager()
    eval_result = EvaluationResult(
        needs_revision=True,
        reasons=["bad"],
        problems_found=[
            ProblemDetail(
                issue_category="style",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="fix",
            )
        ],
    )

    res, _ = await manager.revise_chapter(
        {"plot_points": ["a"]},
        {},
        {},
        "Hello world",
        1,
        eval_result,
        "ctx",
        None,
    )

    assert res[0] == "Rewrite done"
    assert res[2] == []


@pytest.mark.asyncio
async def test_patch_revision_cycle_success(monkeypatch):
    async def fake_generate_and_apply(*_a, **_k):
        return "Hi world", [(0, 2)], None

    async def fake_evaluate(*_a, **_k):
        return EvaluationResult(
            needs_revision=False, reasons=[], problems_found=[]
        ), None

    monkeypatch.setattr(
        patch_generator.PatchGenerator, "generate_and_apply", fake_generate_and_apply
    )
    monkeypatch.setattr(
        ComprehensiveEvaluatorAgent,
        "evaluate_chapter_draft",
        AsyncMock(side_effect=fake_evaluate),
    )

    manager = RevisionManager()
    patched, spans, flag, usage = await manager._patch_revision_cycle(
        {"plot_points": ["a"]},
        {},
        {},
        "Hello world",
        1,
        [
            ProblemDetail(
                issue_category="style",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="fix",
            )
        ],
        "",
        None,
        [],
    )

    assert patched == "Hi world"
    assert spans == [(0, 2)]
    assert flag
    assert usage is None


@pytest.mark.asyncio
async def test_perform_full_rewrite(monkeypatch):
    async def fake_call_llm(*_a, **_k):
        return "Rewrite done", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_llm)
    monkeypatch.setattr(llm_service, "clean_model_response", lambda t: t)
    monkeypatch.setattr(
        truncate_text_by_tokens, "__call__", lambda text, *_a, **_k: text, raising=False
    )

    manager = RevisionManager()
    text, raw, usage = await manager._perform_full_rewrite(
        {"plot_points": ["a"]},
        "Hello world",
        1,
        [
            ProblemDetail(
                issue_category="style",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="fix",
            )
        ],
        "bad",
        "ctx",
        None,
        False,
    )

    assert text == "Rewrite done"
    assert raw == "Rewrite done"
    assert usage is None


@pytest.mark.asyncio
async def test_revision_manager_uses_noop_validator(monkeypatch):
    settings.AGENT_ENABLE_PATCH_VALIDATION = False
    received = None

    async def fake_generate_and_apply(
        self,
        *args,
        **_kwargs,
    ):
        nonlocal received
        if args:
            received = args[-1]
        else:
            received = _kwargs.get("validator")
        return "Hello world", [], None

    monkeypatch.setattr(
        patch_generator.PatchGenerator,
        "generate_and_apply",
        fake_generate_and_apply,
    )
    monkeypatch.setattr(
        llm_service, "async_call_llm", AsyncMock(return_value=("rw", None))
    )
    monkeypatch.setattr(llm_service, "clean_model_response", lambda t: t)
    monkeypatch.setattr(
        truncate_text_by_tokens, "__call__", lambda text, *_a, **_k: text, raising=False
    )

    manager = RevisionManager()
    eval_result = EvaluationResult(
        needs_revision=True,
        reasons=[],
        problems_found=[
            ProblemDetail(
                issue_category="style",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="fix",
            )
        ],
    )

    await manager.revise_chapter(
        {"plot_points": ["a"]},
        {},
        {},
        "Hello world",
        1,
        eval_result,
        "ctx",
        None,
    )

    assert isinstance(received, NoOpPatchValidator)


@pytest.mark.asyncio
async def test_noop_validator_always_passes():
    validator = NoOpPatchValidator()
    ok, reason, usage = await validator.validate_patch("ctx", {}, [])
    assert ok and usage is None and reason is None


@pytest.mark.asyncio
async def test_patch_cycle_receives_extra_problems(monkeypatch):
    captured: dict[str, list] = {}

    async def fake_generate_and_apply(
        self,
        plot_outline,
        original_text,
        problems_to_fix,
        chapter_number,
        hybrid_context_for_revision,
        chapter_plan,
        already_patched_spans,
        validator,
    ):
        captured["problems"] = problems_to_fix
        return original_text, [], None

    async def fake_evaluate(*_a, **_k):
        return EvaluationResult(
            needs_revision=False, reasons=[], problems_found=[]
        ), None

    monkeypatch.setattr(
        patch_generator.PatchGenerator, "generate_and_apply", fake_generate_and_apply
    )
    monkeypatch.setattr(
        ComprehensiveEvaluatorAgent,
        "evaluate_chapter_draft",
        AsyncMock(side_effect=fake_evaluate),
    )

    manager = RevisionManager()
    await manager._patch_revision_cycle(
        {"plot_points": ["a"]},
        {},
        {},
        "Hello world",
        1,
        [
            ProblemDetail(
                issue_category="style",
                problem_description="d",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="fix",
            )
        ],
        "",
        None,
        [],
        continuity_problems=[
            ProblemDetail(
                issue_category="consistency",
                problem_description="c",
                quote_from_original_text="Hello",
                sentence_char_start=0,
                sentence_char_end=5,
                suggested_fix_focus="fix",
            )
        ],
        repetition_problems=[
            ProblemDetail(
                issue_category="repetition_and_redundancy",
                problem_description="r",
                quote_from_original_text="Hello Hello",
                sentence_char_start=0,
                sentence_char_end=11,
                suggested_fix_focus="remove",
            )
        ],
    )

    assert any(
        "consistency"
        in (p["issue_category"] if isinstance(p, dict) else p.issue_category)
        for p in captured["problems"]
    )
    assert any(
        "repetition_and_redundancy"
        in (p["issue_category"] if isinstance(p, dict) else p.issue_category)
        for p in captured["problems"]
    )
