from typing import Any

import structlog
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.patch_validation_agent import PatchValidationAgent
from config import settings
from core.llm_interface import llm_service, truncate_text_by_tokens
from models import (
    CharacterProfile,
    EvaluationResult,
    SceneDetail,
    WorldItem,
)

from . import patch_generator, revision_logic

logger = structlog.get_logger(__name__)


class RevisionManager:
    """Coordinate chapter revision via patches and rewrites."""

    async def revise_chapter(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        original_text: str,
        chapter_number: int,
        evaluation_result: EvaluationResult,
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        is_from_flawed_source: bool = False,
        already_patched_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[
        tuple[str, str | None, list[tuple[int, int]]] | None, dict[str, int] | None
    ]:
        """Revise a chapter draft based on evaluation feedback."""
        if already_patched_spans is None:
            already_patched_spans = []

        cumulative_usage_data: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        def _add_usage(usage: dict[str, int] | None) -> None:
            if usage:
                cumulative_usage_data["prompt_tokens"] += usage.get("prompt_tokens", 0)
                cumulative_usage_data["completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                cumulative_usage_data["total_tokens"] += usage.get("total_tokens", 0)

        if not original_text:
            logger.error(
                "Revision for ch %s aborted: missing original text.", chapter_number
            )
            return None, None

        problems_to_fix: list[dict[str, Any]] = evaluation_result.get(
            "problems_found", []
        )
        problems_to_fix = revision_logic._deduplicate_problems(
            revision_logic._consolidate_overlapping_problems(problems_to_fix)
        )
        if not problems_to_fix and evaluation_result.get("needs_revision"):
            logger.warning(
                "Revision for ch %s explicitly requested, but no specific problems were itemized.",
                chapter_number,
            )
        elif not problems_to_fix:
            logger.info(
                "No specific problems found for ch %s, and not marked for revision. No revision performed.",
                chapter_number,
            )
            return (original_text, "No revision performed.", []), None

        revision_reason_str_list = evaluation_result.get("reasons", [])
        revision_reason_str = (
            "\n- ".join(revision_reason_str_list)
            if revision_reason_str_list
            else "General unspecified issues."
        )
        logger.info(
            "Attempting revision for chapter %s. Reason(s):\n- %s",
            chapter_number,
            revision_reason_str,
        )

        patched_text: str | None = None
        all_spans_in_patched_text: list[tuple[int, int]] = list(already_patched_spans)

        if settings.ENABLE_PATCH_BASED_REVISION:
            logger.info(
                "Attempting patch-based revision for Ch %s with %s problem(s).",
                chapter_number,
                len(problems_to_fix),
            )
            if settings.AGENT_ENABLE_PATCH_VALIDATION:
                validator: PatchValidationAgent | Any = PatchValidationAgent()
            else:

                class _BypassValidator:
                    async def validate_patch(
                        self, *_args: Any, **_kwargs: Any
                    ) -> tuple[bool, None]:
                        return True, None

                validator = _BypassValidator()

            patcher = patch_generator.PatchGenerator()
            (
                patched_text,
                all_spans_in_patched_text,
            ) = await patcher.generate_and_apply(
                plot_outline,
                original_text,
                problems_to_fix,
                chapter_number,
                hybrid_context_for_revision,
                chapter_plan,
                already_patched_spans,
                validator,
            )
            if patched_text != original_text:
                logger.info(
                    "Patch process for Ch %s produced revised text. Original len: %s, Patched text len: %s.",
                    chapter_number,
                    len(original_text),
                    len(patched_text),
                )
            else:
                logger.warning(
                    "Patch-based revision for Ch %s: No valid patch instructions were generated. Will consider full rewrite if needed.",
                    chapter_number,
                )

        final_revised_text: str | None = None
        final_raw_llm_output: str | None = (
            f"Chapter revised using {len(all_spans_in_patched_text) - len(already_patched_spans)} new patches."
        )
        final_spans_for_next_cycle = all_spans_in_patched_text

        use_patched_text_as_final = False
        if patched_text is not None and patched_text != original_text:
            evaluator = ComprehensiveEvaluatorAgent()
            world_ids = {
                cat: [item.id for item in items.values() if isinstance(item, WorldItem)]
                for cat, items in world_building.items()
                if isinstance(items, dict)
            }
            plot_focus, plot_idx = patch_generator._get_plot_point_info(
                plot_outline, chapter_number
            )
            post_eval, post_usage = await evaluator.evaluate_chapter_draft(
                plot_outline,
                list(character_profiles.keys()),
                world_ids,
                patched_text,
                chapter_number,
                plot_focus,
                plot_idx,
                hybrid_context_for_revision,
            )
            _add_usage(post_usage)
            remaining = len(post_eval.get("problems_found", []))
            if remaining <= settings.POST_PATCH_PROBLEM_THRESHOLD:
                use_patched_text_as_final = True

        if use_patched_text_as_final:
            final_revised_text = patched_text
            logger.info(
                "Ch %s: Using patched text as the revised version.", chapter_number
            )

        if not use_patched_text_as_final and evaluation_result.get("needs_revision"):
            logger.info(
                "Proceeding with full chapter rewrite for Ch %s as patching was ineffective or disabled.",
                chapter_number,
            )
            max_original_snippet_tokens = settings.MAX_CONTEXT_TOKENS // 3
            original_snippet = truncate_text_by_tokens(
                original_text,
                settings.REVISION_MODEL,
                max_original_snippet_tokens,
                truncation_marker="\n... (original draft snippet truncated for brevity in rewrite prompt)",
            )
            plan_focus_section_full_rewrite_parts: list[str] = []
            plot_point_focus_full_rewrite, _ = patch_generator._get_plot_point_info(
                plot_outline, chapter_number
            )
            max_plan_tokens_for_full_rewrite = settings.MAX_CONTEXT_TOKENS // 2
            if settings.ENABLE_AGENTIC_PLANNING and chapter_plan:
                formatted_plan_fr = (
                    patch_generator._get_formatted_scene_plan_from_agent_or_fallback(
                        chapter_plan,
                        settings.REVISION_MODEL,
                        max_plan_tokens_for_full_rewrite,
                    )
                )
                plan_focus_section_full_rewrite_parts.append(formatted_plan_fr)
                if "plan truncated" in formatted_plan_fr:
                    logger.warning(
                        "Scene plan token-truncated for Ch %s full rewrite prompt.",
                        chapter_number,
                    )
            else:
                plan_focus_section_full_rewrite_parts.append(
                    f"**Original Chapter Focus (Target):**\n{plot_point_focus_full_rewrite or 'Not specified.'}\n"
                )
            plan_focus_section_full_rewrite_str = "".join(
                plan_focus_section_full_rewrite_parts
            )

            length_issue_explicit_instruction_full_rewrite_parts: list[str] = []
            needs_expansion_from_problems = any(
                (
                    p["issue_category"] == "narrative_depth_and_length"
                    and (
                        "short" in p["problem_description"].lower()
                        or "length" in p["problem_description"].lower()
                        or "expand" in p["suggested_fix_focus"].lower()
                        or "depth" in p["problem_description"].lower()
                    )
                )
                for p in problems_to_fix
            )
            if needs_expansion_from_problems:
                length_issue_explicit_instruction_full_rewrite_parts.extend(
                    [
                        "\n**Specific Focus on Expansion:** A key critique involves insufficient length and/or narrative depth. ",
                        "Your revision MUST substantially expand the narrative by incorporating more descriptive details, character thoughts/introspection, dialogue, actions, and sensory information. ",
                        f"Aim for a chapter length of at least {settings.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.",
                    ]
                )
            length_issue_explicit_instruction_full_rewrite_str = "".join(
                length_issue_explicit_instruction_full_rewrite_parts
            )

            protagonist_name_full_rewrite = plot_outline.get(
                "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
            )

            all_problem_descriptions_parts: list[str] = []
            if problems_to_fix:
                all_problem_descriptions_parts.append(
                    "**Detailed Issues to Address (from evaluation):**\n"
                )
                for prob_idx, prob_item in enumerate(problems_to_fix):
                    all_problem_descriptions_parts.extend(
                        [
                            f"  {prob_idx + 1}. Category: {prob_item['issue_category']}",
                            f"     Description: {prob_item['problem_description']}",
                            f'     Quote Ref: "{prob_item["quote_from_original_text"][:100].replace(chr(10), " ")}..."',
                            f"     Fix Focus: {prob_item['suggested_fix_focus']}\n",
                        ]
                    )
                all_problem_descriptions_parts.append("---\n")
            all_problem_descriptions_str = "".join(all_problem_descriptions_parts)

            deduplication_note = ""
            if is_from_flawed_source:
                deduplication_note = "\n**(Note: The 'Original Draft Snippet' below may have had repetitive content removed prior to evaluation, or other flaws were present. Ensure your rewrite is cohesive and addresses any resulting narrative gaps or inconsistencies.)**\n"

            prompt_full_rewrite_lines: list[str] = []
            if settings.ENABLE_LLM_NO_THINK_DIRECTIVE:
                prompt_full_rewrite_lines.append("/no_think")

            prompt_full_rewrite_lines.extend(
                [
                    f"You are an expert novelist rewriting Chapter {chapter_number} featuring protagonist {protagonist_name_full_rewrite}.",
                    "**Critique/Reason(s) for Revision (MUST be addressed comprehensively):**",
                    "--- FEEDBACK START ---",
                    llm_service.clean_model_response(revision_reason_str).strip(),
                    "--- FEEDBACK END ---",
                    all_problem_descriptions_str,
                    deduplication_note,
                    length_issue_explicit_instruction_full_rewrite_str,
                    plan_focus_section_full_rewrite_str,
                    "**Hybrid Context from Previous Chapters (for consistency with established canon and narrative flow):**",
                    "--- BEGIN HYBRID CONTEXT ---",
                    hybrid_context_for_revision
                    if hybrid_context_for_revision.strip()
                    else "No previous context.",
                    "--- END HYBRID CONTEXT ---",
                    "**Original Draft Snippet (for reference of what went wrong - DO NOT COPY VERBATIM. Your goal is a fresh rewrite addressing all critique and aligning with the plan/focus):**",
                    "--- BEGIN ORIGINAL DRAFT SNIPPET ---",
                    original_snippet,
                    "--- END ORIGINAL DRAFT SNIPPET ---",
                    "",
                    "**Revision Instructions:**",
                    "1.  **ABSOLUTE PRIORITY:** Thoroughly address ALL issues listed in **Critique/Reason(s) for Revision** and **Detailed Issues to Address**. If the original text had content removed (e.g., due to de-duplication) or other flaws as noted, pay special attention to ensuring a smooth, coherent narrative flow and filling any gaps logically.",
                    "2.  **Rewrite the ENTIRE chapter.** Produce a fresh, coherent, and engaging narrative.",
                    "3.  If a Detailed Scene Plan is provided in `plan_focus_section_full_rewrite_str`, follow it closely. Otherwise, align with the `Original Chapter Focus`.",
                    "4.  Ensure seamless narrative flow with the **Hybrid Context**. Pay close attention to any `KEY RELIABLE KG FACTS` mentioned.",
                    f"5.  Maintain the novel's established tone, style, and genre ('{plot_outline.get('genre', 'story')}').",
                    f"6.  Target a substantial chapter length, aiming for at least {settings.MIN_ACCEPTABLE_DRAFT_LENGTH} characters of narrative text.",
                    '7.  Output ONLY the rewritten chapter text.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.',
                    "",
                    f"--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---",
                ]
            )
            prompt_full_rewrite = "\n".join(prompt_full_rewrite_lines)

            logger.info(
                "Calling LLM (%s) for Ch %s full rewrite. Min length: %s chars.",
                settings.REVISION_MODEL,
                chapter_number,
                settings.MIN_ACCEPTABLE_DRAFT_LENGTH,
            )

            (
                raw_revised_llm_output_for_log,
                full_rewrite_usage,
            ) = await llm_service.async_call_llm(
                model_name=settings.REVISION_MODEL,
                prompt=prompt_full_rewrite,
                temperature=settings.TEMPERATURE_REVISION,
                max_tokens=None,
                allow_fallback=True,
                stream_to_disk=True,
                frequency_penalty=settings.FREQUENCY_PENALTY_REVISION,
                presence_penalty=settings.PRESENCE_PENALTY_REVISION,
                auto_clean_response=False,
            )
            _add_usage(full_rewrite_usage)

            final_revised_text = llm_service.clean_model_response(
                raw_revised_llm_output_for_log
            )
            final_raw_llm_output = raw_revised_llm_output_for_log
            final_spans_for_next_cycle = []

            logger.info(
                "Full rewrite for Ch %s generated text of length %s.",
                chapter_number,
                len(final_revised_text),
            )

        if not final_revised_text:
            logger.error(
                "Revision process for ch %s resulted in no usable content.",
                chapter_number,
            )
            return (
                None,
                cumulative_usage_data
                if cumulative_usage_data["total_tokens"] > 0
                else None,
            )

        if len(final_revised_text) < settings.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                "Final revised draft for ch %s is short (%s chars). Min target: %s.",
                chapter_number,
                len(final_revised_text),
                settings.MIN_ACCEPTABLE_DRAFT_LENGTH,
            )

        logger.info(
            "Revision process for ch %s produced a candidate text (Length: %s chars).",
            chapter_number,
            len(final_revised_text),
        )
        return (
            final_revised_text,
            final_raw_llm_output,
            final_spans_for_next_cycle,
        ), cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None
