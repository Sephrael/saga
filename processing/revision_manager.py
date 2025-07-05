# processing/revision_manager.py
from typing import Any

import structlog
import utils
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.patch_validation_agent import NoOpPatchValidator, PatchValidationAgent
from config import settings
from core.llm_interface import llm_service, truncate_text_by_tokens
from core.usage import TokenUsage
from utils.plot import get_plot_point_info

from models import (
    CharacterProfile,
    EvaluationResult,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)

from .patch import (
    PatchGenerator,
    _deduplicate_problems,
    _get_formatted_scene_plan_from_agent_or_fallback,
)

logger = structlog.get_logger(__name__)


class RevisionManager:
    """Coordinate chapter revision via patches and rewrites."""

    def _should_defer_full_rewrite(self, cycle: int) -> bool:
        """Return ``True`` if full rewrite should be deferred for this cycle."""

        if not settings.DEFER_FULL_REWRITE_UNTIL_LAST_CYCLE:
            return False
        return cycle < settings.MAX_REVISION_CYCLES_PER_CHAPTER - 1

    def identify_root_cause(
        self,
        problems: list[dict[str, Any]],
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
    ) -> str | None:
        """Identify a likely root cause for persistent revision issues."""

        search_text = " ".join(
            str(p.get("problem_description", ""))
            + " "
            + str(p.get("quote_from_original_text", ""))
            for p in problems
        ).lower()

        def _find_match(names: list[str], template: str) -> str | None:
            for name in names:
                if name.lower() in search_text and (
                    "inconsistent" in search_text or "contradict" in search_text
                ):
                    return template.format(name=name)
            return None

        char_match = _find_match(
            list(character_profiles.keys()),
            "The error originates from the conflicting description in {name}'s character profile.",
        )
        if char_match:
            return char_match

        world_names: list[str] = []
        for category_dict in world_building.values():
            if not isinstance(category_dict, dict):
                continue
            for item in category_dict.values():
                if getattr(item, "name", None):
                    world_names.append(item.name)

        world_match = _find_match(
            world_names,
            "The error originates from conflicting world element definition for {name}.",
        )
        if world_match:
            return world_match

        plot_points = [str(pp) for pp in plot_outline.get("plot_points", []) if pp]
        for idx, pp in enumerate(plot_points):
            if pp.lower() in search_text and (
                "inconsistent" in search_text or "contradict" in search_text
            ):
                return (
                    f"The error originates from a conflict with plot point {idx + 1}."
                )

        return None

    async def _patch_revision_cycle(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        original_text: str,
        chapter_number: int,
        problems_to_fix: list[ProblemDetail],
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        already_patched_spans: list[tuple[int, int]],
        continuity_problems: list[ProblemDetail] | None = None,
        repetition_problems: list[ProblemDetail] | None = None,
    ) -> tuple[str | None, list[tuple[int, int]], bool, TokenUsage | None]:
        """Run one patch generation and evaluation cycle."""

        cumulative_usage = TokenUsage()

        def add_usage(usage: TokenUsage | dict[str, int] | None) -> None:
            cumulative_usage.add(usage)

        logger.info(
            "Attempting patch-based revision for Ch %s with %s problem(s).",
            chapter_number,
            len(problems_to_fix),
        )
        if settings.AGENT_ENABLE_PATCH_VALIDATION:
            validator: PatchValidationAgent | Any = PatchValidationAgent()
        else:
            validator = NoOpPatchValidator()

        all_problems = list(problems_to_fix)
        if continuity_problems:
            all_problems.extend(continuity_problems)
        if repetition_problems:
            all_problems.extend(repetition_problems)
        all_problems = _deduplicate_problems(all_problems)

        patcher = PatchGenerator()
        patched_text, spans, patch_usage = await patcher.generate_and_apply(
            plot_outline,
            original_text,
            all_problems,
            chapter_number,
            hybrid_context_for_revision,
            chapter_plan,
            already_patched_spans,
            validator,
        )
        add_usage(patch_usage)

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

        use_patched_text_as_final = False
        if patched_text is not None and patched_text != original_text:
            evaluator = ComprehensiveEvaluatorAgent()
            plot_focus, plot_idx = get_plot_point_info(plot_outline, chapter_number)
            post_eval, post_usage = await evaluator.evaluate_chapter_draft(
                plot_outline,
                patched_text,
                chapter_number,
                plot_focus,
                plot_idx,
                hybrid_context_for_revision,
            )
            add_usage(post_usage)
            remaining = len(post_eval.problems_found)
            if remaining <= settings.POST_PATCH_PROBLEM_THRESHOLD:
                use_patched_text_as_final = True

        return (
            patched_text,
            spans,
            use_patched_text_as_final,
            cumulative_usage if cumulative_usage.total_tokens > 0 else None,
        )

    async def _rewrite_problem_scenes(
        self,
        plot_outline: dict[str, Any],
        original_text: str,
        chapter_number: int,
        problems_to_fix: list[ProblemDetail],
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> tuple[str, str, TokenUsage | None]:
        """Rewrite only the paragraphs containing flagged problems."""

        paragraphs = utils.get_text_segments(original_text, "paragraph")
        para_texts = [p[0] for p in paragraphs]
        usage_total = TokenUsage()

        for idx, (_txt, start, end) in enumerate(paragraphs):
            probs = [
                p
                for p in problems_to_fix
                if (
                    (
                        p.sentence_char_start is not None
                        and start <= p.sentence_char_start < end
                    )
                    or (
                        p.quote_char_start is not None
                        and start <= p.quote_char_start < end
                    )
                )
            ]
            if not probs:
                continue
            issues = "\n".join(
                f"- {p.problem_description} (Fix: {p.suggested_fix_focus})"
                for p in probs
            )
            plan_str = ""
            if chapter_plan:
                plan_str = _get_formatted_scene_plan_from_agent_or_fallback(
                    chapter_plan,
                    settings.REVISION_MODEL,
                    settings.MAX_CONTEXT_TOKENS // 4,
                )
            prompt = (
                f"Rewrite the following paragraph from chapter {chapter_number} addressing:\n{issues}\n"
                f"Paragraph:\n{_txt}\n"
                f"{plan_str}\nContext:\n{hybrid_context_for_revision}"
            )
            new_para, usage = await llm_service.async_call_llm(
                model_name=settings.REVISION_MODEL,
                prompt=prompt,
                temperature=settings.TEMPERATURE_REVISION,
                max_tokens=4096,
                allow_fallback=True,
                stream_to_disk=False,
                auto_clean_response=True,
            )
            usage_total.add(usage)
            para_texts[idx] = llm_service.clean_model_response(new_para)

        new_text = "\n\n".join(para_texts)
        return new_text, new_text, usage_total if usage_total.total_tokens > 0 else None

    async def _perform_full_rewrite(
        self,
        plot_outline: dict[str, Any],
        original_text: str,
        chapter_number: int,
        problems_to_fix: list[ProblemDetail],
        revision_reason_str: str,
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        is_from_flawed_source: bool,
    ) -> tuple[str, str, TokenUsage | None]:
        """Rewrite the entire chapter using the LLM."""

        max_original_snippet_tokens = settings.MAX_CONTEXT_TOKENS // 3
        original_snippet = truncate_text_by_tokens(
            original_text,
            settings.REVISION_MODEL,
            max_original_snippet_tokens,
            truncation_marker="\n... (original draft snippet truncated for brevity in rewrite prompt)",
        )

        plan_focus_section_parts: list[str] = []
        plot_point_focus, _ = get_plot_point_info(plot_outline, chapter_number)
        max_plan_tokens_for_full_rewrite = settings.MAX_CONTEXT_TOKENS // 2
        if settings.ENABLE_AGENTIC_PLANNING and chapter_plan:
            formatted_plan = _get_formatted_scene_plan_from_agent_or_fallback(
                chapter_plan,
                settings.REVISION_MODEL,
                max_plan_tokens_for_full_rewrite,
            )
            plan_focus_section_parts.append(formatted_plan)
            if "plan truncated" in formatted_plan:
                logger.warning(
                    "Scene plan token-truncated for Ch %s full rewrite prompt.",
                    chapter_number,
                )
        else:
            plan_focus_section_parts.append(
                f"**Original Chapter Focus (Target):**\n{plot_point_focus or 'Not specified.'}\n"
            )
        plan_focus_section = "".join(plan_focus_section_parts)

        length_instruction_parts: list[str] = []
        needs_expansion = any(
            (
                p.issue_category == "narrative_depth_and_length"
                and (
                    "short" in p.problem_description.lower()
                    or "length" in p.problem_description.lower()
                    or "expand" in p.suggested_fix_focus.lower()
                    or "depth" in p.problem_description.lower()
                )
            )
            for p in problems_to_fix
        )
        if needs_expansion:
            length_instruction_parts.extend(
                [
                    "\n**Specific Focus on Expansion:** A key critique involves insufficient length and/or narrative depth. ",
                    "Your revision MUST substantially expand the narrative by incorporating more descriptive details, character thoughts/introspection, dialogue, actions, and sensory information. ",
                    f"Aim for a chapter length of at least {settings.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.",
                ]
            )
        length_instruction = "".join(length_instruction_parts)

        rewrite_instructions_lines: list[str] = []
        for idx, prob in enumerate(problems_to_fix):
            instruction = prob.rewrite_instruction
            if instruction:
                rewrite_instructions_lines.append(f"  {idx + 1}. {instruction}")
        rewrite_instructions_str = (
            "**Explicit Rewrite Guidelines:**\n"
            + "\n".join(rewrite_instructions_lines)
            + "\n"
            if rewrite_instructions_lines
            else ""
        )

        protagonist_name = plot_outline.get(
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
                        f"  {prob_idx + 1}. Category: {prob_item.issue_category}",
                        f"     Description: {prob_item.problem_description}",
                        f'     Quote Ref: "{prob_item.quote_from_original_text[:100].replace(chr(10), " ")}..."',
                        f"     Fix Focus: {prob_item.suggested_fix_focus}\n",
                    ]
                )
            all_problem_descriptions_parts.append("---\n")
        all_problem_descriptions_str = "".join(all_problem_descriptions_parts)

        deduplication_note = ""
        if is_from_flawed_source:
            deduplication_note = "\n**(Note: The 'Original Draft Snippet' below may have had repetitive content removed prior to evaluation, or other flaws were present. Ensure your rewrite is cohesive and addresses any resulting narrative gaps or inconsistencies.)**\n"

        prompt_lines: list[str] = []
        if settings.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")

        prompt_lines.extend(
            [
                f"You are an expert novelist rewriting Chapter {chapter_number} featuring protagonist {protagonist_name}.",
                "**Critique/Reason(s) for Revision (MUST be addressed comprehensively):**",
                "--- FEEDBACK START ---",
                llm_service.clean_model_response(revision_reason_str).strip(),
                "--- FEEDBACK END ---",
                all_problem_descriptions_str,
                deduplication_note,
                rewrite_instructions_str,
                length_instruction,
                plan_focus_section,
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
                "7.  **Show, Don't Tell:** Convert abstract statements from the original draft into concrete scenes. For example, instead of saying S\xe1g\xe1 felt a conflict, describe an action it takes and then immediately regrets, or a moment of hesitation before interacting with an object.",
                '8.  **Avoid Lexical Repetition:** Do not overuse key thematic words (e.g., "purpose", "legacy", "silence", "unspoken"). If a concept must be revisited, express it through different phrasing, actions, or dialogue.',
                '9.  Output ONLY the rewritten chapter text.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.',
                "",
                f"--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---",
            ]
        )
        prompt_full_rewrite = "\n".join(prompt_lines)

        logger.info(
            "Calling LLM (%s) for Ch %s full rewrite. Min length: %s chars.",
            settings.REVISION_MODEL,
            chapter_number,
            settings.MIN_ACCEPTABLE_DRAFT_LENGTH,
        )

        raw_revised_llm_output, usage = await llm_service.async_call_llm(
            model_name=settings.REVISION_MODEL,
            prompt=prompt_full_rewrite,
            temperature=settings.TEMPERATURE_REVISION,
            max_tokens=None,
            allow_fallback=True,
            stream_to_disk=True,
            auto_clean_response=False,
        )

        cleaned = llm_service.clean_model_response(raw_revised_llm_output)
        logger.info(
            "Full rewrite for Ch %s generated text of length %s.",
            chapter_number,
            len(cleaned),
        )

        return cleaned, raw_revised_llm_output, usage

    def _is_deeply_flawed(self, problems: list[ProblemDetail]) -> bool:
        """Return ``True`` if the draft is considered deeply flawed."""

        return len(problems) > settings.REWRITE_TRIGGER_PROBLEM_COUNT or any(
            p.issue_category == "narrative_depth_and_length" for p in problems
        )

    async def _maybe_direct_full_rewrite(
        self,
        is_deeply_flawed: bool,
        revision_cycle: int,
        plot_outline: dict[str, Any],
        original_text: str,
        chapter_number: int,
        problems_to_fix: list[ProblemDetail],
        revision_reason_str: str,
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        is_from_flawed_source: bool,
    ) -> tuple[str | None, str | None, TokenUsage | None]:
        """Perform a full rewrite immediately if conditions warrant."""

        if (
            is_deeply_flawed
            and settings.ENABLE_STRATEGIC_REWRITES
            and not self._should_defer_full_rewrite(revision_cycle)
        ):
            logger.info(
                "Deeply flawed draft detected. Proceeding directly to full chapter rewrite."
            )
            return await self._perform_full_rewrite(
                plot_outline,
                original_text,
                chapter_number,
                problems_to_fix,
                revision_reason_str,
                hybrid_context_for_revision,
                chapter_plan,
                is_from_flawed_source,
            )
        return None, None, None

    async def _apply_patch_revision(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        original_text: str,
        chapter_number: int,
        problems_to_fix: list[ProblemDetail],
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        already_patched_spans: list[tuple[int, int]],
        continuity_problems: list[ProblemDetail] | None,
        repetition_problems: list[ProblemDetail] | None,
    ) -> tuple[str | None, list[tuple[int, int]], bool, TokenUsage | None]:
        """Run one patch revision cycle if enabled."""

        if not settings.ENABLE_PATCH_BASED_REVISION:
            return None, list(already_patched_spans), False, None

        return await self._patch_revision_cycle(
            plot_outline,
            character_profiles,
            world_building,
            original_text,
            chapter_number,
            problems_to_fix,
            hybrid_context_for_revision,
            chapter_plan,
            already_patched_spans,
            continuity_problems=continuity_problems,
            repetition_problems=repetition_problems,
        )

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
        revision_cycle: int,
        is_from_flawed_source: bool = False,
        already_patched_spans: list[tuple[int, int]] | None = None,
        continuity_problems: list[ProblemDetail] | None = None,
        repetition_problems: list[ProblemDetail] | None = None,
    ) -> tuple[tuple[str, str | None, list[tuple[int, int]]] | None, TokenUsage | None]:
        """Revise a chapter draft based on evaluation feedback.

        Args:
            plot_outline: The overall plot outline for the novel.
            character_profiles: Profiles of all major characters.
            world_building: World building items grouped by category.
            original_text: The initial chapter draft to revise.
            chapter_number: Number of the chapter being revised.
            evaluation_result: Structured feedback from the evaluator agent.
            hybrid_context_for_revision: Context window for maintaining
                continuity with previous chapters.
            chapter_plan: Optional scene plan used when agentic planning is
                enabled.
            revision_cycle: Current revision attempt number (0-indexed).
            is_from_flawed_source: Whether the original text came from a flawed
                generation process.
            already_patched_spans: Spans previously protected from further
                modification.
            continuity_problems: Additional continuity issues from
                the consistency checker.
            repetition_problems: Repetition issues from
                ``RepetitionAnalyzer``.

        Returns:
            A tuple containing either the revised text, the raw LLM output, and
            updated spans, or ``None`` if revision failed. The second element of
            the outer tuple is token usage information accumulated during the
            process.

        Side Effects:
            Communicates with multiple agents and the LLM service and logs
            progress throughout the revision process.
        """
        if already_patched_spans is None:
            already_patched_spans = []

        cumulative_usage_data = TokenUsage()

        if not original_text:
            logger.error("Revision for ch %s aborted: missing original text.", chapter_number)
            return None, None

        problems_to_fix: list[ProblemDetail] = list(evaluation_result.problems_found) # Make a mutable copy
        if continuity_problems:
            problems_to_fix.extend(continuity_problems)
        if repetition_problems:
            problems_to_fix.extend(repetition_problems)
        problems_to_fix = _deduplicate_problems(problems_to_fix)

        if not problems_to_fix and not evaluation_result.needs_revision:
            logger.info("No specific problems found for ch %s, and not marked for revision. No revision performed.", chapter_number)
            return (original_text, "No revision performed.", []), None

        if not problems_to_fix and evaluation_result.needs_revision:
            logger.warning("Revision for ch %s explicitly requested, but no specific problems were itemized.", chapter_number)
            # Proceeding as if general revision is needed, though specific problems list is empty.

        revision_reason_str_list = evaluation_result.reasons
        revision_reason_str = "\n- ".join(revision_reason_str_list) if revision_reason_str_list else "General unspecified issues."
        logger.info("Attempting revision for chapter %s. Reason(s):\n- %s", chapter_number, revision_reason_str)

        is_deeply_flawed = self._is_deeply_flawed(problems_to_fix)

        (
            revised_text,
            raw_output,
            new_spans,
            strategy_usage
        ) = await self._determine_and_execute_revision_strategy(
            plot_outline=plot_outline,
            character_profiles=character_profiles,
            world_building=world_building,
            original_text=original_text,
            chapter_number=chapter_number,
            problems_to_fix=problems_to_fix,
            hybrid_context_for_revision=hybrid_context_for_revision,
            chapter_plan=chapter_plan,
            revision_cycle=revision_cycle,
            is_from_flawed_source=is_from_flawed_source,
            already_patched_spans=already_patched_spans,
            is_deeply_flawed=is_deeply_flawed,
            evaluation_needs_revision=evaluation_result.needs_revision,
            revision_reason_str=revision_reason_str,
        )
        cumulative_usage_data.add(strategy_usage)

        if not revised_text:
            logger.error("Revision process for ch %s resulted in no usable content.", chapter_number)
            return None, cumulative_usage_data.get_if_used()

        if len(revised_text) < settings.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(
                "Final revised draft for ch %s is short (%s chars). Min target: %s.",
                chapter_number, len(revised_text), settings.MIN_ACCEPTABLE_DRAFT_LENGTH
            )

        logger.info(
            "Revision process for ch %s produced a candidate text (Length: %s chars).",
            chapter_number, len(revised_text)
        )
        return (revised_text, raw_output, new_spans), cumulative_usage_data.get_if_used()

    async def _determine_and_execute_revision_strategy(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        original_text: str,
        chapter_number: int,
        problems_to_fix: list[ProblemDetail],
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        revision_cycle: int,
        is_from_flawed_source: bool,
        already_patched_spans: list[tuple[int, int]],
        is_deeply_flawed: bool,
        evaluation_needs_revision: bool,
        revision_reason_str: str,
    ) -> tuple[str | None, str | None, list[tuple[int, int]], TokenUsage | None]:
        """Core logic to decide and execute a revision strategy."""
        cumulative_usage = TokenUsage()

        # 1. Attempt direct full rewrite if conditions met (early exit)
        direct_rewrite_text, direct_rewrite_raw, direct_rewrite_usage = await self._maybe_direct_full_rewrite(
            is_deeply_flawed, revision_cycle, plot_outline, original_text, chapter_number,
            problems_to_fix, revision_reason_str, hybrid_context_for_revision, chapter_plan, is_from_flawed_source
        )
        cumulative_usage.add(direct_rewrite_usage)
        if direct_rewrite_text is not None:
            return direct_rewrite_text, direct_rewrite_raw, [], cumulative_usage.get_if_used()

        # 2. Attempt patch-based revision
        patched_text, spans_after_patch, use_patched_text, patch_usage = await self._apply_patch_revision(
            plot_outline, character_profiles, world_building, original_text, chapter_number,
            problems_to_fix, hybrid_context_for_revision, chapter_plan, already_patched_spans,
            continuity_problems=None, repetition_problems=None # These are already in problems_to_fix
        )
        cumulative_usage.add(patch_usage)

        current_text = patched_text if patched_text is not None else original_text
        current_raw_output = f"Chapter revised using {len(spans_after_patch) - len(already_patched_spans)} new patches." if use_patched_text and patched_text else "Patching did not yield usable text or was not used."
        current_spans = spans_after_patch

        if use_patched_text:
            logger.info("Ch %s: Using patched text as the revised version for this cycle.", chapter_number)
            return current_text, current_raw_output, current_spans, cumulative_usage.get_if_used()

        # 3. If patch wasn't used/effective and revision is still needed & not deferred
        if evaluation_needs_revision and not self._should_defer_full_rewrite(revision_cycle):
            # Try targeted scene rewrite first
            logger.info("Attempting targeted scene rewrite for Ch %s before full rewrite.", chapter_number)
            targeted_rewrite_text, targeted_raw_output, targeted_usage = await self._rewrite_problem_scenes(
                plot_outline, original_text, chapter_number, problems_to_fix,
                hybrid_context_for_revision, chapter_plan
            )
            cumulative_usage.add(targeted_usage)
            if targeted_rewrite_text and targeted_rewrite_text != original_text : # Check if rewrite actually changed text
                logger.info("Ch %s: Targeted scene rewrite produced changes.", chapter_number)
                return targeted_rewrite_text, targeted_raw_output, [], cumulative_usage.get_if_used()

            # If targeted rewrite was ineffective, proceed to full rewrite
            logger.info("Proceeding with full chapter rewrite for Ch %s as targeted rewrite was ineffective or produced no change.", chapter_number)
            full_rewrite_text, full_raw_output, full_rewrite_usage = await self._perform_full_rewrite(
                plot_outline, original_text, chapter_number, problems_to_fix, revision_reason_str,
                hybrid_context_for_revision, chapter_plan, is_from_flawed_source
            )
            cumulative_usage.add(full_rewrite_usage)
            if full_rewrite_text:
                 return full_rewrite_text, full_raw_output, [], cumulative_usage.get_if_used()

        # 4. If revision is needed but deferred
        elif evaluation_needs_revision and self._should_defer_full_rewrite(revision_cycle):
            logger.info("Deferring full rewrite for Ch %s until later cycle. Using original text for now.", chapter_number)
            return original_text, "Rewrite deferred", already_patched_spans, cumulative_usage.get_if_used()

        # 5. Fallback: If no strategy applied or was effective, return original text or patched if available but not "used"
        # This case should ideally be covered by the above logic, but as a safety.
        # If patched_text exists and is different, it means patches were made but _apply_patch_revision decided not to "use_patch_as_final"
        # In this scenario, we might still want to return the patched_text if no other rewrite happened.
        if patched_text is not None and patched_text != original_text:
             logger.warning("Ch %s: No definitive revision strategy applied, but patches were made. Returning patched text.", chapter_number)
             return patched_text, "Patches applied but not validated as final; no further rewrite triggered.", spans_after_patch, cumulative_usage.get_if_used()

        logger.warning("Ch %s: No revision strategy resulted in a change. Returning original text.", chapter_number)
        return original_text, "No effective revision applied.", already_patched_spans, cumulative_usage.get_if_used()
