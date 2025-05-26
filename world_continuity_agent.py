# world_continuity_agent.py
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import config
import llm_interface
import utils # MODIFIED: For spaCy functions
from type import ProblemDetail
from data_access import kg_queries
from prompt_data_getters import (
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
    get_reliable_kg_facts_for_drafting_prompt
)
from parsing_utils import split_text_into_blocks, parse_key_value_block

logger = logging.getLogger(__name__)

PROBLEM_DETAIL_KEY_MAP = {
    "issue_category": "issue_category",
    "problem_description": "problem_description",
    "quote_from_original": "quote_from_original_text", # Maps to text field
    "suggested_fix_focus": "suggested_fix_focus"
}

class WorldContinuityAgent:
    def __init__(self, model_name: str = config.EVALUATION_MODEL):
        self.model_name = model_name
        logger.info(f"WorldContinuityAgent initialized with model: {self.model_name}")
        utils.load_spacy_model_if_needed() # Ensure spaCy model is available

    async def _parse_llm_consistency_output(self, text: str, chapter_number: int, original_draft_text: str) -> List[ProblemDetail]: # Added original_draft_text
        """
        Parses LLM plain text output specifically for consistency problems.
        Populates character offsets for the quote and its containing sentence using spaCy.
        """
        final_problems: List[ProblemDetail] = []
        if not text or not text.strip() or "no significant consistency problems found" in text.lower() or "no significant problems found" in text.lower():
            logger.info(f"Consistency check for Ch {chapter_number} is empty or indicates no problems. No problems parsed.")
            return []

        problem_blocks_text = split_text_into_blocks(text, separator_regex_str=r'\n\s*---\s*\n')

        for block_num, block_content in enumerate(problem_blocks_text):
            if not block_content.strip():
                continue
            logger.debug(f"Parsing consistency problem block {block_num+1} for Ch {chapter_number}:\n{block_content[:150]}...")

            parsed_problem_dict = parse_key_value_block(
                block_text_or_lines=block_content,
                key_map=PROBLEM_DETAIL_KEY_MAP,
                list_internal_keys=[]
            )

            required_keys_internal = set(PROBLEM_DETAIL_KEY_MAP.values())
            missing_keys = required_keys_internal - set(parsed_problem_dict.keys())

            problem_meta: ProblemDetail = { # Initialize with defaults
                "issue_category": "consistency", # Default for this agent
                "problem_description": "N/A",
                "quote_from_original_text": "N/A - Malformed LLM Output",
                "quote_char_start": None, "quote_char_end": None,
                "sentence_char_start": None, "sentence_char_end": None,
                "suggested_fix_focus": "Review LLM output."
            }

            if missing_keys:
                logger.warning(f"Could not parse all required fields from consistency problem block {block_num+1} for Ch {chapter_number}. Missing: {missing_keys}. Block: '{block_content[:150]}...'")
                problem_meta["problem_description"] = f"Malformed consistency problem block from LLM: {block_content}"
                final_problems.append(problem_meta)
                continue

            problem_meta["problem_description"] = parsed_problem_dict.get("problem_description", "N/A")
            problem_meta["quote_from_original_text"] = parsed_problem_dict.get("quote_from_original_text", "N/A - General Issue")
            problem_meta["suggested_fix_focus"] = parsed_problem_dict.get("suggested_fix_focus", "N/A")

            category = str(parsed_problem_dict.get("issue_category", "consistency")).strip().lower()
            if category != "consistency":
                 logger.warning(f"WorldContinuityAgent parsed a non-consistency category '{category}' in block {block_num+1} for Ch {chapter_number}. Forcing to 'consistency'.")
                 problem_meta["issue_category"] = "consistency"
            else:
                problem_meta["issue_category"] = category

            quote_text_from_llm = problem_meta["quote_from_original_text"]
            # Normalize "N/A..." variations to a single canonical form first
            if "N/A - General Issue" in quote_text_from_llm or not quote_text_from_llm.strip():
                problem_meta["quote_from_original_text"] = "N/A - General Issue"
                logger.debug(f"Consistency problem block {block_num+1} for Ch {chapter_number} is 'N/A - General Issue' or empty quote.")
            elif utils.NLP_SPACY is not None and original_draft_text.strip():
                offsets_tuple = await utils.find_quote_and_sentence_offsets_with_spacy(original_draft_text, quote_text_from_llm)
                if offsets_tuple:
                    q_start, q_end, s_start, s_end = offsets_tuple
                    problem_meta["quote_char_start"] = q_start
                    problem_meta["quote_char_end"] = q_end
                    problem_meta["sentence_char_start"] = s_start
                    problem_meta["sentence_char_end"] = s_end
                    logger.debug(f"Ch {chapter_number} consistency problem: Quote '{quote_text_from_llm[:30]}...' found at {q_start}-{q_end}, Sentence: {s_start}-{s_end}")
                else:
                    logger.warning(f"Ch {chapter_number} consistency problem: Could not find quote via spaCy utils: '{quote_text_from_llm[:50]}...'. Offsets will be None.")
            elif not original_draft_text.strip():
                 logger.warning(f"Ch {chapter_number} consistency problem: Original draft text is empty. Cannot find offsets for quote: '{quote_text_from_llm[:50]}...'")
            else: # spaCy not loaded
                logger.info(f"Ch {chapter_number} consistency problem: spaCy not available, quote offsets not determined for: '{quote_text_from_llm[:50]}...'")

            final_problems.append(problem_meta)
        return final_problems

    async def check_consistency(
        self,
        novel_props: Dict[str, Any],
        draft_text: str,
        chapter_number: int,
        previous_chapters_context: str
    ) -> Tuple[List[ProblemDetail], Optional[Dict[str, int]]]:
        if not draft_text:
            logger.warning(f"WorldContinuityAgent: Consistency check skipped for Ch {chapter_number}: empty draft text.")
            return [], None

        logger.info(f"WorldContinuityAgent performing focused consistency check for Chapter {chapter_number}...")

        protagonist_name_str = novel_props.get('protagonist_name', 'The Protagonist')
        char_profiles_plain_text = await get_filtered_character_profiles_for_prompt_plain_text(novel_props, chapter_number - 1)
        world_building_plain_text = await get_filtered_world_data_for_prompt_plain_text(novel_props, chapter_number - 1)

        prompt = f"""/no_think
You are a World & Continuity Expert Editor for Chapter {chapter_number} of the novel "{novel_props.get('title', 'Untitled Novel')}" (Protagonist: {protagonist_name_str}).
Your SOLE TASK is to identify specific CONSISTENCY issues in the **Complete Chapter Text** below. Focus on:
- Contradictions with the Plot Outline summary.
- Contradictions with Character Profiles (descriptions, established traits, known status, relationships).
- Contradictions with World Building rules, descriptions, or established lore.
- Contradictions or inconsistencies with the Previous Chapters Context (which includes semantic flow and Key Reliable KG Facts).
- Internal inconsistencies of fact or established detail within THIS chapter's text.

**Reference Information for CONSISTENCY Check (Summary Format):**
  **Plot Outline Summary:**
  ```text
  Title: {novel_props.get('title', 'N/A')}
  Genre: {novel_props.get('genre', 'N/A')}
  Theme: {novel_props.get('theme', 'N/A')}
  Protagonist: {novel_props.get('protagonist_name', 'N/A')} ({novel_props.get('character_arc', 'N/A')})
  Logline: {novel_props.get('logline', 'N/A')}
  Key Plot Points (summary):
  {chr(10).join([f"- PP {i+1}: {pp[:100]}..." for i, pp in enumerate(novel_props.get('plot_points', []))]) if novel_props.get('plot_points') else "  - Not available"}
  ```
  **Character Profiles (Key Info - check 'prompt_notes' for provisional status):**
  ```text
  {char_profiles_plain_text}
  ```
  **World Building Notes (Key Info - check 'prompt_notes' for provisional status):**
  ```text
  {world_building_plain_text}
  ```
  **Previous Chapters Context (Includes Semantic Flow & Key Reliable KG Facts for Canon):**
  --- PREVIOUS CONTEXT ---
  {previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed)."}
  --- END PREVIOUS CONTEXT ---

**Complete Chapter {chapter_number} Text (to analyze for consistency):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL - PLAIN TEXT ONLY):**
If consistency problems are found, list each problem individually using the following format:

ISSUE CATEGORY: consistency
PROBLEM DESCRIPTION: [A concise description of the specific consistency issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If the issue is general and no single quote captures it, or if a quote is truly inapplicable, use "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote/issue should focus on to resolve the consistency (e.g., "Align character's action with profile trait X", "Correct factual detail Y based on world notes").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO consistency problems are found, output ONLY the phrase: "No significant consistency problems found."

Example if issues are found:
ISSUE CATEGORY: consistency
PROBLEM DESCRIPTION: Character X states they have never left their village, but their profile mentions they studied in the Capital.
QUOTE FROM ORIGINAL: "I've never seen anything beyond these village walls," X sighed.
SUGGESTED FIX FOCUS: Change X's dialogue to reflect their past experience in the Capital, or adjust profile if this is a new character development.
---
ISSUE CATEGORY: consistency
PROBLEM DESCRIPTION: The 'Sunstone' is described as blue, but world notes state it's always red.
QUOTE FROM ORIGINAL: She admired the brilliant blue glow of the Sunstone.
SUGGESTED FIX FOCUS: Change 'blue' to 'red' to match established world canon for the Sunstone.
"""
        logger.info(f"Calling LLM ({self.model_name}) for World/Continuity consistency check of chapter {chapter_number}...")
        raw_consistency_text, usage_data = await llm_interface.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.2,
            allow_fallback=True,
            stream_to_disk=False
        )
        cleaned_consistency_text = llm_interface.clean_model_response(raw_consistency_text)
        # Pass original_draft_text to the parser
        consistency_problems = await self._parse_llm_consistency_output(cleaned_consistency_text, chapter_number, draft_text)

        logger.info(f"World/Continuity consistency check for Ch {chapter_number} found {len(consistency_problems)} problems.")
        return consistency_problems, usage_data

    async def suggest_canon_corrections(self, problems: List[ProblemDetail]) -> List[str]:
        logger.warning("suggest_canon_corrections method is a placeholder and not fully implemented.")
        suggestions = []
        for problem in problems:
            if problem.get("issue_category") == "consistency":
                suggestions.append(f"For consistency problem '{problem.get('problem_description','N/A')}': Focus on '{problem.get('suggested_fix_focus','N/A')}' for canon alignment.")
        return suggestions

    async def query_kg_for_contradiction(self, entity1: str, entity2: Optional[str] = None, relation: Optional[str] = None, chapter_limit: Optional[int] = None) -> List[Dict[str, Any]]:
        logger.info(f"WorldContinuityAgent: Querying KG regarding potential contradiction around '{entity1}'...")
        facts = []
        facts.extend(await kg_queries.query_kg_from_db(subject=entity1, chapter_limit=chapter_limit, include_provisional=False))
        facts.extend(await kg_queries.query_kg_from_db(obj_val=entity1, chapter_limit=chapter_limit, include_provisional=False))

        if entity2 and relation:
            facts.extend(await kg_queries.query_kg_from_db(subject=entity1, predicate=relation, obj_val=entity2, chapter_limit=chapter_limit, include_provisional=False))

        unique_facts_strs = set()
        formatted_facts = []
        for fact_dict in facts:
            fact_str = f"{fact_dict.get('subject')} -{fact_dict.get('predicate')}-> {fact_dict.get('object')} (Ch: {fact_dict.get('chapter_added')}, Prov: {fact_dict.get('is_provisional')})"
            if fact_str not in unique_facts_strs:
                formatted_facts.append(fact_dict)
                unique_facts_strs.add(fact_str)
        logger.info(f"Found {len(formatted_facts)} non-provisional KG facts related to '{entity1}' (up to chapter {chapter_limit}).")
        return formatted_facts