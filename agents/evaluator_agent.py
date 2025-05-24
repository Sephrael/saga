from typing import Any, Dict, List, Optional
import logging
import asyncio
from llm_interface import LLMInterface
from type import EvaluationResult, ProblemDetail
from state_manager import StateManager

logger = logging.getLogger(__name__)

class EvaluatorAgent:
    def __init__(self, agent_name: str, focus_area: str, model_name: str):
        self.agent_name = agent_name
        self.focus_area = focus_area
        self.model_name = model_name
        self.llm_interface = LLMInterface(model_name=model_name)
        self.state_manager = StateManager()
        
    async def evaluate_chapter(self, draft_text: str, chapter_number: int, previous_chapters_context: str) -> EvaluationResult:
        """
        Evaluates the given chapter draft based on the agent's focus area.
        Returns an EvaluationResult dictionary with details about the evaluation.
        """
        logger.info(f"{self.agent_name} evaluating Chapter {chapter_number} for focus area: {self.focus_area}")
        
        # Define prompt based on agent's focus area
        if self.focus_area == "consistency":
            prompt = f"""/no_think
You are a Consistency Evaluator for Chapter {chapter_number} of the novel "{self.state_manager.get_current_novel_title()}".
Your task is to identify specific issues related to:
1.  **CONSISTENCY**: Contradictions with Plot Outline, Character Profiles, World Building, Key Reliable KG Facts, Previous Context, or internal inconsistencies within THIS chapter.

**Reference Information for CONSISTENCY Check (Summary Format):**
  **Plot Outline Summary:**
  ```text
  Title: {self.state_manager.get_current_novel_title()}
  Genre: {self.state_manager.get_current_novel_genre()}
  Theme: {self.state_manager.get_current_novel_theme()}
  Protagonist: {self.state_manager.get_current_novel_protagonist_name()} ({self.state_manager.get_current_novel_character_arc()})
  Logline: {self.state_manager.get_current_novel_logline()}
  Key Plot Points (summary):
  {chr(10).join([f"- PP {i+1}: {pp[:100]}..." for i, pp in enumerate(self.state_manager.get_current_novel_plot_points())]) if self.state_manager.get_current_novel_plot_points() else "  - Not available"}
  ```
  **Character Profiles (Key Info - check 'prompt_notes' for provisional status):**
  ```text
  {await self.state_manager.get_filtered_character_profiles_for_prompt_plain_text(chapter_number - 1)}
  ```
  **World Building Notes (Key Info - check 'prompt_notes' for provisional status):**
  ```text
  {await self.state_manager.get_filtered_world_data_for_prompt_plain_text(chapter_number - 1)}
  ```
  {await self.state_manager.get_reliable_kg_facts_for_drafting_prompt(chapter_number, None)}
  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**
  --- PREVIOUS CONTEXT ---
  {previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed)."}
  --- END PREVIOUS CONTEXT ---

**Complete Chapter {chapter_number} Text (to analyze):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format (ensure keys like "ISSUE CATEGORY" are used, case can vary):

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth | meta]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If general (e.g., overall length), provide a representative short quote or "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."

Example if issues are found:
ISSUE CATEGORY: consistency
PROBLEM DESCRIPTION: Character X acts out of character. Their profile says brave, but here they are cowardly.
QUOTE FROM ORIGINAL: X trembled and hid behind the rock, refusing to move.
SUGGESTED FIX FOCUS: Rewrite X's action to show bravery or internal conflict leading to hesitation, aligning with their profile.
---
ISSUE CATEGORY: narrative_depth
PROBLEM DESCRIPTION: The chapter is too short overall for the events covered and the target length.
QUOTE FROM ORIGINAL: N/A - General Issue
SUGGESTED FIX FOCUS: Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue.

Example if no issues are found:
No significant problems found.
"""
        elif self.focus_area == "plot_arc":
            prompt = f"""/no_think
You are a Plot Arc Evaluator for Chapter {chapter_number} of the novel "{self.state_manager.get_current_novel_title()}".
Your task is to identify specific issues related to:
1.  **PLOT_ARC**: How well this chapter addresses or advances its Intended Plot Point: "{self.state_manager.get_intended_plot_point(chapter_number)}" (Plot Point #{self.state_manager.get_intended_plot_point_index(chapter_number) + 1}).

**Reference Information for PLOT ARC Check (Summary Format):**
  **Plot Point Focus:**
  ```text
  Intended Plot Point: "{self.state_manager.get_intended_plot_point(chapter_number)}" (Plot Point #{self.state_manager.get_intended_plot_point_index(chapter_number) + 1})
  ```
  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**
  --- PREVIOUS CONTEXT ---
  {previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed)."}
  --- END PREVIOUS CONTEXT ---

**Complete Chapter {chapter_number} Text (to analyze):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format (ensure keys like "ISSUE CATEGORY" are used, case can vary):

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth | meta]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If general (e.g., overall length), provide a representative short quote or "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."

Example if issues are found:
ISSUE CATEGORY: plot_arc
PROBLEM DESCRIPTION: The chapter does not advance the intended plot point sufficiently. It lacks key developments related to "{self.state_manager.get_intended_plot_point(chapter_number)}".
QUOTE FROM ORIGINAL: N/A - General Issue
SUGGESTED FIX FOCUS: Identify and expand upon key plot developments related to the intended plot point. Ensure the chapter meaningfully advances the narrative arc.

Example if no issues are found:
No significant problems found.
"""
        elif self.focus_area == "thematic":
            prompt = f"""/no_think
You are a Thematic Alignment Evaluator for Chapter {chapter_number} of the novel "{self.state_manager.get_current_novel_title()}".
Your task is to identify specific issues related to:
1.  **THEMATIC_ALIGNMENT**: Alignment with the novel's core elements (Genre: {self.state_manager.get_current_novel_genre()}, Theme: {self.state_manager.get_current_novel_theme()}, Protagonist's Arc: {self.state_manager.get_current_novel_character_arc()}) and consistency of theme throughout the chapter.

**Reference Information for THEMATIC ALIGNMENT Check (Summary Format):**
  **Novel Core Elements:**
  ```text
  Genre: {self.state_manager.get_current_novel_genre()}
  Theme: {self.state_manager.get_current_novel_theme()}
  Protagonist's Arc: {self.state_manager.get_current_novel_character_arc()}
  ```
  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**
  --- PREVIOUS CONTEXT ---
  {previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed)."}
  --- END PREVIOUS CONTEXT ---

**Complete Chapter {chapter_number} Text (to analyze):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format (ensure keys like "ISSUE CATEGORY" are used, case can vary):

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth | meta]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If general (e.g., overall length), provide a representative short quote or "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."

Example if issues are found:
ISSUE CATEGORY: thematic
PROBLEM DESCRIPTION: The chapter lacks alignment with the novel's theme of "{self.state_manager.get_current_novel_theme()}".
QUOTE FROM ORIGINAL: N/A - General Issue
SUGGESTED FIX FOCUS: Identify sections that need more thematic resonance and revise them to better reflect the novel's core theme.

Example if no issues are found:
No significant problems found.
"""
        elif self.focus_area == "narrative_depth":
            prompt = f"""/no_think
You are a Narrative Depth & Length Evaluator for Chapter {chapter_number} of the novel "{self.state_manager.get_current_novel_title()}".
Your task is to identify specific issues related to:
1.  **NARRATIVE_DEPTH_AND_LENGTH**: Sufficiency of descriptive detail, character introspection, dialogue development, pacing, and overall length (target: at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters).

**Reference Information for NARRATIVE DEPTH & LENGTH Check (Summary Format):**
  **Novel Core Elements:**
  ```text
  Genre: {self.state_manager.get_current_novel_genre()}
  Theme: {self.state_manager.get_current_novel_theme()}
  Protagonist's Arc: {self.state_manager.get_current_novel_character_arc()}
  ```
  **Previous Chapters Context (Semantic Flow & KG Facts for Canon):**
  --- PREVIOUS CONTEXT ---
  {previous_chapters_context if previous_chapters_context.strip() else "N/A (e.g., Chapter 1 or context retrieval failed)."}
  --- END PREVIOUS CONTEXT ---

**Complete Chapter {chapter_number} Text (to analyze):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format (ensure keys like "ISSUE CATEGORY" are used, case can vary):

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth | meta]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
QUOTE FROM ORIGINAL: [**A VERBATIM quote (10-50 words) from the "Complete Chapter Text" that clearly illustrates this specific problem.** If general (e.g., overall length), provide a representative short quote or "N/A - General Issue".]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting").]
---

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."

Example if issues are found:
ISSUE CATEGORY: narrative_depth
PROBLEM DESCRIPTION: The chapter is too short overall for the events covered and the target length.
QUOTE FROM ORIGINAL: N/A - General Issue
SUGGESTED FIX FOCUS: Expand content significantly across multiple scenes/sections to meet the {config.MIN_ACCEPTABLE_DRAFT_LENGTH} character target. Focus on adding descriptive detail, character introspection, and dialogue.

Example if no issues are found:
No significant problems found.
"""
        else:
            logger.warning(f"Unknown focus area '{self.focus_area}' for agent '{self.agent_name}'. Defaulting to meta evaluation.")
            prompt = f"""/no_think
You are a Meta Evaluator for Chapter {chapter_number} of the novel "{self.state_manager.get_current_novel_title()}".
Your task is to identify general issues that don't fall under specific categories.

**Complete Chapter {chapter_number} Text (to analyze):**
--- BEGIN COMPLETE CHAPTER TEXT ---
{draft_text}
--- END COMPLETE CHAPTER TEXT ---

**Output Format (CRITICAL):**
Provide your evaluation as plain text. If problems are found, list each problem individually using the following format (ensure keys like "ISSUE CATEGORY" are used, case can vary):

ISSUE CATEGORY: [consistency | plot_arc | thematic | narrative_depth | meta]
PROBLEM DESCRIPTION: [A concise description of the specific issue.]
SUGGESTED FIX FOCUS: [Brief guidance on what the revision for this specific quote should focus on (e.g., "Clarify character's motivation", "Expand description of setting").]

If multiple problems are found, separate each problem block with a line containing only "---".
If NO problems are found for a category or overall, output ONLY the phrase: "No significant problems found."

Example if issues are found:
ISSUE CATEGORY: meta
PROBLEM DESCRIPTION: The chapter lacks a clear narrative direction or has other unspecified issues.
QUOTE FROM ORIGINAL: N/A - General Issue
SUGGESTED FIX FOCUS: Review the chapter for general issues and revise accordingly.

Example if no issues are found:
No significant problems found.
"""
        
        # Call LLM for evaluation
        logger.info(f"Calling LLM ({self.model_name}) for {self.focus_area} evaluation of chapter {chapter_number}...")
        raw_evaluation_text = await self.llm_interface.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.3, 
            allow_fallback=True, 
            stream_to_disk=False 
        )
        
        # Process the evaluation results
        cleaned_evaluation_text = self.llm_interface.clean_model_response(raw_evaluation_text)
        return await self._process_evaluation_results(cleaned_evaluation_text, chapter_number)

    async def _process_evaluation_results(self, text: str, chapter_number: int) -> Dict[str, Any]:
        """
        Processes the LLM evaluation results into a structured EvaluationResult dictionary.
        """
        problems_data: List[Dict[str, Any]] = []
        if not text or not text.strip() or "no significant problems found" in text.lower():
            logger.info(f"Evaluation for Ch {chapter_number} is empty or indicates no problems.")
            return {
                "needs_revision": False,
                "reasons": ["No significant problems found."],
                "problems_found": []
            }

        problem_blocks_text = self._split_text_into_blocks(text, separator_regex_str=r'\n\s*---\s*\n')

        for block_num, block_content in enumerate(problem_blocks_text):
            if not block_content.strip():
                continue
            
            logger.debug(f"Parsing problem block {block_num+1} for Ch {chapter_number}:\n{block_content[:150]}...")
            
            parsed_problem_dict = self._parse_key_value_block(
                block_text_or_lines=block_content,
                key_map=PROBLEM_DETAIL_KEY_MAP,
                list_internal_keys=[] # No list keys in ProblemDetail
            )

            # Validate required keys and category
            required_keys_internal = set(PROBLEM_DETAIL_KEY_MAP.values())
            missing_keys = required_keys_internal - set(parsed_problem_dict.keys())

            if missing_keys:
                logger.warning(f"Could not parse all required fields from problem block {block_num+1} for Ch {chapter_number}. Missing: {missing_keys}. Block: '{block_content[:150]}...'")
                # Fallback: create a meta-problem
                problems_data.append({
                    "issue_category": "meta",
                    "problem_description": f"Malformed problem block from LLM: {block_content}",
                    "quote_from_original": "N/A - Malformed LLM Output",
                    "suggested_fix_focus": "Review LLM output and prompt for evaluation."
                })
                continue

            category = str(parsed_problem_dict.get("issue_category", "meta")).strip().lower()
            valid_categories = ["consistency", "plot_arc", "thematic", "narrative_depth", "meta"]
            if category not in valid_categories:
                logger.warning(f"Parsed unknown issue category '{category}' in block {block_num+1} for Ch {chapter_number}. Defaulting to 'meta'.")
                parsed_problem_dict["issue_category"] = "meta"
            else:
                parsed_problem_dict["issue_category"] = category # Ensure it's the normalized version
                
            problems_data.append(parsed_problem_dict)
                
        return {
            "needs_revision": len(problems_data) > 0,
            "reasons": ["Issues identified by evaluator agent."],
            "problems_found": problems_data
        }

    def _split_text_into_blocks(self, text: str, separator_regex_str: str) -> List[str]:
        """
        Splits the given text into blocks using the provided regex separator.
        """
        return re.split(separator_regex_str, text)

    def _parse_key_value_block(self, block_text_or_lines: Any, key_map: Dict[str, str], list_internal_keys: List[str]) -> Dict[str, Any]:
        """
        Parses a block of text into a dictionary based on the provided key map.
        """
        # Implementation would go here
        return {}
