import logging
from typing import Dict, List, Optional, Tuple

import config
from core.llm_interface import llm_service
from kg_maintainer.models import PatchInstruction, ProblemDetail
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


class PatchValidationAgent:
    """Validates patch instructions using an LLM."""

    def __init__(self, model_name: str = config.EVALUATION_MODEL) -> None:
        self.model_name = model_name
        logger.info("PatchValidationAgent initialized with model: %s", self.model_name)

    async def validate_patch(
        self,
        context_snippet: str,
        patch: PatchInstruction,
        problems: List[ProblemDetail],
    ) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Return True if patch addresses all problems according to the LLM."""

        issues_list = "\n".join(f"- {p['problem_description']}" for p in problems)
        prompt = render_prompt(
            "patch_validation_agent/validate_patch.j2",
            {
                "context_snippet": context_snippet,
                "patch_text": patch.get("replace_with", ""),
                "issues_list": issues_list,
            },
        )

        response_text, usage = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.Temperatures.EVALUATION,
            max_tokens=64,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_EVALUATION,
            presence_penalty=config.PRESENCE_PENALTY_EVALUATION,
            auto_clean_response=True,
        )

        verdict_line = response_text.splitlines()[0].strip().lower()
        is_pass = verdict_line.startswith("pass")
        if not is_pass:
            logger.info("Patch validation failed: %s", verdict_line)
        return is_pass, usage
