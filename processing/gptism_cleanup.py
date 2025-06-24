from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

from rapidfuzz import fuzz

import utils

logger = logging.getLogger(__name__)

# Common phrases often produced by language models and replacement options
GPT_ISM_PATTERNS: Dict[str, List[str]] = {
    "as an ai language model": [
        "from a broader perspective",
        "considering the context",
    ],
    "i'm sorry": [
        "unfortunately",
        "regrettably",
    ],
    "i do not have personal opinions": [
        "sources differ on this",
        "there is no clear consensus",
    ],
}

DEFAULT_SIMILARITY_THRESHOLD = 80.0


def replace_gptisms(
    text: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> Tuple[str, int]:
    """Replace common GPT-isms in ``text`` with alternative phrasings."""
    utils.load_spacy_model_if_needed()
    nlp = utils.spacy_manager.nlp
    if nlp is None or not text.strip():
        return text, 0

    replacements = 0
    new_sentences: List[str] = []
    for sent in nlp(text).sents:
        sent_text = sent.text
        best_phrase = None
        best_score = 0
        for phrase in GPT_ISM_PATTERNS:
            score = fuzz.partial_ratio(sent_text.lower(), phrase)
            if score > best_score:
                best_score = score
                best_phrase = phrase
        if best_phrase and best_score >= threshold:
            options = GPT_ISM_PATTERNS[best_phrase]
            replacement = random.choice(options)
            logger.info(
                "Replacing GPT-ism '%s' (score %.1f) with '%s'",
                best_phrase,
                best_score,
                replacement,
            )
            sent_text = replacement
            replacements += 1
        new_sentences.append(sent_text.strip())
    cleaned_text = " ".join(new_sentences)
    return cleaned_text, replacements
