import re
from typing import TYPE_CHECKING

from config import settings

from .text_utils import _is_fill_in

__all__ = ["_is_fill_in", "settings"]

import spacy
import structlog
from rapidfuzz.fuzz import partial_ratio_alignment

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    pass


def _normalize_for_id(text: str) -> str:
    """Normalize a string for use in an ID."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    # Remove common leading articles to avoid ID duplicates
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"['\"()]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def normalize_trait_name(trait: str) -> str:
    """Return a canonical representation of a trait name."""
    if not isinstance(trait, str):
        trait = str(trait)
    cleaned = re.sub(r"[^a-z0-9 ]", "", trait.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


class SpaCyModelManager:
    """Lazily loads and stores the spaCy model used across the project."""

    def __init__(self) -> None:
        self._nlp: spacy.language.Language | None = None

    @property
    def nlp(self) -> spacy.language.Language | None:
        return self._nlp

    def load(self) -> None:
        """Load the spaCy model if it hasn't been loaded yet."""
        if self._nlp is not None:
            return
        try:
            self._nlp = spacy.load("en_core_web_lg")
            logger.info("spaCy model 'en_core_web_lg' loaded successfully.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_lg. "
                "spaCy dependent features will be disabled."
            )
            self._nlp = None
        except ImportError:
            logger.error(
                "spaCy library not installed. Please install it: pip install spacy. "
                "spaCy dependent features will be disabled."
            )
            self._nlp = None


spacy_manager = SpaCyModelManager()


def load_spacy_model_if_needed() -> None:
    """Load the spaCy model using the shared manager if needed."""
    spacy_manager.load()


def _normalize_text_for_matching(text: str) -> str:
    """Normalize text for more robust matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(
        r"^[ '\"\(]*(\.\.\.)?[ '\"\(]*|[ '\"\(]*(\.\.\.)?[ '\"\(]*$", "", text
    )
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_similarity(a: str, b: str) -> float:
    """Return Jaccard similarity between token sets of ``a`` and ``b``."""
    tokens_a = set(_normalize_text_for_matching(a).split())
    tokens_b = set(_normalize_text_for_matching(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _direct_substring_search(
    doc_text: str, cleaned_quote: str, spacy_doc: spacy.tokens.Doc
) -> tuple[int, int, int, int] | None:
    """Performs a direct substring search for the cleaned_quote within doc_text."""
    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(cleaned_quote.lower(), current_pos)
        if match_start == -1:
            return None # Not found in the rest of the document

        match_end = match_start + len(cleaned_quote)

        # Check if this match falls within a sentence
        for sent_span in spacy_doc.sents:
            if (
                sent_span.start_char <= match_start < sent_span.end_char and
                sent_span.start_char < match_end <= sent_span.end_char
                # Ensure the match_end is also within or at the end of the sentence.
                # Handles cases where quote might be at the very end of a sentence.
            ):
                logger.info(
                    "Direct Substring Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d",
                    cleaned_quote[:30], match_start, match_end,
                    sent_span.start_char, sent_span.end_char,
                )
                return match_start, match_end, sent_span.start_char, sent_span.end_char

        # If no sentence contains this specific match, try finding the quote again
        # starting after the current match_end to avoid re-finding the same invalid match.
        # However, a simpler approach is to advance beyond the start of the current match.
        current_pos = match_start + 1 # Advance search position
    return None


def _token_similarity_search(
    cleaned_quote: str, spacy_doc: spacy.tokens.Doc
) -> tuple[int, int, int, int] | None:
    """Performs a token-based similarity search against sentences."""
    best_sent_span = None
    best_sim_score = 0.0
    for sent_span in spacy_doc.sents:
        sim = _token_similarity(cleaned_quote, sent_span.text)
        if sim > best_sim_score:
            best_sim_score = sim
            best_sent_span = sent_span

    if best_sent_span and best_sim_score >= 0.45: # Threshold from original code
        logger.info(
            "Token Similarity Match: '%s...' most similar to sentence %d-%d (%.2f)",
            cleaned_quote[:30],
            best_sent_span.start_char, best_sent_span.end_char, best_sim_score,
        )
        # For token similarity, the quote is considered the whole sentence
        return (
            best_sent_span.start_char, best_sent_span.end_char,
            best_sent_span.start_char, best_sent_span.end_char,
        )
    return None


def _fuzzy_search(
    doc_text: str, cleaned_quote: str, spacy_doc: spacy.tokens.Doc
) -> tuple[int, int, int, int] | None:
    """Performs a fuzzy search using partial_ratio_alignment."""
    alignment = partial_ratio_alignment(cleaned_quote, doc_text)
    if alignment.score < 85.0: # Threshold from original code
        return None

    match_start = alignment.dest_start
    match_end = alignment.dest_end

    for sent_span in spacy_doc.sents:
        if (
            sent_span.start_char <= match_start < sent_span.end_char and
            sent_span.start_char < match_end <= sent_span.end_char
        ):
            logger.info(
                "Fuzzy Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d (Score: %.2f)",
                cleaned_quote[:30], match_start, match_end,
                sent_span.start_char, sent_span.end_char, alignment.score,
            )
            return match_start, match_end, sent_span.start_char, sent_span.end_char

    logger.debug("Fuzzy match found but not contained within a single sentence. Quote: '%s', Match: %d-%d", cleaned_quote[:30], match_start, match_end)
    return None


async def _semantic_search(
    doc_text: str, original_llm_quote: str
) -> tuple[int, int, int, int] | None:
    """Performs a semantic search for the quote within the document."""
    from .similarity import find_semantically_closest_segment # Local import

    semantic_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=original_llm_quote, # Use original quote for semantic search
        segment_type="sentence",
        min_similarity_threshold=0.75, # Threshold from original code
    )

    if semantic_match:
        s_start, s_end, similarity = semantic_match
        logger.info(
            "Semantic Match: Found sentence for LLM quote '%s...' from %d-%d (Similarity: %.2f). Using whole sentence as target.",
            original_llm_quote[:30], s_start, s_end, similarity,
        )
        # For semantic search, the quote is considered the whole sentence
        return s_start, s_end, s_start, s_end
    return None


async def find_quote_and_sentence_offsets_with_spacy(
    doc_text: str, quote_text_from_llm: str
) -> tuple[int, int, int, int] | None:
    """Locate quote and sentence offsets within ``doc_text``."""
    load_spacy_model_if_needed()
    if (
        spacy_manager.nlp is None
        or not quote_text_from_llm.strip()
        or not doc_text.strip()
    ):
        if spacy_manager.nlp is None:
            logger.debug("find_quote_offsets: spaCy model not loaded.")
        else:
            logger.debug("find_quote_offsets: Empty quote_text or doc_text.")
        return None

    if "N/A - General Issue" in quote_text_from_llm:
        logger.debug(
            "Quote is '%s', treating as general issue. No offsets.", quote_text_from_llm
        )
        return None

    cleaned_llm_quote_for_direct_search = quote_text_from_llm.strip(" \"'.")
    if not cleaned_llm_quote_for_direct_search:
        logger.debug(
            "LLM quote became empty after basic stripping for direct search, cannot match."
        )
        return None

    spacy_doc = spacy_manager.nlp(doc_text) if spacy_manager.nlp else None
    if spacy_doc is None:
        return None

    # Strategy 1: Direct Substring Search
    direct_match_result = _direct_substring_search(doc_text, cleaned_llm_quote_for_direct_search, spacy_doc)
    if direct_match_result:
        return direct_match_result

    # Strategy 2: Fuzzy Match
    fuzzy_match_result = _fuzzy_search(doc_text, cleaned_llm_quote_for_direct_search, spacy_doc)
    if fuzzy_match_result:
        return fuzzy_match_result

    # Strategy 3: Token Similarity Search
    token_match_result = _token_similarity_search(cleaned_llm_quote_for_direct_search, spacy_doc)
    if token_match_result:
        return token_match_result

    # Strategy 4: Semantic Search
    logger.warning(
        "Direct, fuzzy, and token similarity searches failed for LLM quote '%s...'. Falling back to semantic sentence search.",
        quote_text_from_llm[:50], # Log original quote for better debugging
    )
    semantic_match_result = await _semantic_search(doc_text, quote_text_from_llm)
    if semantic_match_result:
        return semantic_match_result

    logger.warning(
        "All search strategies failed to locate quote from LLM: '%s...' in document.",
        quote_text_from_llm[:50],
    )
    return None


def get_text_segments(
    text: str, segment_level: str = "paragraph"
) -> list[tuple[str, int, int]]:
    """Segment text into paragraphs or sentences with offsets."""
    load_spacy_model_if_needed()
    segments: list[tuple[str, int, int]] = []

    if not text.strip():
        return segments

    if segment_level == "paragraph":
        current_paragraph_lines: list[str] = []
        current_paragraph_start_char = -1

        for line_match in re.finditer(r"([^\r\n]*(?:\r\n|\r|\n)?)", text):
            line_text = line_match.group(0)
            line_text_stripped = line_text.strip()

            if line_text_stripped:
                if not current_paragraph_lines:
                    current_paragraph_start_char = line_match.start()
                current_paragraph_lines.append(line_text)
            else:
                if current_paragraph_lines:
                    full_para_text = "".join(current_paragraph_lines)
                    segments.append(
                        (
                            full_para_text.strip(),
                            current_paragraph_start_char,
                            current_paragraph_start_char + len(full_para_text),
                        )
                    )
                    current_paragraph_lines = []
                    current_paragraph_start_char = -1

        if current_paragraph_lines:
            full_para_text = "".join(current_paragraph_lines)
            segments.append(
                (
                    full_para_text.strip(),
                    current_paragraph_start_char,
                    current_paragraph_start_char + len(full_para_text),
                )
            )

        if not segments and text.strip():
            segments.append((text.strip(), 0, len(text)))

    elif segment_level == "sentence":
        if spacy_manager.nlp:
            doc = spacy_manager.nlp(text)
            for sent in doc.sents:
                sent_text_stripped = sent.text.strip()
                if sent_text_stripped:
                    segments.append(
                        (sent_text_stripped, sent.start_char, sent.end_char)
                    )
        else:
            logger.warning(
                "get_text_segments: spaCy model not loaded. Falling back to basic sentence segmentation (less accurate)."
            )
            for match in re.finditer(r"([^\.!?]+(?:[\.!?]|$))", text):
                sent_text_stripped = match.group(1).strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, match.start(), match.end()))
            if not segments and text.strip():
                segments.append((text.strip(), 0, len(text)))
    else:
        raise ValueError(
            f"Unsupported segment_level for get_text_segments: {segment_level}"
        )

    return segments
