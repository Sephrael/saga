# utils/text_processing.py
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


def _direct_quote_match(
    doc_text: str, quote: str, spacy_doc: spacy.language.Doc
) -> tuple[int, int, int, int] | None:
    """Return offsets for a direct substring match within a sentence."""
    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(quote.lower(), current_pos)
        if match_start == -1:
            break

        match_end = match_start + len(quote)
        for sent in spacy_doc.sents:
            if (
                sent.start_char <= match_start < sent.end_char
                and sent.start_char < match_end <= sent.end_char
            ):
                logger.info(
                    "Direct Substring Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d",
                    quote[:30],
                    match_start,
                    match_end,
                    sent.start_char,
                    sent.end_char,
                )
                return (
                    match_start,
                    match_end,
                    sent.start_char,
                    sent.end_char,
                )

        current_pos = match_end
    return None


def _fuzzy_quote_match(
    doc_text: str, quote: str, spacy_doc: spacy.language.Doc
) -> tuple[int, int, int, int] | None:
    """Return offsets for a fuzzy substring match using rapidfuzz."""
    alignment = partial_ratio_alignment(quote, doc_text)
    if alignment.score >= 85.0:
        match_start = alignment.dest_start
        match_end = alignment.dest_end
        for sent in spacy_doc.sents:
            if (
                sent.start_char <= match_start < sent.end_char
                and sent.start_char < match_end <= sent.end_char
            ):
                logger.info(
                    "Fuzzy Match: Found LLM quote (approx) '%s...' at %d-%d in sentence %d-%d (Score: %.2f)",
                    quote[:30],
                    match_start,
                    match_end,
                    sent.start_char,
                    sent.end_char,
                    alignment.score,
                )
                return (
                    match_start,
                    match_end,
                    sent.start_char,
                    sent.end_char,
                )
    return None


def _token_similarity_sentence_match(
    quote: str, spacy_doc: spacy.language.Doc
) -> tuple[int, int, int, int] | None:
    """Return offsets using simple token overlap similarity."""
    best_sent = None
    best_sim = 0.0
    for sent in spacy_doc.sents:
        sim = _token_similarity(quote, sent.text)
        if sim > best_sim:
            best_sim = sim
            best_sent = sent
    if best_sent and best_sim >= 0.45:
        logger.info(
            "Token Similarity Match: '%s...' most similar to sentence %d-%d (%.2f)",
            quote[:30],
            best_sent.start_char,
            best_sent.end_char,
            best_sim,
        )
        return (
            best_sent.start_char,
            best_sent.end_char,
            best_sent.start_char,
            best_sent.end_char,
        )
    return None


async def _semantic_sentence_search(
    doc_text: str, quote: str
) -> tuple[int, int, int, int] | None:
    """Return offsets using semantic similarity search."""
    from .similarity import find_semantically_closest_segment

    semantic_sentence_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=quote,
        segment_type="sentence",
        min_similarity_threshold=0.75,
    )

    if semantic_sentence_match:
        s_start, s_end, similarity = semantic_sentence_match
        logger.info(
            "Semantic Match: Found sentence for LLM quote '%s...' from %d-%d (Similarity: %.2f). Using whole sentence as target.",
            quote[:30],
            s_start,
            s_end,
            similarity,
        )
        return s_start, s_end, s_start, s_end
    logger.warning(
        "Direct substring match failed for LLM quote '%s...'. Falling back to semantic sentence search.",
        quote[:50],
    )
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

    offsets = _direct_quote_match(
        doc_text, cleaned_llm_quote_for_direct_search, spacy_doc
    )
    if offsets:
        return offsets

    offsets = _fuzzy_quote_match(
        doc_text, cleaned_llm_quote_for_direct_search, spacy_doc
    )
    if offsets:
        return offsets

    offsets = _token_similarity_sentence_match(
        cleaned_llm_quote_for_direct_search, spacy_doc
    )
    if offsets:
        return offsets

    offsets = await _semantic_sentence_search(doc_text, quote_text_from_llm)
    if offsets:
        return offsets

    logger.warning(
        "All search strategies failed to locate quote from LLM: '%s...' in document.",
        quote_text_from_llm[:50],
    )
    return None


def _split_into_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Return paragraph segments with offsets."""
    segments: list[tuple[str, int, int]] = []
    current_lines: list[str] = []
    start = -1
    for match in re.finditer(r"([^\r\n]*(?:\r\n|\r|\n)?)", text):
        line = match.group(0)
        if line.strip():
            if not current_lines:
                start = match.start()
            current_lines.append(line)
            continue
        if current_lines:
            full = "".join(current_lines)
            segments.append((full.strip(), start, start + len(full)))
            current_lines = []
            start = -1
    if current_lines:
        full = "".join(current_lines)
        segments.append((full.strip(), start, start + len(full)))
    if not segments and text.strip():
        segments.append((text.strip(), 0, len(text)))
    return segments


def _split_into_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return sentence segments with offsets."""
    segments: list[tuple[str, int, int]] = []
    if spacy_manager.nlp:
        doc = spacy_manager.nlp(text)
        for sent in doc.sents:
            stripped = sent.text.strip()
            if stripped:
                segments.append((stripped, sent.start_char, sent.end_char))
    else:
        logger.warning(
            "get_text_segments: spaCy model not loaded. Falling back to basic sentence segmentation (less accurate)."
        )
        for match in re.finditer(r"([^\.!?]+(?:[\.!?]|$))", text):
            stripped = match.group(1).strip()
            if stripped:
                segments.append((stripped, match.start(), match.end()))
        if not segments and text.strip():
            segments.append((text.strip(), 0, len(text)))
    return segments


def get_text_segments(
    text: str, segment_level: str = "paragraph"
) -> list[tuple[str, int, int]]:
    """Segment text into paragraphs or sentences with offsets."""
    load_spacy_model_if_needed()
    if not text.strip():
        return []

    if segment_level == "paragraph":
        return _split_into_paragraphs(text)
    if segment_level == "sentence":
        return _split_into_sentences(text)

    raise ValueError(
        f"Unsupported segment_level for get_text_segments: {segment_level}"
    )
