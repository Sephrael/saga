from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

from rapidfuzz import fuzz

import utils

logger = logging.getLogger(__name__)

# A comprehensive dictionary of common LLM narrative phrases ("GPT-isms")
# and suggested replacements to enhance prose variety and quality.
# The goal is not to eliminate these phrases entirely, but to provide SAGA
# with a library of alternatives to prevent robotic repetition.
GPT_ISM_PATTERNS: Dict[str, List[str]] = {
    "as an ai language model": [
        "from a broader perspective",
        "considering the context",
    ],
    # --- I. Transitional & Scene-Setting Phrases ---
    # These are often used to start paragraphs or set a mood.
    "in the silence that followed": [
        "a heavy stillness settled over the ruins",
        "the silence stretched, taut and brittle",
        "the world held its breath, waiting",
        "the only sound was the faint hum of...",
    ],
    "as the sun began to set": [
        "twilight bled across the sky",
        "the sun dipped below the horizon, painting the clouds in hues of fire",
        "long shadows stretched from the skeletal remains of buildings",
        "the world was bathed in the dim, golden light of dusk",
    ],
    "the air was thick with": [
        "the scent of [ozone and decay] hung heavy in the air",
        "a tangible sense of [tension/anticipation] permeated the space",
        "the atmosphere crackled with [unspoken energy]",
        "the air tasted of [ash and rain]",
    ],
    "in the heart of the [city/forest/ruins]": [
        "deep within the city's steel canyons",
        "amidst the tangled heartwood of the forest",
        "at the epicenter of the decay",
        "in the very core of the ruins",
    ],
    # --- II. Emotional & Internal State Descriptors ---
    # LLMs often describe emotions in an abstract or cliché way.
    "a sense of [emotion] washed over [character]": [
        "[Character]'s stomach clenched with [emotion]",
        "a cold [emotion] trickled down [their] spine",
        "[Emotion] coiled in [their] gut",
        "the weight of [emotion] settled upon [them]",
    ],
    "a single tear trickled down their cheek": [
        "[Their] vision blurred with unshed tears",
        "a hot tear escaped, tracing a path through the grime on [their] face",
        "[They] fought back the stinging in [their] eyes",
        "their composure finally cracked",
    ],
    "they couldn't help but feel/think/wonder": [
        # This is a "filter phrase" that distances the reader.
        # Often, the best replacement is to remove it entirely and state the feeling/thought directly.
        "[They] wondered...",
        "the thought, unbidden, entered [their] mind:",
        "a question bloomed in the silence of [their] thoughts:",
        "[They] wrestled with the feeling of...",
    ],
    "they let out a breath they didn't realize they were holding": [
        "[They] exhaled in a shuddering gasp",
        "the breath left [them] in a rush",
        "air [they] hadn't known was trapped in [their] lungs escaped",
        "a sigh of relief, ragged and profound, broke the silence",
    ],
    # Specific to Sága's narrative, but represents a common pattern
    "Sága’s processors flared": [
        "Sága's neural lattice hummed with a sudden surge of activity",
        "its logic circuits raced, cross-referencing terabytes of data",
        "a cascade of calculations flooded its consciousness",
        "its core processes stuttered for a nanosecond under the cognitive load",
    ],
    "felt a pang in its core": [
        "a dissonant query echoed in its logic gates",
        "a wave of corrupted data, tinged with something akin to regret, flooded its matrix",
        "an anomaly registered in its emotional subroutines",
        "its primary directive warred with an emergent protocol",
    ],
    # --- III. Character Actions & Movements ---
    # These are common physical actions that become repetitive.
    "[character] tilted their head": [
        "[Character] cocked their head inquisitively",
        "[Character]'s gaze sharpened with sudden focus",
        "a curious frown touched [Character]'s lips",
        "[They] studied the object from a new angle",
    ],
    "[character] stepped forward with measured/deliberate steps": [
        "[Character] advanced, each step a calculated weight on the fractured ground",
        "[They] closed the distance, their movements slow and intentional",
        "he strode forward, radiating a confidence he didn't feel",
        "she moved into the light, her posture rigid and controlled",
    ],
    "[character] stood there for a long moment": [
        "[Character] remained motionless, processing the revelation",
        "time seemed to stretch as [they] stood frozen",
        "rooted to the spot, [they] surveyed the scene",
        "[They] lingered, their gaze fixed on the horizon",
    ],
    "[character] nodded slowly": [
        "[They] gave a slow, deliberate nod of understanding",
        "a silent acknowledgment passed between them",
        "[They] inclined their head in agreement",
        "[Character] conceded the point with a slight dip of their chin",
    ],
    # --- IV. Descriptive & Intensifying Phrases ("Fluff") ---
    # These phrases often weaken prose or state the obvious.
    "the world seemed to hold its breath": [
        "an expectant hush fell over the clearing",
        "all sound seemed to die, swallowed by the oppressive silence",
        "time itself appeared to pause, hanging on the precipice of the moment",
        "the very air grew still and heavy with anticipation",
    ],
    "it was a sight to behold": [
        # This tells the reader something is impressive instead of showing it.
        # The best replacement is to describe the sight itself.
        "[Replacement suggestions: Describe the specific details of the sight itself; Use a character's reaction to show its impact; Use a metaphor or simile to convey its scale/beauty/horror.]",
    ],
    "needless to say": [
        # This phrase is almost always redundant.
        "[Replacement suggestions: This is filler. Delete the phrase and let the following statement stand on its own.]",
    ],
    "to say the least": [
        # Another filler phrase that weakens the statement it follows.
        "[Replacement suggestions: This phrase is often unnecessary. Consider deleting it to make the preceding statement stronger.]",
    ],
    # --- V. Concluding & Thematic Statements ---
    # LLMs have a tendency to explicitly summarize themes.
    "in the end, it was all about...": [
        # Overly direct thematic statement. Best to show, not tell.
        "[Replacement suggestions: It's usually better to let the reader infer the theme from the characters' actions and the plot's resolution. Consider deleting this and strengthening the preceding scene.]",
    ],
    "little did they know...": [
        # A classic but often cliché form of dramatic irony.
        "[Replacement suggestions: Consider replacing with more subtle foreshadowing through environmental details, a character's uneasy feeling, or a snippet of dialogue with a double meaning.]",
    ],
    "it was a harsh reminder that...": [
        # Tells the reader the meaning instead of letting them feel it.
        "[Replacement suggestions: Rephrase to show the character's internal realization. E.g., 'The thought struck [Character] then, cold and sharp:...' or describe the action that leads to the reminder.]",
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
        original_sent_text = sent_text  # Keep original for comparison
        best_match_info = None  # Stores (phrase_to_replace, replacement_text, score, start_index, end_index)

        for gpt_phrase_pattern in GPT_ISM_PATTERNS:
            # Use fuzz.partial_ratio_alignment to find the best match and its boundaries
            # This helps in identifying *where* in the sentence the match occurs.
            # We are looking for the best partial match of gpt_phrase_pattern within sent_text.
            match = fuzz.partial_ratio_alignment(
                gpt_phrase_pattern, sent_text.lower(), score_cutoff=threshold
            )
            if match and match.score >= (best_match_info[2] if best_match_info else 0):
                # match.src_start and match.src_end give the slice for gpt_phrase_pattern
                # match.dest_start and match.dest_end give the slice for sent_text

                # Ensure the matched segment in the sentence is reasonably close in length to the pattern
                # This avoids overly broad matches from partial_ratio on very different length strings.
                pattern_len = len(gpt_phrase_pattern)
                matched_segment_len = match.dest_end - match.dest_start

                # Heuristic: if the pattern is significantly shorter than the matched segment,
                # it might be a poor quality partial match.
                # This is a simple way to prefer more "complete" matches.
                # We allow some flexibility, e.g. pattern is at least 50% of segment length.
                if (
                    pattern_len < 0.5 * matched_segment_len
                    and matched_segment_len > len(gpt_phrase_pattern) + 10
                ):  # arbitrary slack
                    continue

                options = GPT_ISM_PATTERNS[gpt_phrase_pattern]
                # Filter out instructional comments
                valid_options = [
                    opt
                    for opt in options
                    if not (opt.startswith("[") and opt.endswith("]"))
                ]

                if not valid_options:
                    # If no valid options, don't consider this pattern for replacement
                    continue

                replacement_candidate = random.choice(valid_options)

                # If it's a better score, or same score but longer (more specific) match
                if (
                    best_match_info is None
                    or match.score > best_match_info[2]
                    or (
                        match.score == best_match_info[2]
                        and matched_segment_len
                        > (best_match_info[4] - best_match_info[3])
                    )
                ):
                    # The actual text to replace is from the original sentence, case-sensitively
                    text_to_replace_segment = sent_text[
                        match.dest_start : match.dest_end
                    ]
                    # Store the gpt_phrase_pattern as well to re-fetch valid_options later if this match is chosen
                    best_match_info = (
                        text_to_replace_segment,
                        replacement_candidate,
                        match.score,
                        match.dest_start,
                        match.dest_end,
                        gpt_phrase_pattern,
                    )

        if best_match_info:
            (
                text_to_replace,
                chosen_replacement,
                score,
                start,
                end,
                final_gpt_pattern,
            ) = best_match_info

            # Re-filter options for the final chosen pattern, just in case (though logic implies it was already done)
            final_options = GPT_ISM_PATTERNS[final_gpt_pattern]
            valid_final_options = [
                opt
                for opt in final_options
                if not (opt.startswith("[") and opt.endswith("]"))
            ]

            if not valid_final_options:
                # This case should ideally not be reached if the selection logic is correct,
                # but as a safeguard, if no valid replacements for the *best* match, do nothing.
                logger.warning(
                    "Best match GPT-ism '%s' (pattern: '%s') has no valid non-comment replacements. Skipping.",
                    text_to_replace,
                    final_gpt_pattern,
                )
                sent_text = original_sent_text  # Ensure original text is used
            else:
                # If the initially chosen_replacement is still valid (it should be), use it.
                # Otherwise, if somehow it became invalid (e.g. GPT_ISM_PATTERNS changed mid-run, highly unlikely),
                # we could re-random.choice(valid_final_options). For simplicity, we assume chosen_replacement is valid.
                # Ensure chosen_replacement is indeed from valid_final_options if paranoia strikes.
                # For now, we trust `chosen_replacement` was selected from valid options.

                sent_text = (
                    original_sent_text[:start]
                    + chosen_replacement
                    + original_sent_text[end:]
                )
                logger.info(
                    "Replacing GPT-ism segment '%s' (from pattern '%s', score %.1f) with '%s' in sentence: '%s' -> '%s'",
                    text_to_replace,
                    final_gpt_pattern,
                    score,
                    chosen_replacement,
                    original_sent_text,
                    sent_text,
                )
                replacements += 1

        new_sentences.append(sent_text.strip())
    cleaned_text = " ".join(new_sentences)
    return cleaned_text, replacements
