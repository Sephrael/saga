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
