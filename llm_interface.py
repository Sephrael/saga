# llm_interface.py
"""
Handles all direct interactions with Large Language Models (LLMs)
and embedding models. Includes functions for API calls, response cleaning,
JSON extraction with retry logic, and embedding generation with caching.
"""

import requests
import json
import re
import numpy as np
import logging
import functools
from typing import Optional, Dict, Any

# Import configuration variables directly for clarity
import config

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Embedding Function (No changes needed here based on request) ---

@functools.lru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Retrieves an embedding vector for the given text from the configured
    Ollama endpoint.

    Uses LRU caching to avoid redundant API calls for the same text.
    Includes error handling, dimension checks, and fallback logic.

    Args:
        text: The input string to embed.

    Returns:
        A numpy array representing the embedding, or None if an error occurs
        or the response is invalid.
    """
    # Basic input validation
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.warning("get_embedding called with empty or invalid text.")
        # Return None from cache perspective, but don't cache this None result for empty string
        return None

    # Prepare payload for the Ollama API
    payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()} # Ensure text is stripped
    # Log cache status (simplified check)
    cache_info = get_embedding.cache_info()
    logger.debug(f"Requesting embedding for text snippet: '{text[:80]}...' (Cache info: hits={cache_info.hits}, misses={cache_info.misses}, current_size={cache_info.currsize})")


    try:
        # Make the POST request to the Ollama API
        response = requests.post(
            f"{config.OLLAMA_EMBED_URL}/api/embeddings",
            json=payload,
            timeout=300 # Generous timeout for embedding models
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        # Parse the JSON response
        data = response.json()
        logger.debug(f"Raw embedding response type: {type(data)}, Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

        # --- Primary Key Extraction ---
        embedding_key = "embedding" # Standard key expected from Ollama
        if embedding_key in data and isinstance(data[embedding_key], list):
            embedding_value = data[embedding_key]
            logger.debug(f"Found primary key '{embedding_key}'. Value type: {type(embedding_value)}, Length: {len(embedding_value)}")
            try:
                # Convert to NumPy array with the configured dtype
                embedding = np.array(embedding_value, dtype=config.EMBEDDING_DTYPE)
                logger.debug("Successfully converted primary embedding value to NumPy array.")

                # Flatten if necessary (some models might return nested arrays)
                if embedding.ndim > 1:
                    logger.debug(f"Original embedding shape: {embedding.shape}. Flattening.")
                    embedding = embedding.flatten()

                # Validate dimensions
                if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,):
                    logger.debug(f"Embedding retrieved successfully (shape: {embedding.shape}).")
                    return embedding # Success!
                else:
                    # Log dimension mismatch error
                    logger.error(f"Primary embedding dimension mismatch: Expected ({config.EXPECTED_EMBEDDING_DIM},), Got {embedding.shape}")
                    # Fall through to fallback logic if dimension is wrong

            except (TypeError, ValueError) as e:
                # Handle errors during NumPy conversion
                logger.error(f"Failed to convert primary embedding value to NumPy array: {e}. Value type was {type(embedding_value)}. Falling back.")
                # Fall through to fallback logic

        # --- Fallback Logic (if primary key fails or dimension mismatch) ---
        logger.warning(f"Primary key '{embedding_key}' failed or dimension mismatch. Attempting fallback key search...")
        found_fallback = False
        for key, value in data.items():
            # Check if the value is a list of numbers (floats or ints)
            if isinstance(value, list) and len(value) > 0 and all(isinstance(item, (float, int)) for item in value):
                logger.warning(f"Potential fallback embedding found using key '{key}'.")
                try:
                    # Attempt conversion and validation for the fallback value
                    embedding = np.array(value, dtype=config.EMBEDDING_DTYPE)
                    if embedding.ndim > 1: embedding = embedding.flatten()

                    if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,):
                        logger.info(f"Fallback embedding successful (key: '{key}', shape: {embedding.shape}).")
                        return embedding # Return successfully found fallback
                    else:
                         logger.error(f"Fallback embedding dimension mismatch: Key '{key}', Expected ({config.EXPECTED_EMBEDDING_DIM},), Got {embedding.shape}")
                except (TypeError, ValueError) as e:
                     # Log error if fallback conversion fails, continue checking other keys
                     logger.error(f"Failed to convert fallback embedding value for key '{key}' to NumPy array: {e}")

        # If loop finishes and no suitable fallback is found
        logger.error("Embedding extraction failed. No suitable key/list found or dimension mismatch in response.")
        return None

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logger.error(f"Embedding request timed out after 300 seconds for text: '{text[:80]}...'")
    except requests.exceptions.RequestException as e:
        # Handle general request errors (connection, DNS, etc.)
        logger.error(f"Embedding request failed: {e}", exc_info=True) # Include traceback
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        # Handle errors in processing the response structure or JSON parsing
        logger.error(f"Error processing embedding response: {e}", exc_info=True)
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"An unexpected error occurred during embedding: {e}", exc_info=True)

    # Ensure None is returned if any exception occurred
    return None

# --- LLM Interaction Function (No changes needed here) ---

def call_llm(prompt: str, temperature: float = 0.6, max_tokens: Optional[int] = None) -> str:
    """
    Sends a prompt to the configured OpenAI-compatible LLM endpoint and returns the response content.

    Args:
        prompt: The input prompt string for the LLM.
        temperature: The sampling temperature for generation (controls randomness).
        max_tokens: The maximum number of tokens to generate (overrides config default if provided).

    Returns:
        The generated text content as a string, or an empty string if an error occurs.
    """
    # Basic input validation
    if not prompt or not isinstance(prompt, str):
        logger.error("call_llm received empty or invalid prompt.")
        return ""

    # Determine the effective max_tokens value
    effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS

    # Prepare the payload for the Chat Completions API
    payload = {
        "model": config.MAIN_GENERATION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, # Request a non-streaming response
        "temperature": temperature,
        "top_p": 0.95, # Nucleus sampling parameter
        "max_tokens": effective_max_tokens
    }
    # Prepare headers, including Authorization
    headers = {
        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    logger.debug(f"Calling LLM '{config.MAIN_GENERATION_MODEL}'. Prompt length: {len(prompt)} chars. Max tokens: {effective_max_tokens}. Temp: {temperature}")

    try:
        # Make the POST request
        response = requests.post(
            f"{config.OPENAI_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=600 # Generous timeout for potentially long generations
        )
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        # Parse the JSON response
        response_data = response.json()

        # --- Extract Content and Log Usage ---
        # Navigate the response structure to get the generated content
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message")
            if message and message.get("content"):
                raw_text = message["content"] # The actual generated text

                # Log token usage if the 'usage' field is present
                if 'usage' in response_data:
                    usage = response_data['usage']
                    logger.info(
                        f"LLM Usage - Prompt: {usage.get('prompt_tokens', 'N/A')} tk, "
                        f"Completion: {usage.get('completion_tokens', 'N/A')} tk, "
                        f"Total: {usage.get('total_tokens', 'N/A')} tk"
                    )
                else:
                    # Log a warning if usage data is missing (might indicate issues)
                    logger.warning("LLM response missing 'usage' information.")

                return raw_text # Return the successfully extracted text
            else:
                # Error if the expected message content structure is missing
                logger.error(f"Invalid LLM response structure - missing message content: {response_data}")
        else:
            # Error if the 'choices' array is missing or empty
            logger.error(f"Invalid LLM response structure - missing choices: {response_data}")

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logger.error(f"LLM API request timed out after 600 seconds.")
    except requests.exceptions.RequestException as e:
        # Handle general request errors
        logger.error(f"LLM API request error: {e}", exc_info=True)
        # Log response details if available
        if e.response is not None:
            logger.error(f"LLM API Response Status: {e.response.status_code}")
            try:
                # Log the beginning of the response body for debugging
                logger.error(f"LLM API Response Body: {e.response.text[:500]}...")
            except Exception:
                logger.error("LLM API Response Body could not be decoded or read.")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        # Handle errors related to parsing the expected JSON structure
        logger.error(f"Error processing LLM response JSON structure: {e}", exc_info=True)
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)

    # Return an empty string if any error occurred
    return ""

# --- Response Processing Functions ---

def clean_model_response(text: str) -> str:
    """
    Removes common LLM artifacts (like <think> blocks), extraneous formatting,
    and normalizes whitespace while preserving paragraph structure. Includes
    improved logic for handling improperly closed or leading think blocks.

    Args:
        text: The raw text response from the LLM.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        logger.warning(f"clean_model_response received non-string input: {type(text)}. Returning empty string.")
        return ""

    original_text_len = len(text)
    cleaned_text = text # Start with the original text

    # --- Step 1: Remove Think Blocks ---
    # Primary regex for well-formed think blocks (case-insensitive, dot matches newline)
    think_block_regex = r'<\s*(?:think|thought|thinking)\s*>.*?<\s*/\s*(?:think|thought|thinking)\s*>'
    try:
        # Attempt the primary removal
        text_after_primary_sub = re.sub(think_block_regex, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

        if len(text_after_primary_sub) < len(cleaned_text):
            logger.debug("Primary think block regex removed content.")
            cleaned_text = text_after_primary_sub # Update text if removal occurred
        else:
            # --- Fallback Logic for Leading/Mismatched Tags ---
            # Check if the text *starts* with an opening think tag (ignoring leading whitespace)
            start_tag_pattern = r'^\s*<\s*(?:think|thought|thinking)\s*>'
            start_match = re.search(start_tag_pattern, cleaned_text, re.IGNORECASE)

            if start_match:
                # If it starts with a tag, but primary regex didn't remove it
                logger.warning(f"Primary think block regex failed, but text starts with tag: {cleaned_text[:100]}...")
                # Try to find the *first* corresponding closing tag after the opening tag
                end_tag_pattern = r'<\s*/\s*(?:think|thought|thinking)\s*>'
                # Search only in the part of the string *after* the opening tag
                end_match = re.search(end_tag_pattern, cleaned_text[start_match.end():], re.IGNORECASE)

                if end_match:
                    # If a closing tag is found, remove everything from the start up to the end of that closing tag
                    end_pos_in_original = start_match.end() + end_match.end()
                    cleaned_text = cleaned_text[end_pos_in_original:].strip() # Take only the text *after* the block
                    logger.info("Applied fallback think block removal (removed from start to first closing tag).")
                else:
                    # If no closing tag is found after the opening tag
                    logger.warning("Fallback removal failed: Found start tag but no corresponding end tag. Attempting to strip only start tag.")
                    # As a last resort, remove only the opening tag
                    cleaned_text = re.sub(start_tag_pattern, '', cleaned_text, count=1, flags=re.IGNORECASE).strip()
            else:
                # If no leading tag was found, the primary regex simply found nothing to remove
                logger.debug("No think tags found near the beginning to trigger fallback removal.")

    except Exception as e:
        # Log regex errors but continue with the potentially uncleaned text
        logger.error(f"Error during think block regex substitution: {e}", exc_info=True)

    # --- Step 2: Remove Other Common Artifacts ---
    # Remove common introductory/closing remarks often added by LLMs
    # Made case-insensitive and multiline
    cleaned_text = re.sub(r'^\s*(Okay, here.*?|Here\'s the chapter|Response:|Summary:|List inconsistencies below:|Revised chapter:|Analysis:|JSON Output:)\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE).strip()
    # Remove LaTeX-like blocks (if they appear)
    cleaned_text = re.sub(r'\\boxed{.*?}', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\\begin{.*?}.*?\\end{.*?}', '', cleaned_text, flags=re.DOTALL)
    # Remove markdown code blocks (often used for JSON, but sometimes left in narrative)
    # Made non-greedy (.*?) and handles optional language tags
    cleaned_text = re.sub(r'```(?:json|python|text|)\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    # Remove "Chapter X" lines if they stand alone
    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-]?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove potential markers if the model echoes them back
    cleaned_text = re.sub(r'^\s*---?\s*(BEGIN|END) (CHAPTER|TEXT|DRAFT|CONTEXT|SNIPPET).*?\s*---?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove trailing instructions or sign-offs if they slipped through
    cleaned_text = re.sub(r'\s*Output ONLY the.*?text.*$','', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r'\s*/no_think\s*$', '', cleaned_text, flags=re.IGNORECASE) # Remove /no_think tag


    # --- Step 3: Whitespace Normalization ---
    # Split into lines, process each line, and rejoin
    lines = cleaned_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            # Replace multiple spaces/tabs within the line with a single space
            normalized_line = re.sub(r'[ \t]+', ' ', stripped_line)
            cleaned_lines.append(normalized_line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            # Preserve paragraph breaks: Add at most one empty line
            cleaned_lines.append("")

    # Join the processed lines back together
    final_text = '\n'.join(cleaned_lines).strip()
    # Ensure there's max one blank line between paragraphs globally
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)

    # Log if cleaning significantly changed the length
    if len(final_text) < original_text_len * 0.95: # Log if length reduced by > 5%
        logger.debug(f"Cleaning reduced text length from {original_text_len} to {len(final_text)}.")

    return final_text


def extract_json_block(text: str) -> Optional[str]:
    """
    Attempts to extract a JSON block (specifically a dictionary starting with '{')
    from a string. Checks for markdown code blocks and simple brace matching.
    Does NOT attempt parsing here, only extraction.

    Args:
        text: The string potentially containing a JSON block.

    Returns:
        The extracted potential JSON string, or None if no likely block is found.
    """
    if not isinstance(text, str):
        logger.warning("extract_json_block received non-string input.")
        return None

    logger.debug("Attempting to extract potential JSON block...")

    # 1. Check for Markdown code blocks (```json ... ```)
    match_markdown = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match_markdown:
        json_str = match_markdown.group(1).strip()
        if json_str.startswith('{') and json_str.endswith('}'):
            logger.debug("Potential JSON block found within Markdown ```.")
            return json_str # Return the potential block
        else:
            logger.warning(f"Extracted markdown block content doesn't start/end with braces: {json_str[:100]}...")

    # 2. Find the first '{' and the last '}'
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = text[start_index : end_index + 1].strip()
        logger.debug("Potential JSON block found using first '{' and last '}'.")
        if potential_json.startswith('{') and potential_json.endswith('}'):
             # Basic check for balanced braces (optional, can be slow/imperfect)
             # if potential_json.count('{') == potential_json.count('}'):
             #    return potential_json
             # else:
             #    logger.debug("Block between first '{' and last '}' had unbalanced braces. Skipping.")
             return potential_json # Return even if braces might be unbalanced internally

    # 3. Fallback: Check if the *entire cleaned* response looks like JSON
    # This is less likely if markdown/brace matching failed, but possible
    logger.debug("Attempting fallback: checking if entire cleaned response looks like JSON.")
    cleaned_full = clean_model_response(text) # Clean artifacts first
    if cleaned_full.startswith('{') and cleaned_full.endswith('}'):
        logger.debug("Entire cleaned response appears to be a potential JSON block.")
        return cleaned_full

    logger.warning("Failed to extract a likely JSON block from the text.")
    return None


def parse_llm_json_response(raw_response: str, context_for_log: str) -> Optional[Dict[str, Any]]:
    """
    Cleans raw LLM response, attempts to extract a JSON block, parses it,
    and includes a single retry attempt with the LLM if initial parsing fails.

    Args:
        raw_response: The raw string output from the LLM.
        context_for_log: A string describing the context (e.g., "plot outline") for logging.

    Returns:
        A dictionary parsed from the JSON, or None if extraction or parsing (including retry) fails.
    """
    if not raw_response:
        logger.warning(f"LLM returned empty response for {context_for_log}. Cannot parse JSON.")
        return None

    # Attempt to extract a potential JSON string first
    json_block_str = extract_json_block(raw_response)

    if json_block_str:
        # --- Initial Parsing Attempt ---
        try:
            parsed_json = json.loads(json_block_str)
            if isinstance(parsed_json, dict):
                logger.info(f"Successfully parsed extracted JSON block for {context_for_log} on first attempt.")
                return parsed_json
            else:
                logger.warning(f"Parsed JSON block for {context_for_log} is not a dictionary (type: {type(parsed_json)}). Discarding.")
                return None
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed for {context_for_log}: {e}. Attempting LLM correction...")

            # --- Retry Logic: Ask LLM to Fix JSON ---
            correction_prompt = f"""The following text block was extracted, but it contains JSON syntax errors. Please correct the syntax errors (e.g., missing commas, incorrect quoting, trailing commas, unbalanced braces/brackets) and output ONLY the corrected, valid JSON object. Do not add any explanation or commentary.

Invalid JSON Block:
```json
{json_block_str}
```

Corrected JSON Output Only:
/no_think
"""
            # Call LLM with low temperature to focus on correction
            corrected_raw = call_llm(correction_prompt, temperature=0.6, max_tokens=config.MAX_GENERATION_TOKENS) # Allow enough tokens for large JSON

            if not corrected_raw:
                 logger.error(f"LLM correction attempt failed for {context_for_log} (empty response).")
                 return None

            # Clean and attempt to parse the corrected response
            corrected_cleaned = clean_model_response(corrected_raw)
            # Try extracting again in case the LLM wrapped it
            corrected_block_str = extract_json_block(corrected_cleaned) or corrected_cleaned

            if corrected_block_str:
                try:
                    parsed_corrected_json = json.loads(corrected_block_str)
                    if isinstance(parsed_corrected_json, dict):
                        logger.info(f"Successfully parsed JSON for {context_for_log} after LLM correction.")
                        return parsed_corrected_json
                    else:
                        logger.error(f"Corrected JSON block for {context_for_log} is not a dictionary (type: {type(parsed_corrected_json)}).")
                        return None
                except json.JSONDecodeError as e_retry:
                    logger.error(f"JSON parsing failed AGAIN for {context_for_log} after LLM correction attempt: {e_retry}. Corrected block snippet:\n{corrected_block_str[:200]}...")
                    return None
            else:
                 logger.error(f"Could not extract JSON block from LLM correction response for {context_for_log}.")
                 return None
            # --- End Retry Logic ---

    else:
        # If extract_json_block couldn't find anything initially
        logger.error(f"Could not extract a JSON block from the initial response for {context_for_log}. Raw response snippet:\n{raw_response[:300]}...")
        return None

