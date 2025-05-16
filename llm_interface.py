# llm_interface.py
"""
Handles all direct interactions with Large Language Models (LLMs)
and embedding models. Includes functions for API calls, response cleaning,
JSON extraction with retry logic, and embedding generation with caching.
Also includes asynchronous versions of API call functions.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2025 Dennis Lewis
"""

# Standard library imports
import functools # Keep for potential future sync needs, or remove if strictly async
import logging
import json
import re
# import requests # No longer needed if sync call_llm is removed

# Third-party imports
import numpy as np
import httpx  # For asynchronous HTTP requests
from async_lru import alru_cache  # Import async-aware LRU cache

# Type hints
from typing import List, Optional, Dict, Any, Union, Type
from type import JsonType # Assuming 'type.py' defines JsonType correctly

# Local imports
import config

logger = logging.getLogger(__name__)
def _validate_embedding(embedding_list: List[Union[float, int]], expected_dim: int, dtype: np.dtype) -> Optional[np.ndarray]:
    """Helper to validate and convert a list to a 1D numpy embedding."""
    try:
        embedding = np.array(embedding_list, dtype=dtype)
        if embedding.ndim > 1:
            embedding = embedding.flatten() # Ensure 1D
        if embedding.shape == (expected_dim,):
            return embedding
        logger.error(f"Embedding dimension mismatch: Expected ({expected_dim},), Got {embedding.shape}")
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert embedding list to numpy array: {e}")
    return None

# Synchronous get_embedding REMOVED

@alru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
async def async_get_embedding(text: str) -> Optional[np.ndarray]:
    """Asynchronously retrieves an embedding for the given text."""
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("async_get_embedding: empty or invalid text provided.")
        return None

    payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()}
    cache_info = async_get_embedding.cache_info()
    logger.debug(f"Async Embedding req: '{text[:80]}...' (Cache: h={cache_info.hits},m={cache_info.misses},s={cache_info.currsize})")

    api_response: Optional[httpx.Response] = None # For use in json.JSONDecodeError logging
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            api_response = await client.post(f"{config.OLLAMA_EMBED_URL}/api/embeddings", json=payload)
            api_response.raise_for_status()
            data = api_response.json()

            primary_key = "embedding"
            if primary_key in data and isinstance(data[primary_key], list):
                embedding = _validate_embedding(data[primary_key], config.EXPECTED_EMBEDDING_DIM, config.EMBEDDING_DTYPE)
                if embedding is not None:
                    return embedding
            
            logger.warning(f"Async: Primary embedding key '{primary_key}' failed or not found. Fallback search...")
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(item, (float, int)) for item in value):
                    embedding = _validate_embedding(value, config.EXPECTED_EMBEDDING_DIM, config.EMBEDDING_DTYPE)
                    if embedding is not None:
                        logger.info(f"Async: Fallback embedding success (key: '{key}').")
                        return embedding
            
            logger.error(f"Async: Embedding extraction failed. No suitable embedding list found in response: {data}")
            return None
        except httpx.TimeoutException:
            logger.error(f"Async: Embedding request timed out for text: '{text[:80]}...'")
        except httpx.HTTPStatusError as e_status: # Specifically catch HTTP status errors
            logger.error(f"Async: Embedding request failed with HTTP status {e_status.response.status_code}: {e_status}", exc_info=True)
            # e_status.response is guaranteed by HTTPStatusError
            logger.error(f"Async: Embedding error response body: {e_status.response.text[:200]}")
        except httpx.RequestError as e_req: # Catch other request errors (network, connection, etc.)
            logger.error(f"Async: Embedding request failed: {e_req}", exc_info=True)
        except json.JSONDecodeError as e_json:
             logger.error(
                 f"Async: Failed to decode JSON response for embedding: {e_json}. "
                 f"Response text: {api_response.text[:200] if api_response and hasattr(api_response, 'text') else 'N/A'}"
             )
        except Exception as e_exc: # General exceptions
            logger.error(f"Async: Unexpected error during embedding: {e_exc}", exc_info=True)
        return None


def _process_llm_response(response_data: Dict[str, Any], model_name: str, async_mode: bool = False) -> str:
    """Helper to process LLM response data and log usage."""
    prefix = "Async: " if async_mode else "" 
    if response_data.get("choices") and len(response_data["choices"]) > 0:
        message = response_data["choices"][0].get("message")
        if message and message.get("content"):
            raw_text = message["content"]
            if 'usage' in response_data:
                usage = response_data['usage']
                logger.info(
                    f"{prefix}LLM ('{model_name}') Usage - Prompt: {usage.get('prompt_tokens', 'N/A')} tk, "
                    f"Comp: {usage.get('completion_tokens', 'N/A')} tk, Total: {usage.get('total_tokens', 'N/A')} tk"
                )
            else:
                logger.warning(f"{prefix}LLM ('{model_name}') response missing 'usage' information.")
            return raw_text
        else:
            logger.error(f"{prefix}Invalid LLM ('{model_name}') response - missing message content: {response_data}")
    else:
        logger.error(f"{prefix}Invalid LLM ('{model_name}') response - missing choices: {response_data}")
    return ""

# Synchronous call_llm REMOVED

async def async_call_llm(model_name: str, prompt: str, temperature: float = 0.6, max_tokens: Optional[int] = None) -> str:
    """Asynchronously calls the LLM with the given prompt and parameters."""
    if not model_name:
        logger.error("async_call_llm: model_name is required.")
        return ""
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        logger.error("async_call_llm: empty or invalid prompt.")
        return ""

    effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "top_p": config.LLM_TOP_P,
        "max_tokens": effective_max_tokens
    }
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}

    logger.debug(f"Async Calling LLM '{model_name}'. Prompt len: {len(prompt)}. Max tokens: {effective_max_tokens}. Temp: {temperature}, TopP: {config.LLM_TOP_P}")
    
    api_response: Optional[httpx.Response] = None # For use in json.JSONDecodeError logging
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            api_response = await client.post(f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers)
            api_response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
            response_data = api_response.json()
            return _process_llm_response(response_data, model_name, async_mode=True)
        except httpx.TimeoutException:
            logger.error(f"Async LLM ('{model_name}') API request timed out.")
        except httpx.HTTPStatusError as e_status: # Handle HTTP status errors (4xx, 5xx)
            # This is the corrected block. e_status.response is guaranteed to exist.
            logger.error(f"Async LLM ('{model_name}') API HTTP status error: {e_status}", exc_info=True)
            logger.error(
                f"Async LLM ('{model_name}') API Response Status: {e_status.response.status_code}, "
                f"Body: {e_status.response.text[:500]}..." 
            )
        except httpx.RequestError as e_req: # Handle other request errors (network, connection, etc.)
            # For these errors, e_req.response is not guaranteed or typically not present.
            logger.error(f"Async LLM ('{model_name}') API request error (e.g., network): {e_req}", exc_info=True)
        except json.JSONDecodeError as e_json:
             logger.error(
                 f"Async: Failed to decode JSON response from LLM ('{model_name}'): {e_json}. "
                 f"Response text: {api_response.text[:200] if api_response and hasattr(api_response, 'text') else 'N/A'}"
             )
        except Exception as e_exc: # General exceptions
            logger.error(f"Async Unexpected error during LLM ('{model_name}') call: {e_exc}", exc_info=True)
        return ""


def clean_model_response(text: str) -> str:
    """Cleans common artifacts from LLM text responses."""
    if not isinstance(text, str):
        logger.warning(f"clean_model_response received non-string input: {type(text)}. Returning empty string.")
        return ""
    
    original_length = len(text)
    cleaned_text = text

    # Robustly remove leading/trailing empty <think></think> or similar tags
    # This regex handles variations in spacing and tag names like think, thought, thinking
    leading_think_artifact_regex = r'^\s*<\s*(think|thought|thinking)\s*>\s*(?:<\s*/\s*\1\s*>)?\s*'
    trailing_think_artifact_regex = r'\s*(?:<\s*(think|thought|thinking)\s*>)?\s*<\s*/\s*\1\s*>\s*$'
    
    # Iteratively remove until no more changes, to handle nested or multiple empty tags at ends
    while True:
        prev_len = len(cleaned_text)
        cleaned_text = re.sub(leading_think_artifact_regex, '', cleaned_text, count=1, flags=re.IGNORECASE)
        cleaned_text = re.sub(trailing_think_artifact_regex, '', cleaned_text, count=1, flags=re.IGNORECASE)
        if len(cleaned_text) == prev_len:
            break
            
    # General removal of <think>...</think>, <thought>...</thought>, <thinking>...</thinking> blocks
    think_block_pattern = re.compile(
        r'<\s*(think|thought|thinking)\s*>.*?<\s*/\s*\1\s*>',
        flags=re.DOTALL | re.IGNORECASE
    )
    cleaned_text = think_block_pattern.sub('', cleaned_text)
    
    # Remove markdown code blocks (json, python, text, or no language specified)
    # Ensure it doesn't greedily consume if JSON is the *only* content, which extract_json_block handles.
    # This cleaning is for extraneous code blocks, not the main JSON payload.
    # This might be too aggressive if the desired output *is* a code block.
    # Assuming clean_model_response is for general text, not for extracting specific code.
    cleaned_text = re.sub(r'```(?:json|python|text|)\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    # Remove lines that are just "Chapter X" or "Chapter X:" etc.
    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-—]?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove common framing lines like "--- BEGIN CHAPTER TEXT ---"
    cleaned_text = re.sub(r'^\s*[-=]{3,}\s*(BEGIN|END)\s+(CHAPTER|TEXT|DRAFT|CONTEXT|SNIPPET|JSON|OUTPUT)\b.*?[-=]{3,}\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)

    # Normalize multiple newlines to a maximum of two
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Process lines: strip leading/trailing whitespace from each line, collapse multiple spaces within lines
    lines = cleaned_text.splitlines()
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        # Collapse multiple spaces/tabs to single space, but only if line is not empty
        processed_lines.append(re.sub(r'[ \t]+', ' ', stripped_line) if stripped_line else '')
        
    final_text = '\n'.join(processed_lines).strip()

    if original_length > 0 and len(final_text) < original_length * 0.95 : # Log if significant reduction
        reduction_percentage = ((original_length - len(final_text)) / original_length) * 100
        logger.debug(f"Cleaning reduced text length from {original_length} to {len(final_text)} ({reduction_percentage:.1f}% reduction).")
    
    return final_text


def extract_json_block(text: str, expect_type: Union[Type[dict], Type[list]] = dict) -> Optional[str]:
    """
    Extracts a JSON block from a string.
    Prioritizes JSON within Markdown code blocks, then general brace/bracket matching.
    """
    if not isinstance(text, str):
        logger.warning("extract_json_block: non-string input.")
        return None

    start_char, end_char = ('{', '}') if expect_type == dict else ('[', ']')
    
    # 1. Regex for JSON within markdown ```json ... ``` blocks
    # Handles optional "json" language specifier and potential leading/trailing whitespace within the block.
    markdown_pattern = re.compile(rf'```(?:json)?\s*({start_char}.*?{end_char})\s*```', re.DOTALL | re.IGNORECASE)
    match_markdown = markdown_pattern.search(text)
    if match_markdown:
        json_str = match_markdown.group(1).strip()
        # Basic validation of start/end characters for the extracted string
        if json_str.startswith(start_char) and json_str.endswith(end_char):
            logger.debug(f"Potential JSON {expect_type.__name__} found in Markdown code block.")
            return json_str

    # 2. Fallback: find the first occurrence of start_char and last of end_char (broad match)
    # This is greedy and might grab too much if there are nested structures or multiple JSON objects.
    # It's a common fallback when Markdown blocks are not used.
    first_start_index = text.find(start_char)
    if first_start_index != -1:
        balance = 0
        potential_end_index = -1
        for i in range(first_start_index, len(text)):
            if text[i] == start_char:
                balance += 1
            elif text[i] == end_char:
                balance -= 1
            if balance == 0 and text[i] == end_char : # Found a balanced structure ending with end_char
                potential_end_index = i
                break
        
        if potential_end_index != -1:
            potential_json = text[first_start_index : potential_end_index + 1].strip()
            if potential_json.startswith(start_char) and potential_json.endswith(end_char):
                try:
                    # Attempt to parse to ensure it's valid before returning
                    json.loads(potential_json)
                    logger.debug(f"Potential JSON {expect_type.__name__} found by brace/bracket matching with balance check and validation.")
                    return potential_json
                except json.JSONDecodeError:
                    logger.debug(f"String found by brace/bracket balance check is not valid JSON: '{potential_json[:100]}...'")


    # 3. Fallback: If the entire cleaned response (after removing common LLM chat/cruft) is the JSON
    # This is useful if the LLM *only* returned JSON without any wrappers.
    cleaned_full_response = clean_model_response(text) # Apply basic cleaning
    if cleaned_full_response.startswith(start_char) and cleaned_full_response.endswith(end_char):
        try:
            json.loads(cleaned_full_response) # Test parse
            logger.debug(f"Entire cleaned response appears to be valid JSON {expect_type.__name__}.")
            return cleaned_full_response
        except json.JSONDecodeError:
            logger.debug(f"Entire cleaned response starts/ends with {start_char}/{end_char} but is not valid JSON.")

    logger.warning(f"Failed to extract a likely JSON {expect_type.__name__} block from text: '{text[:150]}...'")
    return None


async def async_parse_llm_json_response(
    raw_response: str, 
    context_for_log: str, 
    expect_type: Union[Type[dict], Type[list]] = dict
) -> Optional[JsonType]:
    """
    Parses a JSON block from an LLM's raw response string, with async LLM-based correction.
    """
    if not raw_response:
        logger.warning(f"LLM returned empty response for {context_for_log}. Cannot parse JSON.")
        return None

    json_block_str = extract_json_block(raw_response, expect_type)
    
    if json_block_str:
        try:
            parsed_json = json.loads(json_block_str)
            if isinstance(parsed_json, expect_type):
                logger.info(f"Successfully parsed JSON {expect_type.__name__} for {context_for_log} on 1st attempt.")
                return parsed_json
            else:
                logger.warning(f"Parsed JSON for {context_for_log} is type {type(parsed_json)}, expected {expect_type.__name__}. Will attempt LLM correction.")
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed for {context_for_log}: {e}. Extracted: '{json_block_str[:100]}...'")
            
            # Heuristic fixes
            fixed_json_str = None
            if e.msg.startswith("Expecting ',' delimiter") and json_block_str.rstrip().endswith((',', chr(0x2c))): # Handle regular comma and unicode comma
                # Try removing trailing comma
                temp_str = json_block_str.rstrip()
                if temp_str.endswith(',') or temp_str.endswith(chr(0x2c)):
                    fixed_json_str = temp_str[:-1]
                    logger.info(f"Attempting heuristic fix for trailing comma in {context_for_log}.")
            elif expect_type == list and e.msg.startswith("Expecting ',' delimiter") and e.pos >= len(json_block_str) - 10:
                fixed_json_str = json_block_str + "]"
                logger.info(f"Attempting heuristic fix for truncated list in {context_for_log}: appending ']'")
            elif expect_type == dict and e.msg.startswith("Expecting property name enclosed in double quotes") and e.pos >= len(json_block_str) -10:
                 fixed_json_str = json_block_str + "}"
                 logger.info(f"Attempting heuristic fix for truncated object in {context_for_log}: appending '}}'")


            if fixed_json_str:
                try:
                    parsed_json = json.loads(fixed_json_str)
                    if isinstance(parsed_json, expect_type):
                        logger.info(f"Heuristically parsed JSON {expect_type.__name__} for {context_for_log}.")
                        return parsed_json
                except json.JSONDecodeError:
                    logger.warning(f"Heuristic fix failed for {context_for_log}. Proceeding to LLM correction...")
    else:
        logger.error(f"Could not extract any JSON {expect_type.__name__} block from initial response for {context_for_log}. Raw: '{raw_response[:200]}...'")
        # If no block was extracted, use the cleaned raw response as input for correction
        json_block_str = clean_model_response(raw_response) 
        if not json_block_str:
            logger.error(f"No content (even after cleaning) to attempt LLM correction for {context_for_log}.")
            return None
    
    expected_start_char = '{' if expect_type == dict else '['
    expected_end_char = '}' if expect_type == dict else ']'
    
    logger.info(f"Attempting LLM JSON correction for {context_for_log} using model {config.JSON_CORRECTION_MODEL}...")
    correction_prompt = f"""/no_think
    
    The following text block was intended to be a valid JSON {expect_type.__name__}, but it contains syntax errors or is not the correct type.
Correct any syntax errors (e.g., missing commas, incorrect quoting, trailing commas, unbalanced braces/brackets, unescaped characters within strings).
Ensure the final output is a single, valid JSON {expect_type.__name__}.
Output ONLY the corrected, valid JSON object or array. Do not include any explanations, apologies, or surrounding text.

Invalid/Problematic JSON Block:
```
{json_block_str}
```

Corrected JSON {expect_type.__name__} Output Only (must start with '{expected_start_char}' and end with '{expected_end_char}'):
"""
    
    corrected_raw = await async_call_llm( 
        model_name=config.JSON_CORRECTION_MODEL, 
        prompt=correction_prompt, 
        temperature=0.6, # Low temperature for deterministic correction
        max_tokens=config.MAX_GENERATION_TOKENS # Ensure sufficient for potentially large JSON
    )
    
    if not corrected_raw:
        logger.error(f"LLM JSON correction attempt returned empty response for {context_for_log}.")
        return None

    # The LLM correction might itself return something wrapped, so extract again.
    corrected_json_block_str = extract_json_block(corrected_raw, expect_type)
    if not corrected_json_block_str:
        # If extract_json_block fails, it could be because the LLM returned *only* the JSON.
        # So, try clean_model_response on the raw output as a fallback.
        corrected_json_block_str = clean_model_response(corrected_raw)
        logger.debug(f"Using cleaned full response from correction LLM for {context_for_log} as no block was extracted.")

    if corrected_json_block_str:
        try:
            parsed_corrected_json = json.loads(corrected_json_block_str)
            if isinstance(parsed_corrected_json, expect_type):
                logger.info(f"Successfully parsed JSON {expect_type.__name__} for {context_for_log} after LLM correction.")
                return parsed_corrected_json
            else:
                logger.error(f"Corrected JSON for {context_for_log} is type {type(parsed_corrected_json)}, expected {expect_type.__name__}. Corrected block: '{corrected_json_block_str[:200]}...'")
                return None
        except json.JSONDecodeError as e_retry:
            logger.error(f"JSON parsing failed AGAIN for {context_for_log} after LLM correction: {e_retry}. Corrected block: '{corrected_json_block_str[:200]}...'")
            return None
    else:
        logger.error(f"Could not extract or clean any JSON {expect_type.__name__} content from LLM correction output for {context_for_log}. Corrected raw: '{corrected_raw[:200]}...'")
        return None