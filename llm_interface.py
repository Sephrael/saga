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
import functools 
import logging
import json
import re
import asyncio # Added for sleep in retry logic
import time # For exponential backoff
import tempfile # For streaming to disk
import os # For temp file cleanup

# Third-party imports
import numpy as np
import httpx  # For asynchronous HTTP requests
from async_lru import alru_cache  # Import async-aware LRU cache

# Type hints
from typing import List, Optional, Dict, Any, Union, Type, Tuple
from type import JsonType 

# Local imports
import config

logger = logging.getLogger(__name__)

def _validate_embedding(embedding_list: List[Union[float, int]], expected_dim: int, dtype: np.dtype) -> Optional[np.ndarray]:
    """Helper to validate and convert a list to a 1D numpy embedding."""
    try:
        # Ensure dtype conversion happens before shape check, esp. for float16
        embedding = np.array(embedding_list).astype(dtype) 
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        if embedding.shape == (expected_dim,):
            return embedding
        logger.error(f"Embedding dimension mismatch: Expected ({expected_dim},), Got {embedding.shape}. Original list length: {len(embedding_list)}")
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert embedding list to numpy array: {e}")
    return None


@alru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
async def async_get_embedding(text: str) -> Optional[np.ndarray]:
    """Asynchronously retrieves an embedding for the given text with retry logic."""
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("async_get_embedding: empty or invalid text provided.")
        return None

    payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()}
    cache_info = async_get_embedding.cache_info()
    logger.debug(f"Async Embedding req: '{text[:80]}...' (Cache: h={cache_info.hits},m={cache_info.misses},s={cache_info.currsize})")

    last_exception = None
    for attempt in range(config.LLM_RETRY_ATTEMPTS):
        api_response: Optional[httpx.Response] = None
        try:
            async with httpx.AsyncClient(timeout=300) as client: # Long timeout for embedding models
                api_response = await client.post(f"{config.OLLAMA_EMBED_URL}/api/embeddings", json=payload)
                api_response.raise_for_status()
                data = api_response.json()

            primary_key = "embedding"
            if primary_key in data and isinstance(data[primary_key], list):
                embedding = _validate_embedding(data[primary_key], config.EXPECTED_EMBEDDING_DIM, config.EMBEDDING_DTYPE)
                if embedding is not None:
                    return embedding
            
            logger.warning(f"Async (Attempt {attempt+1}): Primary embedding key '{primary_key}' failed or not found. Fallback search...")
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(item, (float, int)) for item in value):
                    embedding = _validate_embedding(value, config.EXPECTED_EMBEDDING_DIM, config.EMBEDDING_DTYPE)
                    if embedding is not None:
                        logger.info(f"Async (Attempt {attempt+1}): Fallback embedding success (key: '{key}').")
                        return embedding
            
            logger.error(f"Async (Attempt {attempt+1}): Embedding extraction failed. No suitable embedding list found in response: {data}")
            # This case is not typically a retriable server error, but a data format issue.
            # However, if the server *sometimes* returns malformed data, retrying might help.
            # For now, we'll let it retry, but this could be refined.
            last_exception = ValueError("No suitable embedding list found in response.") # Create an exception for logging if all retries fail

        except httpx.TimeoutException as e_timeout:
            last_exception = e_timeout
            logger.warning(f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Request timed out: {e_timeout}")
        except httpx.HTTPStatusError as e_status:
            last_exception = e_status
            logger.warning(
                f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): HTTP status {e_status.response.status_code}: {e_status}. "
                f"Body: {e_status.response.text[:200]}"
            )
            if 400 <= e_status.response.status_code < 500: # Client-side error, don't retry
                logger.error(f"Async Embedding: Client-side error {e_status.response.status_code}. Aborting retries.")
                break 
        except httpx.RequestError as e_req:
            last_exception = e_req
            logger.warning(f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Request error: {e_req}")
        except json.JSONDecodeError as e_json:
            last_exception = e_json
            logger.warning(
                 f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Failed to decode JSON response: {e_json}. "
                 f"Response text: {api_response.text[:200] if api_response and hasattr(api_response, 'text') else 'N/A'}"
            )
        except Exception as e_exc:
            last_exception = e_exc
            logger.warning(f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Unexpected error: {e_exc}", exc_info=True)
        
        if attempt < config.LLM_RETRY_ATTEMPTS - 1:
            delay = config.LLM_RETRY_DELAY_SECONDS * (2 ** attempt)
            logger.info(f"Async Embedding: Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
        else:
            logger.error(f"Async Embedding: All {config.LLM_RETRY_ATTEMPTS} retry attempts failed. Last error: {last_exception}")
            return None
    return None # Should be unreachable if loop logic is correct, but as a fallback.


def _log_llm_usage(model_name: str, usage_data: Optional[Dict[str, int]], async_mode: bool = False, streamed: bool = False):
    """Helper to log LLM token usage."""
    prefix = "Async: " if async_mode else ""
    stream_prefix = "Streamed " if streamed else ""
    if usage_data:
        logger.info(
            f"{prefix}{stream_prefix}LLM ('{model_name}') Usage - Prompt: {usage_data.get('prompt_tokens', 'N/A')} tk, "
            f"Comp: {usage_data.get('completion_tokens', 'N/A')} tk, Total: {usage_data.get('total_tokens', 'N/A')} tk"
        )
    else:
        logger.warning(f"{prefix}{stream_prefix}LLM ('{model_name}') response missing 'usage' information.")

async def async_call_llm(
    model_name: str, 
    prompt: str, 
    temperature: float = 0.6, 
    max_tokens: Optional[int] = None,
    allow_fallback: bool = False,
    stream_to_disk: bool = False  # New parameter
) -> str:
    """
    Asynchronously calls the LLM with retry and optional model fallback.
    If stream_to_disk is True, streams large responses to a temporary disk file
    before reading back into memory. Otherwise, handles responses in memory.
    Returns the LLM's text response as a string.
    """
    if not model_name:
        logger.error("async_call_llm: model_name is required.")
        return ""
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        logger.error("async_call_llm: empty or invalid prompt.")
        return ""

    effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
    
    current_model_to_try = model_name
    is_fallback_attempt = False

    for attempt_num_overall in range(2): # Max 1 primary attempt (with retries) + 1 fallback attempt (with retries)
        if is_fallback_attempt:
            if not allow_fallback or not config.FALLBACK_GENERATION_MODEL:
                logger.warning(f"Primary model '{model_name}' failed. Fallback not allowed or not configured. Aborting.")
                return ""
            current_model_to_try = config.FALLBACK_GENERATION_MODEL
            logger.info(f"Primary model '{model_name}' failed. Attempting fallback with '{current_model_to_try}'.")
        
        payload = {
            "model": current_model_to_try,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": config.LLM_TOP_P,
            "max_tokens": effective_max_tokens
            # "stream" parameter will be set based on stream_to_disk
        }
        
        last_exception_for_current_model = None
        temp_file_path_for_stream: Optional[str] = None

        for retry_attempt in range(config.LLM_RETRY_ATTEMPTS):
            logger.debug(
                f"Async Calling LLM '{current_model_to_try}' (Attempt {retry_attempt+1}/{config.LLM_RETRY_ATTEMPTS}, Fallback: {is_fallback_attempt}). "
                f"StreamToDisk: {stream_to_disk}. Prompt len: {len(prompt)}. Max tokens: {effective_max_tokens}. Temp: {temperature}, TopP: {config.LLM_TOP_P}"
            )
            api_response: Optional[httpx.Response] = None
            
            try:
                if stream_to_disk:
                    payload["stream"] = True
                    # Create a temp file for this attempt
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".llmstream.txt") as tmp_f:
                        temp_file_path_for_stream = tmp_f.name
                    # tmp_f is closed here. We reopen for writing chunks.

                    accumulated_content_pieces = [] # For safety, if file ops fail but stream works
                    prompt_tokens, completion_tokens, total_tokens = 0,0,0
                    usage_info_from_stream: Optional[Dict[str, int]] = None

                    async with httpx.AsyncClient(timeout=600) as client: # Increased timeout
                        async with client.stream("POST", f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers) as response:
                            response.raise_for_status()
                            
                            with open(temp_file_path_for_stream, "w", encoding="utf-8") as f_out:
                                async for line in response.aiter_lines():
                                    if line.startswith("data: "):
                                        data_json_str = line[len("data: "):].strip()
                                        if data_json_str == "[DONE]":
                                            break
                                        try:
                                            chunk_data = json.loads(data_json_str)
                                            if chunk_data.get("choices"):
                                                delta = chunk_data["choices"][0].get("delta", {})
                                                content_piece = delta.get("content")
                                                if content_piece:
                                                    f_out.write(content_piece)
                                                    accumulated_content_pieces.append(content_piece)
                                                
                                                # Check for usage in the last chunk (OpenAI specific)
                                                if chunk_data["choices"][0].get("finish_reason") is not None:
                                                    # Some LLM providers might put 'usage' in the main chunk_data, not delta
                                                    potential_usage = chunk_data.get("usage") 
                                                    if potential_usage and isinstance(potential_usage, dict):
                                                        usage_info_from_stream = potential_usage
                                                    # Or sometimes it's in x_groq.usage (Groq specific example)
                                                    elif chunk_data.get("x_groq") and chunk_data["x_groq"].get("usage"):
                                                        usage_info_from_stream = chunk_data["x_groq"]["usage"]
                                                    
                                                    # Break if finish_reason is present, signaling end of useful stream
                                                    # This handles cases where [DONE] might be delayed or not sent by some providers
                                                    # and ensures we capture usage if it's in the *same* chunk as finish_reason.
                                                    break 
                                        except json.JSONDecodeError:
                                            logger.warning(f"Async LLM Stream: Could not decode JSON from line: {line}")

                    _log_llm_usage(current_model_to_try, usage_info_from_stream, async_mode=True, streamed=True)
                    
                    # Read content from temp file
                    with open(temp_file_path_for_stream, "r", encoding="utf-8") as f_in:
                        final_content = f_in.read()
                    
                    # Clean up successful stream's temp file
                    if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
                        os.remove(temp_file_path_for_stream)
                        temp_file_path_for_stream = None
                    
                    return final_content

                else: # Not streaming to disk
                    payload["stream"] = False
                    async with httpx.AsyncClient(timeout=600) as client:
                        api_response = await client.post(f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers)
                        api_response.raise_for_status() 
                        response_data = api_response.json()
                        
                        # Process non-streamed response
                        raw_text = ""
                        if response_data.get("choices") and len(response_data["choices"]) > 0:
                            message = response_data["choices"][0].get("message")
                            if message and message.get("content"):
                                raw_text = message["content"]
                        else:
                            logger.error(f"Async LLM ('{current_model_to_try}') Invalid response - missing choices: {response_data}")
                        
                        _log_llm_usage(current_model_to_try, response_data.get("usage"), async_mode=True, streamed=False)
                        return raw_text

            except httpx.TimeoutException as e_timeout:
                last_exception_for_current_model = e_timeout
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): API request timed out: {e_timeout}")
            except httpx.HTTPStatusError as e_status:
                last_exception_for_current_model = e_status
                logger.warning(
                    f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): API HTTP status error: {e_status}. "
                    f"Status: {e_status.response.status_code}, Body: {e_status.response.text[:200]}"
                )
                if 400 <= e_status.response.status_code < 500: # Client-side error
                    logger.error(f"Async LLM ('{current_model_to_try}'): Client-side error {e_status.response.status_code}. Aborting retries for this model.")
                    break # Break from retry loop for this model
            except httpx.RequestError as e_req:
                last_exception_for_current_model = e_req
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): API request error: {e_req}")
            except json.JSONDecodeError as e_json: # Could happen if stream data is malformed or non-streamed response isn't JSON
                last_exception_for_current_model = e_json
                response_text_snippet = ""
                if api_response and hasattr(api_response, 'text'): response_text_snippet = api_response.text[:200]
                elif response and hasattr(response, 'text'): response_text_snippet = response.text[:200] # For stream response object
                logger.warning(
                    f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): Failed to decode JSON: {e_json}. "
                    f"Response text snippet: {response_text_snippet}"
                )
            except Exception as e_exc:
                last_exception_for_current_model = e_exc
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): Unexpected error: {e_exc}", exc_info=True)
            
            # Cleanup temp file if stream attempt failed
            if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
                try:
                    os.remove(temp_file_path_for_stream)
                except Exception as e_clean:
                    logger.error(f"Error cleaning up temp file {temp_file_path_for_stream} after failed attempt: {e_clean}")
                temp_file_path_for_stream = None # Reset for next retry

            if retry_attempt < config.LLM_RETRY_ATTEMPTS - 1:
                delay = config.LLM_RETRY_DELAY_SECONDS * (2 ** retry_attempt)
                logger.info(f"Async LLM ('{current_model_to_try}'): Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Async LLM ('{current_model_to_try}'): All {config.LLM_RETRY_ATTEMPTS} retries failed. Last error: {last_exception_for_current_model}")
        
        if last_exception_for_current_model and not (isinstance(last_exception_for_current_model, httpx.HTTPStatusError) and 400 <= last_exception_for_current_model.response.status_code < 500):
            is_fallback_attempt = True 
        else: 
            break 

    logger.error(f"Async LLM: Call failed for '{model_name}' after all primary and potential fallback attempts.")
    # Ensure any lingering temp file from a failed final attempt is cleaned
    if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
        try:
            os.remove(temp_file_path_for_stream)
        except Exception as e_final_clean:
            logger.error(f"Error cleaning up temp file {temp_file_path_for_stream} after all attempts failed: {e_final_clean}")
    return ""


def clean_model_response(text: str) -> str:
    """Cleans common artifacts from LLM text responses."""
    if not isinstance(text, str):
        logger.warning(f"clean_model_response received non-string input: {type(text)}. Returning empty string.")
        return ""
    
    original_length = len(text)
    cleaned_text = text

    leading_think_artifact_regex = r'^\s*<\s*(think|thought|thinking)\s*>\s*(?:<\s*/\s*\1\s*>)?\s*'
    trailing_think_artifact_regex = r'\s*(?:<\s*(think|thought|thinking)\s*>)?\s*<\s*/\s*\1\s*>\s*$'
    
    while True:
        prev_len = len(cleaned_text)
        cleaned_text = re.sub(leading_think_artifact_regex, '', cleaned_text, count=1, flags=re.IGNORECASE)
        cleaned_text = re.sub(trailing_think_artifact_regex, '', cleaned_text, count=1, flags=re.IGNORECASE)
        if len(cleaned_text) == prev_len:
            break
            
    think_block_pattern = re.compile(
        r'<\s*(think|thought|thinking)\s*>.*?<\s*/\s*\1\s*>',
        flags=re.DOTALL | re.IGNORECASE
    )
    cleaned_text = think_block_pattern.sub('', cleaned_text)
    
    cleaned_text = re.sub(r'```(?:json|python|text|)\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-—]?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*[-=]{3,}\s*(BEGIN|END)\s+(CHAPTER|TEXT|DRAFT|CONTEXT|SNIPPET|JSON|OUTPUT)\b.*?[-=]{3,}\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    lines = cleaned_text.splitlines()
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        processed_lines.append(re.sub(r'[ \t]+', ' ', stripped_line) if stripped_line else '')
        
    final_text = '\n'.join(processed_lines).strip()

    if original_length > 0 and len(final_text) < original_length * 0.95 : 
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
    
    markdown_pattern = re.compile(rf'```(?:json)?\s*({start_char}.*?{end_char})\s*```', re.DOTALL | re.IGNORECASE)
    match_markdown = markdown_pattern.search(text)
    if match_markdown:
        json_str = match_markdown.group(1).strip()
        if json_str.startswith(start_char) and json_str.endswith(end_char):
            logger.debug(f"Potential JSON {expect_type.__name__} found in Markdown code block.")
            return json_str

    first_start_index = text.find(start_char)
    if first_start_index != -1:
        balance = 0
        potential_end_index = -1
        for i in range(first_start_index, len(text)):
            if text[i] == start_char:
                balance += 1
            elif text[i] == end_char:
                balance -= 1
            if balance == 0 and text[i] == end_char : 
                potential_end_index = i
                break
        
        if potential_end_index != -1:
            potential_json = text[first_start_index : potential_end_index + 1].strip()
            if potential_json.startswith(start_char) and potential_json.endswith(end_char):
                try:
                    # Validate if it's actual JSON before returning
                    parsed_val = json.loads(potential_json)
                    if isinstance(parsed_val, expect_type): # Check if the parsed type matches expected
                        logger.debug(f"Potential JSON {expect_type.__name__} found by brace/bracket matching with balance check and validation.")
                        return potential_json
                    else:
                        logger.debug(f"String found by brace/bracket balance is valid JSON but not expected type {expect_type.__name__}. Got {type(parsed_val)}.")
                except json.JSONDecodeError:
                    logger.debug(f"String found by brace/bracket balance check is not valid JSON: '{potential_json[:100]}...'")

    cleaned_full_response = clean_model_response(text) 
    if cleaned_full_response.startswith(start_char) and cleaned_full_response.endswith(end_char):
        try:
            parsed_val = json.loads(cleaned_full_response) 
            if isinstance(parsed_val, expect_type):
                logger.debug(f"Entire cleaned response appears to be valid JSON {expect_type.__name__}.")
                return cleaned_full_response
            else:
                logger.debug(f"Entire cleaned response is valid JSON but not expected type {expect_type.__name__}. Got {type(parsed_val)}.")
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
                # Fall through to LLM correction even if it's valid JSON but wrong type
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed for {context_for_log}: {e}. Extracted: '{json_block_str[:100]}...'")
            # Fall through to LLM correction
            
            fixed_json_str = None
            # Attempt simple heuristic fixes before full LLM correction
            if e.msg.startswith("Expecting ',' delimiter"): # Trailing comma often the issue
                # More robustly find last comma before closing bracket/brace if possible
                if json_block_str.rstrip().endswith((',', chr(0x2c))):
                    temp_str = json_block_str.rstrip()
                    if temp_str.endswith(',') or temp_str.endswith(chr(0x2c)):
                        fixed_json_str = temp_str[:-1]
                        logger.info(f"Attempting heuristic fix for trailing comma in {context_for_log}.")
            elif expect_type == list and e.msg.startswith("Expecting value") and e.pos >= len(json_block_str) - 10 : # Often indicates truncated list
                 # check if it's just missing the closing bracket
                if not json_block_str.strip().endswith("]"):
                    fixed_json_str = json_block_str.strip() + "]"
                    logger.info(f"Attempting heuristic fix for truncated list in {context_for_log}: appending ']'")
            elif expect_type == dict and (e.msg.startswith("Expecting property name enclosed in double quotes") or e.msg.startswith("Expecting ':' delimiter")) and e.pos >= len(json_block_str) -10:
                if not json_block_str.strip().endswith("}"):
                    fixed_json_str = json_block_str.strip() + "}"
                    logger.info(f"Attempting heuristic fix for truncated object in {context_for_log}: appending '}}'")


            if fixed_json_str:
                try:
                    parsed_json = json.loads(fixed_json_str)
                    if isinstance(parsed_json, expect_type):
                        logger.info(f"Heuristically fixed and parsed JSON {expect_type.__name__} for {context_for_log}.")
                        return parsed_json
                except json.JSONDecodeError:
                    logger.warning(f"Heuristic fix failed for {context_for_log}. Proceeding to LLM correction...")
                    # json_block_str remains the originally extracted (or raw) block for LLM correction
    
    # If no block extracted, or initial parse (and heuristic fix) failed, use the raw_response (or cleaned version) for correction
    if not json_block_str:
        logger.error(f"Could not extract any JSON {expect_type.__name__} block from initial response for {context_for_log}. Raw: '{raw_response[:200]}...'")
        json_block_str_for_correction = clean_model_response(raw_response) 
        if not json_block_str_for_correction.strip(): # If cleaning results in empty, nothing to correct
            logger.error(f"No content (even after cleaning) to attempt LLM correction for {context_for_log}.")
            return None
    else: # A block was extracted but failed to parse or was wrong type
        json_block_str_for_correction = json_block_str

    
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
{json_block_str_for_correction}
```

Corrected JSON {expect_type.__name__} Output Only (must start with '{expected_start_char}' and end with '{expected_end_char}'):
"""
    
    corrected_raw = await async_call_llm( 
        model_name=config.JSON_CORRECTION_MODEL, 
        prompt=correction_prompt, 
        temperature=0.6, 
        max_tokens=config.MAX_GENERATION_TOKENS, # Max tokens for the correction output
        allow_fallback=False, # JSON correction is a utility, no need for expensive fallback
        stream_to_disk=False # Correction output should be small
    )
    
    if not corrected_raw:
        logger.error(f"LLM JSON correction attempt returned empty response for {context_for_log}.")
        return None

    # Try to extract JSON from the correction LLM's output
    corrected_json_block_str = extract_json_block(corrected_raw, expect_type)
    
    if not corrected_json_block_str:
        # If extract_json_block fails, it might be because the LLM *only* returned the JSON
        # without Markdown. So, clean_model_response might give the raw JSON.
        cleaned_correction_output = clean_model_response(corrected_raw)
        if cleaned_correction_output.startswith(expected_start_char) and cleaned_correction_output.endswith(expected_end_char):
            corrected_json_block_str = cleaned_correction_output
            logger.debug(f"Using cleaned full response from correction LLM for {context_for_log} as no block was extracted: '{corrected_json_block_str[:100]}...'")
        else:
            logger.error(f"Could not extract a JSON {expect_type.__name__} block from correction LLM output for {context_for_log}. Corrected raw: '{corrected_raw[:200]}...'")
            return None


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
        # This case should ideally be caught by the 'if not corrected_json_block_str' above.
        logger.error(f"No usable JSON content from LLM correction output for {context_for_log}. Corrected raw: '{corrected_raw[:200]}...'")
        return None