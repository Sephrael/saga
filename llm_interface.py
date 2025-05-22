# llm_interface.py
"""
Handles all direct interactions with Large Language Models (LLMs)
and embedding models. Includes functions for API calls, response cleaning,
and embedding generation with caching.
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
import json # Still needed for payload construction, not for parsing LLM responses
import re
import asyncio # Added for sleep in retry logic
import time # For exponential backoff
import tempfile # For streaming to disk
import os # For temp file cleanup

# Third-party imports
import numpy as np
import httpx  # For asynchronous HTTP requests
from async_lru import alru_cache  # Import async-aware LRU cache
import tiktoken # For token counting

# Type hints
from typing import List, Optional, Dict, Any, Union, Type, Tuple
# from type import JsonType # JsonType might be less relevant if we don't parse generic JSON from LLMs

# Local imports
import config

logger = logging.getLogger(__name__)

# --- Tokenizer Cache ---
_tokenizer_cache: Dict[str, tiktoken.Encoding] = {}

@functools.lru_cache(maxsize=config.TOKENIZER_CACHE_SIZE)
def _get_tokenizer(model_name: str) -> Optional[tiktoken.Encoding]:
    """
    Gets a tiktoken encoder for the given model name, with caching.
    Tries model-specific encoding, then a default, then returns None.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        encoder = tiktoken.encoding_for_model(model_name)
        _tokenizer_cache[model_name] = encoder
        logger.debug(f"Tokenizer for model '{model_name}' found and cached.")
        return encoder
    except KeyError:
        logger.warning(
            f"No direct tiktoken encoding found for model '{model_name}'. "
            f"Attempting default encoding '{config.TIKTOKEN_DEFAULT_ENCODING}'."
        )
        try:
            encoder = tiktoken.get_encoding(config.TIKTOKEN_DEFAULT_ENCODING)
            _tokenizer_cache[model_name] = encoder # Cache under original model name
            logger.info(
                f"Using default tiktoken encoding '{config.TIKTOKEN_DEFAULT_ENCODING}' "
                f"for model '{model_name}' and cached."
            )
            return encoder
        except KeyError:
            logger.error(
                f"Default tiktoken encoding '{config.TIKTOKEN_DEFAULT_ENCODING}' also not found. "
                f"Token counting will fall back to character-based heuristic for '{model_name}'."
            )
            return None
    except Exception as e:
        logger.error(f"Unexpected error getting tokenizer for '{model_name}': {e}", exc_info=True)
        return None

def count_tokens(text: str, model_name: str) -> int:
    """
    Counts the number of tokens in a string for a given model.
    Uses tiktoken with caching and fallbacks.
    """
    if not text:
        return 0
    encoder = _get_tokenizer(model_name)
    if encoder:
        return len(encoder.encode(text, allowed_special="all")) # allowed_special avoids errors on special tokens
    else:
        # Fallback to character-based heuristic if tokenizer failed
        char_count = len(text)
        token_estimate = int(char_count / config.FALLBACK_CHARS_PER_TOKEN)
        logger.warning(
            f"count_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based estimate: {char_count} chars -> ~{token_estimate} tokens."
        )
        return token_estimate

def truncate_text_by_tokens(text: str, model_name: str, max_tokens: int, truncation_marker: str = "\n... (truncated)") -> str:
    """
    Truncates text to a maximum number of tokens for a given model.
    Adds a truncation marker if truncation occurs.
    """
    if not text:
        return ""

    encoder = _get_tokenizer(model_name)
    if not encoder:
        # Fallback to character-based truncation if tokenizer failed
        max_chars = int(max_tokens * config.FALLBACK_CHARS_PER_TOKEN)
        logger.warning(
            f"truncate_text_by_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based truncation: {max_tokens} tokens -> ~{max_chars} chars."
        )
        if len(text) > max_chars:
            return text[:max_chars - len(truncation_marker)] + truncation_marker
        return text

    tokens = encoder.encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    marker_tokens_len = len(encoder.encode(truncation_marker, allowed_special="all"))
    
    # Ensure space for the truncation marker
    if max_tokens > marker_tokens_len and len(truncated_tokens) + marker_tokens_len > max_tokens :
        truncated_tokens = tokens[:max_tokens - marker_tokens_len]
    elif max_tokens <= marker_tokens_len: # Not enough space for marker, just truncate hard
        truncated_tokens = tokens[:max_tokens]
        if not truncated_tokens: return "" # Cannot even fit one token
        try: return encoder.decode(truncated_tokens) # No marker if no space
        except: pass # Fallthrough to simpler truncation if decode fails


    try:
        decoded_text = encoder.decode(truncated_tokens)
        return decoded_text + truncation_marker
    except Exception as e:
        logger.error(f"Error decoding truncated tokens for model '{model_name}': {e}. Falling back to simpler truncation.", exc_info=True)
        # Fallback to character-based estimation for truncation point
        # This is a rough estimate
        avg_chars_per_token = len(text) / len(tokens) if len(tokens) > 0 else config.FALLBACK_CHARS_PER_TOKEN
        estimated_char_limit_for_content = int((max_tokens - marker_tokens_len) * avg_chars_per_token)
        if estimated_char_limit_for_content < 0 : estimated_char_limit_for_content = 0
        return text[:estimated_char_limit_for_content] + truncation_marker


def _validate_embedding(embedding_list: List[Union[float, int]], expected_dim: int, dtype: np.dtype) -> Optional[np.ndarray]:
    """Helper to validate and convert a list to a 1D numpy embedding."""
    try:
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
            async with httpx.AsyncClient(timeout=300) as client: 
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
            last_exception = ValueError("No suitable embedding list found in response.")

        except httpx.TimeoutException as e_timeout:
            last_exception = e_timeout
            logger.warning(f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Request timed out: {e_timeout}")
        except httpx.HTTPStatusError as e_status:
            last_exception = e_status
            logger.warning(
                f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): HTTP status {e_status.response.status_code}: {e_status}. "
                f"Body: {e_status.response.text[:200]}"
            )
            if 400 <= e_status.response.status_code < 500: 
                logger.error(f"Async Embedding: Client-side error {e_status.response.status_code}. Aborting retries.")
                break
        except httpx.RequestError as e_req:
            last_exception = e_req
            logger.warning(f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Request error: {e_req}")
        except json.JSONDecodeError as e_json: # For parsing the embedding API response
            last_exception = e_json
            logger.warning(
                 f"Async Embedding (Attempt {attempt+1}/{config.LLM_RETRY_ATTEMPTS}): Failed to decode JSON response from embedding API: {e_json}. "
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
    return None


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
    stream_to_disk: bool = False
) -> str:
    """
    Asynchronously calls the LLM with retry and optional model fallback.
    Returns the LLM's text response as a string.
    """
    if not model_name:
        logger.error("async_call_llm: model_name is required.")
        return ""
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        logger.error("async_call_llm: empty or invalid prompt.")
        return ""

    prompt_token_count = count_tokens(prompt, model_name)
    effective_max_output_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}

    current_model_to_try = model_name
    is_fallback_attempt = False

    for attempt_num_overall in range(2): 
        if is_fallback_attempt:
            if not allow_fallback or not config.FALLBACK_GENERATION_MODEL:
                logger.warning(f"Primary model '{model_name}' failed. Fallback not allowed or not configured. Aborting.")
                return ""
            current_model_to_try = config.FALLBACK_GENERATION_MODEL
            logger.info(f"Primary model '{model_name}' failed. Attempting fallback with '{current_model_to_try}'.")
            prompt_token_count = count_tokens(prompt, current_model_to_try) # Recalculate for fallback model


        payload = {
            "model": current_model_to_try,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": config.LLM_TOP_P,
            "max_tokens": effective_max_output_tokens
        }

        last_exception_for_current_model = None
        temp_file_path_for_stream: Optional[str] = None

        for retry_attempt in range(config.LLM_RETRY_ATTEMPTS):
            logger.debug(
                f"Async Calling LLM '{current_model_to_try}' (Attempt {retry_attempt+1}/{config.LLM_RETRY_ATTEMPTS}, Fallback: {is_fallback_attempt}). "
                f"StreamToDisk: {stream_to_disk}. Prompt tokens (est.): {prompt_token_count}. Max output tokens: {effective_max_output_tokens}. Temp: {temperature}, TopP: {config.LLM_TOP_P}"
            )
            api_response_obj: Optional[httpx.Response] = None 

            try:
                if stream_to_disk:
                    payload["stream"] = True
                    # Create a temporary file to stream the response to disk
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".llmstream.txt") as tmp_f:
                        temp_file_path_for_stream = tmp_f.name
                    
                    usage_info_from_stream: Optional[Dict[str, int]] = None

                    async with httpx.AsyncClient(timeout=600) as client: # Increased timeout for streaming
                        async with client.stream("POST", f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers) as response_stream:
                            response_stream.raise_for_status() # Check for initial HTTP errors

                            with open(temp_file_path_for_stream, "w", encoding="utf-8") as f_out:
                                async for line in response_stream.aiter_lines():
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
                                                
                                                # Check for finish_reason and usage (some models send it with the last content chunk)
                                                if chunk_data["choices"][0].get("finish_reason") is not None:
                                                    potential_usage = chunk_data.get("usage")
                                                    if potential_usage and isinstance(potential_usage, dict):
                                                        usage_info_from_stream = potential_usage
                                                    elif chunk_data.get("x_groq") and chunk_data["x_groq"].get("usage"): # Specific to Groq
                                                        usage_info_from_stream = chunk_data["x_groq"]["usage"]
                                                    # Don't break here if content might still be coming, only on [DONE]
                                        except json.JSONDecodeError:
                                            logger.warning(f"Async LLM Stream: Could not decode JSON from line: {line}")
                    
                    _log_llm_usage(current_model_to_try, usage_info_from_stream, async_mode=True, streamed=True)

                    # Read the complete content from the temporary file
                    with open(temp_file_path_for_stream, "r", encoding="utf-8") as f_in:
                        final_content = f_in.read()
                    
                    if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
                        os.remove(temp_file_path_for_stream)
                        temp_file_path_for_stream = None
                    return final_content

                else: # Not streaming to disk
                    payload["stream"] = False
                    async with httpx.AsyncClient(timeout=600) as client: # Increased timeout
                        api_response_obj = await client.post(f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers)
                        api_response_obj.raise_for_status()
                        response_data = api_response_obj.json()

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
                if 400 <= e_status.response.status_code < 500:
                    if e_status.response.status_code == 400 and "context_length_exceeded" in e_status.response.text.lower():
                         logger.error(
                             f"Async LLM ('{current_model_to_try}'): Context length exceeded. Prompt tokens (est.): {prompt_token_count}. "
                             f"Model's limit might be smaller than estimated or prompt construction needs review."
                         )
                    else: # Other client-side errors
                        logger.error(f"Async LLM ('{current_model_to_try}'): Client-side error {e_status.response.status_code}. Aborting retries for this model.")
                    break # Break retry loop for this model on client errors
            except httpx.RequestError as e_req:
                last_exception_for_current_model = e_req
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): API request error: {e_req}")
            except json.JSONDecodeError as e_json: 
                last_exception_for_current_model = e_json
                response_text_snippet = ""
                if api_response_obj and hasattr(api_response_obj, 'text'): response_text_snippet = api_response_obj.text[:200]
                
                logger.warning(
                    f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): Failed to decode API JSON: {e_json}. "
                    f"Response text snippet: {response_text_snippet}"
                )
            except Exception as e_exc:
                last_exception_for_current_model = e_exc
                logger.warning(f"Async LLM ('{current_model_to_try}' Attempt {retry_attempt+1}): Unexpected error: {e_exc}", exc_info=True)

            # Clean up temp file if stream_to_disk was used and an error occurred
            if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
                try: os.remove(temp_file_path_for_stream)
                except Exception as e_clean: logger.error(f"Error cleaning up temp file {temp_file_path_for_stream} after failed attempt: {e_clean}")
                temp_file_path_for_stream = None

            if retry_attempt < config.LLM_RETRY_ATTEMPTS - 1:
                delay = config.LLM_RETRY_DELAY_SECONDS * (2 ** retry_attempt)
                logger.info(f"Async LLM ('{current_model_to_try}'): Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else: # All retries for the current model failed
                logger.error(f"Async LLM ('{current_model_to_try}'): All {config.LLM_RETRY_ATTEMPTS} retries failed. Last error: {last_exception_for_current_model}")
        
        # Logic to trigger fallback or break from overall attempts
        if last_exception_for_current_model and not (isinstance(last_exception_for_current_model, httpx.HTTPStatusError) and 400 <= last_exception_for_current_model.response.status_code < 500):
            # If the failure was not a non-retryable client error, set up for fallback
            is_fallback_attempt = True 
        else:
            # If successful, or a non-retryable client error occurred, break the outer loop
            break 

    logger.error(f"Async LLM: Call failed for '{model_name}' after all primary and potential fallback attempts.")
    # Final cleanup for temp file if it somehow persisted
    if temp_file_path_for_stream and os.path.exists(temp_file_path_for_stream):
        try: os.remove(temp_file_path_for_stream)
        except Exception as e_final_clean: logger.error(f"Error cleaning up temp file {temp_file_path_for_stream} after all attempts failed: {e_final_clean}")
    return ""


def clean_model_response(text: str) -> str:
    """Cleans common artifacts from LLM text responses, including stray <think> tags."""
    if not isinstance(text, str):
        logger.warning(f"clean_model_response received non-string input: {type(text)}. Returning empty string.")
        return ""

    original_length = len(text)
    cleaned_text = text

    # 1. Remove any complete <think>...</think> blocks anywhere in the text first.
    think_block_pattern = re.compile(
        r'<\s*(think|thought|thinking)\s*>.*?<\s*/\s*\1\s*>',
        flags=re.DOTALL | re.IGNORECASE
    )
    cleaned_text = think_block_pattern.sub('', cleaned_text)

    # 2. Iteratively remove leading/trailing artifacts including unmatched/lone tags.
    leading_stray_closing_tag = re.compile(r"^\s*<\s*/\s*(think|thought|thinking)\s*>\s*", flags=re.IGNORECASE)
    leading_stray_opening_tag = re.compile(r"^\s*<\s*(think|thought|thinking)\s*>\s*", flags=re.IGNORECASE)
    
    trailing_stray_opening_tag = re.compile(r"\s*<\s*(think|thought|thinking)\s*>\s*$", flags=re.IGNORECASE)
    trailing_stray_closing_tag = re.compile(r"\s*<\s*/\s*(think|thought|thinking)\s*>\s*$", flags=re.IGNORECASE)
    
    for _ in range(5): # Limit iterations
        prev_len = len(cleaned_text)
        
        # Apply leading patterns
        cleaned_text = leading_stray_closing_tag.sub('', cleaned_text, count=1)
        cleaned_text = leading_stray_opening_tag.sub('', cleaned_text, count=1)
            
        # Apply trailing patterns
        cleaned_text = trailing_stray_opening_tag.sub('', cleaned_text, count=1)
        cleaned_text = trailing_stray_closing_tag.sub('', cleaned_text, count=1)
            
        if len(cleaned_text) == prev_len: # No change in this iteration, break
            break
            
    # 3. Remove any remaining internal stray tags as a final cleanup.
    # These are unanchored and will remove any occurrence.
    stray_opening_tag_pattern_anywhere = re.compile(r"<\s*(think|thought|thinking)\s*>", flags=re.IGNORECASE)
    stray_closing_tag_pattern_anywhere = re.compile(r"<\s*/\s*(think|thought|thinking)\s*>", flags=re.IGNORECASE)
    cleaned_text = stray_opening_tag_pattern_anywhere.sub('', cleaned_text)
    cleaned_text = stray_closing_tag_pattern_anywhere.sub('', cleaned_text)

    # Remove common markdown code blocks 
    cleaned_text = re.sub(r'```(?:json|python|text|yaml|markdown|)\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    # Remove "Chapter X" or similar headers if they are the primary content of a line.
    # This is less aggressive than removing any line starting with "Title:".
    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-—]?\s*(.*?)\s*$', r'\1', cleaned_text, flags=re.MULTILINE | re.IGNORECASE).strip()
    # The following line was too aggressive and removed "Title:" data fields needed for initial setup.
    # cleaned_text = re.sub(r'^\s*Title:\s*.*?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE).strip()


    # Remove common preamble/postamble
    common_phrases_patterns = [
        r"^\s*Here's the.*?:\s*", r"^\s*Okay, here is the.*?:\s*", r"^\s*Sure, here's the.*?:\s*",
        r"^\s*I've written the.*?as requested:\s*",
        r"\s*Let me know if you have any other questions or need further revisions\.[^\w\n]*$", # Ensure it's at the end
        r"\s*I hope this meets your expectations\.[^\w\n]*$",
        r"\s*Feel free to ask for adjustments\.[^\w\n]*$",
        r"^\s*Output:\s*", r"^\s*Result:\s*", r"^\s*Response:\s*",
    ]
    for pattern_str in common_phrases_patterns:
        cleaned_text = re.sub(pattern_str, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Normalize multiple newlines to a maximum of two
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Trim leading/trailing whitespace from the whole text and from each line
    lines = cleaned_text.splitlines()
    processed_lines = [line.strip() for line in lines] 

    final_text = '\n'.join(processed_lines).strip()

    if original_length > 0 and len(final_text) < original_length: 
        reduction_percentage = ((original_length - len(final_text)) / original_length) * 100
        if reduction_percentage > 1.0: # Log only if reduction is somewhat significant
            logger.debug(f"Cleaning reduced text length from {original_length} to {len(final_text)} ({reduction_percentage:.1f}% reduction).")

    return final_text