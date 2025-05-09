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

import requests
import json
import re
import numpy as np
import logging
import functools
from typing import Optional, Dict, Any, List, Union, Type 

import httpx # For asynchronous HTTP requests

import config

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
def get_embedding(text: str) -> Optional[np.ndarray]:
    if not text or not isinstance(text, str) or len(text.strip()) == 0: logger.warning("get_embedding empty/invalid text."); return None
    payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()}
    # cache_info = get_embedding.cache_info(); logger.debug(f"Embedding req: '{text[:80]}...' (Cache: h={cache_info.hits},m={cache_info.misses},s={cache_info.currsize})")
    try:
        response = requests.post(f"{config.OLLAMA_EMBED_URL}/api/embeddings", json=payload, timeout=300); response.raise_for_status()
        data = response.json(); embedding_key = "embedding"
        if embedding_key in data and isinstance(data[embedding_key], list):
            try:
                embedding = np.array(data[embedding_key], dtype=config.EMBEDDING_DTYPE)
                if embedding.ndim > 1: embedding = embedding.flatten()
                if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,): return embedding
                else: logger.error(f"Primary embedding dim mismatch: Expected ({config.EXPECTED_EMBEDDING_DIM},), Got {embedding.shape}")
            except (TypeError, ValueError) as e: logger.error(f"Failed to convert primary embedding: {e}.")
        logger.warning(f"Primary key '{embedding_key}' failed. Fallback search...")
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and all(isinstance(item, (float, int)) for item in value):
                try:
                    embedding = np.array(value, dtype=config.EMBEDDING_DTYPE)
                    if embedding.ndim > 1: embedding = embedding.flatten()
                    if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,): logger.info(f"Fallback embedding success (key: '{key}')."); return embedding
                    else: logger.error(f"Fallback dim mismatch: Key '{key}', Expected ({config.EXPECTED_EMBEDDING_DIM},), Got {embedding.shape}")
                except (TypeError, ValueError) as e: logger.error(f"Failed convert fallback embedding for key '{key}': {e}")
        logger.error("Embedding extraction failed."); return None
    except requests.exceptions.Timeout: logger.error(f"Embedding request timed out: '{text[:80]}...'")
    except requests.exceptions.RequestException as e: logger.error(f"Embedding request failed: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error during embedding: {e}", exc_info=True)
    return None

# Asynchronous version of get_embedding
@functools.lru_cache(maxsize=config.EMBEDDING_CACHE_SIZE) # Note: LRU cache on async func needs care with event loops if used across different ones. For simple cases, it's fine.
async def async_get_embedding(text: str) -> Optional[np.ndarray]:
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        logger.warning("async_get_embedding empty/invalid text.")
        return None
    payload = {"model": config.EMBEDDING_MODEL, "prompt": text.strip()}
    # cache_info = async_get_embedding.cache_info() # Access cache info if needed
    # logger.debug(f"Async Embedding req: '{text[:80]}...' (Cache: h={cache_info.hits},m={cache_info.misses},s={cache_info.currsize})")
    
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.post(f"{config.OLLAMA_EMBED_URL}/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
            embedding_key = "embedding"
            if embedding_key in data and isinstance(data[embedding_key], list):
                try:
                    embedding = np.array(data[embedding_key], dtype=config.EMBEDDING_DTYPE)
                    if embedding.ndim > 1: embedding = embedding.flatten()
                    if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,): return embedding
                    else: logger.error(f"Async Primary embedding dim mismatch: Expected ({config.EXPECTED_EMBEDDING_DIM},), Got {embedding.shape}")
                except (TypeError, ValueError) as e: logger.error(f"Async Failed to convert primary embedding: {e}.")
            
            logger.warning(f"Async Primary key '{embedding_key}' failed. Fallback search...")
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and all(isinstance(item, (float, int)) for item in value):
                    try:
                        embedding = np.array(value, dtype=config.EMBEDDING_DTYPE)
                        if embedding.ndim > 1: embedding = embedding.flatten()
                        if embedding.shape == (config.EXPECTED_EMBEDDING_DIM,):
                            logger.info(f"Async Fallback embedding success (key: '{key}').")
                            return embedding
                        else: logger.error(f"Async Fallback dim mismatch: Key '{key}', Expected ({config.EXPECTED_EMBEDDING_DIM},), Got {embedding.shape}")
                    except (TypeError, ValueError) as e: logger.error(f"Async Failed convert fallback embedding for key '{key}': {e}")
            
            logger.error("Async Embedding extraction failed.")
            return None
        except httpx.TimeoutException:
            logger.error(f"Async Embedding request timed out: '{text[:80]}...'")
        except httpx.RequestError as e:
            logger.error(f"Async Embedding request failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Async Unexpected error during embedding: {e}", exc_info=True)
        return None


def call_llm(model_name: str, prompt: str, temperature: float = 0.6, max_tokens: Optional[int] = None) -> str:
    if not model_name: logger.error("call_llm: model_name is required."); return ""
    if not prompt or not isinstance(prompt, str): logger.error("call_llm empty/invalid prompt."); return ""
    
    effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
    
    payload = {
        "model": model_name, 
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, 
        "temperature": temperature, 
        "top_p": 0.95,
        "max_tokens": effective_max_tokens
    }
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
    
    logger.debug(f"Calling LLM '{model_name}'. Prompt len: {len(prompt)}. Max tokens: {effective_max_tokens}. Temp: {temperature}")
    try:
        response = requests.post(f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        response_data = response.json()
        
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message")
            if message and message.get("content"):
                raw_text = message["content"]
                if 'usage' in response_data:
                    usage = response_data['usage']
                    logger.info(f"LLM ('{model_name}') Usage - Prompt: {usage.get('prompt_tokens', 'N/A')} tk, Comp: {usage.get('completion_tokens', 'N/A')} tk, Total: {usage.get('total_tokens', 'N/A')} tk")
                else: 
                    logger.warning(f"LLM ('{model_name}') response missing 'usage'.")
                return raw_text
            else: 
                logger.error(f"Invalid LLM ('{model_name}') response - missing message content: {response_data}")
        else: 
            logger.error(f"Invalid LLM ('{model_name}') response - missing choices: {response_data}")
            
    except requests.exceptions.Timeout: 
        logger.error(f"LLM ('{model_name}') API request timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM ('{model_name}') API request error: {e}", exc_info=True)
        if e.response is not None: 
            logger.error(f"LLM ('{model_name}') API Response Status: {e.response.status_code}, Body: {e.response.text[:500]}...")
    except Exception as e: 
        logger.error(f"Unexpected error during LLM ('{model_name}') call: {e}", exc_info=True)
    return ""

# Asynchronous version of call_llm
async def async_call_llm(model_name: str, prompt: str, temperature: float = 0.6, max_tokens: Optional[int] = None) -> str:
    if not model_name:
        logger.error("async_call_llm: model_name is required.")
        return ""
    if not prompt or not isinstance(prompt, str):
        logger.error("async_call_llm empty/invalid prompt.")
        return ""

    effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": effective_max_tokens
    }
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}

    logger.debug(f"Async Calling LLM '{model_name}'. Prompt len: {len(prompt)}. Max tokens: {effective_max_tokens}. Temp: {temperature}")
    
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            response = await client.post(f"{config.OPENAI_API_BASE}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message")
                if message and message.get("content"):
                    raw_text = message["content"]
                    if 'usage' in response_data:
                        usage = response_data['usage']
                        logger.info(f"Async LLM ('{model_name}') Usage - Prompt: {usage.get('prompt_tokens', 'N/A')} tk, Comp: {usage.get('completion_tokens', 'N/A')} tk, Total: {usage.get('total_tokens', 'N/A')} tk")
                    else:
                        logger.warning(f"Async LLM ('{model_name}') response missing 'usage'.")
                    return raw_text
                else:
                    logger.error(f"Async Invalid LLM ('{model_name}') response - missing message content: {response_data}")
            else:
                logger.error(f"Async Invalid LLM ('{model_name}') response - missing choices: {response_data}")

        except httpx.TimeoutException:
            logger.error(f"Async LLM ('{model_name}') API request timed out.")
        except httpx.RequestError as e:
            logger.error(f"Async LLM ('{model_name}') API request error: {e}", exc_info=True)
            if e.response is not None:
                logger.error(f"Async LLM ('{model_name}') API Response Status: {e.response.status_code}, Body: {e.response.text[:500]}...")
        except Exception as e:
            logger.error(f"Async Unexpected error during LLM ('{model_name}') call: {e}", exc_info=True)
    return ""


def clean_model_response(text: str) -> str:
    if not isinstance(text, str): 
        logger.warning(f"clean_model_response non-string input: {type(text)}."); return ""
    
    original_text_len = len(text)
    cleaned_text = text

    leading_no_think_artifact_regex = r'^\s*<\s*think\s*>\s*<\s*/think\s*>\s*'
    text_after_leading_removal = re.sub(leading_no_think_artifact_regex, '', cleaned_text, count=1, flags=re.IGNORECASE)
    if len(text_after_leading_removal) < len(cleaned_text):
        logger.debug("Removed specific leading empty <think></think> artifact.")
        cleaned_text = text_after_leading_removal

    think_block_pattern = re.compile(
        r'<\s*(think|thought|thinking)\s*>.*?<\s*/\s*\1\s*>',
        flags=re.DOTALL | re.IGNORECASE
    )
    text_after_general_think_removal = think_block_pattern.sub('', cleaned_text)
    if len(text_after_general_think_removal) < len(cleaned_text):
        if not (len(text_after_leading_removal) < original_text_len and cleaned_text == text_after_leading_removal):
             logger.debug("General think block regex removed content.")
        cleaned_text = text_after_general_think_removal
    
    cleaned_text = re.sub(r'```(?:json|python|text|)\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-]?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*---?\s*(BEGIN|END) (CHAPTER|TEXT|DRAFT|CONTEXT|SNIPPET).*?\s*---?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text_normalized_newlines = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    lines = cleaned_text_normalized_newlines.splitlines()
    processed_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            processed_lines.append(re.sub(r'[ \t]+', ' ', stripped_line))
        else: 
            processed_lines.append('')
    final_text = '\n'.join(processed_lines).strip()

    if len(final_text) < original_text_len * 0.95 and original_text_len > 0 :
        reduction_percentage = ((original_text_len - len(final_text)) / original_text_len) * 100
        logger.debug(f"Cleaning reduced text length from {original_text_len} to {len(final_text)} ({reduction_percentage:.1f}% reduction).")
    
    return final_text

def extract_json_block(text: str, expect_type: Union[Type[dict], Type[list]] = dict) -> Optional[str]:
    if not isinstance(text, str): logger.warning("extract_json_block non-string input."); return None
    start_char, end_char = ('{', '}') if expect_type == dict else ('[', ']')
    markdown_pattern = rf'```(?:json)?\s*(\{start_char}.*?\{end_char})\s*```'
    match_markdown = re.search(markdown_pattern, text, re.DOTALL | re.IGNORECASE)
    if match_markdown:
        json_str = match_markdown.group(1).strip()
        if json_str.startswith(start_char) and json_str.endswith(end_char):
            logger.debug(f"Potential JSON {expect_type.__name__} found in Markdown."); return json_str
    start_index, end_index = text.find(start_char), text.rfind(end_char)
    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = text[start_index : end_index + 1].strip()
        if potential_json.startswith(start_char) and potential_json.endswith(end_char):
            logger.debug(f"Potential JSON {expect_type.__name__} found by brace/bracket matching."); return potential_json
    cleaned_full = clean_model_response(text) 
    if cleaned_full.startswith(start_char) and cleaned_full.endswith(end_char):
        logger.debug(f"Entire cleaned response appears to be JSON {expect_type.__name__}."); return cleaned_full
    logger.warning(f"Failed to extract likely JSON {expect_type.__name__} block: '{text[:100]}...'")
    return None

def parse_llm_json_response(raw_response: str, context_for_log: str, expect_type: Union[Type[dict], Type[list]] = dict) -> Optional[Union[Dict[str, Any], List[Any]]]:
    if not raw_response: logger.warning(f"LLM empty response for {context_for_log}. Cannot parse JSON."); return None
    json_block_str = extract_json_block(raw_response, expect_type)
    parsed_json = None
    if json_block_str:
        try:
            parsed_json = json.loads(json_block_str)
            if isinstance(parsed_json, expect_type):
                logger.info(f"Successfully parsed JSON {expect_type.__name__} for {context_for_log} on 1st attempt."); return parsed_json
            else: logger.warning(f"Parsed JSON for {context_for_log} is {type(parsed_json)}, expected {expect_type.__name__}. Retrying.")
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed for {context_for_log}: {e}. Extracted: '{json_block_str[:100]}...'")
            if expect_type == list and e.pos >= len(json_block_str) - 10: 
                logger.info(f"Attempting heuristic fix for truncated list in {context_for_log}: appending ']'")
                try:
                    parsed_json = json.loads(json_block_str + "]")
                    if isinstance(parsed_json, list): logger.info(f"Parsed JSON list for {context_for_log} with heuristic ']' append."); return parsed_json
                except json.JSONDecodeError: logger.warning(f"Heuristic ']' append failed for {context_for_log}. LLM correction...")
    else:
        logger.error(f"Could not extract JSON {expect_type.__name__} block from initial response for {context_for_log}. Raw: '{raw_response[:200]}...'")
        json_block_str = clean_model_response(raw_response) 
    if not json_block_str: logger.error(f"No content for LLM correction for {context_for_log}."); return None
    
    logger.info(f"Attempting LLM JSON correction for {context_for_log} using model {config.JSON_CORRECTION_MODEL}...")
    correction_prompt = f"""The following text block was extracted, but it contains JSON syntax errors or is not the expected type. Correct syntax errors (e.g., missing commas, incorrect quoting, trailing commas, unbalanced braces/brackets) and ensure it's a valid JSON {expect_type.__name__}. Output ONLY the corrected, valid JSON. No explanation/commentary.
    Invalid/Problematic JSON Block:\n```json\n{json_block_str}\n```
    Corrected JSON {expect_type.__name__} Output Only:\n/no_think\n"""
    
    # For parse_llm_json_response, it's typically called from within other methods.
    # If those parent methods become async, this call_llm would need to be async_call_llm.
    # For now, keeping it synchronous as its direct callers are mostly synchronous or becoming async.
    # A full async conversion would make this an async function too.
    corrected_raw = call_llm( # If this function is called from an async context, this should be `await async_call_llm`
        model_name=config.JSON_CORRECTION_MODEL, 
        prompt=correction_prompt, 
        temperature=0.2, 
        max_tokens=config.MAX_GENERATION_TOKENS
    )
    
    if not corrected_raw: logger.error(f"LLM correction attempt empty response for {context_for_log}."); return None
    corrected_cleaned = clean_model_response(corrected_raw)
    corrected_block_str_retry = extract_json_block(corrected_cleaned, expect_type) or corrected_cleaned
    
    if corrected_block_str_retry:
        try:
            parsed_corrected_json = json.loads(corrected_block_str_retry)
            if isinstance(parsed_corrected_json, expect_type):
                logger.info(f"Successfully parsed JSON {expect_type.__name__} for {context_for_log} after LLM correction."); return parsed_corrected_json
            else: logger.error(f"Corrected JSON for {context_for_log} is {type(parsed_corrected_json)}, expected {expect_type.__name__}."); return None
        except json.JSONDecodeError as e_retry:
            logger.error(f"JSON parsing failed AGAIN for {context_for_log} after LLM correction: {e_retry}. Corrected: '{corrected_block_str_retry[:200]}...'"); return None
    else: logger.error(f"Could not extract JSON {expect_type.__name__} block from LLM correction for {context_for_log}."); return None
