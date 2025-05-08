# llm_interface.py
"""
Handles all direct interactions with Large Language Models (LLMs)
and embedding models. Includes functions for API calls, response cleaning,
JSON extraction with retry logic, and embedding generation with caching.

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

import config

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=config.EMBEDDING_CACHE_SIZE)
def get_embedding(text: str) -> Optional[np.ndarray]:
    # Same as your last version
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

def call_llm(model_name: str, prompt: str, temperature: float = 0.6, max_tokens: Optional[int] = None) -> str:
    if not model_name: logger.error("call_llm: model_name is required."); return ""
    if not prompt or not isinstance(prompt, str): logger.error("call_llm empty/invalid prompt."); return ""
    
    effective_max_tokens = max_tokens if max_tokens is not None else config.MAX_GENERATION_TOKENS
    
    payload = {
        "model": model_name, 
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, 
        "temperature": temperature, 
        "top_p": 0.95, # Retaining top_p, adjust if needed per model
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

def clean_model_response(text: str) -> str:
    # Same as your last version
    if not isinstance(text, str): logger.warning(f"clean_model_response non-string input: {type(text)}."); return ""
    original_text_len = len(text); cleaned_text = text
    leading_no_think_artifact_regex = r'^\s*<\s*think\s*>\s*<\s*/think\s*>\s*'
    text_after_leading_removal = re.sub(leading_no_think_artifact_regex, '', cleaned_text, count=1, flags=re.IGNORECASE)
    if len(text_after_leading_removal) < len(cleaned_text): logger.debug("Removed specific leading /no_think artifact."); cleaned_text = text_after_leading_removal
    think_block_regex = r'<\s*(?:think|thought|thinking)\s*>.*?<\s*/\s*(?:think|thought|thinking)\s*>'
    text_after_general_think_removal = re.sub(think_block_regex, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    if len(text_after_general_think_removal) < len(cleaned_text): logger.debug("General think block regex removed content."); cleaned_text = text_after_general_think_removal
    elif not (len(text_after_leading_removal) < len(text)):
        start_tag_pattern = r'^\s*<\s*(?:think|thought|thinking)\s*>'
        start_match = re.search(start_tag_pattern, cleaned_text, re.IGNORECASE)
        if start_match:
            logger.warning(f"Think block regexes failed, but text starts with tag: {cleaned_text[:100]}...")
            end_tag_pattern = r'<\s*/\s*(?:think|thought|thinking)\s*>'
            end_match = re.search(end_tag_pattern, cleaned_text[start_match.end():], re.IGNORECASE)
            if end_match: cleaned_text = cleaned_text[start_match.end() + end_match.end():].strip(); logger.info("Applied fallback think block removal.")
            else: logger.warning("Fallback removal: Found start tag but no end tag. Stripping start tag only."); cleaned_text = re.sub(start_tag_pattern, '', cleaned_text, count=1, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r'^\s*(Okay, here.*?|Here\'s the chapter|Response:|Summary:|List inconsistencies below:|Revised chapter:|Analysis:|JSON Output:)\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE).strip()
    cleaned_text = re.sub(r'```(?:json|python|text|)\s*.*?\s*```', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*Chapter \d+\s*[:\-]?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'^\s*---?\s*(BEGIN|END) (CHAPTER|TEXT|DRAFT|CONTEXT|SNIPPET).*?\s*---?\s*$', '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    cleaned_text = re.sub(r'\s*Output ONLY the.*?text.*$','', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r'\s*/no_think\s*$', '', cleaned_text, flags=re.IGNORECASE)
    lines = cleaned_text.splitlines()
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line.strip()) for line in lines if line.strip()]
    final_text = re.sub(r'\n{2,}', '\n\n', '\n'.join(cleaned_lines)).strip()
    if len(final_text) < original_text_len * 0.95: logger.debug(f"Cleaning reduced text length from {original_text_len} to {len(final_text)}.")
    return final_text

def extract_json_block(text: str, expect_type: Union[Type[dict], Type[list]] = dict) -> Optional[str]:
    # Same as your last version
    if not isinstance(text, str): logger.warning("extract_json_block non-string input."); return None
    # logger.debug(f"Attempting to extract JSON block (expecting {expect_type})...")
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
            if expect_type == list and e.pos >= len(json_block_str) - 10: # Heuristic for list truncation
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
    
    corrected_raw = call_llm(
        model_name=config.JSON_CORRECTION_MODEL, 
        prompt=correction_prompt, 
        temperature=0.2, 
        max_tokens=config.MAX_GENERATION_TOKENS # Use general max tokens, or a specific one for corrections
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
