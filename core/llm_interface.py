# core/llm_interface.py
"""
Handles all direct interactions with Large Language Models (LLMs)
and embedding models (via Ollama). Includes functions for API calls,
response cleaning, and embedding generation with caching.
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
import asyncio
import functools
import json
import os
import random
import re
import tempfile

# Type hints
from typing import Any

import httpx

# Third-party imports
import numpy as np
import structlog
import tiktoken
from async_lru import alru_cache

# Local imports
from config import settings

logger = structlog.get_logger(__name__)


# Token parameter handling
def _completion_token_param(api_base: str) -> str:
    """Return the token count parameter expected by the provider."""
    if "api.openai.com" in api_base or "api.anthropic.com" in api_base:
        return "max_completion_tokens"
    return "max_tokens"


# --- Tokenizer Cache and Utility Functions (Module Level) ---
_tokenizer_cache: dict[str, tiktoken.Encoding] = {}


@functools.lru_cache(maxsize=settings.TOKENIZER_CACHE_SIZE)
def _get_tokenizer(model_name: str) -> tiktoken.Encoding | None:
    """
    Gets a tiktoken encoder for the given model name, with caching.
    Tries model-specific encoding, then a default, then returns None.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        try:
            encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.debug(
                f"No direct tiktoken encoding for '{model_name}'. Using default '{settings.TIKTOKEN_DEFAULT_ENCODING}'."
            )
            encoder = tiktoken.get_encoding(settings.TIKTOKEN_DEFAULT_ENCODING)

        _tokenizer_cache[model_name] = encoder
        logger.debug(
            f"Tokenizer for model '{model_name}' (using actual encoder '{encoder.name}') found and cached."
        )
        return encoder
    except KeyError:
        logger.error(
            f"Default tiktoken encoding '{settings.TIKTOKEN_DEFAULT_ENCODING}' also not found. "
            f"Token counting will fall back to character-based heuristic for '{model_name}'."
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error getting tokenizer for '{model_name}': {e}",
            exc_info=True,
        )
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
        return len(encoder.encode(text, allowed_special="all"))
    else:
        char_count = len(text)
        token_estimate = int(char_count / settings.FALLBACK_CHARS_PER_TOKEN)
        logger.warning(
            f"count_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based estimate: {char_count} chars -> ~{token_estimate} tokens."
        )
        return token_estimate


def truncate_text_by_tokens(
    text: str,
    model_name: str,
    max_tokens: int,
    truncation_marker: str = "\n... (truncated)",
) -> str:
    """
    Truncates text to a maximum number of tokens for a given model.
    Adds a truncation marker if truncation occurs.
    """
    if not text:
        return ""

    encoder = _get_tokenizer(model_name)

    if not encoder:
        max_chars = int(max_tokens * settings.FALLBACK_CHARS_PER_TOKEN)
        logger.warning(
            f"truncate_text_by_tokens: Failed to get tokenizer for '{model_name}'. "
            f"Falling back to character-based truncation: {max_tokens} tokens -> ~{max_chars} chars."
        )
        if len(text) > max_chars:
            effective_max_chars = max_chars - len(truncation_marker)
            if effective_max_chars < 0:
                effective_max_chars = 0
            return text[:effective_max_chars] + truncation_marker
        return text

    tokens = encoder.encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return text

    marker_tokens_len = 0
    if truncation_marker:
        marker_tokens_len = len(
            encoder.encode(truncation_marker, allowed_special="all")
        )

    content_tokens_to_keep = max_tokens - marker_tokens_len
    effective_truncation_marker = truncation_marker

    if content_tokens_to_keep < 0:
        logger.debug(
            f"Truncation marker ('{truncation_marker}' -> {marker_tokens_len} tokens) is longer than max_tokens ({max_tokens}). Using empty marker."
        )
        content_tokens_to_keep = max_tokens
        effective_truncation_marker = ""

    truncated_content_tokens = tokens[:content_tokens_to_keep]

    if not truncated_content_tokens and max_tokens > 0 and tokens:
        logger.debug(
            "Truncated content to 0 tokens due to marker length. Attempting to"
            " keep 1 token of content."
        )
        truncated_content_tokens = tokens[:1]
        effective_truncation_marker = ""

    try:
        decoded_text = encoder.decode(truncated_content_tokens)
        return decoded_text + effective_truncation_marker
    except Exception as e:
        logger.error(
            f"Error decoding truncated tokens for model '{model_name}': {e}. Falling back to simpler char-based truncation.",
            exc_info=True,
        )
        avg_chars_per_token = (
            len(text) / len(tokens)
            if len(tokens) > 0
            else settings.FALLBACK_CHARS_PER_TOKEN
        )
        estimated_char_limit_for_content = int(
            content_tokens_to_keep * avg_chars_per_token
        )
        return text[:estimated_char_limit_for_content] + effective_truncation_marker


class LLMService:
    """Utility class for interacting with LLM and embedding endpoints."""

    def __init__(self, timeout: float = settings.HTTPX_TIMEOUT):
        # Use a single async client for all requests to reuse connections
        self._client = httpx.AsyncClient(timeout=timeout)
        # Add a semaphore to limit concurrent requests
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_CALLS)
        self.request_count = 0
        logger.info(
            f"LLMService initialized with a concurrency limit of {settings.MAX_CONCURRENT_LLM_CALLS}."
        )

    async def _backoff_delay(self, attempt: int) -> None:
        """Sleep for an exponentially increasing delay with jitter."""
        delay = settings.LLM_RETRY_DELAY_SECONDS * (2**attempt)
        jitter = random.uniform(0, delay / 2)
        await asyncio.sleep(delay + jitter)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _validate_embedding(
        self,
        embedding_list: list[float | int],
        expected_dim: int,
        dtype: np.dtype,
    ) -> np.ndarray | None:
        """Helper to validate and convert a list to a 1D numpy embedding."""
        try:
            embedding = np.array(embedding_list).astype(dtype)
            if embedding.ndim > 1:
                logger.warning(
                    f"Embedding from source had unexpected ndim > 1: {embedding.ndim}. Flattening."
                )
                embedding = embedding.flatten()
            if embedding.shape == (expected_dim,):
                logger.debug(
                    f"Embedding validated successfully. Shape: {embedding.shape}, Dtype: {embedding.dtype}"
                )
                return embedding
            logger.error(
                f"Embedding dimension mismatch: Expected ({expected_dim},), Got {embedding.shape}. Original list length: {len(embedding_list)}"
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to convert embedding list to numpy array: {e}")
        return None

    @alru_cache(maxsize=settings.EMBEDDING_CACHE_SIZE)
    async def async_get_embedding(self, text: str) -> np.ndarray | None:
        """
        Asynchronously retrieves an embedding for the given text from Ollama with retry logic.
        """
        async with self._semaphore:
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning(
                    "async_get_embedding: empty or invalid text provided. Returning None."
                )
                return None

            payload = {"model": settings.EMBEDDING_MODEL, "prompt": text.strip()}
            logger.debug(
                f"Async Embedding req to Ollama for model '{settings.EMBEDDING_MODEL}': '{text[:80].replace(chr(10), ' ')}...'"
            )

            last_exception: Exception | None = None
            for attempt in range(settings.LLM_RETRY_ATTEMPTS):
                api_response: httpx.Response | None = None
                try:
                    self.request_count += 1
                    api_response = await self._client.post(
                        f"{settings.OLLAMA_EMBED_URL}/api/embeddings", json=payload
                    )
                    api_response.raise_for_status()
                    data = api_response.json()

                    primary_key = "embedding"
                    if primary_key in data and isinstance(data[primary_key], list):
                        embedding = self._validate_embedding(
                            data[primary_key],
                            settings.EXPECTED_EMBEDDING_DIM,
                            settings.EMBEDDING_DTYPE,
                        )
                        if embedding is not None:
                            return embedding
                    else:
                        logger.warning(
                            f"Ollama (Attempt {attempt + 1}): Primary embedding key '{primary_key}' not found or not a list. Data: {data}"
                        )
                        for key, value in data.items():
                            if isinstance(value, list) and all(
                                isinstance(item, float | int) for item in value
                            ):
                                embedding = self._validate_embedding(
                                    value,
                                    settings.EXPECTED_EMBEDDING_DIM,
                                    settings.EMBEDDING_DTYPE,
                                )
                                if embedding is not None:
                                    logger.info(
                                        f"Ollama (Attempt {attempt + 1}): Found embedding using fallback key '{key}'."
                                    )
                                    return embedding

                    logger.error(
                        f"Ollama (Attempt {attempt + 1}): Embedding extraction failed. No suitable embedding list found in response: {data}"
                    )
                    last_exception = ValueError(
                        "No suitable embedding list found in Ollama response after parsing."
                    )

                except httpx.TimeoutException as e_timeout:
                    last_exception = e_timeout
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{settings.LLM_RETRY_ATTEMPTS}): Request timed out: {e_timeout}"
                    )
                except httpx.HTTPStatusError as e_status:
                    last_exception = e_status
                    error_message_detail = f"HTTP status {e_status.response.status_code}: {e_status}. Body: {e_status.response.text[:200]}"
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{settings.LLM_RETRY_ATTEMPTS}): {error_message_detail}"
                    )
                    if 400 <= e_status.response.status_code < 500:
                        logger.error(
                            f"Ollama Embedding: Client-side error {e_status.response.status_code}. Aborting retries."
                        )
                        break
                except httpx.RequestError as e_req:
                    last_exception = e_req
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{settings.LLM_RETRY_ATTEMPTS}): Request error: {e_req}"
                    )
                except json.JSONDecodeError as e_json:
                    last_exception = e_json
                    response_text_snippet = (
                        api_response.text[:200]
                        if api_response and hasattr(api_response, "text")
                        else "N/A"
                    )
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{settings.LLM_RETRY_ATTEMPTS}): Failed to decode JSON response: {e_json}. "
                        f"Response text: {response_text_snippet}"
                    )
                except Exception as e_exc:
                    last_exception = e_exc
                    logger.warning(
                        f"Ollama Embedding (Attempt {attempt + 1}/{settings.LLM_RETRY_ATTEMPTS}): Unexpected error: {e_exc}",
                        exc_info=True,
                    )

                if attempt < settings.LLM_RETRY_ATTEMPTS - 1:
                    delay = settings.LLM_RETRY_DELAY_SECONDS * (2**attempt)
                    retry_reason = (
                        type(last_exception).__name__
                        if last_exception
                        else "Unknown reason"
                    )
                    logger.info(
                        f"Ollama Embedding: Retrying in {delay:.2f} seconds due to: {retry_reason}."
                    )
                    await self._backoff_delay(attempt)
                else:
                    logger.error(
                        f"Ollama Embedding: All {settings.LLM_RETRY_ATTEMPTS} retry attempts failed. Last error: {last_exception}"
                    )
                    return None
            return None

    def _log_llm_usage(
        self,
        model_name: str,
        usage_data: dict[str, int] | None,
        async_mode: bool = False,
        streamed: bool = False,
    ) -> None:
        """Helper to log LLM token usage if available in the response."""
        prefix = "Async: " if async_mode else ""
        stream_prefix = "Streamed " if streamed else ""
        if usage_data and isinstance(usage_data, dict):
            logger.info(
                f"{prefix}{stream_prefix}LLM ('{model_name}') Usage - Prompt: {usage_data.get('prompt_tokens', 'N/A')} tk, "
                f"Comp: {usage_data.get('completion_tokens', 'N/A')} tk, Total: {usage_data.get('total_tokens', 'N/A')} tk"
            )
        else:
            logger.debug(
                f"{prefix}{stream_prefix}LLM ('{model_name}') response missing 'usage' information or 'usage' was not a dictionary."
            )

    async def _post_streaming(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[str, dict[str, int] | None, str]:
        """Send a streaming chat completion request."""
        payload["stream"] = True
        _tmp_fd, temp_path = tempfile.mkstemp(suffix=".llmstream.txt", text=True)
        os.close(_tmp_fd)
        accumulated = ""
        usage: dict[str, int] | None = None
        async with self._client.stream(
            "POST",
            f"{settings.OPENAI_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
        ) as response_stream:
            response_stream.raise_for_status()
            with open(temp_path, "w", encoding="utf-8") as tmp_write:
                async for line in response_stream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_json_str = line[len("data: ") :].strip()
                    if data_json_str == "[DONE]":
                        break
                    chunk_data = json.loads(data_json_str)
                    if chunk_data.get("choices"):
                        delta = chunk_data["choices"][0].get("delta", {})
                        content_piece = delta.get("content")
                        if content_piece:
                            accumulated += content_piece
                            tmp_write.write(content_piece)
                        if chunk_data["choices"][0].get("finish_reason") is not None:
                            usage = chunk_data.get("usage") or chunk_data.get(
                                "x_groq", {}
                            ).get("usage")
        return accumulated, usage, temp_path

    async def _post_non_streaming(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[str, dict[str, int] | None]:
        """Send a regular chat completion request."""
        payload["stream"] = False
        response = await self._client.post(
            f"{settings.OPENAI_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()
        raw_text = ""
        if data.get("choices") and len(data["choices"]) > 0:
            message = data["choices"][0].get("message")
            if message and message.get("content"):
                raw_text = message["content"]
        else:
            logger.error(
                f"Async LLM ('{payload['model']}') Invalid response structure - missing choices/content despite 200 OK: {data}"
            )
        return raw_text, data.get("usage")

    async def _call_model_with_retries(
        self,
        model_name: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        stream_to_disk: bool,
        auto_clean_response: bool,
    ) -> tuple[str, dict[str, int] | None, Exception | None, str | None]:
        """Try calling the model with retry logic."""
        last_exc: Exception | None = None
        final_text = ""
        usage: dict[str, int] | None = None
        temp_path: str | None = None
        for retry_attempt in range(settings.LLM_RETRY_ATTEMPTS):
            try:
                self.request_count += 1
                if stream_to_disk:
                    final_text, usage, temp_path = await self._post_streaming(
                        payload, headers
                    )
                else:
                    final_text, usage = await self._post_non_streaming(payload, headers)
                self._log_llm_usage(
                    model_name, usage, async_mode=True, streamed=stream_to_disk
                )
                if auto_clean_response:
                    final_text = self.clean_model_response(final_text)
                return final_text, usage, None, temp_path
            except Exception as exc:  # Consolidated error handling
                last_exc = exc
                logger.warning(
                    f"Async LLM ('{model_name}' Attempt {retry_attempt + 1}): {exc}",
                    exc_info=isinstance(exc, Exception),
                )
            finally:
                if (
                    stream_to_disk
                    and temp_path
                    and os.path.exists(temp_path)
                    and last_exc is not None
                ):
                    try:
                        logger.info(
                            f"Cleaning up temp stream file due to error: {temp_path}"
                        )
                        os.remove(temp_path)
                    except Exception as clean_err:
                        logger.error(
                            f"Error cleaning up temp file {temp_path} after failed LLM attempt: {clean_err}"
                        )
            if retry_attempt < settings.LLM_RETRY_ATTEMPTS - 1 and last_exc is not None:
                await self._backoff_delay(retry_attempt)
        return final_text, usage, last_exc, temp_path

    async def _async_call_llm(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        stream_to_disk: bool = False,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        auto_clean_response: bool = True,
    ) -> tuple[str, dict[str, int] | None]:
        """Call the LLM asynchronously with optional fallback and retries."""
        async with self._semaphore:
            if not model_name:
                logger.error("async_call_llm: model_name is required.")
                return "", None
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                logger.error("async_call_llm: empty or invalid prompt.")
                return "", None

            prompt_token_count = count_tokens(prompt, model_name)
            effective_max_output_tokens = (
                max_tokens if max_tokens is not None else settings.MAX_GENERATION_TOKENS
            )
            effective_temperature = (
                temperature if temperature is not None else settings.TEMPERATURE_DEFAULT
            )

            headers = {
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            current_model_to_try = model_name
            final_text_response = ""
            current_usage_data: dict[str, int] | None = None
            last_exc: Exception | None = None
            temp_file_path: str | None = None
            is_fallback_attempt = False

            for attempt_num_overall in range(2):
                if is_fallback_attempt:
                    if not allow_fallback or not settings.FALLBACK_GENERATION_MODEL:
                        logger.warning(
                            f"Primary model '{model_name}' failed. Fallback not allowed or no fallback model configured. Aborting call."
                        )
                        return final_text_response, current_usage_data
                    current_model_to_try = settings.FALLBACK_GENERATION_MODEL
                    logger.info(
                        f"Primary model '{model_name}' failed. Attempting fallback with '{current_model_to_try}'."
                    )
                    prompt_token_count = count_tokens(prompt, current_model_to_try)
                    current_usage_data = None

                token_param_name = _completion_token_param(settings.OPENAI_API_BASE)
                payload: dict[str, Any] = {
                    "model": current_model_to_try,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": effective_temperature,
                    "top_p": settings.LLM_TOP_P,
                    token_param_name: effective_max_output_tokens,
                }
                if frequency_penalty is not None:
                    payload["frequency_penalty"] = frequency_penalty
                if presence_penalty is not None:
                    payload["presence_penalty"] = presence_penalty

                logger.debug(
                    f"Async Calling LLM '{current_model_to_try}' (OverallAttempt: {attempt_num_overall + 1}). "
                    f"StreamToDisk: {stream_to_disk}. Prompt tokens (est.): {prompt_token_count}. "
                    f"Max output tokens: {effective_max_output_tokens}. Temp: {effective_temperature}, TopP: {settings.LLM_TOP_P}"
                )

                (
                    final_text_response,
                    current_usage_data,
                    last_exc,
                    temp_file_path,
                ) = await self._call_model_with_retries(
                    current_model_to_try,
                    payload,
                    headers,
                    stream_to_disk,
                    auto_clean_response,
                )

                if last_exc is None:
                    break

                is_fallback_attempt = True
                if (
                    attempt_num_overall == 0
                    and isinstance(last_exc, httpx.HTTPStatusError)
                    and last_exc.response
                    and 400 <= last_exc.response.status_code < 500
                    and last_exc.response.status_code != 429
                ):
                    logger.error(
                        f"Async LLM: Primary model '{model_name}' failed with non-429 client error. Checking fallback conditions."
                    )

            if stream_to_disk and temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_final_clean:
                    logger.error(
                        f"Error during final cleanup of temp file {temp_file_path}: {e_final_clean}"
                    )

            if last_exc is not None:
                logger.error(
                    f"Async LLM: Call failed for '{model_name}' after all primary and potential fallback attempts. Returning last captured text ('{final_text_response[:50]}...') and usage."
                )

            return final_text_response, current_usage_data

    @alru_cache(maxsize=settings.LLM_CALL_CACHE_SIZE)
    async def async_call_llm(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        stream_to_disk: bool = False,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        auto_clean_response: bool = True,
    ) -> tuple[str, dict[str, int] | None]:
        return await self._async_call_llm(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            allow_fallback=allow_fallback,
            stream_to_disk=stream_to_disk,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            auto_clean_response=auto_clean_response,
        )

    def clean_model_response(self, text: str) -> str:
        """Cleans common artifacts from LLM text responses, including content within <think> tags and normalizes newlines."""
        if not isinstance(text, str):
            logger.warning(
                f"clean_model_response received non-string input: {type(text)}. Returning empty string."
            )
            return ""

        original_length = len(text)
        cleaned_text = text

        text_before_think_removal = cleaned_text
        think_tags_to_remove = [
            "think",
            "thought",
            "thinking",
            "reasoning",
            "rationale",
            "meta",
            "reflection",
            "internal_monologue",
            "plan",
            "analysis",
            "no_think",
        ]
        for tag_name in think_tags_to_remove:
            block_pattern = re.compile(
                rf"<\s*{tag_name}\s*>.*?<\s*/\s*{tag_name}\s*>",
                flags=re.DOTALL | re.IGNORECASE,
            )
            cleaned_text = block_pattern.sub("", cleaned_text)

            self_closing_pattern = re.compile(
                rf"<\s*{tag_name}\s*/\s*>", flags=re.IGNORECASE
            )
            cleaned_text = self_closing_pattern.sub("", cleaned_text)

            lone_opening_pattern = re.compile(
                rf"<\s*{tag_name}\s*>", flags=re.IGNORECASE
            )
            cleaned_text = lone_opening_pattern.sub("", cleaned_text)

            lone_closing_pattern = re.compile(
                rf"<\s*/\s*{tag_name}\s*>", flags=re.IGNORECASE
            )
            cleaned_text = lone_closing_pattern.sub("", cleaned_text)

        if len(cleaned_text) < len(text_before_think_removal):
            logger.debug(
                f"clean_model_response: Removed content associated with <think>/similar tags. Length before: {len(text_before_think_removal)}, after: {len(cleaned_text)}."
            )

        cleaned_text = re.sub(
            r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```",
            r"\1",
            cleaned_text,
            flags=re.DOTALL,
        )

        cleaned_text = re.sub(
            r"^\s*Chapter \d+\s*[:\-—]?\s*(.*?)\s*$",
            r"\1",
            cleaned_text,
            flags=re.MULTILINE | re.IGNORECASE,
        ).strip()

        common_phrases_patterns = [
            r"^\s*(Okay,\s*)?(Sure,\s*)?(Here's|Here is)\s+(the|your)\s+[\w\s]+?:\s*",
            r"^\s*I've written the\s+[\w\s]+?\s+as requested:\s*",
            r"^\s*Certainly! Here is the text:\s*",
            r"^\s*(?:Output|Result|Response|Answer)\s*:\s*",
            r"^\s*\[SYSTEM OUTPUT\]\s*",
            r"^\s*USER:\s*.*?ASSISTANT:\s*",
            r"\s*Let me know if you (need|have) any(thing else| other questions| further revisions| adjustments)\b.*?\.?[^\w\n]*$",
            r"\s*I hope this (meets your expectations|helps|is what you were looking for)\b.*?\.?[^\w\n]*$",
            r"\s*Feel free to ask for (adjustments|anything else)\b.*?\.?[^\w\n]*$",
            r"\s*Is there anything else I can help you with\b.*?(\?|.)[^\w\n]*$",
            r"\s*\[END SYSTEM OUTPUT\]\s*$",
        ]
        for pattern_str in common_phrases_patterns:
            if pattern_str.startswith("^"):
                while True:
                    new_text = re.sub(
                        pattern_str,
                        "",
                        cleaned_text,
                        count=1,
                        flags=re.IGNORECASE | re.MULTILINE,
                    ).strip()
                    if new_text == cleaned_text:
                        break
                    cleaned_text = new_text
            else:
                cleaned_text = re.sub(
                    pattern_str,
                    "",
                    cleaned_text,
                    count=1,
                    flags=re.IGNORECASE | re.MULTILINE,
                ).strip()

        cleaned_text = re.sub(
            r"^\s*\*?\s*replace_with\s*:\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()

        final_text = cleaned_text.strip()
        final_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", final_text)
        final_text = re.sub(r"\n{3,}", "\n\n", final_text)

        if original_length > 0 and len(final_text) < original_length:
            reduction_percentage = (
                (original_length - len(final_text)) / original_length
            ) * 100
            if reduction_percentage > 0.5:
                logger.debug(
                    f"Cleaning reduced text length from {original_length} to {len(final_text)} ({reduction_percentage:.1f}% reduction)."
                )

        return final_text


# Instantiate the service for other modules to import and use
llm_service = LLMService()
