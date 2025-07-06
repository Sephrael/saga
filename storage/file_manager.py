# storage/file_manager.py
"""Utility class for asynchronous file operations."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from config import CHAPTER_LOGS_DIR, CHAPTERS_DIR, DEBUG_OUTPUTS_DIR


class FileManager:
    """Handle reading and writing chapter artifacts."""

    def __init__(
        self,
        chapters_dir: str = CHAPTERS_DIR,
        logs_dir: str = CHAPTER_LOGS_DIR,
        debug_dir: str = DEBUG_OUTPUTS_DIR,
    ) -> None:
        self.chapters_dir = chapters_dir
        self.logs_dir = logs_dir
        self.debug_dir = debug_dir
        os.makedirs(self.chapters_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    async def save_chapter_and_log(
        self, chapter_number: int, text: str, raw_llm_log: str
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._save_chapter_and_log_sync, chapter_number, text, raw_llm_log
        )

    def _save_chapter_and_log_sync(
        self, chapter_number: int, text: str, raw_llm_log: str
    ) -> None:
        chapter_path = os.path.join(
            self.chapters_dir, f"chapter_{chapter_number:04d}.txt"
        )
        log_path = os.path.join(
            self.logs_dir, f"chapter_{chapter_number:04d}_raw_llm_log.txt"
        )
        os.makedirs(os.path.dirname(chapter_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(raw_llm_log)

    async def save_debug_output(
        self, chapter_number: int, stage_description: str, content: Any
    ) -> None:
        if content is None:
            return
        content_str = str(content) if not isinstance(content, str) else content
        if not content_str.strip():
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._save_debug_output_sync,
            chapter_number,
            stage_description,
            content_str,
        )

    def _save_debug_output_sync(
        self, chapter_number: int, stage_description: str, content_str: str
    ) -> None:
        safe_stage_desc = "".join(
            c if c.isalnum() or c in ["_", "-"] else "_" for c in stage_description
        )
        file_name = f"chapter_{chapter_number:04d}_{safe_stage_desc}.txt"
        file_path = os.path.join(self.debug_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content_str)

    async def read_text(self, file_path: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_text_sync, file_path)

    def _read_text_sync(self, file_path: str) -> str:
        """Read the contents of ``file_path`` synchronously.

        Args:
            file_path: Path to the file to read.

        Returns:
            The full text of the file.
        """

        with open(file_path, encoding="utf-8") as f:
            return f.read()
