# utils/logging.py

"""Logging helpers for the Saga system."""

from __future__ import annotations

import logging
import logging.handlers
import os

import structlog
from config import settings

_RichHandler: type[logging.Handler]

logger = structlog.get_logger(__name__)

try:
    import rich.logging as rich_logging

    _RichHandler = rich_logging.RichHandler

    RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when Rich is missing
    RICH_AVAILABLE = False

    class _FallbackRichHandler(logging.Handler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
            logging.getLogger(__name__).handle(record)

    _RichHandler = _FallbackRichHandler

RichHandler = _RichHandler


__all__ = ["setup_logging_nana"]


def setup_logging_nana() -> None:
    """Configure structlog and standard logging for NANA."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.render_to_log_kwargs,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(settings.LOG_LEVEL_STR)

    if settings.LOG_FILE:
        try:
            file_path = (
                settings.LOG_FILE
                if os.path.isabs(settings.LOG_FILE)
                else os.path.join(settings.BASE_OUTPUT_DIR, settings.LOG_FILE)
            )
            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                mode="a",
                encoding="utf-8",
            )
            file_formatter = logging.Formatter(
                settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:  # pragma: no cover - path issues
            logger.error("Error setting up file logger: %s", e)

    if RICH_AVAILABLE and settings.ENABLE_RICH_PROGRESS:
        console_handler = RichHandler(
            level=settings.LOG_LEVEL_STR,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
            show_time=True,
            show_level=True,
        )
        root_logger.addHandler(console_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            settings.LOG_FORMAT, datefmt=settings.LOG_DATE_FORMAT
        )
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)

    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    log = structlog.get_logger()
    log.info(
        "NANA Logging setup complete.",
        log_level=logging.getLevelName(settings.LOG_LEVEL_STR),
    )
