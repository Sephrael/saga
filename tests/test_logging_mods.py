# tests/test_logging_mods.py
import importlib
import logging
import logging as std_logging

import pytest
import reset_neo4j
from config import settings

import utils.logging as logging_utils


@pytest.mark.asyncio
async def test_reset_cancel_logs_info(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    root_logger = std_logging.getLogger()

    # Configure structlog before reloading module so logger uses logging
    logging_utils.structlog.configure(
        logger_factory=logging_utils.structlog.stdlib.LoggerFactory()
    )
    importlib.reload(reset_neo4j)

    logging_utils.setup_logging_nana()
    root_logger.addHandler(caplog.handler)

    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")
    result = await reset_neo4j.reset_neo4j_database_async(
        None, None, None, confirm=False
    )
    assert result is False
    assert any("Operation cancelled." in record.message for record in caplog.records)


def test_setup_logging_file_error(monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    root_logger = std_logging.getLogger()

    class Handlers(list):
        def clear(self):
            pass

    root_logger.handlers = Handlers([caplog.handler])

    logging_utils.structlog.configure(
        logger_factory=logging_utils.structlog.stdlib.LoggerFactory()
    )
    importlib.reload(logging_utils)

    def raise_handler(*_a, **_k):
        raise OSError("fail")

    monkeypatch.setattr(std_logging.handlers, "RotatingFileHandler", raise_handler)
    monkeypatch.setattr(settings, "LOG_FILE", "temp.log")

    logging_utils.setup_logging_nana()

    assert any(
        "Error setting up file logger" in record.message for record in caplog.records
    )
