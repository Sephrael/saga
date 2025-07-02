# orchestration/cli_runner.py
"""Command-line runner for the NANA orchestrator."""

from __future__ import annotations

import asyncio

import structlog
from utils.logging import setup_logging_nana

from orchestration.nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


async def _run(orchestrator: NANA_Orchestrator, ingest: str | None) -> None:
    if ingest:
        await orchestrator.run_ingestion_process(ingest)
    else:
        await orchestrator.run_novel_generation_loop()


def run(ingest: str | None) -> None:
    """Initialize the orchestrator and run the requested operation."""
    setup_logging_nana()
    orchestrator = NANA_Orchestrator()
    try:
        asyncio.run(_run(orchestrator, ingest))
    except KeyboardInterrupt:
        logger.info(
            "NANA Orchestrator shutting down gracefully due to KeyboardInterrupt..."
        )
    except Exception as main_err:  # pragma: no cover - entry point catch
        logger.critical(
            "NANA Orchestrator encountered an unhandled main exception: %s",
            main_err,
            exc_info=True,
        )
    finally:

        async def _shutdown() -> None:
            await orchestrator.shutdown()

        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                loop.create_task(_shutdown())
        except RuntimeError:
            asyncio.run(_shutdown())
        except Exception as e:
            logger.warning(
                "Could not explicitly shutdown orchestrator from CLI runner: %s",
                e,
            )
