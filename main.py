import argparse
import asyncio
import logging

from core.db_manager import neo4j_manager
from orchestration.nana_orchestrator import NANA_Orchestrator, setup_logging_nana

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging_nana()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", default=None, help="Path to text file to ingest")
    args = parser.parse_args()

    orchestrator = NANA_Orchestrator()
    try:
        if args.ingest:
            asyncio.run(orchestrator.run_ingestion_process(args.ingest))
        else:
            asyncio.run(orchestrator.run_novel_generation_loop())
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
        if neo4j_manager.driver is not None:
            logger.info("Ensuring Neo4j driver is closed from main entry point.")

            async def _close_driver_main() -> None:
                await neo4j_manager.close()

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running() and not loop.is_closed():
                    asyncio.ensure_future(_close_driver_main())
                elif not loop.is_running() and not loop.is_closed():
                    loop.run_until_complete(_close_driver_main())
                else:
                    asyncio.run(_close_driver_main())
            except RuntimeError as e:
                logger.warning(
                    "Could not explicitly close driver from main (event loop might be closed or other issue): %s",
                    e,
                )


if __name__ == "__main__":
    main()
