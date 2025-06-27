import argparse
import asyncio

import structlog
from orchestration.nana_orchestrator import NANA_Orchestrator
from utils.logging import setup_logging_nana

logger = structlog.get_logger(__name__)


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
                "Could not explicitly shutdown orchestrator from main: %s",
                e,
            )


if __name__ == "__main__":
    main()
