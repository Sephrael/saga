import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import main
from orchestration import cli_runner


def test_main_invokes_shutdown(monkeypatch):
    orchestrator = SimpleNamespace(
        run_novel_generation_loop=AsyncMock(),
        run_ingestion_process=AsyncMock(),
        shutdown=AsyncMock(),
    )

    monkeypatch.setattr(cli_runner, "NANA_Orchestrator", lambda: orchestrator)
    monkeypatch.setattr(sys, "argv", ["prog"])

    main.main()

    orchestrator.shutdown.assert_awaited_once()
