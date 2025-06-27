import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import main


def test_main_invokes_shutdown(monkeypatch):
    orchestrator = SimpleNamespace(
        run_novel_generation_loop=AsyncMock(),
        run_ingestion_process=AsyncMock(),
        shutdown=AsyncMock(),
    )

    monkeypatch.setattr(main, "NANA_Orchestrator", lambda: orchestrator)
    monkeypatch.setattr(sys, "argv", ["prog"])

    main.main()

    orchestrator.shutdown.assert_awaited_once()
