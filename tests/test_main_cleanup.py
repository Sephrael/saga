import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import main
from core.llm_interface import llm_service


def test_main_invokes_llm_aclose(monkeypatch):
    orchestrator = SimpleNamespace(
        run_novel_generation_loop=AsyncMock(),
        run_ingestion_process=AsyncMock(),
    )

    monkeypatch.setattr(main, "NANA_Orchestrator", lambda: orchestrator)
    monkeypatch.setattr(sys, "argv", ["prog"])

    monkeypatch.setattr(main.neo4j_manager, "driver", object())
    monkeypatch.setattr(main.neo4j_manager, "close", AsyncMock())

    llm_close = AsyncMock()
    monkeypatch.setattr(llm_service, "aclose", llm_close)

    main.main()

    llm_close.assert_awaited_once()
