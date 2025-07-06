# tests/test_config_validators.py

import config
import pytest
from config import SagaSettings


def test_openai_key_placeholder_raises():
    with pytest.raises(ValueError):
        SagaSettings(OPENAI_API_KEY="nope")


def test_default_neo4j_password_warns(monkeypatch):
    warnings: list[str] = []

    def fake_warning(msg: str, **_kw: object) -> None:
        warnings.append(msg)

    monkeypatch.setattr(config.logger, "warning", fake_warning)
    SagaSettings(OPENAI_API_KEY="valid", NEO4J_PASSWORD="saga_password")
    assert any("NEO4J_PASSWORD" in msg for msg in warnings)
