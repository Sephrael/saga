import logging

from orchestration.token_tracker import TokenTracker


def test_token_tracker_add_completion_tokens(caplog):
    caplog.set_level(logging.INFO)
    tracker = TokenTracker()
    tracker.add("draft", {"completion_tokens": 5})
    assert tracker.total == 5
    assert any("tokens from" in record.message.lower() for record in caplog.records)


def test_token_tracker_add_total_tokens_only(caplog):
    caplog.set_level(logging.INFO)
    tracker = TokenTracker()
    tracker.add("draft", {"total_tokens": 10})
    assert tracker.total == 0
    assert any("total tokens" in record.message.lower() for record in caplog.records)


def test_token_tracker_add_invalid_usage(caplog):
    caplog.set_level(logging.WARNING)
    tracker = TokenTracker()
    tracker.add("draft", {"other": 1})
    assert tracker.total == 0
    assert any("missing" in record.message.lower() for record in caplog.records)
