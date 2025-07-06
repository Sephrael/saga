# tests/orchestration/test_token_accountant.py

from core.usage import TokenUsage
from orchestration.token_accountant import Stage, TokenAccountant


def test_record_usage_with_tokenusage():
    tracker = TokenAccountant()
    usage = TokenUsage(prompt_tokens=1, completion_tokens=4, total_tokens=5)
    tracker.record_usage(Stage.GENESIS_PHASE, usage)
    assert tracker.get_stage_total(Stage.GENESIS_PHASE) == 4
    assert tracker.total == 4


def test_record_usage_with_dict_and_accumulation():
    tracker = TokenAccountant()
    tracker.record_usage(Stage.DRAFTING, {"completion_tokens": 3})
    tracker.record_usage(Stage.EVALUATION, TokenUsage(0, 2, 2))
    tracker.record_usage(Stage.DRAFTING, {"completion_tokens": 7})

    assert tracker.get_stage_total(Stage.DRAFTING) == 10
    assert tracker.get_stage_total(Stage.EVALUATION) == 2
    assert tracker.total == 12
