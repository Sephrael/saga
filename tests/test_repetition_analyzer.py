import pytest
from processing.repetition_analyzer import RepetitionAnalyzer


@pytest.mark.asyncio
async def test_repetition_analyzer_basic():
    text = "hello world " * 4
    analyzer = RepetitionAnalyzer(n=2, threshold=3)
    problems = await analyzer.analyze(text)
    assert problems
    assert problems[0]["issue_category"] == "repetition_and_redundancy"
