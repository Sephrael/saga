import pytest
from processing.repetition_analyzer import RepetitionAnalyzer
from processing.repetition_tracker import RepetitionTracker


@pytest.mark.asyncio
async def test_repetition_tracker_overuse(tmp_path):
    stats_file = tmp_path / "stats.json"
    tracker = RepetitionTracker(file_path=str(stats_file), n=2)
    tracker.update_from_text("alpha beta gamma alpha beta")
    analyzer = RepetitionAnalyzer(n=2, threshold=3, tracker=tracker, cross_threshold=1)
    problems = await analyzer.analyze("alpha beta")
    assert any("overused" in p.problem_description for p in problems)
