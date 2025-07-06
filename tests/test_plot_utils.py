from config import settings

from utils.plot import get_plot_point_info


def test_plot_point_info_span(monkeypatch):
    monkeypatch.setattr(settings, "PLOT_POINT_CHAPTER_SPAN", 2)
    outline = {"plot_points": ["pp1", "pp2", "pp3"]}
    text, idx = get_plot_point_info(outline, 1)
    assert text == "pp1" and idx == 0
    text, idx = get_plot_point_info(outline, 2)
    assert text == "pp1" and idx == 0
    text, idx = get_plot_point_info(outline, 3)
    assert text == "pp2" and idx == 1
