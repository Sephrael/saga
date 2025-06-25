import prompt_renderer
from jinja2 import DictLoader, Environment


def test_render_prompt_with_custom_env(monkeypatch):
    env = Environment(
        loader=DictLoader({"greet.j2": "Hello {{ name }}"}), autoescape=False
    )
    monkeypatch.setattr(prompt_renderer, "_env", env)
    result = prompt_renderer.render_prompt("greet.j2", {"name": "Bob"})
    assert result == "Hello Bob"
