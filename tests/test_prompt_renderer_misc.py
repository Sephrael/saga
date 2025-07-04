import prompt_renderer
from jinja2 import DictLoader, Environment
from pydantic import BaseModel


def test_render_prompt_with_custom_env(monkeypatch):
    env = Environment(
        loader=DictLoader({"greet.j2": "Hello {{ name }}"}), autoescape=False
    )
    monkeypatch.setattr(prompt_renderer, "_env", env)
    result = prompt_renderer.render_prompt("greet.j2", {"name": "Bob"})
    assert result == "Hello Bob"


class Person(BaseModel):
    name: str


def test_render_prompt_with_pydantic_object(monkeypatch):
    env = Environment(
        loader=DictLoader({"greet.j2": "Hello {{ person.name }}"}), autoescape=False
    )
    monkeypatch.setattr(prompt_renderer, "_env", env)
    result = prompt_renderer.render_prompt("greet.j2", {"person": Person(name="Alice")})
    assert result == "Hello Alice"


def test_tojson_with_pydantic_object(monkeypatch):
    env = Environment(
        loader=DictLoader({"obj.j2": "{{ person | tojson }}"}),
        autoescape=False,
    )
    env.filters["tojson"] = prompt_renderer._tojson
    monkeypatch.setattr(prompt_renderer, "_env", env)
    result = prompt_renderer.render_prompt("obj.j2", {"person": Person(name="Alice")})
    assert result == '{"name": "Alice"}'
