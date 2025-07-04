# prompt_renderer.py
"""Utilities for rendering LLM prompts using Jinja2 templates."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from jinja2.utils import htmlsafe_json_dumps
from pydantic import BaseModel

PROMPTS_PATH = Path(__file__).parent / "prompts"
_env = Environment(loader=FileSystemLoader(PROMPTS_PATH), autoescape=False)


def _default_json_serializer(value: Any) -> Any:
    """Serialize pydantic models for JSON output."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


_env.policies["json.dumps_function"] = lambda obj, **kw: json.dumps(
    obj,
    default=_default_json_serializer,
    **kw,
)


def _tojson(value: Any, indent: int | None = None) -> str:
    """JSON filter that supports pydantic models."""
    dumps: Callable[..., str] = lambda obj, **kwargs: json.dumps(
        obj, default=_default_json_serializer, **kwargs
    )
    kwargs: dict[str, Any] = {}
    if indent is not None:
        kwargs["indent"] = indent
    return htmlsafe_json_dumps(value, dumps=dumps, **kwargs)


_env.filters["tojson"] = _tojson


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 template from the prompts directory."""
    template = _env.get_template(template_name)
    return template.render(**context)
