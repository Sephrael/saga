from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

PROMPTS_PATH = Path(__file__).parent / "prompts"
_env = Environment(
    loader=FileSystemLoader(PROMPTS_PATH), autoescape=select_autoescape()
)


def render_prompt(template_name: str, context: Dict[str, Any]) -> str:
    """Render a Jinja2 template from the prompts directory."""
    template = _env.get_template(template_name)
    return template.render(**context)
