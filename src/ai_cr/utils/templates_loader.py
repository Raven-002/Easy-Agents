from typing import Any

from jinja2 import Environment, PackageLoader

jinja_env = Environment(
    loader=PackageLoader("ai_cr", "templates"),
)


def render_template(template_name: str, **context: Any) -> str:
    template = jinja_env.get_template(template_name)
    return template.render(**context)


__all__ = ["render_template"]
