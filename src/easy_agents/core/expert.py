import re
from typing import Any

import yaml
from pydantic import BaseModel


class Expert(BaseModel):
    name: str
    description: str
    content: str

    @classmethod
    def from_skill_md(cls, skill_md_content: str) -> "Expert":
        """
        Parse a SKILL.md file with YAML frontmatter.

        Expected format:
        ---
        name: skill_name
        description: skill description
        ---

        Rest of the Markdown content...
        """
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, skill_md_content.strip(), re.DOTALL)

        if not match:
            raise ValueError("Invalid SKILL.md format. Expected YAML frontmatter enclosed in '---' delimiters.")

        yaml_content = match.group(1)
        markdown_content = match.group(2).strip()

        try:
            yaml_data: dict[str, Any] | Any = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in frontmatter: {yaml_content}") from e

        if not isinstance(yaml_data, dict):
            raise ValueError("YAML frontmatter must be a dictionary")

        name: str | Any | None = yaml_data.get("name", None)  # pyright: ignore [reportUnknownVariableType, reportUnknownMemberType]
        description: str | Any | None = yaml_data.get("description", None)  # pyright: ignore [reportUnknownVariableType, reportUnknownMemberType]

        if not name:
            raise ValueError("SKILL.md must contain 'name' in frontmatter")
        assert isinstance(name, str)

        if not description:
            raise ValueError("SKILL.md must contain 'description' in frontmatter")
        assert isinstance(description, str)

        return cls(name=str(name), description=str(description), content=markdown_content)
