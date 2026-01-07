from enum import StrEnum
from typing import Any

import yaml
from pydantic import BaseModel

from easy_agents.agent.model import Model


class AiExpertise(StrEnum):
    ORCHESTRATION = "orchestration"
    CODE_ANALYSIS = "code_analysis"
    CODE_WRITING = "code_writing"
    CONTEXT_SUMMARIZATION = "context_summarization"


class Settings(BaseModel):
    models: dict[str, Model]  # The string refers to an alias name for the model.
    model_choices: dict[AiExpertise, str]  # expertise: model_name

    def model_post_init(self, _context: Any) -> None:
        for model_id in self.models.keys():
            if model_id == "":
                raise ValueError("Model name cannot be empty.")

        def validate_model_name(model_name: str | None, expertise: AiExpertise) -> None:
            if model_name is None or model_name == "":
                raise ValueError(
                    f"Model name for {expertise} cannot be empty. "
                    "Use None to specify no model for this expertise, or use a valid model name."
                )
            if model_name not in self.models.keys():
                raise ValueError(f"Model {model_name} for expertise {expertise} not found in models.")

        for e in AiExpertise:
            validate_model_name(self.model_choices.get(e), e)


__settings: Settings | None = None


def get_settings() -> Settings:
    global __settings
    if __settings is None:
        raise RuntimeError("Settings not initialized")
    return __settings


def init_settings(settings: Settings) -> None:
    global __settings
    if __settings is not None:
        raise RuntimeError("Settings already initialized")
    __settings = settings


def load_settings_from_yaml(yaml_content: str) -> None:
    raw = yaml.safe_load(yaml_content)
    # Strict does not work with enums, so we disable it.
    settings = Settings.model_validate(raw, strict=False)
    init_settings(settings)


def reset_settings() -> None:
    """Ment to be used in tests only."""
    global __settings
    __settings = None
