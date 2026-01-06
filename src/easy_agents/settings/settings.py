from enum import StrEnum
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider


class AiExpertise(StrEnum):
    ORCHESTRATION = "orchestration"
    CODE_ANALYSIS = "code_analysis"
    CODE_WRITING = "code_writing"
    CONTEXT_SUMMARIZATION = "context_summarization"


class AiModel(BaseModel):
    model_name: str  # The actual model name in the api.
    api_base: str
    api_key: str | None = None

    def create_completion_model(self) -> OpenAIChatModel:
        return OpenAIChatModel(
            model_name=self.model_name,
            provider=LiteLLMProvider(api_base=self.api_base, api_key=(self.api_key if self.api_key else "")),
        )


class Settings(BaseModel):
    models: dict[str, AiModel]  # The string refers to an alias name for the model.
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

    def create_model_object(self, name: AiExpertise | str) -> OpenAIChatModel:
        if name in AiExpertise:
            model_name = self.model_choices[name]
        else:
            model_name = name

        if model_name not in self.models.keys():
            raise ValueError(f"Model {model_name} not found in models.")

        return self.models[model_name].create_completion_model()


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
