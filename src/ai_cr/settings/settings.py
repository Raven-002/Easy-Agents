from typing import Any

import yaml
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from pydantic import BaseModel


class AiModel(BaseModel):
    model_name: str  # The actual model name in the api.
    api_base: str
    api_key: str | None = None

    def create_completion_model(self) -> OpenAIChatCompletionsModel:
        return OpenAIChatCompletionsModel(
            model=self.model_name,
            openai_client=AsyncOpenAI(base_url=self.api_base, api_key=(self.api_key if self.api_key else "")),
        )


class Settings(BaseModel):
    models: dict[str, AiModel]  # The string refers to an alias name for the model.
    orchestrator_model_name: str
    code_analysis_model_name: str
    code_review_model_name: str

    def model_post_init(self, _context: Any) -> None:
        for model_id in self.models.keys():
            if model_id == "":
                raise ValueError("Model name cannot be empty.")

        def validate_model_name(model_name: str | None, category: str) -> None:
            if model_name is None:
                return
            if model_name == "":
                raise ValueError(
                    f"Model name for {category} cannot be empty. "
                    "Use None to specify no model for this category, or use a valid model name."
                )
            if model_name not in self.models.keys():
                raise ValueError(f"Model {model_name} for category {category} not found in models.")

        validate_model_name(self.orchestrator_model_name, "orchestrator")
        validate_model_name(self.code_analysis_model_name, "code analysis")
        validate_model_name(self.code_review_model_name, "code review")

    def create_model_object(self, model_name: str) -> OpenAIChatCompletionsModel:
        if model_name not in self.models.keys():
            raise ValueError(f"Model {model_name} not found in models.")
        return self.models[model_name].create_completion_model()

    @property
    def orchestrator_model(self) -> OpenAIChatCompletionsModel:
        if not self.orchestrator_model_name:
            raise ValueError("Orchestrator model not found in settings.")
        return self.create_model_object(self.orchestrator_model_name)

    @property
    def code_analysis_model(self) -> OpenAIChatCompletionsModel:
        if not self.code_analysis_model_name:
            raise ValueError("Code analysis model not found in settings.")
        return self.create_model_object(self.code_analysis_model_name)

    @property
    def code_review_model(self) -> OpenAIChatCompletionsModel:
        if not self.code_review_model_name:
            raise ValueError("Code review model not found in settings.")
        return self.create_model_object(self.code_review_model_name)


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
    settings = Settings.model_validate(raw, strict=True)
    init_settings(settings)


def reset_settings() -> None:
    """Ment to be used in tests only."""
    global __settings
    __settings = None
