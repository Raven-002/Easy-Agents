from typing import Any

import yaml
from pydantic import BaseModel

from ..utils.ai_utils import AiRunner, AiRunnerType, create_runner
from ..utils.common import JsonType


class AiModel(BaseModel):
    runner_type: AiRunnerType
    extra_args: JsonType

    def create_runner(self) -> AiRunner:
        return create_runner(self.runner_type, self.extra_args)


class Settings(BaseModel):
    models: dict[str, AiModel]
    orchestrator_model: str
    code_analysis_model: str
    code_review_model: str

    def model_post_init(self, _context: Any) -> None:
        for model_id in self.models.keys():
            if model_id == "":
                raise ValueError("Model name cannot be empty.")

        if self.orchestrator_model not in self.models:
            raise ValueError(f"Orchestrator model {self.orchestrator_model} not found in models.")

        if self.code_analysis_model not in self.models:
            raise ValueError(f"Code analysis model {self.code_analysis_model} not found in models.")

        if self.code_review_model not in self.models:
            raise ValueError(f"Code review model {self.code_review_model} not found in models.")

    def create_orchestrator_runner(self) -> AiRunner:
        return self.models[self.orchestrator_model].create_runner()

    def create_code_analysis_runner(self) -> AiRunner:
        return self.models[self.code_analysis_model].create_runner()

    def create_code_review_runner(self) -> AiRunner:
        return self.models[self.code_review_model].create_runner()


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
