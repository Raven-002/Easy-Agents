import yaml
from pydantic import BaseModel

from ..utils.ai_utils.ai_runner import (
    AiRunner,
    AiRunnerExpertise,
    AiRunnerExtraArgs,
    ParsableAiExpertise,
    ParsableAiRunnerType,
    create_ai_runner,
)


class AiModel(BaseModel):
    runner_type: ParsableAiRunnerType
    expertise: list[ParsableAiExpertise]
    extra_args: AiRunnerExtraArgs

    def __post_init__(self) -> None:
        if not self.expertise:
            raise ValueError("Model must have at least one expertise.")

    def create_runner(self) -> AiRunner:
        return create_ai_runner(runner_type=self.runner_type, expertise=self.expertise, extra_args=self.extra_args)


class Settings(BaseModel):
    models: dict[str, AiModel]
    orchestrator_model: str
    code_analysis_model: str
    code_review_model: str

    def __post_init__(self) -> None:
        for model_id in self.models.keys():
            if model_id == "":
                raise ValueError("Model name cannot be empty.")

        if self.orchestrator_model not in self.models:
            raise ValueError(f"Orchestrator model {self.orchestrator_model} not found in models.")

        if AiRunnerExpertise.ORCHESTRATION not in self.models[self.orchestrator_model].expertise:
            raise ValueError(f"Orchestrator model {self.orchestrator_model} is missing the orchestration expertise.")

        if self.code_analysis_model not in self.models:
            raise ValueError(f"Code analysis model {self.code_analysis_model} not found in models.")

        if AiRunnerExpertise.CODE_UNDERSTANDING not in self.models[self.code_analysis_model].expertise:
            raise ValueError(
                f"Code analysis model {self.code_analysis_model} is missing the code understanding expertise."
            )

        if self.code_review_model not in self.models:
            raise ValueError(f"Code review model {self.code_review_model} not found in models.")

        if AiRunnerExpertise.CODE_REVIEW not in self.models[self.code_review_model].expertise:
            raise ValueError(f"Code review model {self.code_review_model} is missing the code review expertise.")

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
