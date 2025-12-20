from typing import Any

from .base_ai_runner import AiRunner


class OpenAiRunner(AiRunner):
    """Runner pre-configured for OpenAI style api."""

    def __init__(self, model_name: str, api_base: str, api_key: str) -> None:
        super().__init__(model_name=model_name, api_base=api_base, api_key=api_key)

    @classmethod
    def create_runner(cls, runner_params: dict[str, Any]) -> "OpenAiRunner":
        return cls(**runner_params)
