from typing import Literal

from easy_agents.utils.types.json_dict import JsonType

from .base_ai_runner import AiRunner
from .ollama_runners import LocalOllamaRunner
from .openai_runner import OpenAiRunner

AiRunnerType = Literal["local_ollama", "openai"]

runner_types: dict[AiRunnerType, type[AiRunner]] = {"local_ollama": LocalOllamaRunner, "openai": OpenAiRunner}


def create_runner(runner_type: AiRunnerType, runner_params: JsonType) -> AiRunner:
    if runner_type not in runner_types:
        raise ValueError(f"Invalid runner type: {runner_type}")
    return runner_types[runner_type].create_runner(runner_params)
