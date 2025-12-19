from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from ollama import Client

from ..logging_utils import dlog, status, vlog


class AiRunnerExpertise(Enum):
    CODE_UNDERSTANDING = "code_understanding"
    CODE_REVIEW = "code_review"
    ORCHESTRATION = "orchestration"


class AiRunner(ABC):
    def __init__(self, expertise: list[AiRunnerExpertise]):
        self.__expertise = expertise

    @property
    def expertise(self) -> list[AiRunnerExpertise]:
        return self.__expertise

    def run(self, prompt: str) -> str:
        vlog(f"Calling AI model with prompt: {prompt}")
        with status("Running AI model… Waiting for response"):
            answer = self._run(prompt)
        dlog(f"AI model response: {answer}")
        return answer

    @classmethod
    @abstractmethod
    def create(cls, expertise: list[AiRunnerExpertise], *args: Any, **kwargs: Any) -> "AiRunner":
        raise NotImplementedError

    @abstractmethod
    def _run(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaAiRunner(AiRunner):
    """Run AI model locally."""

    def __init__(self, expertise: list[AiRunnerExpertise], client: Client, model: str):
        super().__init__(expertise)
        self.client = client
        self.model = model

    def _run(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response

    @classmethod
    def create(cls, expertise: list[AiRunnerExpertise], *args: Any, **kwargs: Any) -> "OllamaAiRunner":
        return cls(expertise, *args, **kwargs)


class LocalOllamaAiRunner(OllamaAiRunner):
    """Run AI model locally."""

    def __init__(self, expertise: list[AiRunnerExpertise], model: str):
        super().__init__(expertise, Client(), model)

    @classmethod
    def create(cls, expertise: list[AiRunnerExpertise], *args: Any, **kwargs: Any) -> "LocalOllamaAiRunner":
        return cls(expertise, *args, **kwargs)


runner_types: dict[str, type[AiRunner]] = {"local_ollama": LocalOllamaAiRunner}


def create_ai_runner(runner_type: str, *args: Any, **kwargs: Any) -> AiRunner:
    return runner_types[runner_type].create(*args, **kwargs)
