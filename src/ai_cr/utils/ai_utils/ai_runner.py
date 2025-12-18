from abc import ABC, abstractmethod

from ollama import Client
from rich.console import Console


class AiRunner(ABC):
    def run(self, prompt: str) -> str:
        Console().print(f"Calling AI model with prompt: {prompt}")
        answer = self._run(prompt)
        Console().print(f"AI model response: {answer}")
        return answer

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> "AiRunner":
        raise NotImplementedError

    @abstractmethod
    def _run(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaAiRunner(AiRunner):
    """Run AI model locally."""

    def __init__(self, client: Client, model: str):
        self.client = client
        self.model = model

    def _run(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response

    @classmethod
    def create(cls, *args, **kwargs) -> "OllamaAiRunner":
        return cls(*args, **kwargs)


class LocalOllamaAiRunner(OllamaAiRunner):
    """Run AI model locally."""

    def __init__(self, model: str):
        super().__init__(Client(), model)

    @classmethod
    def create(cls, *args, **kwargs) -> "LocalOllamaAiRunner":
        return cls(*args, **kwargs)


runner_types: dict[str, type[AiRunner]] = {"local_ollama": LocalOllamaAiRunner}


def create_ai_runner(runner_type: str, *args, **kwargs) -> AiRunner:
    return runner_types[runner_type].create(*args, **kwargs)
