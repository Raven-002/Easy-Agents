from abc import ABC, abstractmethod

from ollama import Client

class AiRunner(ABC):
    @abstractmethod
    def run(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaAiRunner(AiRunner):
    """Run AI model locally."""

    def __init__(self, client: Client, model: str):
        self.client = client
        self.model = model

    def run(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response

class LocalOllamaAiRunner(OllamaAiRunner):
    """Run AI model locally."""

    def __init__(self, model: str):
        super().__init__(Client(), model)


runner_types: dict[str, type[AiRunner]] = {"local_ollama": LocalOllamaAiRunner}

def create_ai_runner(type: str, *args, **kwargs) -> AiRunner:
    return runner_types.get(type)(*args, **kwargs)
