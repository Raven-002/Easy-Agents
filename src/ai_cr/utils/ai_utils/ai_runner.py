from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any

from ollama import Client
from pydantic import BeforeValidator

from ..logging_utils import dlog, status, vlog


class AiRunnerExpertise(Enum):
    CODE_UNDERSTANDING = "code_understanding"
    CODE_REVIEW = "code_review"
    ORCHESTRATION = "orchestration"


class AiRunnerType(Enum):
    LOCAL_OLLAMA = "local_ollama"


def parse_ai_expertise(v: Any) -> AiRunnerExpertise:
    if isinstance(v, AiRunnerExpertise):
        return v
    if isinstance(v, str):
        try:
            return AiRunnerExpertise(v)
        except ValueError:
            raise ValueError(f"Invalid AiRunnerExpertise: {v}") from None
    raise TypeError("AiRunnerExpertise must be str or AiRunnerExpertise")


def parse_ai_runner_type(v: Any) -> AiRunnerType:
    if isinstance(v, AiRunnerType):
        return v
    if isinstance(v, str):
        try:
            return AiRunnerType(v)
        except ValueError:
            raise ValueError(f"Invalid AiRunnerType: {v}") from None
    raise TypeError("AiRunnerType must be str or AiRunnerType")


ParsableAiRunnerType = Annotated[AiRunnerType, BeforeValidator(parse_ai_runner_type)]
ParsableAiExpertise = Annotated[AiRunnerExpertise, BeforeValidator(parse_ai_expertise)]


AiRunnerExtraArgs = dict[str, str]


class AiRunner(ABC):
    def __init__(self, expertise: list[AiRunnerExpertise]):
        self.__expertise = expertise

    @property
    def expertise(self) -> list[AiRunnerExpertise]:
        return self.__expertise

    def run(self, prompt: str, system_prompt: str | None = None) -> str:
        vlog(f"Calling AI model with system prompt: {system_prompt}")
        vlog(f"Calling AI model with prompt: {prompt}")
        with status("Running AI model… Waiting for response"):
            answer = self._run(prompt=prompt, system_prompt=system_prompt)
        dlog(f"AI model response: {answer}")
        return answer

    @classmethod
    @abstractmethod
    def create(cls, expertise: list[AiRunnerExpertise], extra_args: AiRunnerExtraArgs) -> "AiRunner":
        raise NotImplementedError

    @abstractmethod
    def _run(self, prompt: str, system_prompt: str | None = None) -> str:
        raise NotImplementedError


class OllamaAiRunner(AiRunner, ABC):
    """Run AI model locally."""

    def __init__(self, expertise: list[AiRunnerExpertise], client: Client, model: str):
        super().__init__(expertise)
        self.client = client
        self.model = model

    def _run(self, prompt: str, system_prompt: str | None = None) -> str:
        response = self.client.generate(model=self.model, prompt=prompt, system=system_prompt if system_prompt else "")
        return response.response


class LocalOllamaAiRunner(OllamaAiRunner):
    """Run AI model locally."""

    def __init__(self, expertise: list[AiRunnerExpertise], model: str):
        super().__init__(expertise, Client(), model)

    @classmethod
    def create(cls, expertise: list[AiRunnerExpertise], extra_args: AiRunnerExtraArgs) -> "LocalOllamaAiRunner":
        return cls(expertise, extra_args["model"])


runner_types: dict[AiRunnerType, type[AiRunner]] = {AiRunnerType.LOCAL_OLLAMA: LocalOllamaAiRunner}


def create_ai_runner(
    runner_type: AiRunnerType, expertise: list[AiRunnerExpertise], extra_args: AiRunnerExtraArgs
) -> AiRunner:
    return runner_types[runner_type].create(expertise, extra_args)
