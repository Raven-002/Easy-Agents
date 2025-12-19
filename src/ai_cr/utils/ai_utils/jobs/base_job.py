from abc import ABC, abstractmethod
from dataclasses import dataclass

from diff_parser import Diff

from ...common import RecursiveStrDict
from ..ai_runner import AiRunner, AiRunnerExpertise


@dataclass
class CommonJobParams:
    diff: Diff


class BaseJob(ABC):
    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_expertise(cls) -> AiRunnerExpertise:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_extra_details_description(cls) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def run(self, ai_runner: AiRunner, common_params: CommonJobParams, extra_details: RecursiveStrDict) -> str:
        raise NotImplementedError
