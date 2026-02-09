from abc import ABC, abstractmethod
from typing import Any

from ..types import (
    ModelCompletionRequest,
    ModelCompletionResponse,
)


class ModelApiAdapter(ABC):
    @abstractmethod
    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        pass

    @abstractmethod
    def adjust_response(self, response: ModelCompletionResponse) -> None:
        pass
