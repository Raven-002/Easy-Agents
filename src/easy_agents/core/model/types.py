from dataclasses import dataclass
from typing import Literal

import litellm
from pydantic import BaseModel

from ..context import AnyChatCompletionMessage, AssistantMessage
from ..tool import ToolAny

type ToolChoiceType = Literal["auto", "none", "required"]
type FinishReasonType = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


@dataclass
class AssistantResponse[T: str | BaseModel]:
    message: AssistantMessage[T]
    finish_reason: FinishReasonType


class ModelTokenLimitExceededError(Exception):
    def __init__(self, partial_response: AssistantResponse[str]) -> None:
        super().__init__(f"Model token limit exceeded: {partial_response}")
        self.partial_response = partial_response


@dataclass
class ModelCompletionRequest[T: BaseModel | str]:
    messages: list[AnyChatCompletionMessage]
    tools: list[ToolAny] | None = None
    tool_choice: ToolChoiceType | None = "auto"
    response_format: type[T] = str  # type: ignore[assignment]
    assistant_name: str = ""
    token_limit: int = 0


@dataclass
class ModelCompletionResponse:
    msg: litellm.Message
    finish_reason: FinishReasonType
