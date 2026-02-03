from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .context import AnyChatCompletionMessage, AssistantMessage
from .router import Router

type MessagesEndsWithAssistantMessage[T: str | BaseModel] = tuple[
    *tuple[AnyChatCompletionMessage, ...], AssistantMessage[T]
]


def to_messages_endswith_assistant_message(
    messages: Sequence[AnyChatCompletionMessage],
) -> MessagesEndsWithAssistantMessage[str]:
    if not messages:
        raise ValueError("The message sequence is empty")

    messages_tuple = tuple(messages)
    last_message = messages_tuple[-1]

    if not isinstance(last_message, AssistantMessage):
        raise TypeError(f"Last message in context is not an assistant message: {last_message}")
    if not isinstance(last_message.content, str):
        raise TypeError(f"Last assistant message content is not a string: {last_message.content}")

    return *messages_tuple[:-1], last_message  # type: ignore


class ContextRefiner(ABC):
    @abstractmethod
    async def refine_new_messages(
        self,
        router: Router,
        raw_messages: Sequence[AnyChatCompletionMessage],
        refined_messages: Sequence[AnyChatCompletionMessage],
    ) -> Sequence[AnyChatCompletionMessage]:
        pass

    @abstractmethod
    async def refine_pre_tool_assistant_message(
        self,
        router: Router,
        raw_messages: MessagesEndsWithAssistantMessage[str],
        refined_messages: MessagesEndsWithAssistantMessage[str],
    ) -> AssistantMessage[str]:
        pass


class ContextRefinerFactory[T: ContextRefiner]:
    def __init__(self, refiner_type: type[T], *args: Any, **kwargs: Any) -> None:
        self._refiner_type = refiner_type
        self._args = args
        self._kwargs = kwargs

    def get_refiner(self) -> T:
        return self._refiner_type(*self._args, **self._kwargs)
