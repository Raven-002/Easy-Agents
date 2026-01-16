from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import litellm
from pydantic import BaseModel

__all__ = [
    "Context",
    "ChatCompletionMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolCall",
    "ToolCallFunction",
    "ToolMessage",
    "AnyChatCompletionMessage",
]


type RoleType = Literal["system", "user", "assistant", "tool"]


class ChatCompletionMessage[RoleT: RoleType](BaseModel, ABC):
    role: RoleT

    @abstractmethod
    def to_litellm_message(
        self,
    ) -> litellm.Message:
        raise NotImplementedError


class SystemMessage(ChatCompletionMessage[Literal["system"]]):
    role: Literal["system"] = "system"
    content: str
    name: str | None = None

    def to_litellm_message(self) -> litellm.Message:
        raise NotImplementedError  # TODO


class UserMessage(ChatCompletionMessage[Literal["user"]]):
    role: Literal["user"] = "user"
    content: str
    name: str | None = None

    def to_litellm_message(self) -> litellm.Message:
        raise NotImplementedError  # TODO


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage[T: BaseModel | str](ChatCompletionMessage[Literal["assistant"]]):
    role: Literal["assistant"] = "assistant"
    content: T
    reasoning: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None

    @staticmethod
    def from_litellm_message[ST: BaseModel | str](
        completion_message: litellm.Message[Any], response_format: type[ST], assistant_name: str
    ) -> "AssistantMessage[ST]":
        raise NotImplementedError  # TODO

    def to_litellm_message(self) -> litellm.Message:
        raise NotImplementedError  # TODO


class ToolMessage(ChatCompletionMessage[Literal["tool"]]):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    name: str | None = None

    def to_litellm_message(self) -> litellm.Message:
        raise NotImplementedError  # TODO


type AnyChatCompletionMessage = SystemMessage | UserMessage | AssistantMessage[Any] | ToolMessage


@dataclass
class Context:
    messages: list[AnyChatCompletionMessage]

    @classmethod
    def simple(cls, prompt: str, system_prompt: str | None = None) -> "Context":
        messages_list: list[AnyChatCompletionMessage] = []
        if system_prompt:
            messages_list.append(SystemMessage(content=system_prompt, name=""))
        messages_list.append(UserMessage(content=prompt, name=""))
        return cls(messages=messages_list)
