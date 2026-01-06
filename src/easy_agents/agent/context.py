from dataclasses import dataclass
from typing import Literal

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
]


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]


class SystemMessage(ChatCompletionMessage):
    role: Literal["system"] = "system"
    content: str
    name: str | None = None


class UserMessage(ChatCompletionMessage):
    role: Literal["user"] = "user"
    content: str
    name: str | None = None


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class AssistantMessage[T](ChatCompletionMessage):
    role: Literal["assistant"] = "assistant"
    content: T
    reasoning: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None


class ToolMessage(ChatCompletionMessage):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    name: str | None = None


@dataclass
class Context:
    messages: list[ChatCompletionMessage]

    @classmethod
    def simple(cls, prompt: str, system_prompt: str | None = None) -> "Context":
        messages_list: list[ChatCompletionMessage] = []
        if system_prompt:
            messages_list.append(SystemMessage(content=system_prompt, role="system", name=""))
        messages_list.append(UserMessage(content=prompt, role="user", name=""))
        return cls(messages=messages_list)
