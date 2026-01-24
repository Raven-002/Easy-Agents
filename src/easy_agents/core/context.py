from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field

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
    refinement_metadata: dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def to_litellm_message(
        self,
    ) -> litellm.Message:
        raise NotImplementedError


class SystemMessage(ChatCompletionMessage[Literal["system"]]):
    role: Literal["system"] = "system"
    content: str

    def to_litellm_message(self) -> litellm.Message:
        return litellm.Message(role="system", content=self.content)


class UserMessage(ChatCompletionMessage[Literal["user"]]):
    role: Literal["user"] = "user"
    content: str
    name: str | None = None

    def to_litellm_message(self) -> litellm.Message:
        return litellm.Message(role="user", content=self.content)


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
        completion_message: litellm.Message, response_format: type[ST], assistant_name: str
    ) -> "AssistantMessage[ST]":
        content: ST
        if issubclass(response_format, BaseModel):
            if not completion_message.content:
                raise ValueError("Response format is a pydantic model, but no content was returned.")
            content = response_format.model_validate_json(completion_message.content)
        else:
            content = completion_message.content or ""  # type: ignore

        tool_calls: list[ToolCall] | None = None
        if completion_message.tool_calls:
            tool_calls = []
            tc: litellm.ChatCompletionMessageToolCall
            for tc in completion_message.tool_calls:
                if tc.type != "function":
                    raise ValueError(f"Unsupported tool call type: {tc.type}")
                if not tc.function:
                    raise ValueError("Tool call function is missing.")
                if not tc.function.name:
                    raise ValueError("Tool call function name is missing.")
                if not tc.function.arguments:
                    raise ValueError("Tool call function arguments are missing.")
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=ToolCallFunction(name=tc.function.name, arguments=tc.function.arguments),
                    )
                )

        return AssistantMessage(
            content=content,
            reasoning=getattr(completion_message, "reasoning_content", None),
            refusal=getattr(completion_message, "refusal", None),
            tool_calls=tool_calls,
            name=assistant_name,
        )

    def to_litellm_message(self) -> litellm.Message:
        content_str: str | None = None
        if isinstance(self.content, BaseModel):
            content_str = self.content.model_dump_json()
        elif isinstance(self.content, str):  # pyright: ignore [reportUnnecessaryIsInstance]
            content_str = self.content

        tool_calls = None
        if self.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]

        return litellm.Message(
            role="assistant",
            content=content_str,
            tool_calls=tool_calls,
        )


class ToolMessage(ChatCompletionMessage[Literal["tool"]]):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    name: str | None = None

    def to_litellm_message(self) -> litellm.Message:
        return litellm.Message(role="tool", content=self.content, tool_call_id=self.tool_call_id)


type AnyChatCompletionMessage = SystemMessage | UserMessage | AssistantMessage[Any] | ToolMessage


class Context:
    def __init__(self, messages: list[AnyChatCompletionMessage]) -> None:
        self.__raw_messages: list[AnyChatCompletionMessage] = messages
        self.__refined_messages: list[AnyChatCompletionMessage] = messages

    @property
    def messages(self) -> Sequence[AnyChatCompletionMessage]:
        return tuple(self.__refined_messages)

    @property
    def raw_messages(self) -> Sequence[AnyChatCompletionMessage]:
        return tuple(self.__raw_messages)

    def extend_messages(self, messages: Sequence[AnyChatCompletionMessage]) -> None:
        self.__refined_messages.extend(messages)
        self.__raw_messages.extend(messages)

    def append_message(self, message: AnyChatCompletionMessage) -> None:
        self.extend_messages([message])

    def override_with_refined_messages(self, messages: Sequence[AnyChatCompletionMessage]) -> None:
        self.__refined_messages = list(messages)

    @classmethod
    def simple(cls, prompt: str, system_prompt: str | None = None) -> "Context":
        messages_list: list[AnyChatCompletionMessage] = []
        if system_prompt:
            messages_list.append(SystemMessage(content=system_prompt))
        messages_list.append(UserMessage(content=prompt))
        return cls(messages=messages_list)
