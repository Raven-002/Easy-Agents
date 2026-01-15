from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    ParsedChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
    Function,
)
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
    def to_openai_param(
        self,
    ) -> (
        ChatCompletionDeveloperMessageParam
        | ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
        | ChatCompletionFunctionMessageParam
    ):
        raise NotImplementedError


class SystemMessage(ChatCompletionMessage[Literal["system"]]):
    role: Literal["system"] = "system"
    content: str
    name: str | None = None

    def to_openai_param(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(content=self.content, name=self.name or "", role="system")


class UserMessage(ChatCompletionMessage[Literal["user"]]):
    role: Literal["user"] = "user"
    content: str
    name: str | None = None

    def to_openai_param(self) -> ChatCompletionUserMessageParam:
        return ChatCompletionUserMessageParam(content=self.content, name=self.name or "", role="user")


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
    def from_openai_parsed_message[ST: BaseModel | str](
        completion_message: ParsedChatCompletionMessage[Any], response_format: type[ST], assistant_name: str
    ) -> "AssistantMessage[ST]":
        content: ST
        if issubclass(response_format, BaseModel):
            assert isinstance(completion_message.content, str)
            content = response_format.model_validate_json(completion_message.content, strict=True, extra="forbid")
        elif response_format is str:
            content = response_format(completion_message.content or "")
        else:
            raise TypeError(f"Unsupported response format: {response_format}")

        # Some apis return reasoning in a separate field, even though it's not part of the openai message.
        reasoning: str | None | Any = getattr(completion_message, "reasoning", None)
        if reasoning and not isinstance(reasoning, str):
            raise TypeError(f"Unexpected reasoning type: {type(reasoning)}")

        return AssistantMessage(
            name=assistant_name,
            content=content,
            reasoning=reasoning,
            tool_calls=[
                ToolCall(
                    id=t.id,
                    type=t.type,
                    function=ToolCallFunction(name=t.function.name, arguments=t.function.arguments),
                )
                for t in completion_message.tool_calls
            ]
            if completion_message.tool_calls
            else [],
        )

    def to_openai_param(self) -> ChatCompletionAssistantMessageParam:
        output_content: str
        if isinstance(self.content, str):
            output_content = self.content
        elif isinstance(self.content, BaseModel):  # pyright: ignore [reportUnnecessaryIsInstance]
            output_content = self.content.model_dump_json()
        else:
            raise TypeError(f"Unsupported content type: {type(self.content)}")

        content: str = f"<think>{self.reasoning}</think>{output_content}" if self.reasoning else output_content
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = [
            ChatCompletionMessageFunctionToolCallParam(
                id=t.id, type="function", function=Function(arguments=t.function.arguments, name=t.function.name)
            )
            for t in self.tool_calls or []
        ]
        return ChatCompletionAssistantMessageParam(content=content, role="assistant", tool_calls=tool_calls)


class ToolMessage(ChatCompletionMessage[Literal["tool"]]):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    name: str | None = None

    def to_openai_param(self) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(content=self.content, role="tool", tool_call_id=self.tool_call_id)


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
