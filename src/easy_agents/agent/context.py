from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionAssistantMessageParam as AssistantMessage,
)
from openai.types.chat import (
    ChatCompletionMessageParam as ChatCompletionMessage,
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam as SystemMessage,
)
from openai.types.chat import (
    ChatCompletionToolMessageParam as ToolMessage,
)
from openai.types.chat import (
    ChatCompletionUserMessageParam as UserMessage,
)

__all__ = [
    "Context",
    "ChatCompletionMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
]


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
