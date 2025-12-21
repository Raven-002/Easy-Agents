from typing import Any, TypedDict


# Define the shapes of the different message Context
class UserMessage(TypedDict):
    role: str  # "user"
    content: str


class AssistantMessage(TypedDict, total=False):
    role: str  # "assistant"
    content: str
    tool_calls: list[Any]


class ToolMessage(TypedDict):
    role: str  # "tool"
    content: str
    tool_call_id: str


# Use a Union to define the list type
MessageType = UserMessage | AssistantMessage | ToolMessage


class Context:
    def __init__(self) -> None:
        # Explicitly type the list to handle different dictionary shapes
        self._messages: list[MessageType] = []

    def add_user_message(self, content: str) -> None:
        self._messages.append(UserMessage(role="user", content=content))

    def add_assistant_message(self, content: str, tool_calls: list[Any] | None = None) -> None:
        # Construct the message; total=False allows us to omit tool_calls if empty
        msg: AssistantMessage = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._messages.append(msg)

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        # Tool messages require the specific ID to link back to the request
        self._messages.append(ToolMessage(role="tool", tool_call_id=tool_call_id, content=content))

    @property
    def messages(self) -> list[MessageType]:
        return self._messages
