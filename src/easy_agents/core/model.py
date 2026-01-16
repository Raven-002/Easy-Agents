from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import litellm
from pydantic import BaseModel

from .context import AnyChatCompletionMessage, AssistantMessage, UserMessage
from .tool import ToolAny


@dataclass
class AssistantResponse[T: str | BaseModel]:
    message: AssistantMessage[T]
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class Model(BaseModel):
    api_base: str
    api_key: str
    model_name: str
    description: str
    thinking: bool = False

    def is_available(self) -> bool:
        try:
            self.chat_completion([UserMessage(content="do not think. reply yes")], tools=[], token_limit=1)
            return True
        except litellm.exceptions.APIError:
            # If there is an API, it is not available.
            return False
        except Exception as e:
            raise RuntimeError(f"Unexpected error while checking model availability: {e}") from e

    def chat_completion[T: BaseModel | str](
        self,
        messages: Iterable[AnyChatCompletionMessage],
        tools: list[ToolAny] | None = None,
        tool_choice: TODO_TOOL_CHOISE_LITERAL_TYPE = "auto",
        response_format: type[T] = str,  # type: ignore
        assistant_name: str = "",
        temperature: float = 0.0,
        token_limit: int = 0,
    ) -> AssistantResponse[T]:
        tools_param = [t.get_json_schema() for t in tools] if tools else []

        response_format_param: type[BaseModel] | None
        if issubclass(response_format, BaseModel):
            response_format_param = response_format
        else:
            assert response_format is str
            response_format_param = None

        stream = litellm.completion(
            model=self.model_name,
            messages=[m.to_litellm_message().model_dump(exclude_none=True) for m in messages],
            tools=tools_param,
            tool_choice=tool_choice,
            response_format=response_format_param,
            temperature=temperature,
            stream=True,
            max_tokens=token_limit if token_limit > 0 else None,
        )

        if not isinstance(stream, litellm.CustomStreamWrapper):  # pyright: ignore [reportUnnecessaryIsInstance]
            raise TypeError(f"Expected Stream, got {type(stream)}")

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function and tool_call.function.name:
                        print(f"\nTool Call: {tool_call.function.name}")

        final: litellm.ModelResponse = None  # TODO: get a final complete response
        msg: litellm.Message = final.choices[0].message

        return AssistantResponse[T](
            message=AssistantMessage.from_litellm_message(msg, response_format, assistant_name),
            finish_reason=final.choices[0].finish_reason,
        )
