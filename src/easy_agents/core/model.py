from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import openai
from openai import APIError, LengthFinishReasonError, Stream
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionToolChoiceOptionParam
from openai.types.shared_params import ResponseFormatJSONSchema, ResponseFormatText
from openai.types.shared_params.response_format_json_schema import JSONSchema
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

    def create_openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.api_key, base_url=self.api_base)

    def is_available(self) -> bool:
        try:
            self.chat_completion([UserMessage(content="do not think. reply yes")], tools=[], token_limit=1)
            return True
        except LengthFinishReasonError:
            # Since it is limited to a single token, it is almost expected to fail.
            return True
        except APIError:
            # If there is an API, it is not available.
            return False
        except Exception as e:
            raise RuntimeError(f"Unexpected error while checking model availability: {e}") from e

    def chat_completion[T: BaseModel | str](
        self,
        messages: Iterable[AnyChatCompletionMessage],
        tools: list[ToolAny] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam = "auto",
        response_format: type[T] = str,  # type: ignore
        assistant_name: str = "",
        temperature: float = 0.0,
        token_limit: int = 0,
    ) -> AssistantResponse[T]:
        tools_param = [t.get_json_schema() for t in tools] if tools else []

        if issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            response_format_param: ResponseFormatJSONSchema | ResponseFormatText = ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name=schema.get("title", response_format.__name__),
                    description=schema.get("description", ""),
                    schema=schema,
                    strict=True,
                ),
            )
        else:
            assert response_format is str
            response_format_param = ResponseFormatText(type="text")

        client = self.create_openai_client()

        state = ChatCompletionStreamState(
            input_tools=openai.NOT_GIVEN,  # type: ignore
            response_format=openai.NOT_GIVEN,  # type: ignore
        )

        stream = client.chat.completions.create(
            model=self.model_name,
            messages=[m.to_openai_param() for m in messages],
            tools=tools_param,
            tool_choice=tool_choice,
            response_format=response_format_param,
            temperature=temperature,
            stream=True,
            max_tokens=token_limit if token_limit > 0 else None,
        )

        if not isinstance(stream, Stream):  # pyright: ignore [reportUnnecessaryIsInstance]
            raise TypeError(f"Expected Stream, got {type(stream)}")

        for chunk in stream:
            state.handle_chunk(chunk)

            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function and tool_call.function.name:
                        print(f"\nTool Call: {tool_call.function.name}")

        final = state.get_final_completion()
        msg = final.choices[0].message

        return AssistantResponse[T](
            message=AssistantMessage.from_openai_parsed_message(msg, response_format, assistant_name),
            finish_reason=final.choices[0].finish_reason,
        )
