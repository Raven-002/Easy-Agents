from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import litellm
from pydantic import BaseModel

from .context import AnyChatCompletionMessage, AssistantMessage, UserMessage
from .tool import ToolAny

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


class Model(BaseModel):
    api_base: str | None = None
    api_key: str | None = None
    model_provider: Literal["openai", "ollama_chat"]
    model_name: str
    description: str
    thinking: bool = False

    def model_post_init(self, context: Any, /) -> None:
        if self.model_provider == "openai":
            if self.api_base is None or self.api_key is None:
                raise ValueError("OpenAI API base and key must be provided.")
        elif self.model_provider == "ollama_chat":
            if self.api_base is not None or self.api_key is not None:
                raise ValueError("Ollama API base and key must not be provided.")

    async def is_available(self) -> bool:
        try:
            await self.chat_completion([UserMessage(content="do not think. reply yes")], tools=[], token_limit=1)
            return True
        except ModelTokenLimitExceededError:
            return True
        except (litellm.exceptions.APIError, litellm.exceptions.APIConnectionError):
            # If there is an API, it is not available.
            return False
        except Exception as e:
            raise RuntimeError(f"Unexpected error while checking model availability: {e}") from e

    @staticmethod
    def _handle_completion_stream(stream: litellm.CustomStreamWrapper) -> litellm.ModelResponse:
        chunks: list[litellm.ModelResponseStream] = []
        state: Literal["Start", "Reasoning", "Content"] = "Start"
        for chunk in stream:
            chunks.append(chunk)
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                assert isinstance(reasoning_content, str)
                if state == "Start":
                    print("Reasoning:")
                if state == "Content":
                    print("\nReasoning:")
                state = "Reasoning"
                print(reasoning_content, end="", flush=True)
            elif delta.content:  # pyright: ignore [reportUnknownMemberType]
                # noinspection PySimplifyBooleanCheck
                if state == "Reasoning":
                    print("\n\n", end="")
                state = "Content"
                print(delta.content, end="", flush=True)  # pyright: ignore
            elif delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function and tool_call.function.name:
                        print(f"\nTool Call: {tool_call.function.name}")
        print("\n\n", end="")

        final = litellm.stream_chunk_builder(chunks)  # pyright: ignore [reportUnknownMemberType]
        if not isinstance(final, litellm.ModelResponse):
            raise TypeError(f"Expected ModelResponse, got {type(final)}")
        return final

    async def _chat_completion[T: BaseModel | str](
        self,
        messages: Iterable[AnyChatCompletionMessage],
        tools: list[ToolAny] | None = None,
        tool_choice: ToolChoiceType = "auto",
        response_format: type[T] = str,  # type: ignore
        assistant_name: str = "",
        temperature: float = 0.0,
        token_limit: int = 0,
    ) -> AssistantResponse[T]:
        tools_param: list[dict[str, object]] | None = [t.get_json_schema() for t in tools] if tools else None

        response_format_param: type[BaseModel] | None
        if issubclass(response_format, BaseModel):
            response_format_param = response_format
        else:
            assert response_format is str
            response_format_param = None

        should_stream = False

        completion = await litellm.acompletion(  # pyright: ignore [reportUnknownMemberType]
            model=f"{self.model_provider}/{self.model_name}",
            api_base=self.api_base,
            api_key=self.api_key,
            messages=[m.to_litellm_message().model_dump(exclude_none=True) for m in messages],
            tools=tools_param,
            tool_choice=tool_choice if tools else None,
            response_format=response_format_param,
            temperature=temperature,
            stream=should_stream,
            max_tokens=token_limit if token_limit > 0 else None,
        )

        final: litellm.ModelResponse
        if should_stream:
            if not isinstance(completion, litellm.CustomStreamWrapper):  # pyright: ignore [reportUnnecessaryIsInstance]
                raise TypeError(f"Expected Stream, got {type(completion)}")
            final = self._handle_completion_stream(completion)
        else:
            if not isinstance(completion, litellm.ModelResponse):  # pyright: ignore [reportUnnecessaryIsInstance]
                raise TypeError(f"Expected Stream, got {type(completion)}")
            final = completion

        msg: litellm.Message = final.choices[0].message  # pyright: ignore
        assert isinstance(msg, litellm.Message)

        finish_reason: FinishReasonType = final.choices[0].finish_reason or "stop"  # pyright: ignore
        assert finish_reason in get_args(FinishReasonType.__value__)

        if finish_reason == "length":
            raise ModelTokenLimitExceededError(
                AssistantResponse[str](AssistantMessage.from_litellm_message(msg, str, assistant_name), finish_reason)
            )
        return AssistantResponse[T](
            message=AssistantMessage.from_litellm_message(msg, response_format, assistant_name),
            finish_reason=finish_reason,
        )

    async def chat_completion[T: BaseModel | str](
        self,
        messages: Iterable[AnyChatCompletionMessage],
        tools: list[ToolAny] | None = None,
        tool_choice: ToolChoiceType = "auto",
        response_format: type[T] = str,  # type: ignore
        assistant_name: str = "",
        temperature: float = 0.0,
        token_limit: int = 0,
    ) -> AssistantResponse[T]:
        assistant_response = await self._chat_completion(
            messages, tools, tool_choice, response_format, assistant_name, temperature, token_limit
        )
        print(assistant_response)
        return assistant_response
