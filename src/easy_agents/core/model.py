from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import litellm
from pydantic import BaseModel

from .context import AnyChatCompletionMessage, AssistantMessage, SystemMessage, UserMessage
from .run_context import RunContext
from .tool import Tool, ToolAny

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
    assume_available: bool = False
    temperature: float = 0.7
    override_auto_as_none: bool = False
    thinking_in_content: Literal["xml_think"] | None = None
    response_format_handling: Literal["system_prompt", "tool_call"] | None = None
    required_tools_in_system_prompt: bool = False

    def model_post_init(self, context: Any, /) -> None:
        if self.model_provider == "openai":
            if self.api_base is None or self.api_key is None:
                raise ValueError("OpenAI API base and key must be provided.")
        elif self.model_provider == "ollama_chat":
            if self.api_base is not None or self.api_key is not None:
                raise ValueError("Ollama API base and key must not be provided.")

    async def is_available(self) -> bool:
        if self.assume_available:
            return True

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

    def _fix_reasoning_in_content(self, msg: litellm.Message) -> litellm.Message:
        if not self.thinking or not self.thinking_in_content or getattr(msg, "reasoning_content", None):
            return msg
        if self.thinking_in_content == "xml_think":
            if not msg.content:
                raise ValueError("Expected content in message with thinking_in_content=xml_think, got empty content.")
            parts = msg.content.split("</think>")
            if 2 < len(parts):
                raise ValueError(f"Expected a max of one </think> tag in content: {msg}")
            if len(parts) == 1:
                return msg
            return msg.model_copy(update={"content": parts[1], "reasoning_content": parts[0].removeprefix("<think>")})
        raise ValueError(f"Unknown thinking_in_content: {self.thinking_in_content}")

    async def _chat_completion[T: BaseModel | str](
        self,
        messages: Iterable[AnyChatCompletionMessage],
        tools: list[ToolAny] | None = None,
        tool_choice: ToolChoiceType | None = None,
        response_format: type[T] = str,  # type: ignore
        assistant_name: str = "",
        token_limit: int = 0,
    ) -> AssistantResponse[T]:
        # Handle response format normalization
        response_format_param: type[BaseModel] | None
        if issubclass(response_format, BaseModel):
            response_format_param = response_format
            if tools:
                raise ValueError("Cannot specify response format and tools at the same time.")
        else:
            assert response_format is str
            response_format_param = None

        # Handle special response format cases
        if response_format_param is not None and self.response_format_handling == "system_prompt":
            messages = [
                SystemMessage(
                    content="\n\nYou MUST response with a valid json object. Your entire output must ONLY be a valid "
                    "json object. The json must be of the following json object schema(a description of the fields): "
                    f"{response_format_param.model_json_schema()}\n\n"
                )
            ] + list(messages)
            response_format_param = None
        elif issubclass(response_format, BaseModel) and self.response_format_handling == "tool_call":
            tool_choice = "required"
            response_format_param = None

            async def final_output_fn(_1: RunContext, _2: BaseModel) -> None:
                return

            messages = [
                SystemMessage(
                    content="\n\nYou MUST provide your response by using the 'final_output' tool call, even if you do "
                    "not have enough data to answer. The final_output tool MUST BE CALLED. If you do not have enough "
                    "data, fill the fields with fake data to make it clear you are missing data if possible, but you "
                    "must stick to the tool call.\n\n"
                )
            ] + list(messages)
            tools = [Tool("final_output", "Give the final output", final_output_fn, response_format)]

        # Handle tool choice
        if tool_choice == "required" and not tools:
            raise ValueError("Cannot use tool_choice=required without specifying tools.")

        if self.required_tools_in_system_prompt and tool_choice == "required":
            tool_choice = "auto"
            messages = [SystemMessage(content="\n\nYou MUST response with a valid tool call.\n\n")] + list(messages)

        if tool_choice in [None, "auto"]:
            tool_choice = None if self.override_auto_as_none else "auto"

        tools_param: list[dict[str, object]] | None = [t.get_json_schema() for t in tools] if tools else None

        # Complete
        should_stream = False
        completion = await litellm.acompletion(  # pyright: ignore [reportUnknownMemberType]
            model=f"{self.model_provider}/{self.model_name}",
            api_base=self.api_base,
            api_key=self.api_key,
            messages=[m.to_litellm_message().model_dump(exclude_none=True) for m in messages],
            tools=tools_param,
            tool_choice=tool_choice if tools else None,
            response_format=response_format_param,
            temperature=self.temperature,
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

        msg = self._fix_reasoning_in_content(msg)

        if finish_reason == "length":
            raise ModelTokenLimitExceededError(
                AssistantResponse[str](AssistantMessage.from_litellm_message(msg, str, assistant_name), finish_reason)
            )

        if issubclass(response_format, BaseModel) and self.response_format_handling == "tool_call":
            if not msg.tool_calls or len(msg.tool_calls) != 1:
                raise ValueError(f"Expected tool calls in response with tool_call response format handling. got: {msg}")
            if (
                not msg.tool_calls[0].function
                or not msg.tool_calls[0].function.arguments
                or msg.tool_calls[0].function.name != "final_output"
            ):
                raise ValueError(
                    "Expected tool call with name final_output in response with tool_call response format handling. "
                    f"got{msg}"
                )
            msg.content = msg.tool_calls[0].function.arguments
            msg.tool_calls = None
            finish_reason = "stop"

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
        token_limit: int = 0,
    ) -> AssistantResponse[T]:
        assistant_response = await self._chat_completion(
            messages, tools, tool_choice, response_format, assistant_name, token_limit
        )
        print(assistant_response)
        return assistant_response
