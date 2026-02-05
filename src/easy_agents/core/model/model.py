from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import litellm
from pydantic import BaseModel

from ..context import AnyChatCompletionMessage, AssistantMessage, SystemMessage, UserMessage
from ..run_context import RunContext
from ..tool import Tool, ToolAny

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


@dataclass
class ModelCompletionRequest[T: BaseModel | str]:
    messages: list[AnyChatCompletionMessage]
    tools: list[ToolAny] | None = None
    tool_choice: ToolChoiceType | None = "auto"
    response_format: type[T] = str  # type: ignore[assignment]
    assistant_name: str = ""
    token_limit: int = 0


@dataclass
class ModelCompletionResponse:
    msg: litellm.Message
    finish_reason: FinishReasonType


class ModelApiAdapter(ABC):
    @abstractmethod
    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        pass

    @abstractmethod
    def adjust_response(self, response: ModelCompletionResponse) -> None:
        pass


class ApiAdapterStructuredOutputAsTool(ModelApiAdapter):
    def __init__(self) -> None:
        self.is_expecting_structured_output = False

    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        if request.response_format is str:
            return

        async def final_output_fn(_1: RunContext, _2: BaseModel) -> None:
            return

        request.messages = [
            SystemMessage(
                content="\n\nYou MUST provide your response by using the 'final_output' tool call, even if you do "
                "not have enough data to answer. The final_output tool MUST BE CALLED. If you do not have enough "
                "data, fill the fields with fake data to make it clear you are missing data if possible, but you "
                "must stick to the tool call.\n\n"
            )
        ] + list(request.messages)
        request.tools = [Tool("final_output", "Give the final output", final_output_fn, request.response_format)]
        request.tool_choice = "required"
        request.response_format = str
        self.is_expecting_structured_output = True

    def adjust_response(self, response: ModelCompletionResponse) -> None:
        if not self.is_expecting_structured_output:
            return

        if not response.msg.tool_calls or len(response.msg.tool_calls) != 1:
            raise ValueError(
                f"Expected tool calls in response with tool_call response format handling. got: {response.msg}"
            )

        if (
            not response.msg.tool_calls[0].function
            or not response.msg.tool_calls[0].function.arguments
            or response.msg.tool_calls[0].function.name != "final_output"
        ):
            raise ValueError(
                "Expected tool call with name final_output in response with tool_call response format handling. "
                f"got{response.msg}"
            )
        response.msg.content = response.msg.tool_calls[0].function.arguments
        response.msg.tool_calls = None
        response.finish_reason = "stop"
        self.is_expecting_structured_output = False


class ApiAdapterExtractXmlReasoningFromContent(ModelApiAdapter):
    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        return

    def adjust_response(self, response: ModelCompletionResponse) -> None:
        if response.msg.reasoning_content or not response.msg.content:
            # Reasoning content already set or no content - no thinking to extract
            return
        parts = response.msg.content.split("</think>")

        if len(parts) == 1:
            # No </think> tag found - no thinking generated
            return

        if len(parts) != 2:
            raise ValueError(f"Expected a max of one </think> tag in content: {response.msg}")

        response.msg.content = parts[1]
        response.msg.reasoning_content = parts[0].removeprefix("<think>")


class ApiAdapterSetRequiredToolsAsPartOfSystemPrompt(ModelApiAdapter):
    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        if request.tool_choice == "required":
            request.tool_choice = "auto"

    def adjust_response(self, response: ModelCompletionResponse) -> None:
        return


class ApiAdapterSetAutoToolsAsNone(ModelApiAdapter):
    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        if request.tool_choice == "auto":
            request.tool_choice = None

    def adjust_response(self, response: ModelCompletionResponse) -> None:
        return


class Model(BaseModel):
    api_base: str | None = None
    api_key: str | None = None
    model_provider: Literal["openai", "ollama_chat"]
    model_name: str
    description: str
    thinking: bool = False
    assume_available: bool = False
    temperature: float | None = None
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

    def _get_adapters_chain(self) -> list[ModelApiAdapter]:
        chain: list[ModelApiAdapter] = []
        if self.thinking_in_content:
            chain.append(ApiAdapterExtractXmlReasoningFromContent())
        match self.response_format_handling:
            case "tool_call":
                chain.append(ApiAdapterStructuredOutputAsTool())
            case "system_prompt":
                chain.append(ApiAdapterSetRequiredToolsAsPartOfSystemPrompt())
            case None:
                pass
        if self.override_auto_as_none:
            chain.append(ApiAdapterSetAutoToolsAsNone())
        if self.required_tools_in_system_prompt:
            chain.append(ApiAdapterSetRequiredToolsAsPartOfSystemPrompt())
        return chain

    @staticmethod
    def _completion_to_response(completion: litellm.ModelResponse | Any) -> ModelCompletionResponse:
        if not isinstance(completion, litellm.ModelResponse):  # pyright: ignore [reportUnnecessaryIsInstance]
            raise TypeError(f"Expected Stream, got {type(completion)}")
        final: litellm.ModelResponse = completion
        assert isinstance(final.choices[0], litellm.Choices)
        msg: litellm.Message | Any = final.choices[0].message
        assert isinstance(msg, litellm.Message)
        finish_reason: FinishReasonType | Any = final.choices[0].finish_reason or "stop"  # pyright: ignore
        assert finish_reason in get_args(FinishReasonType.__value__)
        return ModelCompletionResponse(msg, finish_reason)

    async def _chat_completion[T: BaseModel | str](self, request: ModelCompletionRequest[T]) -> AssistantResponse[T]:
        if issubclass(request.response_format, BaseModel) and request.tools:
            raise ValueError("Cannot specify response format and tools at the same time.")

        original_response_format = request.response_format

        adapters = self._get_adapters_chain()
        for adapter in adapters:
            adapter.adjust_request(request)

        completion = await litellm.acompletion(  # pyright: ignore [reportUnknownMemberType]
            model=f"{self.model_provider}/{self.model_name}",
            api_base=self.api_base,
            api_key=self.api_key,
            messages=[m.to_litellm_message().model_dump(exclude_none=True) for m in request.messages],
            tools=[t.get_json_schema() for t in request.tools] if request.tools else None,
            tool_choice=request.tool_choice if request.tools else None,
            response_format=request.response_format if issubclass(request.response_format, BaseModel) else None,
            temperature=self.temperature,
            stream=False,
            max_tokens=request.token_limit if request.token_limit > 0 else None,
        )

        response = self._completion_to_response(completion)
        for adapter in adapters[::-1]:
            adapter.adjust_response(response)

        if response.finish_reason == "length":
            raise ModelTokenLimitExceededError(
                AssistantResponse[str](
                    AssistantMessage.from_litellm_message(response.msg, str, request.assistant_name),
                    response.finish_reason,
                )
            )

        return AssistantResponse[T](
            message=AssistantMessage.from_litellm_message(
                response.msg, original_response_format, request.assistant_name
            ),
            finish_reason=response.finish_reason,
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
        request = ModelCompletionRequest[T](
            list(messages),
            tools,
            tool_choice,
            response_format,
            assistant_name,
            token_limit,
        )
        assistant_response = await self._chat_completion(request)
        print(assistant_response)
        return assistant_response  # type: ignore
