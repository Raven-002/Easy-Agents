from typing import Any

from pydantic import BaseModel

from ...context import SystemMessage
from ...run_context import RunContext
from ...tool import Tool
from ..types import (
    ModelCompletionRequest,
    ModelCompletionResponse,
)
from .api_adapter import ModelApiAdapter


class ApiAdapterStructuredOutputAsTool(ModelApiAdapter):
    def __init__(self) -> None:
        self.is_expecting_structured_output = False

    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        if not issubclass(request.response_format, BaseModel):
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
        request.tools = [
            Tool(
                "final_output",
                "Give the final output as a structured response in this tool's arguments. The arguments are "
                f"of type: {request.response_format.__name__} (description: {request.response_format.__doc__})",
                final_output_fn,
                request.response_format,
            )
        ]
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


class ApiAdapterInjectSystemPrompt(ModelApiAdapter):
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def adjust_request(self, request: ModelCompletionRequest[Any]) -> None:
        request.messages = [SystemMessage(content=self.system_prompt)] + list(request.messages)

    def adjust_response(self, response: ModelCompletionResponse) -> None:
        return
