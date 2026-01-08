from collections.abc import Iterable
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCallParam
from pydantic import BaseModel

from .context import Context, ToolMessage
from .model import Model
from .tool import Tool

type AgentLoopInputType = BaseModel | None
type AgentLoopOutputType = BaseModel | None


class Agent[InputT: AgentLoopInputType, OutputT: AgentLoopOutputType, DepsT: Any]:
    def __init__(self, input_type: type[InputT], output_type: type[OutputT], deps_type: type[DepsT], model: Model):
        self._input_type = input_type
        self._output_type = output_type
        self._deps_type = deps_type
        self._model = model

    def _get_tools(self, deps: DepsT) -> list[Tool]:
        """Get the list of tools to use."""
        raise NotImplementedError

    def _generate_initial_context(self, input_args: InputT, deps: DepsT) -> Context:
        """Generate the initial context based on the input arguments."""
        raise NotImplementedError

    def _get_response_format_schema(self) -> str:
        """Get the schema of the response format."""
        raise NotImplementedError

    def _generate_final_output(self, ctx: Context) -> OutputT:
        """Generate a reply based on the current context and tools."""
        raise NotImplementedError

    def _handle_tools(
        self, tool_calls: Iterable[ChatCompletionMessageFunctionToolCallParam], tools: list[Tool], deps: DepsT
    ) -> list[ToolMessage]:
        """Handle tool calls and return a list of tool messages."""
        raise NotImplementedError

    def run(self, input_args: InputT, deps: DepsT) -> OutputT:
        """Run the core loop."""
        ctx = self._generate_initial_context(input_args, deps)
        tools = self._get_tools(deps)

        while True:
            reply = self._model.chat_completion(ctx.messages, tools)
            if not reply["tool_calls"]:
                break

            tool_messages = self._handle_tools(reply["tool_calls"], tools, deps)
            ctx.messages.extend(tool_messages)

        return self._generate_final_output(ctx)
