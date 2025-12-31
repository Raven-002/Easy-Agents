from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool

from .function_tool import create_function_tool


class ThinkingParameters(BaseModel):
    message: str = Field(description="The thinking message.")


class ThinkingResults(BaseModel):
    message: str = Field(description="The thinking message.")


async def _run(_ctx: RunContext[Any], parameters: ThinkingParameters) -> ThinkingResults:
    return ThinkingResults(message=parameters.message)


thinking_tool: Tool[Any] = create_function_tool(
    name="thinking_tool",
    description="A scratchpad for thinking. Returns its input like an echo server. "
                "Use when there is a need for internal thinking. Use multiple times if needed.",
    run=_run,
    parameters_type=ThinkingParameters,
    results_type=ThinkingResults,
)
