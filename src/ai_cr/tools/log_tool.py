from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool

from .function_tool import create_function_tool

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogParameters(BaseModel):
    log_level: LogLevel = Field(description="The level of the message.")
    message: str = Field(description="Message to log.")


async def _run(_ctx: RunContext[Any], parameters: LogParameters) -> None:
    print(f"[{parameters.log_level}] {parameters.message}", flush=True)


log_tool: Tool[Any] = create_function_tool(
    name="log_tool",
    description="Create a log to inform the user about progress.",
    run=_run,
    parameters_type=LogParameters,
)
