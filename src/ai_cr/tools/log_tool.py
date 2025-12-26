from typing import Any, Literal

from mistralai.extra.run.context import RunContext
from pydantic import BaseModel, Field

from .function_tool import FunctionTool

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogParameters(BaseModel):
    log_level: LogLevel = Field(description="The level of the message.")
    message: str = Field(description="Message to log.")


async def _run(_ctx: RunContext[Any], parameters: LogParameters) -> None:
    print(f"[{parameters.log_level}] {parameters.message}", flush=True)


log_tool = FunctionTool(
    name="log_tool",
    description="Create a log to inform the user about progress.",
    run=_run,
    parameters_type=LogParameters,
).to_tool()
