from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from easy_agents.agent.tool import BaseTool

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogParameters(BaseModel):
    log_level: LogLevel = Field(description="The level of the message.")
    message: str = Field(description="Message to log.")


async def _run(_ctx: RunContext[Any], parameters: LogParameters) -> None:
    print(f"[{parameters.log_level}] {parameters.message}", flush=True)


class LogTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="log_tool",
            description="Create a log to inform the user about progress.",
            run=_run,
            parameters_type=LogParameters,
        )
