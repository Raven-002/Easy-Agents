from typing import Any, Literal

from agents import FunctionTool, RunContextWrapper
from pydantic import BaseModel, Field

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _Parameters(BaseModel):
    log_level: LogLevel = Field(..., description="The level of the message.")
    message: str = Field(..., description="Message to log.")


async def _run(_ctx: RunContextWrapper[Any], args: str) -> str:  # type: ignore
    params = _Parameters.model_validate_json(args)

    print(f"[{params.log_level}] {params.message}", flush=True)


log_tool = FunctionTool(
    name="log",
    description="Create a log to inform the user about progress.",
    params_json_schema=_Parameters.model_json_schema(),
    on_invoke_tool=_run,
)
