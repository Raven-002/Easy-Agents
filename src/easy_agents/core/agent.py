from collections.abc import Callable
from dataclasses import dataclass, field
from types import NoneType
from typing import Any

from pydantic import BaseModel

from .agent_loop import AgentLoopOutputType, run_agent_loop
from .context import Context
from .router import Router
from .tool import Tool

type AgentInputType = BaseModel | None | str
type AgentOutputType = AgentLoopOutputType
type ContextFactoryFunctionType[InputT: AgentInputType, DepsT: Any] = Callable[[InputT, DepsT], Context]


@dataclass()
class SimpleContextFactory:
    system_prompt: str | None = None

    def __call__(self, input_args: str, _deps: Any) -> Context:
        return Context.simple(input_args, system_prompt=self.system_prompt)


@dataclass
class Agent[InputT: AgentInputType, OutputT: AgentOutputType, DepsT: Any]:
    router: Router
    context_factory: ContextFactoryFunctionType[InputT, DepsT]
    input_type: type[InputT] = str
    output_type: type[OutputT] = str
    deps_type: type[DepsT] = NoneType
    tools: list[Tool] = field(default_factory=list)

    async def run(self, input_args: InputT, deps: DepsT) -> OutputT:
        ctx = self.context_factory(input_args, deps)
        return await run_agent_loop(self.router, ctx, self.output_type, self.tools, deps=deps)
