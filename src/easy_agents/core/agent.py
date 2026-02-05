from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from .agent_loop import AgentLoopOutputType, run_agent_loop
from .context import Context
from .context_refiner import ContextRefiner, ContextRefinerFactory
from .router import Router
from .run_context import ToolDepsRegistry
from .tool import ToolAny

type AgentInputType = BaseModel | None | str
type AgentOutputType = AgentLoopOutputType
type ContextFactoryFunctionType[InputT: AgentInputType] = Callable[[InputT, ToolDepsRegistry], Context]


@dataclass()
class SimpleContextFactory:
    system_prompt: str | None = None

    def __call__(self, input_args: str, _deps: ToolDepsRegistry) -> Context:
        return Context.simple(input_args, system_prompt=self.system_prompt)


@dataclass
class Agent[InputT: AgentInputType, OutputT: AgentOutputType]:
    context_factory: ContextFactoryFunctionType[InputT]
    input_type: type[InputT] = str  # type: ignore
    output_type: type[OutputT] = str  # type: ignore
    refiners_factories: list[ContextRefinerFactory[ContextRefiner]] | None = None
    tools: list[ToolAny] = field(default_factory=list)

    def _verify_deps(self, deps: ToolDepsRegistry) -> None:
        for tool in self.tools:
            tool.verify_deps(deps)

    async def run(self, input_args: InputT, router: Router, deps: ToolDepsRegistry | None = None) -> OutputT:
        if deps is None:
            deps = ToolDepsRegistry.empty()
        self._verify_deps(deps)
        ctx = self.context_factory(input_args, deps)
        refiners: list[ContextRefiner] | None = None
        if self.refiners_factories:
            refiners = [refiner_factory.get_refiner() for refiner_factory in self.refiners_factories]
        return await run_agent_loop(
            router,
            ctx,
            self.output_type,
            refiners,
            self.tools,
            deps=deps,
        )
