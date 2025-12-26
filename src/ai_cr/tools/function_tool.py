from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

type OptionalBaseModel = BaseModel | None


type ToolRunFunctionWithContext[ParametersType: OptionalBaseModel, ResultsType: OptionalBaseModel] = Callable[
    [RunContext[Any], ParametersType], Coroutine[Any, Any, ResultsType]
]

type ToolRunFunctionWithDepsAndContext[
    ParametersType: OptionalBaseModel,
    ResultsType: OptionalBaseModel,
    DepsType: Any,
] = Callable[[RunContext[Any], DepsType, ParametersType], Coroutine[Any, Any, ResultsType]]

type ToolRunFunction[
    ParametersType: OptionalBaseModel,
    ResultsType: OptionalBaseModel,
    DepsType: Any,
] = (
    ToolRunFunctionWithContext[ParametersType, ResultsType]
    | ToolRunFunctionWithDepsAndContext[ParametersType, ResultsType, DepsType]
)


class FunctionTool[
    ParametersType: OptionalBaseModel,
    ResultsType: OptionalBaseModel,
    AppDepsType: Any,
    DepsType: Any,
]:
    def __init__(
        self,
        name: str,
        description: str,
        run: ToolRunFunction[ParametersType, ResultsType, DepsType],
        parameters_type: type[ParametersType] = None,
        results_type: type[ResultsType] = None,
        deps_extractor: Callable[[RunContext[AppDepsType]], DepsType] | None = None,
        deps_type: type[DepsType] = None,
    ) -> None:
        has_deps_conditions: list[bool] = [
            deps_type is not None,
            deps_extractor is not None,
        ]
        if not (all(has_deps_conditions) or not any(has_deps_conditions)):
            raise TypeError(f"Not all conditions matched: {has_deps_conditions}")

        self._name = name
        self._description = description
        self._run = run
        self._parameters_type = parameters_type
        self._results_type = results_type
        self._deps_extractor = deps_extractor
        self._deps_type = deps_type

    @property
    def need_deps(self) -> bool:
        # The matching of all the ways to get whether we have deps or not was done in init, so we can use the simple way
        # to check without verifying everything again.
        return self._deps_type is not None

    def to_tool(self) -> Tool[AppDepsType]:
        async def run_wrapper(ctx: RunContext[AppDepsType], parameters: ParametersType) -> ResultsType:
            if self.need_deps:
                tool_deps = self._deps_extractor(ctx)
                if not isinstance(tool_deps, self._deps_type):
                    raise TypeError("Bad type returned from tool context extractor.")
                return await self._run(ctx, tool_deps, parameters)
            return await self._run(ctx, parameters)

        # Manually inject the actual parameter type into the wrapper's annotations
        # so Pydantic AI knows how to deserialize the incoming JSON.
        if self._parameters_type:
            run_wrapper.__annotations__["parameters"] = self._parameters_type

        # Also helpful for the return type if you want the LLM to understand the output schema
        if self._results_type:
            run_wrapper.__annotations__["return"] = self._results_type

        return Tool[AppDepsType](function=run_wrapper, takes_ctx=True, name=self._name, description=self._description)
