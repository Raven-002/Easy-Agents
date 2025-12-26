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


def create_function_tool[
    ParametersType: OptionalBaseModel,
    ResultsType: OptionalBaseModel,
    AppDepsType,
    DepsType,
](
    name: str,
    description: str,
    run: ToolRunFunction[ParametersType, ResultsType, DepsType],
    parameters_type: type[ParametersType] | None = None,
    results_type: type[ResultsType] | None = None,
    deps_extractor: Callable[[AppDepsType], DepsType] | None = None,
    deps_type: type[DepsType] | None = None,
    app_deps_type: type[AppDepsType] | None = None,
) -> Tool[AppDepsType]:
    has_deps_conditions: list[bool] = [
        deps_type is not None,
        deps_extractor is not None,
    ]
    if not (all(has_deps_conditions) or not any(has_deps_conditions)):
        raise TypeError(f"Not all conditions matched: {has_deps_conditions}")

    async def run_wrapper(ctx: RunContext[AppDepsType], parameters: ParametersType) -> ResultsType:
        if deps_type is not None:
            assert deps_extractor
            assert app_deps_type
            if isinstance(ctx.deps, app_deps_type):
                raise TypeError("Bad type returned from tool context extractor.")
            tool_deps = deps_extractor(ctx.deps)
            if isinstance(tool_deps, deps_type):
                raise TypeError("Bad type returned from tool context extractor.")
            return await run(ctx, tool_deps, parameters)  # type: ignore
        return await run(ctx, parameters)  # type: ignore

    # Manually inject the actual parameter type into the wrapper's annotations
    # so Pydantic AI knows how to deserialize the incoming JSON.
    if parameters_type:
        run_wrapper.__annotations__["parameters"] = parameters_type

    # Also helpful for the return type if you want the LLM to understand the output schema
    if results_type:
        run_wrapper.__annotations__["return"] = results_type

    return Tool[AppDepsType](function=run_wrapper, takes_ctx=True, name=name, description=description)
