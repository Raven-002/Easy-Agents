from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import NoneType
from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

__all__ = ["Tool", "RunContext"]

from .context import Context

type ParametersBaseType = BaseModel | str | None
type ResultsBaseType = BaseModel | str | None


@dataclass
class RunContext:
    deps: dict[str, Any]
    ctx: Context


type ToolRunFunctionWithContext[ParametersType: ParametersBaseType, ResultsType: ResultsBaseType] = Callable[
    [RunContext, ParametersType], Coroutine[Any, Any, ResultsType]
]

type ToolRunFunctionWithDepsAndContext[
    ParametersType: ParametersBaseType,
    ResultsType: ResultsBaseType,
    DepsType: Any,
] = Callable[[RunContext, DepsType, ParametersType], Coroutine[Any, Any, ResultsType]]

type ToolRunFunction[
    ParametersType: ParametersBaseType,
    ResultsType: ResultsBaseType,
    DepsType: Any,
] = (
    ToolRunFunctionWithContext[ParametersType, ResultsType]
    | ToolRunFunctionWithDepsAndContext[ParametersType, ResultsType, DepsType]
)


class Tool[ParametersType: ParametersBaseType, ResultsType: ResultsBaseType, DepsType]:
    def __init__(
        self,
        name: str,
        description: str,
        run: ToolRunFunction[ParametersType, ResultsType, DepsType],
        parameters_type: type[ParametersType] = NoneType,
        results_type: type[ResultsType] = NoneType,
        deps_type: type[DepsType] = NoneType,
        deps_entry: str | None = None,
    ) -> None:
        if deps_type is not NoneType and deps_entry is None:
            raise TypeError(f"Missing deps extractor for tool {name}.")
        if deps_type is NoneType and deps_entry is not None:
            raise TypeError(f"Missing deps type for tool {name}.")

        self._name = name
        self._description = description
        self._run = run
        self._parameters_type = parameters_type
        self._results_type = results_type
        self._deps_type = deps_type
        self._deps_entry = deps_entry

        if issubclass(self._parameters_type, BaseModel):
            self._parameters_shema = self._parameters_type.model_json_schema()
        elif self._parameters_type is str:
            self._parameters_shema = {"type": "string"}
        else:
            self._parameters_shema = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def verify_deps(self, deps: dict[str, Any]) -> None:
        if self._deps_entry is None:
            return
        if self._deps_entry not in deps:
            raise KeyError(f'Deps entry "{self._deps_entry}" for tool "{self.name}" not found in context.')
        if not isinstance(deps[self._deps_entry], self._deps_type):
            raise TypeError("Bad type returned from tool context extractor.")

    def _extract_deps(self, ctx: RunContext) -> DepsType:
        assert self._deps_type
        assert self._deps_entry

        self.verify_deps(ctx.deps)

        return ctx.deps[self._deps_entry]

    async def run(self, ctx: RunContext, arguments: str) -> ResultsType:
        if issubclass(self._parameters_type, BaseModel):
            parameters = self._parameters_type.model_validate_json(arguments, strict=True, extra="forbid")
        elif self._parameters_type is str:
            parameters = arguments
        else:
            parameters = None
        if self._deps_type is not NoneType:
            return await self._run(ctx, self._extract_deps(ctx), parameters)
        return await self._run(ctx, parameters)

    def get_json_schema(self) -> ChatCompletionFunctionToolParam:
        return ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self._parameters_shema,
                strict=True,
            ),
        )
