from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from types import NoneType
from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from .context import Context

type ParametersBaseType = BaseModel | str | None
type ResultsBaseType = BaseModel | str | None


@dataclass(frozen=True)
class ToolDependency[T]:
    key: str
    value_type: type[T]

    def to_entry(self, value: T) -> "ToolDepEntry[T]":
        return ToolDepEntry(type=self, value=value)


NoneToolDep = ToolDependency[None](key="None", value_type=NoneType)


@dataclass(frozen=True)
class ToolDepEntry[T]:
    type: ToolDependency[T]
    value: T

    def __post_init__(self):
        if not isinstance(self.value, self.type.value_type):
            raise TypeError(
                f"Bad type for dependency {self.type.key}: expected {self.type.value_type}, got {type(self.value)}"
            )


@dataclass
class ToolDepsRegistry:
    list: list[ToolDepEntry[Any]]
    map: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.map: dict[str, Any] = {d.type.key: d.value for d in self.list}

    @staticmethod
    def empty() -> "ToolDepsRegistry":
        return ToolDepsRegistry(list=[])

    @staticmethod
    def from_map(deps: dict[ToolDependency[Any], Any]) -> "ToolDepsRegistry":
        return ToolDepsRegistry([ToolDepEntry(type=k, value=v) for k, v in deps.items()])


@dataclass
class RunContext:
    deps: ToolDepsRegistry
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
        deps_type: ToolDependency[DepsType] = NoneToolDep,
    ) -> None:
        self._name = name
        self._description = description
        self._run = run
        self._parameters_type = parameters_type
        self._results_type = results_type
        self._deps_type = deps_type

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

    def verify_deps(self, deps: ToolDepsRegistry) -> None:
        if self._deps_type is NoneToolDep:
            return
        if self._deps_type.key not in deps.map:
            raise KeyError(f'Deps entry "{self._deps_type.key}" for tool "{self.name}" not found in context.')
        if not isinstance(deps.map[self._deps_type.key], self._deps_type.value_type):
            raise TypeError("Bad type returned from tool context extractor.")

    def _extract_deps(self, ctx: RunContext) -> DepsType:
        assert self._deps_type is not NoneToolDep
        self.verify_deps(ctx.deps)
        return ctx.deps.map[self._deps_type.key]

    async def run(self, ctx: RunContext, arguments: str) -> ResultsType:
        if issubclass(self._parameters_type, BaseModel):
            parameters = self._parameters_type.model_validate_json(arguments, strict=True, extra="forbid")
        elif self._parameters_type is str:
            parameters = arguments
        else:
            parameters = None
        if self._deps_type is not NoneToolDep:
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
