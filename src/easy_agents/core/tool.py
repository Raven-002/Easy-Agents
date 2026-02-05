from collections.abc import Callable, Coroutine
from types import NoneType
from typing import Any

from pydantic import BaseModel

from .run_context import NoneToolDep, RunContext, ToolDependency, ToolDepsRegistry

type ParametersBaseType = BaseModel | str | None
type ResultsBaseType = BaseModel | str | None


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
    class StringParameter(BaseModel):
        value: str

    def __init__(
        self,
        name: str,
        description: str,
        run: ToolRunFunction[ParametersType, ResultsType, DepsType],
        parameters_type: type[ParametersType] = NoneType,  # type: ignore
        results_type: type[ResultsType] = NoneType,  # type: ignore
        deps_type: ToolDependency[DepsType] = NoneToolDep,  # type: ignore
    ) -> None:
        self._name = name
        self._description = description
        self._run = run
        self._parameters_type = parameters_type
        self._results_type = results_type
        self._deps_type = deps_type

        self._parameters_schema: dict[str, Any] = {}
        if issubclass(self._parameters_type, BaseModel):
            self._parameters_schema = self._parameters_type.model_json_schema()
        elif self._parameters_type is str:
            self._parameters_schema = self.StringParameter.model_json_schema()
        else:
            self._parameters_schema = {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def verify_deps(self, deps: ToolDepsRegistry) -> None:
        if self._deps_type is NoneToolDep:
            return
        if self._deps_type.key not in deps.deps_map:
            raise KeyError(f'Deps entry "{self._deps_type.key}" for tool "{self.name}" not found in context.')
        if not isinstance(deps.deps_map[self._deps_type.key], self._deps_type.value_type):
            raise TypeError("Bad type returned from tool context extractor.")

    def _extract_deps(self, ctx: RunContext) -> DepsType:
        assert self._deps_type is not NoneToolDep
        self.verify_deps(ctx.deps)
        deps = ctx.deps.deps_map[self._deps_type.key]
        if not isinstance(deps, self._deps_type.value_type):
            raise TypeError("Bad type returned from tool context extractor.")
        return deps

    async def run(self, ctx: RunContext, arguments: str) -> ResultsType:
        parameters: ParametersType
        if issubclass(self._parameters_type, BaseModel):
            parameters = self._parameters_type.model_validate_json(arguments, strict=True, extra="forbid")
        elif self._parameters_type is str:
            parameters_object = self.StringParameter.model_validate_json(arguments, strict=True, extra="forbid")
            assert isinstance(parameters_object.value, self._parameters_type)
            parameters = parameters_object.value  # pyright: ignore [reportAssignmentType]
        elif self._parameters_type is NoneType:
            parameters = None  # type: ignore
        else:
            raise TypeError(f"Unsupported parameters type: {self._parameters_type}")
        if self._deps_type is not NoneToolDep:
            return await self._run(ctx, self._extract_deps(ctx), parameters)  # type: ignore
        return await self._run(ctx, parameters)  # type: ignore

    def get_json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters_schema,
                "strict": True,
            },
        }


type ToolAny = Tool[Any, Any, Any]
