from dataclasses import dataclass, field
from types import NoneType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import Context
    from .model import Model
    from .router import Router


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

    def __post_init__(self) -> None:
        if not isinstance(self.value, self.type.value_type):
            raise TypeError(
                f"Bad type for dependency {self.type.key}: expected {self.type.value_type}, got {type(self.value)}"
            )


@dataclass
class ToolDepsRegistry:
    list: list[ToolDepEntry[Any]]
    deps_map: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.deps_map = {d.type.key: d.value for d in self.list}

    @staticmethod
    def empty() -> "ToolDepsRegistry":
        return ToolDepsRegistry(list=[])

    @staticmethod
    def from_map(deps: dict[ToolDependency[Any], Any]) -> "ToolDepsRegistry":
        return ToolDepsRegistry([ToolDepEntry(type=k, value=v) for k, v in deps.items()])


@dataclass(frozen=True)
class RunContext:
    deps: ToolDepsRegistry
    ctx: Context
    router: Router
    main_model: Model
