from dataclasses import dataclass

from easy_agents.core import ToolAny, ToolDependency


@dataclass
class ToolsRegistryDeps:
    available_tools: list[ToolAny]


tools_registry_deps_type = ToolDependency[ToolsRegistryDeps](key="available_tools", value_type=ToolsRegistryDeps)
