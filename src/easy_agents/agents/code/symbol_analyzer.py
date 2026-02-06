from typing import Any

from pydantic import BaseModel, Field

from easy_agents.core import Agent, Context, ToolDepsRegistry
from easy_agents.refiners import memory_refiner_factory
from easy_agents.tools import find_tool, read_tool
from easy_agents.tools.deps.project_files_deps import ProjectFilesDeps


class SymbolAnalysisRequest(BaseModel):
    symbol_name: str


class SymbolAnalysis(BaseModel):
    symbol_name: str = Field(description="The name of the analyzed symbol.")
    type: str = Field(description="The type of the analyzed symbol.")
    brief: str = Field(description="A brief description of the analyzed symbol.")
    details: str = Field(description="Detailed information about the analyzed symbol and how it is used.")
    use_cases: list[str] = Field(description="A list of use cases for the analyzed symbol.")
    edge_cases: list[str] = Field(
        description="A list of potential edge cases for the analyzed symbol, and whether they are handled (and how)."
    )
    related_symbols: dict[str, str] = Field(
        description="A dictionary of related symbols. Key is the symbol name, value is a short description and how "
        "they relate."
    )


SYMBOL_ANALYZER_SYSTEM_PROMPT = f"""You are a senior developer. Your job is to analyze code for new developers.

# GUIDELINES
- Do not make up data, use tools to get it.
- Think of use cases for the analyzed symbol.
- Think of edge cases for the analyzed symbol.
- Describe the analyzed symbol in detail.
- Show step by step thinking process.
- If you think the steps, summerize them in your output. The thinking content will be erased between tool calls.
- Depending on the type, add extra information to the details:
  - For Functions, add parameters and return type.
  - For Classes, add external, protected, and private fields and methods.
  - For Variables/Constants, add type and scope.
  - For APIs, add api description and parameters.
  - For other symbols, add relevant information.

# OUTPUT
Your final result, once you are done analyzing the symbol, will be of the following format:
{SymbolAnalysis.model_json_schema()}
"""


def symbol_analyzer_context_creator(input_args: SymbolAnalysisRequest, deps: ToolDepsRegistry) -> Context:
    project_files: ProjectFilesDeps | None | Any = deps.deps_map.get("project_files")
    if project_files is None or not isinstance(project_files, ProjectFilesDeps):
        raise ValueError("Project files dependency is required for symbol analyzer.")
    return Context.simple(
        f'Analyze the symbol "{input_args.symbol_name}" in {project_files.project_root}',
        system_prompt=SYMBOL_ANALYZER_SYSTEM_PROMPT,
    )


symbol_analyzer = Agent[SymbolAnalysisRequest, SymbolAnalysis](
    context_factory=symbol_analyzer_context_creator,
    input_type=SymbolAnalysisRequest,
    output_type=SymbolAnalysis,
    refiners_factories=[memory_refiner_factory],
    tools=[read_tool, find_tool],
)
