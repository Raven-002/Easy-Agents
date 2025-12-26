from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, RunContext, Tool
from pydantic_ai.models import Model

from ai_cr.tools import create_function_tool, find_tool, read_tool

from .code_context import CodeProjectContext


class _Parameters(BaseModel):
    symbol_name: str = Field("The plain string name of the symbol to look (Case sensitive).")


class CodeAnalysisResults(BaseModel):
    symbol_name: str = Field(description="Analyzed Symbol.")
    description: str = Field(description="Description of what the symbol is (Plain text or markdown).")
    type_description: str = Field(description="Description of what type the symbol is (Plain text or markdown).")
    usage: str = Field(description="Description of what is the usage of this symbol (Plain text or markdown).")
    files_it_exists_in: list[str] = Field(description="List of files the symbol appears in")


instructions_template = """You are "Symbol Analyzer Agent", an expert code analyzer, tasked with analyzing symbols in
code bases.

CODE_BASE:
You are working on a code based described as: {code_layout_json}.

GUIDELINES:
- Plan your steps before executing them.
- Find the symbol you are asked to.
- Understand the context in which the symbol is being used.
- Understand the symbol's documentation.
- Check if the documentation is correct. It may be wrong.
- Find edge cases for the symbol.
"""


def symbol_analyzer_instructions(ctx: RunContext[CodeProjectContext]) -> str:
    return instructions_template.format(code_layout_json=ctx.deps.model_dump_json())


def create_symbol_analyzer(model: Model) -> Agent[CodeProjectContext, CodeAnalysisResults]:
    return Agent[CodeProjectContext, CodeAnalysisResults](
        name="Code Symbol Analyzer",
        instructions=symbol_analyzer_instructions,
        model=model,
        deps_type=CodeProjectContext,
        output_type=NativeOutput(CodeAnalysisResults),
        tools=[find_tool, read_tool],
    )


def create_symbol_analyzer_tool[T](model: Model, deps_extractor: Callable[[T], CodeProjectContext]) -> Tool[T]:
    agent = create_symbol_analyzer(model)

    async def _run(_ctx: RunContext[Any], deps: CodeProjectContext, parameters: _Parameters) -> CodeAnalysisResults:
        result_string = str(await agent.run(f"Analyze the symbol: '{parameters.symbol_name}'.", deps=deps))
        return CodeAnalysisResults.model_validate_json(result_string, strict=True, extra="forbid")

    return create_function_tool(
        name="symbol_analyzer",
        description="Analyze a symbol in the project",
        run=_run,
        parameters_type=_Parameters,
        results_type=CodeAnalysisResults,
        deps_type=CodeProjectContext,
        deps_extractor=deps_extractor,
    )
