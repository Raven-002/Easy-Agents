from agents import Agent, RunContextWrapper, Model, FunctionTool
from .code_context import CodeProjectContext, CodeDir


instructions_template = """You are "{agent_name}", an expert code analyzer, tasked with analyzing symbols in code bases.
You are working on a code based described as: {code_layout_json}.

You are tasked with understanding symbols in the code.

GUIDELINES:
- Plan your steps before executing them.
- Find the symbol you are asked to.
- Search for symbols with similar names and meaning.
- Understand the context in which the symbol is being used.
- Understand the symbol's documentation.
- Check if the documentation is correct. It may be wrong.
- Find edge cases for the symbol.

FINAL_OUTPUT:
- Provide a detailed description of the symbol.
- Include any conflict you found with its docs.
- Include any potential edge cases/bugs you found with the symbol usage.
- Include any caveat you found with the symbol or its usage.
"""


def symbol_analyzer_instructions(context: RunContextWrapper[CodeProjectContext],
                                 agent: Agent[CodeProjectContext]) -> str:
    return instructions_template.format(
        agent_name=agent.name,
        code_layout_json=CodeDir.get_code_layout(context),
    )


def create_symbol_analyzer(model: Model) -> Agent[CodeProjectContext]:
    return Agent[CodeProjectContext](
        name="Code Symbol Analyzer",
        instructions=symbol_analyzer_instructions,
        model=model,
    )


def create_symbol_analyzer_tool(model: Model) -> FunctionTool:
    agent = create_symbol_analyzer(model)
    return agent.as_tool(
        tool_name=agent.name,
        tool_description="Analyze a single symbol in the entire code in the project."
    )

